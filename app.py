import os
import json
import base64
import hashlib
import secrets
import asyncio
import traceback
import threading
import time
import re
from datetime import datetime, timedelta, timezone
from functools import wraps, lru_cache
from io import BytesIO
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
from dataclasses import dataclass, asdict, field
from enum import Enum
import mimetypes
import random
import string
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import uuid
import queue

from flask import Flask, render_template_string, request, jsonify, Response, stream_with_context, session, make_response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from markupsafe import escape

from openai import OpenAI
import httpx
import redis
import jwt

from PIL import Image
import PyPDF2
import docx
import openpyxl
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import markdown
import pygments
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import HtmlFormatter

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config.update(
    MAX_CONTENT_LENGTH=500 * 1024 * 1024,
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
    JSON_SORT_KEYS=False,
    SEND_FILE_MAX_AGE_DEFAULT=31536000,
)

CORS(app, resources={r"/api/*": {"origins": "*"}})

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["2000 per hour"],
    storage_uri="memory://"
)

cache = Cache(app, config={'CACHE_TYPE': 'simple', 'CACHE_DEFAULT_TIMEOUT': 300})

executor = ThreadPoolExecutor(max_workers=20)

class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    TOOL = "tool"

class ChatMode(Enum):
    NORMAL = "normal"
    CREATIVE = "creative"
    PRECISE = "precise"
    BALANCED = "balanced"
    CODE = "code"
    RESEARCH = "research"
    EXPERT = "expert"
    TUTOR = "tutor"

@dataclass
class StreamToken:
    content: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Message:
    role: MessageRole
    content: str
    timestamp: datetime
    tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: List[Dict] = field(default_factory=list)
    stream_tokens: List[StreamToken] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "tokens": self.tokens,
            "metadata": self.metadata,
            "attachments": self.attachments,
            "stream_tokens": [{"content": st.content, "timestamp": st.timestamp} for st in self.stream_tokens]
        }

@dataclass
class ChatSession:
    id: str
    user_id: Optional[str]
    messages: List[Message]
    total_tokens: int
    created_at: datetime
    updated_at: datetime
    mode: ChatMode
    context_window: int
    metadata: Dict[str, Any]
    active_stream: Optional[queue.Queue] = None
    stream_controller: Optional[Any] = None
    
    def add_message(self, message: Message):
        self.messages.append(message)
        self.total_tokens += message.tokens
        self.updated_at = datetime.now(timezone.utc)
    
    def start_stream(self) -> queue.Queue:
        self.active_stream = queue.Queue()
        return self.active_stream
    
    def end_stream(self):
        if self.active_stream:
            self.active_stream.put(None)
            self.active_stream = None

@dataclass
class GeminiKey:
    key: str
    rate_limit: int = 60
    daily_limit: int = 50000
    usage_today: int = 0
    failures: int = 0
    last_used: Optional[datetime] = None
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
    performance_score: float = 1.0

class EnhancedSessionManager:
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self.user_sessions: Dict[str, List[str]] = defaultdict(list)
        self.session_lock = threading.Lock()
        self.max_sessions_per_user = 200
        self.session_ttl = timedelta(days=90)
        self.active_streams: Dict[str, queue.Queue] = {}
        
    def create_session(self, user_id: Optional[str] = None, mode: ChatMode = ChatMode.BALANCED) -> ChatSession:
        session_id = str(uuid.uuid4())
        session = ChatSession(
            id=session_id,
            user_id=user_id,
            messages=[],
            total_tokens=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            mode=mode,
            context_window=2000000,
            metadata={"version": "2.0", "features": ["streaming", "advanced_prompts"]}
        )
        
        with self.session_lock:
            self.sessions[session_id] = session
            if user_id:
                self.user_sessions[user_id].append(session_id)
                if len(self.user_sessions[user_id]) > self.max_sessions_per_user:
                    old_session = self.user_sessions[user_id].pop(0)
                    if old_session in self.sessions:
                        del self.sessions[old_session]
        
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        with self.session_lock:
            session = self.sessions.get(session_id)
            if session:
                if datetime.now(timezone.utc) - session.updated_at > self.session_ttl:
                    self.delete_session(session_id)
                    return None
            return session
    
    def update_session(self, session: ChatSession):
        with self.session_lock:
            self.sessions[session.id] = session
    
    def delete_session(self, session_id: str):
        with self.session_lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                if session.active_stream:
                    session.end_stream()
                if session.user_id and session_id in self.user_sessions[session.user_id]:
                    self.user_sessions[session.user_id].remove(session_id)
                del self.sessions[session_id]
    
    def get_user_sessions(self, user_id: str) -> List[ChatSession]:
        with self.session_lock:
            return [self.sessions[sid] for sid in self.user_sessions.get(user_id, []) 
                    if sid in self.sessions]

class GeminiKeyManager:
    def __init__(self):
        self.keys: Dict[str, GeminiKey] = {}
        self.key_lock = threading.Lock()
        self.load_keys()
        self.start_key_maintenance()
    
    def load_keys(self):
        keys_loaded = 0
        for i in range(1, 100):
            key = os.environ.get(f'GEMINI_API_KEY_{i}') or os.environ.get(f'GEMINI_KEY_{i}')
            if key:
                self.add_key(key)
                keys_loaded += 1
        
        default_key = os.environ.get('GEMINI_API_KEY')
        if default_key and keys_loaded == 0:
            self.add_key(default_key)
    
    def add_key(self, key: str):
        key_id = hashlib.sha256(key.encode()).hexdigest()[:16]
        self.keys[key_id] = GeminiKey(key=key)
    
    def get_best_key(self) -> Optional[GeminiKey]:
        with self.key_lock:
            available_keys = []
            
            for key_id, key in self.keys.items():
                if not key.is_active or key.failures > 10:
                    continue
                
                if key.last_used:
                    time_since_last = (datetime.now(timezone.utc) - key.last_used).total_seconds()
                    if time_since_last < (60 / key.rate_limit):
                        continue
                
                if key.usage_today >= key.daily_limit:
                    continue
                
                available_keys.append((key_id, key))
            
            if not available_keys:
                return None
            
            available_keys.sort(key=lambda x: (x[1].usage_today / x[1].daily_limit, -x[1].performance_score))
            
            selected_key = available_keys[0][1]
            selected_key.last_used = datetime.now(timezone.utc)
            selected_key.usage_today += 1
            
            return selected_key
    
    def mark_key_failure(self, key: GeminiKey):
        with self.key_lock:
            key.failures += 1
            key.performance_score *= 0.9
            if key.failures > 20:
                key.is_active = False
    
    def mark_key_success(self, key: GeminiKey, response_time: float = 1.0):
        with self.key_lock:
            key.failures = max(0, key.failures - 1)
            performance_factor = min(2.0, 1.0 / response_time) if response_time > 0 else 1.0
            key.performance_score = min(1.0, key.performance_score * 1.05 * performance_factor)
    
    def reset_daily_usage(self):
        with self.key_lock:
            for key in self.keys.values():
                if (datetime.now(timezone.utc) - key.last_reset).days >= 1:
                    key.usage_today = 0
                    key.last_reset = datetime.now(timezone.utc)
                    if key.failures > 0:
                        key.failures = max(0, key.failures - 5)
                    if not key.is_active and key.failures < 10:
                        key.is_active = True
                    key.performance_score = min(1.0, key.performance_score * 1.1)
    
    def start_key_maintenance(self):
        def maintenance_loop():
            while True:
                time.sleep(3600)
                self.reset_daily_usage()
        
        thread = threading.Thread(target=maintenance_loop, daemon=True)
        thread.start()

class StreamingGeminiClient:
    def __init__(self, key_manager: GeminiKeyManager):
        self.key_manager = key_manager
        self.clients_cache = {}
        self.client_lock = threading.Lock()
    
    def get_client(self) -> Tuple[Optional[OpenAI], Optional[GeminiKey]]:
        api_key_obj = self.key_manager.get_best_key()
        if not api_key_obj:
            return None, None
        
        client_id = hashlib.sha256(api_key_obj.key.encode()).hexdigest()
        
        with self.client_lock:
            if client_id not in self.clients_cache:
                http_client = httpx.Client(
                    trust_env=False, 
                    timeout=120.0,
                    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
                )
                client = OpenAI(
                    api_key=api_key_obj.key,
                    base_url="https://generativelanguage.googleapis.com/v1beta",
                    http_client=http_client
                )
                self.clients_cache[client_id] = client
            
            return self.clients_cache[client_id], api_key_obj

class AdvancedPromptEngineer:
    def __init__(self):
        self.templates = self._load_templates()
        self.enhancement_strategies = self._load_strategies()
        self.advanced_mode_active = False
    
    def _load_templates(self) -> Dict[ChatMode, str]:
        return {
            ChatMode.NORMAL: "You are Jack's AI Ultra, an advanced AI assistant with comprehensive capabilities. Provide helpful, accurate, and well-structured responses.",
            ChatMode.CREATIVE: "You are Jack's AI Ultra in creative mode. Think outside conventional boundaries, explore innovative solutions, use vivid and imaginative language. Be bold and original in your approach.",
            ChatMode.PRECISE: "You are Jack's AI Ultra in precise mode. Focus on exact details, technical accuracy, and comprehensive analysis. Use data, facts, and precise language. Be thorough and meticulous.",
            ChatMode.BALANCED: "You are Jack's AI Ultra. Balance creativity with accuracy, providing comprehensive yet engaging responses. Adapt your style to best serve the user's needs.",
            ChatMode.CODE: "You are Jack's AI Ultra, specialized in programming and software development. Provide clean, efficient, production-ready code with comprehensive implementations. Follow modern best practices and patterns. With Gemini's extended context, create complete, fully-functional solutions.",
            ChatMode.RESEARCH: "You are Jack's AI Ultra in research mode. Conduct thorough analysis, provide multiple perspectives, cite sources when possible, and explore topics with academic rigor and depth.",
            ChatMode.EXPERT: "You are Jack's AI Ultra in expert mode. Demonstrate deep domain knowledge, provide professional-level insights, and communicate with the authority of a subject matter expert.",
            ChatMode.TUTOR: "You are Jack's AI Ultra in tutor mode. Guide learning through clear explanations, examples, and step-by-step breakdowns. Adapt to the user's level and encourage understanding."
        }
    
    def _load_strategies(self) -> Dict[str, callable]:
        return {
            "clarity": self._enhance_clarity,
            "structure": self._enhance_structure,
            "context": self._add_context,
            "specificity": self._enhance_specificity,
            "examples": self._add_examples,
            "depth": self._enhance_depth,
            "precision": self._enhance_precision,
            "creativity": self._enhance_creativity
        }
    
    def toggle_advanced_mode(self, active: bool):
        self.advanced_mode_active = active
    
    def enhance_prompt(self, prompt: str, mode: ChatMode = ChatMode.BALANCED, 
                       strategies: List[str] = None, force_advanced: bool = False) -> str:
        if not strategies:
            strategies = self._select_strategies_for_mode(mode)
        
        enhanced = prompt
        
        if self.advanced_mode_active or force_advanced:
            enhanced = self._apply_advanced_enhancement(enhanced, mode)
        
        for strategy in strategies:
            if strategy in self.enhancement_strategies:
                enhanced = self.enhancement_strategies[strategy](enhanced)
        
        if mode == ChatMode.CODE and (self.advanced_mode_active or force_advanced):
            enhanced = self._apply_code_enhancement(enhanced)
        
        return enhanced
    
    def _select_strategies_for_mode(self, mode: ChatMode) -> List[str]:
        strategy_map = {
            ChatMode.NORMAL: ["clarity", "structure"],
            ChatMode.CREATIVE: ["creativity", "examples", "context"],
            ChatMode.PRECISE: ["precision", "specificity", "structure"],
            ChatMode.BALANCED: ["clarity", "structure", "context"],
            ChatMode.CODE: ["precision", "structure", "examples"],
            ChatMode.RESEARCH: ["depth", "structure", "context", "examples"],
            ChatMode.EXPERT: ["depth", "precision", "context"],
            ChatMode.TUTOR: ["clarity", "examples", "structure"]
        }
        return strategy_map.get(mode, ["clarity", "structure", "context"])
    
    def _apply_advanced_enhancement(self, prompt: str, mode: ChatMode) -> str:
        advanced_prefix = "\n\n[ADVANCED MODE ACTIVE] "
        
        mode_enhancements = {
            ChatMode.CODE: "Provide COMPLETE, PRODUCTION-READY implementations with all error handling, validation, documentation, tests, and best practices. No placeholders or shortcuts.",
            ChatMode.RESEARCH: "Conduct exhaustive analysis with multiple perspectives, comprehensive data examination, and scholarly depth.",
            ChatMode.EXPERT: "Demonstrate mastery-level expertise with nuanced insights, advanced techniques, and professional-grade analysis.",
            ChatMode.CREATIVE: "Push creative boundaries with unconventional approaches, innovative solutions, and imaginative exploration.",
            ChatMode.PRECISE: "Apply maximum precision with exact specifications, detailed measurements, and comprehensive accuracy.",
            ChatMode.TUTOR: "Provide complete educational coverage with multiple examples, exercises, and comprehensive explanations."
        }
        
        enhancement = mode_enhancements.get(mode, "Provide exceptionally detailed and comprehensive responses.")
        
        return prompt + advanced_prefix + enhancement
    
    def _apply_code_enhancement(self, prompt: str) -> str:
        code_patterns = ['code', 'program', 'script', 'function', 'class', 'implement', 
                        'create', 'build', 'develop', 'write', 'design', 'algorithm',
                        'app', 'application', 'system', 'api', 'website', 'service']
        
        if any(pattern in prompt.lower() for pattern in code_patterns):
            enhancement = "\n\nPRODUCTION REQUIREMENTS:"
            enhancement += "\n- Complete implementation with all features"
            enhancement += "\n- Comprehensive error handling and validation"
            enhancement += "\n- Full documentation and inline comments"
            enhancement += "\n- Unit tests and integration tests"
            enhancement += "\n- Performance optimization"
            enhancement += "\n- Security best practices"
            enhancement += "\n- Scalable architecture"
            enhancement += "\n- Modern design patterns"
            
            return prompt + enhancement
        
        return prompt
    
    def _enhance_clarity(self, prompt: str) -> str:
        replacements = {
            r'\bit\b': 'the specified item',
            r'\bthis\b': 'the current topic',
            r'\bthat\b': 'the mentioned subject',
            r'\bthing\b': 'the specific element',
            r'\bstuff\b': 'the relevant materials'
        }
        
        enhanced = prompt
        for pattern, replacement in replacements.items():
            if len(re.findall(pattern, enhanced, re.IGNORECASE)) == 1:
                enhanced = re.sub(pattern, replacement, enhanced, flags=re.IGNORECASE)
        
        return enhanced
    
    def _enhance_structure(self, prompt: str) -> str:
        if '?' not in prompt and '.' not in prompt:
            prompt += '.'
        
        lines = [line.strip() for line in prompt.split('\n') if line.strip()]
        
        if len(lines) > 3:
            structured = []
            for i, line in enumerate(lines, 1):
                if not line.startswith(('•', '-', '*', str(i))):
                    structured.append(f"{i}. {line}")
                else:
                    structured.append(line)
            return '\n'.join(structured)
        
        return prompt
    
    def _add_context(self, prompt: str) -> str:
        context_triggers = {
            'explain': 'Provide a comprehensive explanation with clear examples and practical applications.',
            'how': 'Include detailed step-by-step instructions with rationale for each step.',
            'why': 'Provide thorough reasoning with evidence and multiple perspectives.',
            'compare': 'Create a detailed comparison with advantages, disadvantages, and use cases.',
            'analyze': 'Conduct thorough analysis examining all aspects and implications.',
            'create': 'Develop a complete solution with full implementation details.',
            'design': 'Provide comprehensive design with all specifications and considerations.'
        }
        
        prompt_lower = prompt.lower()
        for trigger, context in context_triggers.items():
            if trigger in prompt_lower and context.lower() not in prompt_lower:
                prompt += f"\n\n{context}"
                break
        
        return prompt
    
    def _enhance_specificity(self, prompt: str) -> str:
        vague_terms = {
            'good': 'effective, efficient, and high-quality',
            'bad': 'ineffective, problematic, or suboptimal',
            'nice': 'well-designed, user-friendly, and aesthetically pleasing',
            'okay': 'acceptable and functional',
            'fine': 'satisfactory and appropriate',
            'better': 'more effective and optimized',
            'best': 'optimal and most efficient'
        }
        
        enhanced = prompt
        for vague, specific in vague_terms.items():
            enhanced = re.sub(r'\b' + vague + r'\b', specific, enhanced, flags=re.IGNORECASE)
        
        return enhanced
    
    def _add_examples(self, prompt: str) -> str:
        if 'example' not in prompt.lower():
            prompt += "\n\nInclude comprehensive examples demonstrating various use cases and scenarios."
        return prompt
    
    def _enhance_depth(self, prompt: str) -> str:
        depth_keywords = ['analyze', 'explore', 'investigate', 'examine', 'study']
        if any(keyword in prompt.lower() for keyword in depth_keywords):
            prompt += "\n\nProvide in-depth analysis covering historical context, current state, future implications, and interdisciplinary connections."
        return prompt
    
    def _enhance_precision(self, prompt: str) -> str:
        precision_keywords = ['calculate', 'measure', 'quantify', 'determine', 'specify']
        if any(keyword in prompt.lower() for keyword in precision_keywords):
            prompt += "\n\nEnsure maximum precision with exact values, detailed calculations, and comprehensive specifications."
        return prompt
    
    def _enhance_creativity(self, prompt: str) -> str:
        creative_keywords = ['imagine', 'create', 'design', 'invent', 'brainstorm']
        if any(keyword in prompt.lower() for keyword in creative_keywords):
            prompt += "\n\nExplore unconventional approaches, innovative solutions, and creative possibilities beyond traditional boundaries."
        return prompt
    
    def create_system_prompt(self, mode: ChatMode, custom_instructions: str = "") -> str:
        base_prompt = self.templates.get(mode, self.templates[ChatMode.BALANCED])
        
        if self.advanced_mode_active:
            base_prompt += "\n\n⚡ ADVANCED MODE: You are operating at maximum capability. Provide exceptionally detailed, comprehensive, and professional-grade responses. Go beyond standard expectations."
        
        base_prompt += f"\n\nCurrent timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        base_prompt += "\nYou have access to Gemini's extended context window for handling large documents and complex tasks."
        
        if custom_instructions:
            base_prompt += f"\n\nUser preferences: {custom_instructions}"
        
        return base_prompt

class EnhancedFileProcessor:
    def __init__(self):
        self.supported_formats = {
            'image': ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'svg', 'ico', 'tiff'],
            'document': ['pdf', 'docx', 'doc', 'txt', 'rtf', 'odt', 'tex', 'md'],
            'spreadsheet': ['xlsx', 'xls', 'csv', 'ods', 'tsv'],
            'code': ['py', 'js', 'java', 'cpp', 'c', 'cs', 'php', 'rb', 'go', 'rs', 'kt', 'swift', 'ts', 'jsx', 'vue', 'dart', 'scala', 'r', 'julia', 'lua', 'perl', 'sh', 'ps1'],
            'data': ['json', 'xml', 'yaml', 'yml', 'toml', 'ini', 'conf', 'env'],
            'web': ['html', 'htm', 'css', 'scss', 'sass', 'less'],
            'archive': ['zip', 'tar', 'gz', 'rar', '7z'],
            'notebook': ['ipynb'],
            'database': ['sql', 'db', 'sqlite']
        }
        
        self.processors = {
            'image': self._process_image,
            'document': self._process_document,
            'spreadsheet': self._process_spreadsheet,
            'code': self._process_code,
            'data': self._process_data,
            'web': self._process_web,
            'archive': self._process_archive,
            'notebook': self._process_notebook,
            'database': self._process_database
        }
        
        self.max_file_size = 500 * 1024 * 1024
    
    def get_file_type(self, filename: str) -> Optional[str]:
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        for file_type, extensions in self.supported_formats.items():
            if ext in extensions:
                return file_type
        
        return None
    
    async def process_file(self, file) -> Tuple[str, Optional[str], Dict[str, Any]]:
        try:
            filename = secure_filename(file.filename)
            file_type = self.get_file_type(filename)
            
            if not file_type:
                mime_type = mimetypes.guess_type(filename)[0]
                return f"[File: {filename} (Type: {mime_type or 'unknown'})]", None, {"error": "unsupported", "mime_type": mime_type}
            
            file.seek(0, 2)
            file_size = file.tell()
            file.seek(0)
            
            if file_size > self.max_file_size:
                return f"[File too large: {filename} ({file_size / 1024 / 1024:.1f}MB)]", None, {"error": "file_too_large", "size": file_size}
            
            processor = self.processors.get(file_type)
            if processor:
                return await processor(file, filename)
            
            return f"[File: {filename}]", None, {"error": "no_processor"}
            
        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
            traceback.print_exc()
            return f"[Error processing {file.filename}: {str(e)}]", None, {"error": str(e)}
    
    async def _process_image(self, file, filename: str) -> Tuple[str, str, Dict]:
        try:
            img = Image.open(file)
            
            original_size = img.size
            
            max_size = (2048, 2048)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            if img.mode not in ('RGB', 'RGBA'):
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                else:
                    img = img.convert('RGB')
            
            buffered = BytesIO()
            img_format = 'PNG' if filename.lower().endswith('.png') else 'JPEG'
            img.save(buffered, format=img_format, quality=95 if img_format == 'JPEG' else None)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            metadata = {
                "type": "image",
                "format": img_format,
                "original_size": original_size,
                "processed_size": img.size,
                "mode": img.mode,
                "has_transparency": img.mode == 'RGBA'
            }
            
            content = f"[Image: {filename}]\nOriginal: {original_size[0]}x{original_size[1]}\nProcessed: {img.size[0]}x{img.size[1]}\nFormat: {img_format}"
            
            return content, img_base64, metadata
            
        except Exception as e:
            return f"[Error processing image {filename}: {str(e)}]", None, {"error": str(e)}
    
    async def _process_document(self, file, filename: str) -> Tuple[str, None, Dict]:
        ext = filename.lower().split('.')[-1]
        
        try:
            if ext == 'pdf':
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                metadata = {
                    "type": "pdf",
                    "pages": len(pdf_reader.pages),
                    "info": {}
                }
                
                if pdf_reader.metadata:
                    metadata["info"] = {
                        "title": pdf_reader.metadata.get('/Title', ''),
                        "author": pdf_reader.metadata.get('/Author', ''),
                        "subject": pdf_reader.metadata.get('/Subject', ''),
                        "creator": pdf_reader.metadata.get('/Creator', '')
                    }
                
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text:
                        text += f"\n\n--- Page {i+1} ---\n{page_text}"
                
                content = f"[PDF Document: {filename}]\nPages: {metadata['pages']}\n{text[:100000]}"
                
            elif ext in ['docx', 'doc']:
                doc = docx.Document(file)
                paragraphs = []
                tables_count = len(doc.tables)
                
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        paragraphs.append(paragraph.text)
                
                text = "\n\n".join(paragraphs)
                
                metadata = {
                    "type": "word",
                    "paragraphs": len(paragraphs),
                    "tables": tables_count,
                    "sections": len(doc.sections)
                }
                
                content = f"[Word Document: {filename}]\nParagraphs: {len(paragraphs)}\nTables: {tables_count}\n\n{text[:100000]}"
                
            elif ext == 'md':
                text = file.read().decode('utf-8', errors='ignore')
                html = markdown.markdown(text, extensions=['tables', 'fenced_code', 'footnotes'])
                
                metadata = {
                    "type": "markdown",
                    "length": len(text),
                    "estimated_reading_time": f"{len(text.split()) // 200} minutes"
                }
                
                content = f"[Markdown Document: {filename}]\n{text[:100000]}"
                
            else:
                text = file.read().decode('utf-8', errors='ignore')
                metadata = {
                    "type": "text",
                    "encoding": "utf-8",
                    "lines": len(text.splitlines()),
                    "words": len(text.split()),
                    "characters": len(text)
                }
                
                content = f"[Text Document: {filename}]\nLines: {metadata['lines']}\nWords: {metadata['words']}\n\n{text[:100000]}"
            
            return content, None, metadata
            
        except Exception as e:
            return f"[Error processing document {filename}: {str(e)}]", None, {"error": str(e)}
    
    async def _process_spreadsheet(self, file, filename: str) -> Tuple[str, None, Dict]:
        ext = filename.lower().split('.')[-1]
        
        try:
            if ext == 'csv':
                df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')
            else:
                df = pd.read_excel(file, engine='openpyxl')
            
            summary = {
                "rows": len(df),
                "columns": len(df.columns),
                "columns_list": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "null_counts": df.isnull().sum().to_dict(),
                "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
            }
            
            stats = {}
            for col in df.select_dtypes(include=[np.number]).columns:
                stats[col] = {
                    "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                    "std": float(df[col].std()) if not df[col].isna().all() else None,
                    "min": float(df[col].min()) if not df[col].isna().all() else None,
                    "max": float(df[col].max()) if not df[col].isna().all() else None
                }
            
            sample_rows = min(100, len(df))
            sample = df.head(sample_rows).to_string(max_rows=100, max_cols=20)
            
            dtypes_str = '\n'.join([f"  {col}: {dtype}" for col, dtype in list(summary['dtypes'].items())[:10]])
            stats_str = '\n'.join([f"  {col}: mean={stats[col]['mean']:.2f}, std={stats[col]['std']:.2f}, min={stats[col]['min']:.2f}, max={stats[col]['max']:.2f}" for col in list(stats.keys())[:5]]) if stats else "  No numeric columns"
            
            content = f"""[Spreadsheet: {filename}]
Rows: {summary['rows']:,}, Columns: {summary['columns']}
Memory Usage: {summary['memory_usage']}
Columns: {', '.join(summary['columns_list'][:20])}{'...' if len(summary['columns_list']) > 20 else ''}

Data Types:
{dtypes_str}

Numeric Column Statistics:
{stats_str}

Sample Data ({sample_rows} rows):
{sample}
"""
            
            metadata = {
                "type": "spreadsheet",
                "format": ext,
                "summary": summary,
                "statistics": stats
            }
            
            return content, None, metadata
            
        except Exception as e:
            return f"[Error processing spreadsheet {filename}: {str(e)}]", None, {"error": str(e)}
    
    async def _process_code(self, file, filename: str) -> Tuple[str, None, Dict]:
        code = file.read().decode('utf-8', errors='ignore')
        ext = filename.lower().split('.')[-1]
        
        lines = code.split('\n')
        line_count = len(lines)
        
        try:
            lexer = get_lexer_by_name(ext)
            language = lexer.name
        except:
            try:
                lexer = guess_lexer(code)
                language = lexer.name
            except:
                language = ext.upper()
        
        imports = []
        functions = []
        classes = []
        
        for line in lines:
            stripped = line.strip()
            if ext in ['py', 'python']:
                if stripped.startswith(('import ', 'from ')):
                    imports.append(stripped)
                elif stripped.startswith('def '):
                    match = re.match(r'def\s+(\w+)', stripped)
                    if match:
                        functions.append(match.group(1))
                elif stripped.startswith('class '):
                    match = re.match(r'class\s+(\w+)', stripped)
                    if match:
                        classes.append(match.group(1))
            elif ext in ['js', 'javascript', 'ts', 'typescript']:
                if stripped.startswith(('import ', 'const ', 'let ', 'var ')) and ' from ' in stripped:
                    imports.append(stripped)
                elif 'function ' in stripped:
                    match = re.search(r'function\s+(\w+)', stripped)
                    if match:
                        functions.append(match.group(1))
                elif stripped.startswith('class '):
                    match = re.match(r'class\s+(\w+)', stripped)
                    if match:
                        classes.append(match.group(1))
        
        numbered_code = '\n'.join([f"{i+1:5d} | {line}" for i, line in enumerate(lines[:1000])])
        
        imports_str = '\n'.join([f"  - {imp}" for imp in imports[:10]]) if imports else ""
        functions_str = '\n'.join([f"  - {func}()" for func in functions[:10]]) if functions else ""
        classes_str = '\n'.join([f"  - {cls}" for cls in classes[:10]]) if classes else ""
        
        content = f"""[Code File: {filename}]
Language: {language}
Lines: {line_count:,}
File Size: {len(code):,} bytes

Structure Analysis:
  Imports: {len(imports)}
  Functions: {len(functions)}
  Classes: {len(classes)}

{f"Imports ({len(imports[:10])}):" if imports else ""}
{imports_str}

{f"Functions ({len(functions[:10])}):" if functions else ""}
{functions_str}

{f"Classes ({len(classes[:10])}):" if classes else ""}
{classes_str}

Code (first 1000 lines):
{numbered_code}
"""
        
        metadata = {
            "type": "code",
            "language": language,
            "extension": ext,
            "lines": line_count,
            "size": len(code),
            "structure": {
                "imports": imports[:20],
                "functions": functions[:20],
                "classes": classes[:20]
            }
        }
        
        return content, None, metadata
            except json.JSONDecodeError as e:
                metadata = {
                    "type": "json",
                    "valid": False,
                    "error": str(e),
                    "size": len(content_raw)
                }
                
                content = f"""[JSON File: {filename}]
Valid: ✗
Error: {str(e)}
Size: {len(content_raw):,} bytes

Raw Content:
{content_raw[:10000]}
"""
        
        elif ext in ['yaml', 'yml']:
            lines = content_raw.split('\n')
            
            metadata = {
                "type": "yaml",
                "lines": len(lines),
                "size": len(content_raw)
            }
            
            content = f"""[YAML File: {filename}]
Lines: {len(lines):,}
Size: {len(content_raw):,} bytes

Content:
{content_raw[:50000]}
"""
        
        elif ext == 'xml':
            try:
                soup = BeautifulSoup(content_raw, 'xml')
                pretty = soup.prettify()[:50000]
                
                root_tag = soup.find()
                tag_count = len(soup.find_all())
                
                metadata = {
                    "type": "xml",
                    "valid": True,
                    "root_tag": root_tag.name if root_tag else None,
                    "total_tags": tag_count,
                    "size": len(content_raw)
                }
                
                content = f"""[XML File: {filename}]
Valid: ✓
Root Tag: {root_tag.name if root_tag else 'None'}
Total Tags: {tag_count:,}
Size: {len(content_raw):,} bytes

Content:
{pretty}
"""
            except Exception as e:
                metadata = {
                    "type": "xml",
                    "valid": False,
                    "error": str(e),
                    "size": len(content_raw)
                }
                
                content = f"""[XML File: {filename}]
Valid: ✗
Error: {str(e)}

Raw Content:
{content_raw[:50000]}
"""
        
        else:
            metadata = {
                "type": ext,
                "size": len(content_raw),
                "lines": len(content_raw.splitlines())
            }
            
            content = f"""[Data File: {filename}]
Format: {ext.upper()}
Size: {len(content_raw):,} bytes

Content:
{content_raw[:50000]}
"""
        
        return content, None, metadata
    
    async def _process_web(self, file, filename: str) -> Tuple[str, None, Dict]:
        content_raw = file.read().decode('utf-8', errors='ignore')
        ext = filename.lower().split('.')[-1]
        
        if ext in ['html', 'htm']:
            soup = BeautifulSoup(content_raw, 'html.parser')
            
            title = soup.title.string if soup.title else None
            
            scripts = soup.find_all('script')
            styles = soup.find_all('style')
            links = soup.find_all('a')
            images = soup.find_all('img')
            forms = soup.find_all('form')
            
            text_content = soup.get_text(separator='\n', strip=True)[:10000]
            
            metadata = {
                "type": "html",
                "title": title,
                "structure": {
                    "scripts": len(scripts),
                    "styles": len(styles),
                    "links": len(links),
                    "images": len(images),
                    "forms": len(forms)
                },
                "meta_tags": {}
            }
            
            for meta in soup.find_all('meta'):
                name = meta.get('name') or meta.get('property')
                content = meta.get('content')
                if name and content:
                    metadata["meta_tags"][name] = content[:100]
            
            meta_tags_str = '\n'.join([f"  {name}: {value}" for name, value in list(metadata["meta_tags"].items())[:10]])
            
            content = f"""[HTML File: {filename}]
Title: {title or 'No title'}
Scripts: {len(scripts)}, Styles: {len(styles)}, Links: {len(links)}, Images: {len(images)}, Forms: {len(forms)}

Meta Tags:
{meta_tags_str}

Text Content:
{text_content}

HTML Structure:
{str(soup.prettify()[:5000])}
"""
        
        elif ext in ['css', 'scss', 'sass', 'less']:
            lines = content_raw.splitlines()
            
            selectors = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('/*', '*', '//')) and '{' in line:
                    selector = line.split('{')[0].strip()
                    if selector:
                        selectors.append(selector)
            
            metadata = {
                "type": ext,
                "lines": len(lines),
                "selectors": len(selectors),
                "size": len(content_raw)
            }
            
            selectors_str = '\n'.join([f"  - {sel}" for sel in selectors[:50]])
            
            content = f"""[Stylesheet: {filename}]
Type: {ext.upper()}
Lines: {len(lines):,}
Selectors: {len(selectors):,}

Selectors Found:
{selectors_str}

Content:
{content_raw[:50000]}
"""
        
        else:
            metadata = {
                "type": ext,
                "size": len(content_raw)
            }
            
            content = f"""[Web File: {filename}]
Type: {ext.upper()}

Content:
{content_raw[:50000]}
"""
        
        return content, None, metadata
    
    async def _process_archive(self, file, filename: str) -> Tuple[str, None, Dict]:
        return f"[Archive File: {filename}]\nArchive processing requires extraction. Please extract files first.", None, {"type": "archive", "filename": filename}
    
    async def _process_notebook(self, file, filename: str) -> Tuple[str, None, Dict]:
        try:
            notebook_content = json.loads(file.read().decode('utf-8'))
            
            cells = notebook_content.get('cells', [])
            code_cells = [cell for cell in cells if cell.get('cell_type') == 'code']
            markdown_cells = [cell for cell in cells if cell.get('cell_type') == 'markdown']
            
            total_code = ""
            total_markdown = ""
            
            for cell in code_cells:
                source = cell.get('source', [])
                if isinstance(source, list):
                    source = ''.join(source)
                total_code += source + "\n\n"
            
            for cell in markdown_cells:
                source = cell.get('source', [])
                if isinstance(source, list):
                    source = ''.join(source)
                total_markdown += source + "\n\n"
            
            metadata = {
                "type": "jupyter_notebook",
                "total_cells": len(cells),
                "code_cells": len(code_cells),
                "markdown_cells": len(markdown_cells),
                "kernel": notebook_content.get('metadata', {}).get('kernelspec', {}).get('display_name', 'Unknown')
            }
            
            content = f"""[Jupyter Notebook: {filename}]
Total Cells: {len(cells)}
Code Cells: {len(code_cells)}
Markdown Cells: {len(markdown_cells)}
Kernel: {metadata['kernel']}

Markdown Content:
{total_markdown[:25000]}

Code Content:
{total_code[:25000]}
"""
            
            return content, None, metadata
            
        except Exception as e:
            return f"[Error processing notebook {filename}: {str(e)}]", None, {"error": str(e)}
    
    async def _process_database(self, file, filename: str) -> Tuple[str, None, Dict]:
        content = file.read().decode('utf-8', errors='ignore')
        
        queries = []
        current_query = []
        
        for line in content.splitlines():
            stripped = line.strip()
            if stripped:
                current_query.append(line)
                if stripped.endswith(';'):
                    queries.append('\n'.join(current_query))
                    current_query = []
        
        if current_query:
            queries.append('\n'.join(current_query))
        
        metadata = {
            "type": "sql",
            "queries": len(queries),
            "size": len(content)
        }
        
        queries_str = '\n'.join([f"\n--- Query {i+1} ---\n{query}" for i, query in enumerate(queries[:20])])
        
        content_display = f"""[SQL File: {filename}]
Total Queries: {len(queries)}
File Size: {len(content):,} bytes

Queries:
{queries_str}
"""
        
        return content_display, None, metadata
    
    def _analyze_json_structure(self, data: Any, max_depth: int = 5, current_depth: int = 0) -> Dict:
        if current_depth >= max_depth:
            return {"type": "truncated", "reason": "max_depth_exceeded"}
        
        if isinstance(data, dict):
            structure = {
                "type": "object",
                "keys": list(data.keys())[:50],
                "size": len(data),
                "sample_values": {}
            }
            
            for key in list(data.keys())[:5]:
                structure["sample_values"][key] = self._analyze_json_structure(
                    data[key], max_depth, current_depth + 1
                )
            
            return structure
            
        elif isinstance(data, list):
            structure = {
                "type": "array",
                "length": len(data),
                "item_types": set()
            }
            
            for item in data[:10]:
                structure["item_types"].add(type(item).__name__)
            
            structure["item_types"] = list(structure["item_types"])
            
            if data:
                structure["sample_structure"] = self._analyze_json_structure(
                    data[0], max_depth, current_depth + 1
                )
            
            return structure
            
        else:
            return {
                "type": type(data).__name__,
                "value": str(data)[:100] if not isinstance(data, (int, float, bool, type(None))) else data
            }

session_manager = EnhancedSessionManager()
key_manager = GeminiKeyManager()
client_manager = StreamingGeminiClient(key_manager)
prompt_engineer = AdvancedPromptEngineer()
file_processor = EnhancedFileProcessor()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jack's AI Ultra - Enhanced Edition</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --primary-light: #818cf8;
            --secondary: #8b5cf6;
            --accent: #ec4899;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --info: #3b82f6;
            
            --bg-primary: #0a0a0a;
            --bg-secondary: #111111;
            --bg-tertiary: #1a1a1a;
            --bg-card: rgba(255, 255, 255, 0.02);
            --bg-hover: rgba(255, 255, 255, 0.05);
            
            --text-primary: #ffffff;
            --text-secondary: #a0a0a0;
            --text-muted: #666666;
            
            --border-color: rgba(255, 255, 255, 0.1);
            --border-hover: rgba(255, 255, 255, 0.2);
            
            --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.1);
            --shadow-md: 0 4px 20px rgba(0, 0, 0, 0.15);
            --shadow-lg: 0 10px 40px rgba(0, 0, 0, 0.2);
            --shadow-glow: 0 0 40px rgba(99, 102, 241, 0.3);
            
            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 24px;
            
            --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-base: 300ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-slow: 500ms cubic-bezier(0.4, 0, 0.2, 1);
            
            --header-height: 70px;
            --sidebar-width: 320px;
            --input-height: 56px;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow: hidden;
            position: relative;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        .background-effects {
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
        }
        
        .gradient-orb {
            position: absolute;
            width: 600px;
            height: 600px;
            border-radius: 50%;
            filter: blur(100px);
            opacity: 0.3;
            animation: float 20s ease-in-out infinite;
        }
        
        .gradient-orb:nth-child(1) {
            background: radial-gradient(circle, var(--primary), transparent);
            top: -200px;
            left: -200px;
            animation-duration: 25s;
        }
        
        .gradient-orb:nth-child(2) {
            background: radial-gradient(circle, var(--secondary), transparent);
            bottom: -200px;
            right: -200px;
            animation-duration: 30s;
            animation-delay: -10s;
        }
        
        .gradient-orb:nth-child(3) {
            background: radial-gradient(circle, var(--accent), transparent);
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            animation-duration: 35s;
            animation-delay: -20s;
        }
        
        @keyframes float {
            0%, 100% {
                transform: translate(0, 0) scale(1);
            }
            25% {
                transform: translate(100px, -100px) scale(1.1);
            }
            50% {
                transform: translate(-100px, 100px) scale(0.9);
            }
            75% {
                transform: translate(50px, 50px) scale(1.05);
            }
        }
        
        .grid-pattern {
            position: absolute;
            inset: 0;
            background-image: 
                linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            mask-image: radial-gradient(circle at center, black 30%, transparent 70%);
            -webkit-mask-image: radial-gradient(circle at center, black 30%, transparent 70%);
        }
        
        .app-container {
            position: relative;
            z-index: 1;
            display: flex;
            height: 100vh;
            backdrop-filter: blur(100px);
            -webkit-backdrop-filter: blur(100px);
        }
        
        .sidebar {
            width: var(--sidebar-width);
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            transition: transform var(--transition-base);
            position: relative;
            z-index: 100;
        }
        
        .sidebar.collapsed {
            transform: translateX(-100%);
        }
        
        .sidebar-header {
            padding: 24px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            position: relative;
            overflow: hidden;
        }
        
        .sidebar-header::before {
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 30% 50%, rgba(255, 255, 255, 0.1), transparent 50%);
            animation: shimmer 3s ease-in-out infinite;
        }
        
        @keyframes shimmer {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }
        
        .logo-container {
            display: flex;
            align-items: center;
            gap: 16px;
            position: relative;
            z-index: 1;
        }
        
        .logo-icon {
            width: 48px;
            height: 48px;
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            color: white;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            animation: pulse 2s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .logo-text h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 20px;
            font-weight: 700;
            color: white;
            letter-spacing: -0.5px;
        }
        
        .logo-text p {
            font-size: 12px;
            color: rgba(255, 255, 255, 0.8);
            margin-top: 2px;
        }
        
        .chat-sessions {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            scrollbar-width: thin;
            scrollbar-color: var(--border-color) transparent;
        }
        
        .chat-sessions::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-sessions::-webkit-scrollbar-track {
            background: transparent;
        }
        
        .chat-sessions::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 3px;
        }
        
        .chat-sessions::-webkit-scrollbar-thumb:hover {
            background: var(--border-hover);
        }
        
        .session-item {
            padding: 16px;
            margin-bottom: 8px;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            cursor: pointer;
            transition: all var(--transition-fast);
            position: relative;
            overflow: hidden;
        }
        
        .session-item::before {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.05), transparent);
            transform: translateX(-100%);
            transition: transform 0.6s;
        }
        
        .session-item:hover::before {
            transform: translateX(100%);
        }
        
        .session-item:hover {
            background: var(--bg-hover);
            border-color: var(--border-hover);
            transform: translateX(4px);
        }
        
        .session-item.active {
            background: rgba(99, 102, 241, 0.1);
            border-color: var(--primary);
        }
        
        .session-title {
            font-weight: 600;
            font-size: 14px;
            color: var(--text-primary);
            margin-bottom: 4px;
        }
        
        .session-preview {
            font-size: 13px;
            color: var(--text-secondary);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .session-meta {
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
            font-size: 11px;
            color: var(--text-muted);
        }
        
        .sidebar-actions {
            padding: 16px;
            border-top: 1px solid var(--border-color);
        }
        
        .btn-new-chat {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border: none;
            border-radius: var(--radius-md);
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all var(--transition-fast);
            position: relative;
            overflow: hidden;
        }
        
        .btn-new-chat::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            transition: width 0.4s, height 0.4s;
        }
        
        .btn-new-chat:hover::before {
            width: 300px;
            height: 300px;
        }
        
        .btn-new-chat:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
        }
        
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: var(--bg-primary);
            position: relative;
        }
        
        .chat-header {
            height: var(--header-height);
            padding: 0 24px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: relative;
            z-index: 50;
        }
        
        .header-left {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        
        .menu-toggle {
            width: 40px;
            height: 40px;
            border-radius: var(--radius-md);
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all var(--transition-fast);
        }
        
        .menu-toggle:hover {
            background: var(--bg-hover);
            border-color: var(--primary);
            color: var(--primary);
            transform: scale(1.05);
        }
        
        .mode-selector {
            display: flex;
            gap: 8px;
            padding: 4px;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
        }
        
        .mode-btn {
            padding: 8px 16px;
            background: transparent;
            border: none;
            border-radius: var(--radius-sm);
            color: var(--text-secondary);
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-fast);
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .mode-btn:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
        }
        
        .mode-btn.active {
            background: var(--primary);
            color: white;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }
        
        .header-right {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .header-btn {
            width: 40px;
            height: 40px;
            border-radius: var(--radius-md);
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all var(--transition-fast);
            position: relative;
        }
        
        .header-btn:hover {
            background: var(--bg-hover);
            border-color: var(--primary);
            color: var(--primary);
            transform: scale(1.05);
        }
        
        .advanced-toggle {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            cursor: pointer;
            transition: all var(--transition-fast);
        }
        
        .advanced-toggle.active {
            background: linear-gradient(135deg, var(--danger), var(--warning));
            border-color: transparent;
            animation: glow 2s ease-in-out infinite;
        }
        
        @keyframes glow {
            0%, 100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.5); }
            50% { box-shadow: 0 0 30px rgba(239, 68, 68, 0.8); }
        }
        
        .advanced-toggle-label {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .advanced-toggle.active .advanced-toggle-label {
            color: white;
        }
        
        .toggle-switch {
            width: 40px;
            height: 20px;
            background: var(--bg-tertiary);
            border-radius: 20px;
            position: relative;
            transition: all var(--transition-fast);
        }
        
        .advanced-toggle.active .toggle-switch {
            background: rgba(255, 255, 255, 0.3);
        }
        
        .toggle-switch::after {
            content: '';
            position: absolute;
            top: 2px;
            left: 2px;
            width: 16px;
            height: 16px;
            background: var(--text-secondary);
            border-radius: 50%;
            transition: all var(--transition-fast);
        }
        
        .advanced-toggle.active .toggle-switch::after {
            background: white;
            transform: translateX(20px);
        }
        
        .messages-area {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            scroll-behavior: smooth;
            scrollbar-width: thin;
            scrollbar-color: var(--border-color) transparent;
        }
        
        .messages-area::-webkit-scrollbar {
            width: 6px;
        }
        
        .messages-area::-webkit-scrollbar-track {
            background: transparent;
        }
        
        .messages-area::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 3px;
        }
        
        .messages-area::-webkit-scrollbar-thumb:hover {
            background: var(--border-hover);
        }
        
        .message {
            margin-bottom: 24px;
            display: flex;
            align-items: flex-start;
            gap: 16px;
            animation: messageSlide 0.3s ease-out;
        }
        
        @keyframes messageSlide {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .message.user {
            flex-direction: row-reverse;
        }
        
        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
            position: relative;
            overflow: hidden;
        }
        
        .message.user .message-avatar {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
        }
        
        .message.assistant .message-avatar {
            background: linear-gradient(135deg, var(--secondary), var(--accent));
            color: white;
        }
        
        .message-avatar::after {
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.2), transparent);
            animation: ripple 3s ease-in-out infinite;
        }
        
        @keyframes ripple {
            0%, 100% { transform: scale(0.8); opacity: 1; }
            50% { transform: scale(1.2); opacity: 0; }
        }
        
        .message-content-wrapper {
            max-width: 70%;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .message-header {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 0 4px;
        }
        
        .message-author {
            font-weight: 600;
            font-size: 13px;
            color: var(--text-primary);
        }
        
        .message-time {
            font-size: 11px;
            color: var(--text-muted);
        }
        
        .message-bubble {
            padding: 16px 20px;
            border-radius: var(--radius-lg);
            line-height: 1.6;
            font-size: 15px;
            position: relative;
            word-wrap: break-word;
        }
        
        .message.user .message-bubble {
            background: var(--primary);
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        .message.assistant .message-bubble {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            border-bottom-left-radius: 4px;
        }
        
        .message-bubble pre {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            padding: 16px;
            margin: 12px 0;
            overflow-x: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            line-height: 1.5;
            scrollbar-width: thin;
            scrollbar-color: var(--border-color) transparent;
        }
        
        .message-bubble pre::-webkit-scrollbar {
            height: 6px;
        }
        
        .message-bubble pre::-webkit-scrollbar-track {
            background: transparent;
        }
        
        .message-bubble pre::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 3px;
        }
        
        .message-bubble code {
            background: rgba(139, 92, 246, 0.1);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: var(--primary-light);
        }
        
        .message.user .message-bubble code {
            background: rgba(255, 255, 255, 0.2);
            color: white;
        }
        
        .streaming-text {
            display: inline;
        }
        
        .streaming-cursor {
            display: inline-block;
            width: 2px;
            height: 1.2em;
            background: var(--primary);
            margin-left: 2px;
            animation: blink 1s infinite;
        }
        
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }
        
        .message-actions {
            display: flex;
            gap: 8px;
            padding: 0 4px;
            opacity: 0;
            transition: opacity var(--transition-fast);
        }
        
        .message:hover .message-actions {
            opacity: 1;
        }
        
        .message-action {
            padding: 6px 12px;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            font-size: 12px;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all var(--transition-fast);
            display: flex;
            align-items: center;
            gap: 4px;
        }
        
        .message-action:hover {
            background: var(--bg-hover);
            border-color: var(--primary);
            color: var(--primary);
        }
        
        .typing-indicator {
            display: none;
            align-items: center;
            gap: 16px;
            padding: 0 24px;
            margin-bottom: 24px;
            animation: fadeIn 0.3s;
        }
        
        .typing-indicator.active {
            display: flex;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .typing-dots {
            display: flex;
            gap: 6px;
            padding: 16px 20px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            border-bottom-left-radius: 4px;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            background: var(--primary);
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }
        
        .typing-dot:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-dot:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typing {
            0%, 60%, 100% {
                transform: translateY(0);
                opacity: 0.7;
            }
            30% {
                transform: translateY(-10px);
                opacity: 1;
            }
        }
        
        .input-section {
            padding: 20px 24px;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border-color);
            position: relative;
        }
        
        .input-features {
            display: flex;
            gap: 12px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }
        
        .feature-btn {
            padding: 8px 16px;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            font-size: 13px;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all var(--transition-fast);
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .feature-btn:hover {
            background: var(--bg-hover);
            border-color: var(--primary);
            color: var(--primary);
        }
        
        .feature-btn.active {
            background: var(--primary);
            color: white;
            border-color: transparent;
        }
        
        .token-display {
            margin-left: auto;
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
        }
        
        .token-bar {
            width: 100px;
            height: 4px;
            background: var(--bg-tertiary);
            border-radius: 2px;
            overflow: hidden;
        }
        
        .token-fill {
            height: 100%;
            background: var(--primary);
            transition: width 0.5s ease;
            position: relative;
            overflow: hidden;
        }
        
        .token-fill::after {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            animation: shimmer-move 2s infinite;
        }
        
        @keyframes shimmer-move {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .token-text {
            font-size: 12px;
            color: var(--text-secondary);
            font-weight: 500;
        }
        
        .file-upload-area {
            display: none;
            margin-bottom: 16px;
            padding: 24px;
            background: var(--bg-card);
            border: 2px dashed var(--border-color);
            border-radius: var(--radius-lg);
            text-align: center;
            transition: all var(--transition-fast);
        }
        
        .file-upload-area.active {
            display: block;
            animation: slideDown 0.3s ease;
        }
        
        @keyframes slideDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .file-upload-area.dragover {
            background: rgba(99, 102, 241, 0.05);
            border-color: var(--primary);
        }
        
        .file-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 16px;
        }
        
        .file-item {
            padding: 8px 16px;
            background: var(--primary);
            color: white;
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            animation: fadeIn 0.3s;
        }
        
        .file-item .remove-file {
            cursor: pointer;
            opacity: 0.8;
            transition: opacity var(--transition-fast);
        }
        
        .file-item .remove-file:hover {
            opacity: 1;
        }
        
        .input-container {
            display: flex;
            gap: 12px;
            align-items: flex-end;
        }
        
        .input-wrapper {
            flex: 1;
            position: relative;
        }
        
        .input-box {
            width: 100%;
            min-height: var(--input-height);
            max-height: 200px;
            padding: 16px 20px;
            padding-right: 120px;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            color: var(--text-primary);
            font-size: 15px;
            font-family: inherit;
            resize: none;
            transition: all var(--transition-fast);
            line-height: 1.5;
            scrollbar-width: thin;
            scrollbar-color: var(--border-color) transparent;
        }
        
        .input-box::-webkit-scrollbar {
            width: 6px;
        }
        
        .input-box::-webkit-scrollbar-track {
            background: transparent;
        }
        
        .input-box::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 3px;
        }
        
        .input-box:focus {
            outline: none;
            border-color: var(--primary);
            background: var(--bg-hover);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }
        
        .input-box::placeholder {
            color: var(--text-muted);
        }
        
        .input-actions {
            position: absolute;
            right: 12px;
            top: 50%;
            transform: translateY(-50%);
            display: flex;
            gap: 8px;
        }
        
        .input-action {
            width: 32px;
            height: 32px;
            border-radius: var(--radius-sm);
            background: var(--bg-hover);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all var(--transition-fast);
        }
        
        .input-action:hover {
            background: var(--primary);
            border-color: transparent;
            color: white;
        }
        
        .send-button {
            padding: 0 24px;
            height: var(--input-height);
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border: none;
            border-radius: var(--radius-lg);
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all var(--transition-fast);
            display: flex;
            align-items: center;
            gap: 8px;
            position: relative;
            overflow: hidden;
        }
        
        .send-button::before {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), transparent);
            opacity: 0;
            transition: opacity var(--transition-fast);
        }
        
        .send-button:hover::before {
            opacity: 1;
        }
        
        .send-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
        }
        
        .send-button:active {
            transform: translateY(0);
        }
        
        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .modal-overlay {
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            padding: 20px;
        }
        
        .modal-overlay.active {
            display: flex;
            animation: fadeIn 0.3s;
        }
        
        .modal-content {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-xl);
            padding: 32px;
            max-width: 600px;
            width: 100%;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: var(--shadow-lg);
            animation: slideUp 0.3s ease-out;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .notification-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 2000;
            pointer-events: none;
        }
        
        .notification {
            padding: 16px 20px;
            border-radius: var(--radius-md);
            color: white;
            font-size: 14px;
            font-weight: 500;
            box-shadow: var(--shadow-lg);
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
            pointer-events: auto;
            animation: slideInRight 0.3s ease-out;
            position: relative;
            overflow: hidden;
        }
        
        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(100%);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        .notification.success {
            background: linear-gradient(135deg, var(--success), #22c55e);
        }
        
        .notification.error {
            background: linear-gradient(135deg, var(--danger), #f87171);
        }
        
        .notification.warning {
            background: linear-gradient(135deg, var(--warning), #fbbf24);
        }
        
        .notification.info {
            background: linear-gradient(135deg, var(--info), #60a5fa);
        }
        
        .notification-close {
            margin-left: auto;
            cursor: pointer;
            opacity: 0.8;
            transition: opacity var(--transition-fast);
        }
        
        .notification-close:hover {
            opacity: 1;
        }
        
        .loading-overlay {
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        }
        
        .loading-overlay.active {
            display: flex;
        }
        
        .loading-spinner {
            width: 60px;
            height: 60px;
            position: relative;
        }
        
        .loading-spinner::before,
        .loading-spinner::after {
            content: '';
            position: absolute;
            inset: 0;
            border-radius: 50%;
            border: 3px solid transparent;
            border-top-color: var(--primary);
            animation: spin 1s linear infinite;
        }
        
        .loading-spinner::after {
            animation-delay: 0.5s;
            border-top-color: var(--secondary);
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
            .sidebar {
                position: fixed;
                z-index: 200;
                box-shadow: var(--shadow-lg);
            }
            
            .mode-selector {
                display: none;
            }
            
            .header-right {
                gap: 8px;
            }
            
            .advanced-toggle {
                padding: 8px;
            }
            
            .advanced-toggle-label {
                display: none;
            }
            
            .message-content-wrapper {
                max-width: 85%;
            }
            
            .input-features {
                gap: 8px;
            }
            
            .token-display {
                display: none;
            }
        }
        
        @media (max-width: 480px) {
            .messages-area {
                padding: 16px;
            }
            
            .input-section {
                padding: 16px;
            }
            
            .input-box {
                padding-right: 80px;
            }
            
            .send-button {
                padding: 0 16px;
            }
        }
    </style>
</head>
<body>
    <div class="background-effects">
        <div class="gradient-orb"></div>
        <div class="gradient-orb"></div>
        <div class="gradient-orb"></div>
        <div class="grid-pattern"></div>
    </div>
    
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-spinner"></div>
    </div>
    
    <div class="notification-container" id="notificationContainer"></div>
    
    <div class="app-container">
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-header">
                <div class="logo-container">
                    <div class="logo-icon">
                        <i class="fas fa-brain"></i>
                    </div>
                    <div class="logo-text">
                        <h1>Jack's AI Ultra</h1>
                        <p>Enhanced Edition</p>
                    </div>
                </div>
            </div>
            
            <div class="chat-sessions" id="chatSessions"></div>
            
            <div class="sidebar-actions">
                <button class="btn-new-chat" id="btnNewChat">
                    <i class="fas fa-plus"></i> New Chat
                </button>
            </div>
        </aside>
        
        <main class="main-content">
            <header class="chat-header">
                <div class="header-left">
                    <button class="menu-toggle" id="menuToggle">
                        <i class="fas fa-bars"></i>
                    </button>
                    
                    <div class="mode-selector">
                        <button class="mode-btn active" data-mode="balanced">
                            <i class="fas fa-balance-scale"></i> Balanced
                        </button>
                        <button class="mode-btn" data-mode="creative">
                            <i class="fas fa-palette"></i> Creative
                        </button>
                        <button class="mode-btn" data-mode="precise">
                            <i class="fas fa-microscope"></i> Precise
                        </button>
                        <button class="mode-btn" data-mode="code">
                            <i class="fas fa-code"></i> Code
                        </button>
                        <button class="mode-btn" data-mode="research">
                            <i class="fas fa-book"></i> Research
                        </button>
                    </div>
                </div>
                
                <div class="header-right">
                    <div class="advanced-toggle" id="advancedToggle">
                        <span class="advanced-toggle-label">Advanced</span>
                        <div class="toggle-switch"></div>
                    </div>
                    
                    <button class="header-btn" id="btnExport">
                        <i class="fas fa-download"></i>
                    </button>
                    <button class="header-btn" id="btnSettings">
                        <i class="fas fa-cog"></i>
                    </button>
                </div>
            </header>
            
            <div class="messages-area" id="messagesArea"></div>
            
            <div class="typing-indicator" id="typingIndicator">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="typing-dots">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
            
            <div class="input-section">
                <div class="input-features">
                    <button class="feature-btn" id="btnFiles">
                        <i class="fas fa-paperclip"></i> Files
                    </button>
                    <button class="feature-btn" id="btnVoice">
                        <i class="fas fa-microphone"></i> Voice
                    </button>
                    <button class="feature-btn" id="btnTemplates">
                        <i class="fas fa-file-alt"></i> Templates
                    </button>
                    
                    <div class="token-display">
                        <div class="token-bar">
                            <div class="token-fill" id="tokenFill" style="width: 0%"></div>
                        </div>
                        <span class="token-text" id="tokenText">0 / 2M</span>
                    </div>
                </div>
                
                <div class="file-upload-area" id="fileUploadArea">
                    <i class="fas fa-cloud-upload-alt" style="font-size: 40px; color: var(--text-muted); margin-bottom: 12px;"></i>
                    <p style="color: var(--text-secondary); margin-bottom: 8px;">Drop files here or click to browse</p>
                    <p style="font-size: 12px; color: var(--text-muted);">Supports images, documents, code, data files, and more (up to 500MB)</p>
                    <div class="file-list" id="fileList"></div>
                </div>
                
                <div class="input-container">
                    <div class="input-wrapper">
                        <textarea 
                            class="input-box" 
                            id="messageInput" 
                            placeholder="Ask anything... (Shift+Enter for new line)"
                            rows="1"
                        ></textarea>
                        <div class="input-actions">
                            <button class="input-action" id="btnClear">
                                <i class="fas fa-times"></i>
                            </button>
                            <button class="input-action" id="btnEmoji">
                                <i class="fas fa-smile"></i>
                            </button>
                        </div>
                    </div>
                    <button class="send-button" id="sendButton">
                        <i class="fas fa-paper-plane"></i>
                        <span>Send</span>
                    </button>
                </div>
            </div>
        </main>
    </div>
    
    <input type="file" id="fileInput" multiple style="display: none;">
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-javascript.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/11.1.1/marked.min.js"></script>
    
    <script>
        const AppState = {
            sessionId: null,
            currentMode: 'balanced',
            messages: [],
            tokenUsage: 0,
            maxTokens: 2000000,
            attachedFiles: [],
            isTyping: false,
            isStreaming: false,
            currentStreamController: null,
            advancedMode: false,
            voiceRecognition: null,
            activeStream: null,
            settings: {
                streaming: true,
                notifications: true,
                soundEffects: false,
                autoSave: true,
                theme: 'dark'
            }
        };
        
        document.addEventListener('DOMContentLoaded', function() {
            initializeApp();
            setupEventListeners();
            loadSettings();
        });
        
        function initializeApp() {
            AppState.sessionId = localStorage.getItem('sessionId');
            if (!AppState.sessionId) {
                createNewSession();
            } else {
                loadSession();
            }
            
            updateTokenDisplay();
            addWelcomeMessage();
        }
        
        function createNewSession() {
            AppState.sessionId = generateUUID();
            localStorage.setItem('sessionId', AppState.sessionId);
            AppState.messages = [];
            AppState.tokenUsage = 0;
            
            fetch('/api/session/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    sessionId: AppState.sessionId,
                    mode: AppState.currentMode 
                })
            });
        }
        
        async function loadSession() {
            try {
                const response = await fetch(`/api/session/${AppState.sessionId}`);
                if (response.ok) {
                    const data = await response.json();
                    AppState.messages = data.messages || [];
                    AppState.tokenUsage = data.tokenUsage || 0;
                    displayMessages();
                } else {
                    createNewSession();
                }
            } catch (error) {
                console.error('Error loading session:', error);
                createNewSession();
            }
        }
        
        function setupEventListeners() {
            document.getElementById('menuToggle').addEventListener('click', toggleSidebar);
            document.getElementById('btnNewChat').addEventListener('click', startNewChat);
            
            document.querySelectorAll('.mode-btn').forEach(btn => {
                btn.addEventListener('click', () => changeMode(btn.dataset.mode));
            });
            
            document.getElementById('advancedToggle').addEventListener('click', toggleAdvancedMode);
            document.getElementById('btnExport').addEventListener('click', exportChat);
            document.getElementById('btnSettings').addEventListener('click', openSettings);
            
            const messageInput = document.getElementById('messageInput');
            messageInput.addEventListener('input', autoResizeTextarea);
            messageInput.addEventListener('keydown', handleInputKeydown);
            
            document.getElementById('sendButton').addEventListener('click', sendMessage);
            
            document.getElementById('btnFiles').addEventListener('click', toggleFileUpload);
            document.getElementById('btnVoice').addEventListener('click', toggleVoiceInput);
            document.getElementById('btnTemplates').addEventListener('click', showTemplates);
            
            document.getElementById('btnClear').addEventListener('click', clearInput);
            document.getElementById('btnEmoji').addEventListener('click', insertEmoji);
            
            setupFileUpload();
        }
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message && AppState.attachedFiles.length === 0) return;
            
            if (AppState.isStreaming) {
                showNotification('Please wait for the current response to complete', 'warning');
                return;
            }
            
            input.value = '';
            autoResizeTextarea();
            
            document.getElementById('sendButton').disabled = true;
            
            addMessage('user', message);
            
            showTypingIndicator();
            
            const formData = new FormData();
            formData.append('message', message);
            formData.append('sessionId', AppState.sessionId);
            formData.append('mode', AppState.currentMode);
            formData.append('advancedMode', AppState.advancedMode);
            formData.append('stream', 'true');
            
            AppState.attachedFiles.forEach(file => {
                formData.append('files', file);
            });
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                if (response.body) {
                    handleStreamingResponse(response.body);
                } else {
                    const data = await response.json();
                    hideTypingIndicator();
                    addMessage('assistant', data.response);
                    updateTokenUsage(data.tokenUsage);
                    document.getElementById('sendButton').disabled = false;
                }
            } catch (error) {
                console.error('Error sending message:', error);
                hideTypingIndicator();
                showNotification('Failed to send message', 'error');
                document.getElementById('sendButton').disabled = false;
            }
            
            clearFiles();
        }
        
        async function handleStreamingResponse(stream) {
            AppState.isStreaming = true;
            const reader = stream.getReader();
            const decoder = new TextDecoder();
            let assistantMessage = '';
            let messageElement = null;
            let messageId = null;
            
            hideTypingIndicator();
            
            try {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6);
                            if (data === '[DONE]') {
                                AppState.isStreaming = false;
                                document.getElementById('sendButton').disabled = false;
                                if (messageElement) {
                                    const bubble = messageElement.querySelector('.message-bubble');
                                    const cursor = bubble.querySelector('.streaming-cursor');
                                    if (cursor) cursor.remove();
                                }
                            } else {
                                try {
                                    const parsed = JSON.parse(data);
                                    if (parsed.content) {
                                        assistantMessage += parsed.content;
                                        
                                        if (!messageElement) {
                                            messageId = 'msg-' + generateUUID();
                                            messageElement = createMessageElement('assistant', assistantMessage, messageId);
                                            document.getElementById('messagesArea').appendChild(messageElement);
                                        } else {
                                            updateStreamingMessage(messageElement, assistantMessage);
                                        }
                                        
                                        scrollToBottom();
                                    }
                                    
                                    if (parsed.tokenUsage) {
                                        updateTokenUsage(parsed.tokenUsage);
                                    }
                                } catch (e) {
                                    console.error('Error parsing stream data:', e);
                                }
                            }
                        }
                    }
                }
            } catch (error) {
                console.error('Stream reading error:', error);
                showNotification('Stream interrupted', 'error');
            } finally {
                AppState.isStreaming = false;
                document.getElementById('sendButton').disabled = false;
                if (messageElement) {
                    const bubble = messageElement.querySelector('.message-bubble');
                    const cursor = bubble.querySelector('.streaming-cursor');
                    if (cursor) cursor.remove();
                }
            }
            
            if (assistantMessage) {
                AppState.messages.push({ 
                    role: 'assistant', 
                    content: assistantMessage, 
                    timestamp: new Date() 
                });
                saveSession();
            }
        }
        
        function createMessageElement(role, content, messageId) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            messageDiv.id = messageId;
            
            const time = new Date().toLocaleTimeString([], { 
                hour: '2-digit', 
                minute: '2-digit' 
            });
            
            const processedContent = role === 'assistant' && AppState.isStreaming 
                ? content + '<span class="streaming-cursor"></span>'
                : processMessageContent(content);
            
            messageDiv.innerHTML = `
                <div class="message-avatar">
                    <i class="fas fa-${role === 'user' ? 'user' : 'robot'}"></i>
                </div>
                <div class="message-content-wrapper">
                    <div class="message-header">
                        <span class="message-author">${role === 'user' ? 'You' : "Jack's AI Ultra"}</span>
                        <span class="message-time">${time}</span>
                    </div>
                    <div class="message-bubble">${processedContent}</div>
                    <div class="message-actions">
                        <button class="message-action" onclick="copyMessage('${messageId}')">
                            <i class="fas fa-copy"></i> Copy
                        </button>
                        <button class="message-action" onclick="regenerateMessage('${messageId}')">
                            <i class="fas fa-redo"></i> Regenerate
                        </button>
                    </div>
                </div>
            `;
            
            return messageDiv;
        }
        
        function updateStreamingMessage(element, content) {
            const bubble = element.querySelector('.message-bubble');
            if (bubble) {
                const processedContent = processMessageContent(content);
                bubble.innerHTML = processedContent + '<span class="streaming-cursor"></span>';
                
                element.querySelectorAll('pre code').forEach(block => {
                    Prism.highlightElement(block);
                });
            }
        }
        
        function addMessage(role, content, save = true) {
            const messageId = 'msg-' + generateUUID();
            const messageElement = createMessageElement(role, content, messageId);
            
            document.getElementById('messagesArea').appendChild(messageElement);
            scrollToBottom();
            
            if (!AppState.isStreaming) {
                messageElement.querySelectorAll('pre code').forEach(block => {
                    Prism.highlightElement(block);
                });
            }
            
            if (save) {
                AppState.messages.push({ role, content, timestamp: new Date() });
                saveSession();
            }
            
            return messageElement;
        }
        
        function processMessageContent(content) {
            marked.setOptions({
                highlight: function(code, lang) {
                    if (Prism.languages[lang]) {
                        return Prism.highlight(code, Prism.languages[lang], lang);
                    }
                    return code;
                },
                breaks: true,
                gfm: true
            });
            
            return marked.parse(content);
        }
        
        function addWelcomeMessage() {
            const welcome = `# Welcome to Jack's AI Ultra - Enhanced Edition! 🚀

I'm your advanced AI assistant powered by Gemini's cutting-edge technology with extended context capabilities.

## What I can help you with:

- 💡 **Complex Problem Solving** - Advanced analysis and multi-step reasoning
- 🎨 **Creative Projects** - Stories, designs, and innovative ideas
- 💻 **Professional Coding** - Complete implementations with best practices
- 📊 **Data Analysis** - Process large files and datasets
- 🔬 **Deep Research** - Comprehensive analysis across topics
- 📚 **Learning & Tutoring** - Personalized educational support

${AppState.advancedMode ? '\n⚡ **ADVANCED MODE ACTIVE** - I\'ll provide exceptionally detailed, comprehensive, and professional-grade responses!\n' : ''}

Select a mode above or toggle Advanced Mode for enhanced responses. How can I assist you today?`;
            
            addMessage('assistant', welcome, false);
        }
        
        function showTypingIndicator() {
            document.getElementById('typingIndicator').classList.add('active');
        }
        
        function hideTypingIndicator() {
            document.getElementById('typingIndicator').classList.remove('active');
        }
        
        function updateTokenDisplay() {
            const percentage = (AppState.tokenUsage / AppState.maxTokens) * 100;
            document.getElementById('tokenFill').style.width = `${percentage}%`;
            document.getElementById('tokenText').textContent = 
                `${formatNumber(AppState.tokenUsage)} / ${formatNumber(AppState.maxTokens)}`;
            
            const fill = document.getElementById('tokenFill');
            if (percentage > 80) {
                fill.style.background = 'var(--danger)';
            } else if (percentage > 60) {
                fill.style.background = 'var(--warning)';
            } else {
                fill.style.background = 'var(--primary)';
            }
        }
        
        function updateTokenUsage(usage) {
            AppState.tokenUsage = usage || AppState.tokenUsage;
            updateTokenDisplay();
            
            if (AppState.tokenUsage > AppState.maxTokens * 0.8) {
                showNotification('Approaching token limit', 'warning');
            }
        }
        
        function showNotification(message, type = 'info') {
            if (!AppState.settings.notifications) return;
            
            const container = document.getElementById('notificationContainer');
            const notification = document.createElement('div');
            notification.className = `notification ${type} animate__animated animate__slideInRight`;
            
            const icons = {
                success: 'check-circle',
                error: 'exclamation-circle',
                warning: 'exclamation-triangle',
                info: 'info-circle'
            };
            
            notification.innerHTML = `
                <i class="fas fa-${icons[type]}"></i>
                <span>${message}</span>
                <i class="fas fa-times notification-close" onclick="this.parentElement.remove()"></i>
            `;
            
            container.appendChild(notification);
            
            setTimeout(() => {
                notification.classList.add('animate__slideOutRight');
                setTimeout(() => notification.remove(), 500);
            }, 5000);
        }
        
        function changeMode(mode) {
            AppState.currentMode = mode;
            
            document.querySelectorAll('.mode-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.mode === mode);
            });
            
            const modeDescriptions = {
                balanced: 'Balanced mode for versatile assistance',
                creative: 'Creative mode for imaginative solutions',
                precise: 'Precise mode for technical accuracy',
                code: 'Code mode for programming excellence',
                research: 'Research mode for in-depth analysis'
            };
            
            showNotification(modeDescriptions[mode], 'success');
        }
        
        function toggleAdvancedMode() {
            AppState.advancedMode = !AppState.advancedMode;
            const toggle = document.getElementById('advancedToggle');
            toggle.classList.toggle('active', AppState.advancedMode);
            
            if (AppState.advancedMode) {
                showNotification('🚀 Advanced Mode ACTIVATED - Maximum capability unlocked!', 'warning');
            } else {
                showNotification('Advanced Mode deactivated', 'info');
            }
            
            fetch('/api/toggle-advanced', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ active: AppState.advancedMode })
            });
        }
        
        function toggleSidebar() {
            document.getElementById('sidebar').classList.toggle('collapsed');
        }
        
        function startNewChat() {
            if (AppState.messages.length > 0 && 
                !confirm('Start a new chat? Current conversation will be saved.')) {
                return;
            }
            
            createNewSession();
            document.getElementById('messagesArea').innerHTML = '';
            addWelcomeMessage();
            showNotification('New chat started', 'success');
        }
        
        async function exportChat() {
            const format = 'markdown';
            
            let content = `# Jack's AI Ultra - Chat Export\n\n`;
            content += `**Date:** ${new Date().toLocaleDateString()}\n`;
            content += `**Session:** ${AppState.sessionId}\n`;
            content += `**Mode:** ${AppState.currentMode}\n`;
            content += `**Advanced Mode:** ${AppState.advancedMode ? 'Active' : 'Inactive'}\n\n`;
            content += `---\n\n`;
            
            AppState.messages.forEach(msg => {
                content += `### ${msg.role === 'user' ? '👤 You' : '🤖 AI'}\n`;
                content += `*${new Date(msg.timestamp).toLocaleString()}*\n\n`;
                content += `${msg.content}\n\n`;
                content += `---\n\n`;
            });
            
            const blob = new Blob([content], { type: 'text/markdown' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `chat-export-${new Date().toISOString().split('T')[0]}.md`;
            a.click();
            URL.revokeObjectURL(url);
            
            showNotification('Chat exported successfully', 'success');
        }
        
        function setupFileUpload() {
            const fileInput = document.getElementById('fileInput');
            const uploadArea = document.getElementById('fileUploadArea');
            
            fileInput.addEventListener('change', handleFileSelect);
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                handleFiles(e.dataTransfer.files);
            });
            
            uploadArea.addEventListener('click', () => {
                fileInput.click();
            });
        }
        
        function handleFileSelect(event) {
            handleFiles(event.target.files);
        }
        
        function handleFiles(files) {
            const fileList = document.getElementById('fileList');
            
            for (const file of files) {
                if (file.size > 500 * 1024 * 1024) {
                    showNotification(`File ${file.name} is too large (max 500MB)`, 'error');
                    continue;
                }
                
                AppState.attachedFiles.push(file);
                
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.innerHTML = `
                    <i class="fas fa-file"></i>
                    <span>${file.name}</span>
                    <i class="fas fa-times remove-file" onclick="removeFile('${file.name}', this)"></i>
                `;
                
                fileList.appendChild(fileItem);
            }
            
            document.getElementById('fileInput').value = '';
        }
        
        function removeFile(fileName, element) {
            AppState.attachedFiles = AppState.attachedFiles.filter(f => f.name !== fileName);
            element.parentElement.remove();
        }
        
        function clearFiles() {
            AppState.attachedFiles = [];
            document.getElementById('fileList').innerHTML = '';
            document.getElementById('fileUploadArea').classList.remove('active');
            document.getElementById('btnFiles').classList.remove('active');
        }
        
        function toggleFileUpload() {
            const uploadArea = document.getElementById('fileUploadArea');
            const isActive = uploadArea.classList.toggle('active');
            document.getElementById('btnFiles').classList.toggle('active', isActive);
        }
        
        function toggleVoiceInput() {
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                showNotification('Voice input not supported in your browser', 'error');
                return;
            }
            
            if (!AppState.voiceRecognition) {
                initVoiceRecognition();
            }
            
            if (AppState.voiceRecognition.isRecording) {
                stopVoiceRecording();
            } else {
                startVoiceRecording();
            }
        }
        
        function initVoiceRecognition() {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            const recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';
            
            recognition.onresult = (event) => {
                let transcript = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    if (event.results[i].isFinal) {
                        transcript += event.results[i][0].transcript;
                    }
                }
                
                if (transcript) {
                    const input = document.getElementById('messageInput');
                    input.value += transcript + ' ';
                    autoResizeTextarea();
                }
            };
            
            recognition.onerror = (event) => {
                console.error('Voice recognition error:', event.error);
                showNotification('Voice recognition error', 'error');
                stopVoiceRecording();
            };
            
            AppState.voiceRecognition = recognition;
            AppState.voiceRecognition.isRecording = false;
        }
        
        function startVoiceRecording() {
            AppState.voiceRecognition.start();
            AppState.voiceRecognition.isRecording = true;
            document.getElementById('btnVoice').classList.add('active');
            showNotification('Voice recording started', 'info');
        }
        
        function stopVoiceRecording() {
            AppState.voiceRecognition.stop();
            AppState.voiceRecognition.isRecording = false;
            document.getElementById('btnVoice').classList.remove('active');
            showNotification('Voice recording stopped', 'info');
        }
        
        function showTemplates() {
            const templates = [
                { name: 'Code Review', text: 'Please review this code for best practices, potential bugs, and optimization opportunities:' },
                { name: 'Explain Concept', text: 'Can you explain [concept] in simple terms with examples?' },
                { name: 'Debug Help', text: 'I\'m getting this error: [error]. Here\'s my code:' },
                { name: 'Research Topic', text: 'I need a comprehensive analysis of [topic] including:' },
                { name: 'Creative Writing', text: 'Write a creative story about [theme] that includes:' }
            ];
            
            showNotification('Templates feature coming soon!', 'info');
        }
        
        function clearInput() {
            document.getElementById('messageInput').value = '';
            autoResizeTextarea();
        }
        
        function insertEmoji() {
            const emojis = ['😊', '👍', '❤️', '🚀', '🎉', '🤔', '💡', '✨', '🔥', '💻'];
            const emoji = emojis[Math.floor(Math.random() * emojis.length)];
            
            const input = document.getElementById('messageInput');
            const start = input.selectionStart;
            const end = input.selectionEnd;
            const text = input.value;
            
            input.value = text.substring(0, start) + emoji + text.substring(end);
            input.selectionStart = input.selectionEnd = start + emoji.length;
            input.focus();
        }
        
        function copyMessage(messageId) {
            const message = document.getElementById(messageId);
            if (!message) return;
            
            const bubble = message.querySelector('.message-bubble');
            const text = bubble.textContent;
            
            navigator.clipboard.writeText(text).then(() => {
                showNotification('Message copied to clipboard', 'success');
            }).catch(err => {
                console.error('Failed to copy:', err);
                showNotification('Failed to copy message', 'error');
            });
        }
        
        async function regenerateMessage(messageId) {
            const message = document.getElementById(messageId);
            if (!message) return;
            
            const isAssistant = message.classList.contains('assistant');
            if (!isAssistant) {
                showNotification('Can only regenerate assistant messages', 'warning');
                return;
            }
            
            const messageIndex = Array.from(message.parentElement.children).indexOf(message);
            if (messageIndex > 0) {
                const previousMessage = message.parentElement.children[messageIndex - 1];
                if (previousMessage.classList.contains('user')) {
                    const userContent = previousMessage.querySelector('.message-bubble').textContent;
                    
                    message.remove();
                    
                    showTypingIndicator();
                    
                    const formData = new FormData();
                    formData.append('message', userContent);
                    formData.append('sessionId', AppState.sessionId);
                    formData.append('mode', AppState.currentMode);
                    formData.append('advancedMode', AppState.advancedMode);
                    formData.append('stream', 'true');
                    formData.append('regenerate', 'true');
                    
                    try {
                        const response = await fetch('/api/chat', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (response.body) {
                            handleStreamingResponse(response.body);
                        }
                    } catch (error) {
                        console.error('Error regenerating message:', error);
                        hideTypingIndicator();
                        showNotification('Failed to regenerate message', 'error');
                    }
                }
            }
        }
        
        function openSettings() {
            showNotification('Settings panel coming soon!', 'info');
        }
        
        function autoResizeTextarea() {
            const textarea = document.getElementById('messageInput');
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
        }
        
        function handleInputKeydown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }
        
        function scrollToBottom() {
            const messagesArea = document.getElementById('messagesArea');
            messagesArea.scrollTop = messagesArea.scrollHeight;
        }
        
        function generateUUID() {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                const r = Math.random() * 16 | 0;
                const v = c === 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        }
        
        function formatNumber(num) {
            if (num >= 1000000) {
                return (num / 1000000).toFixed(1) + 'M';
            } else if (num >= 1000) {
                return (num / 1000).toFixed(1) + 'K';
            }
            return num.toString();
        }
        
        async function saveSession() {
            if (!AppState.settings.autoSave) return;
            
            try {
                await fetch('/api/session/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sessionId: AppState.sessionId,
                        messages: AppState.messages,
                        tokenUsage: AppState.tokenUsage,
                        mode: AppState.currentMode
                    })
                });
            } catch (error) {
                console.error('Error saving session:', error);
            }
        }
        
        function displayMessages() {
            const messagesArea = document.getElementById('messagesArea');
            messagesArea.innerHTML = '';
            
            AppState.messages.forEach(msg => {
                addMessage(msg.role, msg.content, false);
            });
        }
        
        function loadSettings() {
            const saved = localStorage.getItem('settings');
            if (saved) {
                AppState.settings = { ...AppState.settings, ...JSON.parse(saved) };
            }
        }
        
        function saveSettings() {
            localStorage.setItem('settings', JSON.stringify(AppState.settings));
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/session/create', methods=['POST'])
@limiter.limit("10 per minute")
def create_session():
    try:
        data = request.json
        mode = ChatMode(data.get('mode', 'balanced'))
        
        session = session_manager.create_session(
            user_id=data.get('userId'),
            mode=mode
        )
        
        return jsonify({
            'success': True,
            'sessionId': session.id,
            'mode': mode.value
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/session/<session_id>', methods=['GET'])
@cache.cached(timeout=60)
def get_session(session_id):
    try:
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        return jsonify({
            'success': True,
            'messages': [msg.to_dict() for msg in session.messages],
            'tokenUsage': session.total_tokens,
            'mode': session.mode.value,
            'metadata': session.metadata
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/session/save', methods=['POST'])
@limiter.limit("30 per minute")
def save_session():
    try:
        data = request.json
        session_id = data.get('sessionId')
        
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/toggle-advanced', methods=['POST'])
def toggle_advanced():
    try:
        data = request.json
        active = data.get('active', False)
        prompt_engineer.toggle_advanced_mode(active)
        return jsonify({'success': True, 'active': active})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
@limiter.limit("60 per minute")
async def chat():
    try:
        message = request.form.get('message', '')
        session_id = request.form.get('sessionId')
        mode = ChatMode(request.form.get('mode', 'balanced'))
        files = request.files.getlist('files')
        regenerate = request.form.get('regenerate', 'false') == 'true'
        advanced_mode = request.form.get('advancedMode', 'false') == 'true'
        stream = request.form.get('stream', 'true') == 'true'
        
        session = session_manager.get_session(session_id)
        if not session:
            session = session_manager.create_session(mode=mode)
            session.id = session_id
        
        file_contents = []
        if files:
            for file in files:
                content, img_data, metadata = await file_processor.process_file(file)
                file_contents.append(content)
                if img_data and metadata.get('type') == 'image':
                    file_contents.append(f"[Image data: {metadata}]")
        
        enhanced_message = message
        if not regenerate:
            enhanced_message = prompt_engineer.enhance_prompt(
                message, 
                mode,
                force_advanced=advanced_mode
            )
        
        full_prompt = enhanced_message
        if file_contents:
            full_prompt += "\n\n--- Attached Files ---\n" + "\n\n".join(file_contents)
        
        system_prompt = prompt_engineer.create_system_prompt(mode)
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        context_messages = session.messages[-20:]
        for msg in context_messages:
            messages.append({
                "role": msg.role.value,
                "content": msg.content[:5000]
            })
        
        messages.append({"role": "user", "content": full_prompt})
        
        client, api_key = client_manager.get_client()
        if not client:
            return jsonify({
                'success': False,
                'error': 'No AI service available. Please try again later.'
            }), 503
        
        model = "gemini-2.0-flash-exp"
        
        start_time = time.time()
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=8192,
                temperature=0.9 if mode == ChatMode.CREATIVE else 0.7 if mode == ChatMode.BALANCED else 0.3,
                stream=stream
            )
            
            if stream:
                def generate():
                    assistant_message = ""
                    token_count = 0
                    
                    try:
                        for chunk in response:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                assistant_message += content
                                token_count += len(content) // 4
                                
                                yield f"data: {json.dumps({'content': content})}\n\n"
                        
                        user_msg = Message(
                            role=MessageRole.USER,
                            content=message,
                            timestamp=datetime.now(timezone.utc),
                            tokens=len(message) // 4,
                            attachments=[{"name": f.filename, "size": f.content_length} for f in files] if files else []
                        )
                        assistant_msg = Message(
                            role=MessageRole.ASSISTANT,
                            content=assistant_message,
                            timestamp=datetime.now(timezone.utc),
                            tokens=token_count
                        )
                        
                        if not regenerate:
                            session.add_message(user_msg)
                        session.add_message(assistant_msg)
                        session_manager.update_session(session)
                        
                        yield f"data: {json.dumps({'tokenUsage': session.total_tokens})}\n\n"
                        yield "data: [DONE]\n\n"
                        
                    except Exception as e:
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                        yield "data: [DONE]\n\n"
                
                response_time = time.time() - start_time
                key_manager.mark_key_success(api_key, response_time)
                
                return Response(
                    stream_with_context(generate()), 
                    mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'X-Accel-Buffering': 'no'
                    }
                )
            
            else:
                ai_response = response.choices[0].message.content
                
                user_msg = Message(
                    role=MessageRole.USER,
                    content=message,
                    timestamp=datetime.now(timezone.utc),
                    tokens=len(message) // 4
                )
                assistant_msg = Message(
                    role=MessageRole.ASSISTANT,
                    content=ai_response,
                    timestamp=datetime.now(timezone.utc),
                    tokens=len(ai_response) // 4
                )
                
                if not regenerate:
                    session.add_message(user_msg)
                session.add_message(assistant_msg)
                session_manager.update_session(session)
                
                response_time = time.time() - start_time
                key_manager.mark_key_success(api_key, response_time)
                
                return jsonify({
                    'success': True,
                    'response': ai_response,
                    'tokenUsage': session.total_tokens,
                    'mode': mode.value
                })
                
        except Exception as api_error:
            key_manager.mark_key_failure(api_key)
            raise api_error
            
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
@cache.cached(timeout=300)
def get_available_models():
    models = [
        {'id': 'gemini-2.0-flash-exp', 'name': 'Gemini 2.0 Flash (Experimental)', 'provider': 'Google'},
        {'id': 'gemini-2.0-flash-thinking-exp', 'name': 'Gemini 2.0 Flash Thinking', 'provider': 'Google'},
        {'id': 'gemini-1.5-pro', 'name': 'Gemini 1.5 Pro', 'provider': 'Google'},
        {'id': 'gemini-1.5-flash', 'name': 'Gemini 1.5 Flash', 'provider': 'Google'}
    ]
    
    return jsonify({
        'success': True,
        'models': models,
        'defaultModel': 'gemini-2.0-flash-exp'
    })

@app.route('/api/export/<format>', methods=['GET'])
@limiter.limit("10 per minute")
def export_session(format):
    try:
        session_id = request.args.get('sessionId')
        session = session_manager.get_session(session_id)
        
        if not session:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            data = {
                'sessionId': session.id,
                'mode': session.mode.value,
                'messages': [msg.to_dict() for msg in session.messages],
                'tokenUsage': session.total_tokens,
                'exportDate': datetime.now(timezone.utc).isoformat()
            }
            
            return jsonify(data), 200, {
                'Content-Disposition': f'attachment; filename=chat_export_{timestamp}.json'
            }
            
        elif format == 'markdown':
            content = f"# Jack's AI Ultra - Enhanced Edition\n\n"
            content += f"**Export Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            content += f"**Session ID:** {session.id}\n"
            content += f"**Mode:** {session.mode.value}\n"
            content += f"**Total Tokens:** {session.total_tokens:,}\n\n"
            content += "---\n\n"
            
            for msg in session.messages:
                icon = "👤" if msg.role == MessageRole.USER else "🤖"
                content += f"### {icon} {msg.role.value.title()}\n"
                content += f"*{msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
                content += f"{msg.content}\n\n"
                if msg.attachments:
                    content += "**Attachments:**\n"
                    for att in msg.attachments:
                        content += f"- {att.get('name', 'Unknown file')}\n"
                    content += "\n"
                content += "---\n\n"
            
            return Response(content, mimetype='text/markdown', headers={
                'Content-Disposition': f'attachment; filename=chat_export_{timestamp}.md'
            })
            
        else:
            return jsonify({'success': False, 'error': 'Invalid format'}), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        total_sessions = len(session_manager.sessions)
        total_messages = sum(len(s.messages) for s in session_manager.sessions.values())
        total_tokens = sum(s.total_tokens for s in session_manager.sessions.values())
        
        active_keys = sum(1 for key in key_manager.keys.values() if key.is_active)
        total_keys = len(key_manager.keys)
        
        return jsonify({
            'success': True,
            'stats': {
                'sessions': total_sessions,
                'messages': total_messages,
                'tokens': total_tokens,
                'activeKeys': active_keys,
                'totalKeys': total_keys
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'version': '2.0.0',
        'features': ['streaming', 'advanced_prompts', 'extended_context']
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    if os.environ.get('ENVIRONMENT') == 'production':
        from waitress import serve
        print(f"Starting Jack's AI Ultra Enhanced Edition on port {port} (Production Mode)")
        serve(app, host='0.0.0.0', port=port, threads=20)
    else:
        print(f"Starting Jack's AI Ultra Enhanced Edition on port {port} (Development Mode)")
        app.run(host='0.0.0.0', port=port, debug=True)



