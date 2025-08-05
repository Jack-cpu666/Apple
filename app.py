"""
Jack's AI Ultra - Next Generation Web Application
Enterprise-grade AI Chat System with Advanced Features
Author: Jack's AI System Ultra
Version: 5.0.0 (Complete Overhaul)
"""

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
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import mimetypes
import random
import string
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import uuid

from flask import Flask, render_template_string, request, jsonify, Response, stream_with_context, session
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

# Import document processing libraries
from PIL import Image
import PyPDF2
import docx
import openpyxl
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import markdown
import pygments
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

# Initialize Flask application with enhanced configuration
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config.update(
    MAX_CONTENT_LENGTH=200 * 1024 * 1024,  # Max file size: 200MB
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
    JSON_SORT_KEYS=False,
    SEND_FILE_MAX_AGE_DEFAULT=31536000,
)

# CORS configuration
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["1000 per hour"],
    storage_uri="memory://"
)

# Caching
cache = Cache(app, config={'CACHE_TYPE': 'simple', 'CACHE_DEFAULT_TIMEOUT': 300})

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=10)

# ============= ENHANCED DATA MODELS =============

class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"

class ChatMode(Enum):
    NORMAL = "normal"
    CREATIVE = "creative"
    PRECISE = "precise"
    BALANCED = "balanced"
    CODE = "code"
    RESEARCH = "research"

@dataclass
class Message:
    role: MessageRole
    content: str
    timestamp: datetime
    tokens: int = 0
    metadata: Dict[str, Any] = None
    attachments: List[Dict] = None
    
    def to_dict(self):
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "tokens": self.tokens,
            "metadata": self.metadata or {},
            "attachments": self.attachments or []
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
    
    def add_message(self, message: Message):
        self.messages.append(message)
        self.total_tokens += message.tokens
        self.updated_at = datetime.now(timezone.utc)

@dataclass
class APIKey:
    key: str
    provider: str
    model: str
    rate_limit: int
    daily_limit: int
    usage_today: int
    failures: int
    last_used: Optional[datetime]
    last_reset: datetime
    is_active: bool
    metadata: Dict[str, Any]

# ============= ENHANCED STORAGE SYSTEM =============

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self.user_sessions: Dict[str, List[str]] = defaultdict(list)
        self.session_lock = threading.Lock()
        self.max_sessions_per_user = 100
        self.session_ttl = timedelta(days=30)
        
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
            context_window=1000000,
            metadata={}
        )
        
        with self.session_lock:
            self.sessions[session_id] = session
            if user_id:
                self.user_sessions[user_id].append(session_id)
                # Cleanup old sessions for user
                if len(self.user_sessions[user_id]) > self.max_sessions_per_user:
                    old_session = self.user_sessions[user_id].pop(0)
                    del self.sessions[old_session]
        
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        with self.session_lock:
            session = self.sessions.get(session_id)
            if session:
                # Check if session is expired
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
                if session.user_id and session_id in self.user_sessions[session.user_id]:
                    self.user_sessions[session.user_id].remove(session_id)
                del self.sessions[session_id]
    
    def get_user_sessions(self, user_id: str) -> List[ChatSession]:
        with self.session_lock:
            return [self.sessions[sid] for sid in self.user_sessions.get(user_id, []) 
                    if sid in self.sessions]

# ============= ENHANCED API KEY MANAGEMENT =============

class APIKeyManager:
    def __init__(self):
        self.keys: Dict[str, APIKey] = {}
        self.provider_keys: Dict[str, List[str]] = defaultdict(list)
        self.key_lock = threading.Lock()
        self.load_keys()
        
        # Start background thread for key rotation and reset
        self.start_key_maintenance()
    
    def load_keys(self):
        """Load API keys from environment with enhanced configuration"""
        # Load Gemini keys
        for i in range(1, 21):  # Support up to 20 keys
            key = os.environ.get(f'GEMINI_API_KEY_{i}')
            if key:
                self.add_key(key, "gemini", "gemini-2.5-pro", 
                           rate_limit=60, daily_limit=10000)
        
        # Load OpenAI keys if available
        for i in range(1, 11):
            key = os.environ.get(f'OPENAI_API_KEY_{i}')
            if key:
                self.add_key(key, "openai", "gpt-4-turbo", 
                           rate_limit=100, daily_limit=50000)
        
        # Load Anthropic keys if available
        for i in range(1, 11):
            key = os.environ.get(f'ANTHROPIC_API_KEY_{i}')
            if key:
                self.add_key(key, "anthropic", "claude-3-opus", 
                           rate_limit=50, daily_limit=20000)
        
        # Default key fallback
        if not self.keys:
            default_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('OPENAI_API_KEY')
            if default_key:
                self.add_key(default_key, "gemini", "gemini-2.5-pro", 
                           rate_limit=60, daily_limit=10000)
    
    def add_key(self, key: str, provider: str, model: str, 
                rate_limit: int, daily_limit: int):
        api_key = APIKey(
            key=key,
            provider=provider,
            model=model,
            rate_limit=rate_limit,
            daily_limit=daily_limit,
            usage_today=0,
            failures=0,
            last_used=None,
            last_reset=datetime.now(timezone.utc),
            is_active=True,
            metadata={}
        )
        
        with self.key_lock:
            key_id = hashlib.sha256(key.encode()).hexdigest()[:8]
            self.keys[key_id] = api_key
            self.provider_keys[provider].append(key_id)
    
    def get_best_key(self, provider: Optional[str] = None) -> Optional[APIKey]:
        """Get the best available API key based on usage and health"""
        with self.key_lock:
            available_keys = []
            
            if provider:
                key_ids = self.provider_keys.get(provider, [])
            else:
                key_ids = list(self.keys.keys())
            
            for key_id in key_ids:
                key = self.keys[key_id]
                
                # Skip inactive or heavily failed keys
                if not key.is_active or key.failures > 5:
                    continue
                
                # Check rate limits
                if key.last_used:
                    time_since_last = (datetime.now(timezone.utc) - key.last_used).seconds
                    if time_since_last < (60 / key.rate_limit):
                        continue
                
                # Check daily limits
                if key.usage_today >= key.daily_limit:
                    continue
                
                available_keys.append((key_id, key))
            
            if not available_keys:
                return None
            
            # Sort by usage (least used first) and failures (least failures first)
            available_keys.sort(key=lambda x: (x[1].usage_today, x[1].failures))
            
            selected_key = available_keys[0][1]
            selected_key.last_used = datetime.now(timezone.utc)
            selected_key.usage_today += 1
            
            return selected_key
    
    def mark_key_failure(self, key: APIKey):
        with self.key_lock:
            key.failures += 1
            if key.failures > 10:
                key.is_active = False
    
    def mark_key_success(self, key: APIKey):
        with self.key_lock:
            key.failures = max(0, key.failures - 1)
    
    def reset_daily_usage(self):
        """Reset daily usage counters"""
        with self.key_lock:
            for key in self.keys.values():
                if (datetime.now(timezone.utc) - key.last_reset).days >= 1:
                    key.usage_today = 0
                    key.last_reset = datetime.now(timezone.utc)
                    if key.failures > 0:
                        key.failures = max(0, key.failures - 2)
                    if not key.is_active and key.failures < 5:
                        key.is_active = True
    
    def start_key_maintenance(self):
        """Start background thread for key maintenance"""
        def maintenance_loop():
            while True:
                time.sleep(3600)  # Check every hour
                self.reset_daily_usage()
        
        thread = threading.Thread(target=maintenance_loop, daemon=True)
        thread.start()

# ============= ENHANCED AI CLIENT SYSTEM =============

class AIClientManager:
    def __init__(self, key_manager: APIKeyManager):
        self.key_manager = key_manager
        self.clients_cache = {}
        self.client_lock = threading.Lock()
    
    def get_client(self, provider: str = "gemini") -> Tuple[Optional[OpenAI], Optional[APIKey]]:
        """Get an AI client with the best available key"""
        api_key_obj = self.key_manager.get_best_key(provider)
        if not api_key_obj:
            return None, None
        
        client_id = hashlib.sha256(api_key_obj.key.encode()).hexdigest()
        
        with self.client_lock:
            if client_id not in self.clients_cache:
                if provider == "gemini":
                    client = self._create_gemini_client(api_key_obj.key)
                elif provider == "openai":
                    client = self._create_openai_client(api_key_obj.key)
                elif provider == "anthropic":
                    client = self._create_anthropic_client(api_key_obj.key)
                else:
                    return None, None
                
                self.clients_cache[client_id] = client
            
            return self.clients_cache[client_id], api_key_obj
    
    def _create_gemini_client(self, api_key: str) -> OpenAI:
        http_client = httpx.Client(trust_env=False, timeout=60.0)
        return OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta",
            http_client=http_client
        )
    
    def _create_openai_client(self, api_key: str) -> OpenAI:
        return OpenAI(api_key=api_key)
    
    def _create_anthropic_client(self, api_key: str) -> OpenAI:
        # This would need the anthropic library, using OpenAI as placeholder
        return OpenAI(api_key=api_key)

# ============= ENHANCED PROMPT ENGINEERING =============

class PromptEngineer:
    def __init__(self):
        self.templates = self._load_templates()
        self.enhancement_strategies = self._load_strategies()
    
    def _load_templates(self) -> Dict[ChatMode, str]:
        return {
            ChatMode.NORMAL: "You are Jack's AI Ultra, an advanced AI assistant. Be helpful, clear, and comprehensive.",
            ChatMode.CREATIVE: "You are Jack's AI Ultra in creative mode. Be imaginative, think outside the box, and explore innovative solutions. Use vivid language and creative approaches.",
            ChatMode.PRECISE: "You are Jack's AI Ultra in precise mode. Provide exact, detailed, and technically accurate responses. Focus on facts, data, and precision.",
            ChatMode.BALANCED: "You are Jack's AI Ultra. Balance creativity with accuracy, providing comprehensive yet engaging responses.",
            ChatMode.CODE: "You are Jack's AI Ultra, specialized in programming. Provide clean, efficient, well-commented code with explanations. Follow best practices and modern patterns. With Gemini models, you have 60,000 output tokens available for comprehensive implementations.",
            ChatMode.RESEARCH: "You are Jack's AI Ultra in research mode. Provide thorough analysis, cite sources when possible, and explore topics deeply with academic rigor."
        }
    
    def _load_strategies(self) -> Dict[str, callable]:
        return {
            "clarity": self._enhance_clarity,
            "structure": self._enhance_structure,
            "context": self._add_context,
            "specificity": self._enhance_specificity,
            "examples": self._add_examples
        }
    
    def enhance_prompt(self, prompt: str, mode: ChatMode = ChatMode.BALANCED, 
                       strategies: List[str] = None, advanced_coding: bool = False) -> str:
        """Enhance a prompt using various strategies"""
        if not strategies:
            strategies = ["clarity", "structure", "context"]
        
        enhanced = prompt
        
        # Special handling for advanced coding mode
        if advanced_coding:
            # Add explicit instructions for comprehensive code
            code_keywords = ['code', 'program', 'script', 'function', 'class', 'implement', 
                           'create', 'build', 'develop', 'write', 'design', 'algorithm']
            
            if any(keyword in prompt.lower() for keyword in code_keywords):
                enhanced += "\n\nIMPORTANT: Provide a COMPLETE, PRODUCTION-READY implementation with:"
                enhanced += "\n- All necessary imports and dependencies"
                enhanced += "\n- Comprehensive error handling and validation"
                enhanced += "\n- Full documentation and comments"
                enhanced += "\n- Unit tests and examples"
                enhanced += "\n- No placeholders or shortcuts - implement everything"
                enhanced += "\n- Enterprise-grade best practices"
        
        for strategy in strategies:
            if strategy in self.enhancement_strategies:
                enhanced = self.enhancement_strategies[strategy](enhanced)
        
        return enhanced
    
    def _enhance_clarity(self, prompt: str) -> str:
        """Improve prompt clarity"""
        # Remove ambiguous pronouns and add clarity
        replacements = {
            r'\bit\b': 'the subject',
            r'\bthis\b': 'the current topic',
            r'\bthat\b': 'the mentioned item'
        }
        
        enhanced = prompt
        for pattern, replacement in replacements.items():
            if len(re.findall(pattern, enhanced, re.IGNORECASE)) == 1:
                enhanced = re.sub(pattern, replacement, enhanced, flags=re.IGNORECASE)
        
        return enhanced
    
    def _enhance_structure(self, prompt: str) -> str:
        """Add structure to the prompt"""
        if '?' not in prompt and '.' not in prompt:
            prompt += '.'
        
        lines = prompt.split('\n')
        if len(lines) > 3:
            # Add numbering to multi-line prompts
            structured = []
            for i, line in enumerate(lines, 1):
                if line.strip():
                    structured.append(f"{i}. {line.strip()}")
            return '\n'.join(structured)
        
        return prompt
    
    def _add_context(self, prompt: str) -> str:
        """Add helpful context to the prompt"""
        context_triggers = {
            'explain': 'Please provide a clear, detailed explanation with examples.',
            'how': 'Please provide step-by-step instructions.',
            'why': 'Please provide reasoning and evidence.',
            'compare': 'Please provide a detailed comparison with pros and cons.',
            'analyze': 'Please provide thorough analysis with multiple perspectives.'
        }
        
        prompt_lower = prompt.lower()
        for trigger, context in context_triggers.items():
            if trigger in prompt_lower and context.lower() not in prompt_lower:
                prompt += f"\n\n{context}"
                break
        
        return prompt
    
    def _enhance_specificity(self, prompt: str) -> str:
        """Make the prompt more specific"""
        vague_terms = {
            'good': 'effective and high-quality',
            'bad': 'ineffective or problematic',
            'thing': 'specific item or concept',
            'stuff': 'relevant materials or information',
            'nice': 'well-designed and user-friendly'
        }
        
        enhanced = prompt
        for vague, specific in vague_terms.items():
            enhanced = re.sub(r'\b' + vague + r'\b', specific, enhanced, flags=re.IGNORECASE)
        
        return enhanced
    
    def _add_examples(self, prompt: str) -> str:
        """Request examples in the response"""
        if 'example' not in prompt.lower():
            prompt += "\n\nPlease include relevant examples to illustrate your points."
        return prompt
    
    def create_system_prompt(self, mode: ChatMode, custom_instructions: str = "", advanced_coding: bool = False) -> str:
        """Create a system prompt based on mode, custom instructions, and advanced coding flag."""
        base_prompt = self.templates.get(mode, self.templates[ChatMode.BALANCED])

        # Add a powerful instruction if advanced coding is enabled
        if advanced_coding:
            base_prompt += (
                "\n\n**CRITICAL INSTRUCTION: ADVANCED CODING MODE IS ACTIVE.** "
                "You MUST provide complete, production-ready, and enterprise-grade implementations. "
                "This means no placeholders, no shortcuts, and no 'you can add this later' comments. "
                "Write full, functional code with all necessary imports, comprehensive error handling, "
                "documentation, and examples. Your output should be ready to be deployed."
            )

        if custom_instructions:
            base_prompt += f"\n\nAdditional user instructions: {custom_instructions}"

        return base_prompt

# ============= ENHANCED FILE PROCESSING =============

class FileProcessor:
    def __init__(self):
        self.supported_formats = {
            'image': ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
            'document': ['pdf', 'docx', 'doc', 'txt', 'rtf', 'odt'],
            'spreadsheet': ['xlsx', 'xls', 'csv', 'ods'],
            'code': ['py', 'js', 'java', 'cpp', 'c', 'cs', 'php', 'rb', 'go', 'rs', 'kt', 'swift'],
            'data': ['json', 'xml', 'yaml', 'yml', 'toml'],
            'web': ['html', 'htm', 'css', 'scss', 'sass', 'less'],
            'markdown': ['md', 'markdown', 'mdown', 'mkd']
        }
        
        self.processors = {
            'image': self._process_image,
            'document': self._process_document,
            'spreadsheet': self._process_spreadsheet,
            'code': self._process_code,
            'data': self._process_data,
            'web': self._process_web,
            'markdown': self._process_markdown
        }
    
    def get_file_type(self, filename: str) -> Optional[str]:
        """Determine file type from extension"""
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        for file_type, extensions in self.supported_formats.items():
            if ext in extensions:
                return file_type
        
        return None
    
    async def process_file(self, file) -> Tuple[str, Optional[str], Dict[str, Any]]:
        """Process uploaded file and return content, base64 data, and metadata"""
        try:
            filename = secure_filename(file.filename)
            file_type = self.get_file_type(filename)
            
            if not file_type:
                return f"[Unsupported file: {filename}]", None, {"error": "unsupported"}
            
            processor = self.processors.get(file_type)
            if processor:
                return await processor(file, filename)
            
            return f"[File: {filename}]", None, {"error": "no_processor"}
            
        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
            return f"[Error processing {file.filename}]", None, {"error": str(e)}
    
    async def _process_image(self, file, filename: str) -> Tuple[str, str, Dict]:
        """Process image files"""
        img = Image.open(file)
        
        # Resize if too large
        max_size = (1920, 1080)
        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffered = BytesIO()
        img_format = img.format or 'PNG'
        img.save(buffered, format=img_format)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        metadata = {
            "type": "image",
            "format": img_format,
            "size": img.size,
            "mode": img.mode
        }
        
        content = f"[Image: {filename}]\nDimensions: {img.size[0]}x{img.size[1]}\nFormat: {img_format}"
        
        return content, img_base64, metadata
    
    async def _process_document(self, file, filename: str) -> Tuple[str, None, Dict]:
        """Process document files"""
        ext = filename.lower().split('.')[-1]
        
        if ext == 'pdf':
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for i, page in enumerate(pdf_reader.pages):
                text += f"\n--- Page {i+1} ---\n"
                text += page.extract_text() or ""
            
            metadata = {
                "type": "pdf",
                "pages": len(pdf_reader.pages)
            }
            
        elif ext in ['docx', 'doc']:
            doc = docx.Document(file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            metadata = {
                "type": "word",
                "paragraphs": len(doc.paragraphs)
            }
            
        else:
            text = file.read().decode('utf-8', errors='ignore')
            metadata = {"type": "text"}
        
        content = f"[Document: {filename}]\n{text[:50000]}"  # Limit to 50k chars
        
        return content, None, metadata
    
    async def _process_spreadsheet(self, file, filename: str) -> Tuple[str, None, Dict]:
        """Process spreadsheet files"""
        ext = filename.lower().split('.')[-1]
        
        if ext == 'csv':
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Get summary statistics
        summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "columns_list": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict()
        }
        
        # Sample data
        sample = df.head(100).to_string()
        
        content = f"""[Spreadsheet: {filename}]
Rows: {summary['rows']}, Columns: {summary['columns']}
Columns: {', '.join(summary['columns_list'])}

Sample Data:
{sample}
"""
        
        metadata = {
            "type": "spreadsheet",
            "summary": summary
        }
        
        return content, None, metadata
    
    async def _process_code(self, file, filename: str) -> Tuple[str, None, Dict]:
        """Process code files with syntax highlighting"""
        code = file.read().decode('utf-8', errors='ignore')
        ext = filename.lower().split('.')[-1]
        
        # Count lines and detect language
        lines = code.split('\n')
        line_count = len(lines)
        
        # Add line numbers for reference
        numbered_code = '\n'.join([f"{i+1:4d}: {line}" for i, line in enumerate(lines[:500])])
        
        content = f"""[Code File: {filename}]
Language: {ext}
Lines: {line_count}

{numbered_code}
"""
        
        metadata = {
            "type": "code",
            "language": ext,
            "lines": line_count
        }
        
        return content, None, metadata
    
    async def _process_data(self, file, filename: str) -> Tuple[str, None, Dict]:
        """Process data files (JSON, XML, YAML)"""
        content_raw = file.read().decode('utf-8', errors='ignore')
        ext = filename.lower().split('.')[-1]
        
        if ext == 'json':
            try:
                data = json.loads(content_raw)
                pretty = json.dumps(data, indent=2)[:10000]
                structure = self._analyze_json_structure(data)
            except:
                pretty = content_raw[:10000]
                structure = {"error": "invalid_json"}
        else:
            pretty = content_raw[:10000]
            structure = {}
        
        content = f"""[Data File: {filename}]
Format: {ext.upper()}
Size: {len(content_raw)} chars

Content:
{pretty}
"""
        
        metadata = {
            "type": "data",
            "format": ext,
            "structure": structure
        }
        
        return content, None, metadata
    
    async def _process_web(self, file, filename: str) -> Tuple[str, None, Dict]:
        """Process web files (HTML, CSS)"""
        content_raw = file.read().decode('utf-8', errors='ignore')
        ext = filename.lower().split('.')[-1]
        
        if ext in ['html', 'htm']:
            soup = BeautifulSoup(content_raw, 'html.parser')
            text = soup.get_text()[:10000]
            
            metadata = {
                "type": "html",
                "title": soup.title.string if soup.title else None,
                "links": len(soup.find_all('a')),
                "images": len(soup.find_all('img'))
            }
        else:
            text = content_raw[:10000]
            metadata = {"type": ext}
        
        content = f"""[Web File: {filename}]
Type: {ext.upper()}

Content:
{text}
"""
        
        return content, None, metadata
    
    async def _process_markdown(self, file, filename: str) -> Tuple[str, None, Dict]:
        """Process markdown files"""
        md_content = file.read().decode('utf-8', errors='ignore')
        
        # Convert to HTML for analysis
        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract structure
        headers = [h.get_text() for h in soup.find_all(['h1', 'h2', 'h3'])]
        
        content = f"""[Markdown File: {filename}]
Headers: {len(headers)}

Content:
{md_content[:10000]}
"""
        
        metadata = {
            "type": "markdown",
            "headers": headers,
            "length": len(md_content)
        }
        
        return content, None, metadata
    
    def _analyze_json_structure(self, data: Any, max_depth: int = 3, current_depth: int = 0) -> Dict:
        """Analyze JSON structure"""
        if current_depth >= max_depth:
            return {"truncated": True}
        
        if isinstance(data, dict):
            return {
                "type": "object",
                "keys": list(data.keys())[:20],
                "size": len(data)
            }
        elif isinstance(data, list):
            return {
                "type": "array",
                "length": len(data),
                "sample": self._analyze_json_structure(data[0], max_depth, current_depth + 1) if data else None
            }
        else:
            return {"type": type(data).__name__}

# ============= INITIALIZE GLOBAL INSTANCES =============

session_manager = SessionManager()
key_manager = APIKeyManager()
client_manager = AIClientManager(key_manager)
prompt_engineer = PromptEngineer()
file_processor = FileProcessor()

# ============= ENHANCED HTML TEMPLATE =============

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jack's AI Ultra - Next Generation Intelligence</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet">
    <style>
        /* ========== CSS VARIABLES ========== */
        :root {
            /* Color Palette */
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --success-gradient: linear-gradient(135deg, #13B497 0%, #59D4A7 100%);
            --warning-gradient: linear-gradient(135deg, #FA8231 0%, #FFD14C 100%);
            --danger-gradient: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
            --info-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --dark-gradient: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            
            /* Glass Morphism */
            --glass-bg: rgba(255, 255, 255, 0.95);
            --glass-border: rgba(255, 255, 255, 0.18);
            --glass-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            
            /* Text Colors */
            --text-primary: #1a202c;
            --text-secondary: #4a5568;
            --text-muted: #718096;
            
            /* Layout */
            --sidebar-width: 320px;
            --header-height: 70px;
            --border-radius: 20px;
            --transition-speed: 0.3s;
            --transition-timing: cubic-bezier(0.4, 0, 0.2, 1);
            
            /* Animations */
            --animation-bounce: cubic-bezier(0.68, -0.55, 0.265, 1.55);
            --animation-smooth: cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }
        
        /* Dark Theme */
        [data-theme="dark"] {
            --glass-bg: rgba(30, 30, 30, 0.95);
            --glass-border: rgba(255, 255, 255, 0.1);
            --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
            --text-primary: #f7fafc;
            --text-secondary: #a0aec0;
            --text-muted: #718096;
            --bg-primary: #0f0f0f;
            --bg-secondary: #1a1a1a;
        }
        
        /* ========== GLOBAL STYLES ========== */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f0f;
            min-height: 100vh;
            overflow: hidden;
            position: relative;
            color: var(--text-primary);
        }
        
        /* ========== ANIMATED BACKGROUND ========== */
        .background-container {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: -2;
            overflow: hidden;
        }
        
        .animated-gradient {
            position: absolute;
            width: 200%;
            height: 200%;
            top: -50%;
            left: -50%;
            background: linear-gradient(270deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe, #00f2fe);
            background-size: 1200% 1200%;
            animation: gradientShift 30s ease infinite;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            25% { background-position: 50% 100%; }
            50% { background-position: 100% 50%; }
            75% { background-position: 50% 0%; }
            100% { background-position: 0% 50%; }
        }
        
        .particle-canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        
        /* ========== MAIN LAYOUT ========== */
        .app-container {
            width: 100%;
            height: 100vh;
            display: flex;
            backdrop-filter: blur(20px);
            position: relative;
        }
        
        /* ========== SIDEBAR ========== */
        .sidebar {
            width: var(--sidebar-width);
            height: 100vh;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border-right: 1px solid var(--glass-border);
            display: flex;
            flex-direction: column;
            transition: transform var(--transition-speed) var(--transition-timing);
            position: relative;
            z-index: 100;
        }
        
        .sidebar.collapsed {
            transform: translateX(-100%);
        }
        
        .sidebar-header {
            padding: 30px 25px;
            background: var(--primary-gradient);
            position: relative;
            overflow: hidden;
        }
        
        .sidebar-header::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: pulse 3s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.5; }
            50% { transform: scale(1.1); opacity: 0.8; }
        }
        
        .logo-container {
            display: flex;
            align-items: center;
            gap: 15px;
            position: relative;
            z-index: 1;
        }
        
        .logo-icon {
            width: 50px;
            height: 50px;
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            color: white;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
        
        .logo-text h1 {
            color: white;
            font-size: 22px;
            font-weight: 800;
            margin-bottom: 4px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        .logo-text p {
            color: rgba(255, 255, 255, 0.9);
            font-size: 12px;
            font-weight: 500;
        }
        
        /* Chat Sessions */
        .chat-sessions {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        .session-item {
            padding: 15px;
            margin-bottom: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            cursor: pointer;
            transition: all var(--transition-speed) var(--transition-timing);
            border: 1px solid transparent;
            position: relative;
            overflow: hidden;
        }
        
        .session-item::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 0;
            height: 100%;
            background: var(--primary-gradient);
            opacity: 0.1;
            transition: width var(--transition-speed) var(--transition-timing);
        }
        
        .session-item:hover::before {
            width: 100%;
        }
        
        .session-item:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: var(--glass-border);
            transform: translateX(5px);
        }
        
        .session-item.active {
            background: rgba(102, 126, 234, 0.1);
            border-color: #667eea;
        }
        
        .session-title {
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 5px;
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
        
        /* Sidebar Actions */
        .sidebar-actions {
            padding: 20px;
            border-top: 1px solid var(--glass-border);
        }
        
        .btn-new-chat {
            width: 100%;
            padding: 15px;
            background: var(--primary-gradient);
            color: white;
            border: none;
            border-radius: 15px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all var(--transition-speed) var(--transition-timing);
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .btn-new-chat::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: translate(-50%, -50%);
            transition: width 0.4s, height 0.4s;
        }
        
        .btn-new-chat:hover::before {
            width: 300px;
            height: 300px;
        }
        
        .btn-new-chat:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
        }
        
        /* ========== MAIN CONTENT ========== */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            position: relative;
            overflow: hidden;
        }
        
        /* Header */
        .chat-header {
            height: var(--header-height);
            padding: 0 30px;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--glass-border);
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
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid var(--glass-border);
            color: var(--text-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all var(--transition-speed) var(--transition-timing);
        }
        
        .menu-toggle:hover {
            background: var(--primary-gradient);
            color: white;
            transform: scale(1.1);
        }
        
        .mode-selector {
            display: flex;
            gap: 10px;
            padding: 5px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            border: 1px solid var(--glass-border);
        }
        
        .mode-btn {
            padding: 8px 16px;
            background: transparent;
            border: none;
            border-radius: 10px;
            color: var(--text-secondary);
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-speed) var(--transition-timing);
            position: relative;
        }
        
        .mode-btn.active {
            background: var(--primary-gradient);
            color: white;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        
        .mode-btn:hover:not(.active) {
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
        }
        
        .header-right {
            display: flex;
            gap: 10px;
        }
        
        .header-btn {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid var(--glass-border);
            color: var(--text-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all var(--transition-speed) var(--transition-timing);
            position: relative;
        }
        
        .header-btn:hover {
            background: var(--primary-gradient);
            color: white;
            transform: scale(1.1);
        }
        
        .header-btn .tooltip {
            position: absolute;
            bottom: -40px;
            left: 50%;
            transform: translateX(-50%);
            background: #333;
            color: white;
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 12px;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity var(--transition-speed);
        }
        
        .header-btn:hover .tooltip {
            opacity: 1;
        }
        
        /* Messages Area */
        .messages-area {
            flex: 1;
            overflow-y: auto;
            padding: 30px;
            position: relative;
            scroll-behavior: smooth;
        }
        
        .messages-area::-webkit-scrollbar {
            width: 10px;
        }
        
        .messages-area::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
        }
        
        .messages-area::-webkit-scrollbar-thumb {
            background: var(--primary-gradient);
            border-radius: 10px;
        }
        
        /* Message Styles */
        .message {
            margin-bottom: 25px;
            display: flex;
            align-items: flex-start;
            gap: 15px;
            animation: messageSlide 0.3s ease-out;
            position: relative;
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
            width: 45px;
            height: 45px;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            flex-shrink: 0;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        .message.user .message-avatar {
            background: var(--primary-gradient);
            color: white;
        }
        
        .message.assistant .message-avatar {
            background: var(--secondary-gradient);
            color: white;
        }
        
        .message-avatar::after {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
            animation: rotate 10s linear infinite;
        }
        
        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .message-content-wrapper {
            max-width: 70%;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .message-header {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0 10px;
        }
        
        .message-author {
            font-weight: 600;
            font-size: 14px;
            color: var(--text-primary);
        }
        
        .message-time {
            font-size: 12px;
            color: var(--text-muted);
        }
        
        .message-bubble {
            padding: 18px 24px;
            border-radius: 20px;
            position: relative;
            word-wrap: break-word;
            line-height: 1.6;
            font-size: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        }
        
        .message.user .message-bubble {
            background: var(--primary-gradient);
            color: white;
            border-bottom-right-radius: 5px;
        }
        
        .message.assistant .message-bubble {
            background: rgba(255, 255, 255, 0.9);
            color: var(--text-primary);
            border: 1px solid var(--glass-border);
            border-bottom-left-radius: 5px;
        }
        
        [data-theme="dark"] .message.assistant .message-bubble {
            background: rgba(40, 40, 40, 0.9);
        }
        
        /* Code blocks in messages */
        .message-bubble pre {
            background: rgba(0, 0, 0, 0.8);
            color: #e4e4e4;
            padding: 15px;
            border-radius: 10px;
            overflow-x: auto;
            margin: 10px 0;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            line-height: 1.5;
            max-height: 600px;
            overflow-y: auto;
        }
        
        /* Extended code blocks in advanced mode */
        .advanced-code-active .message-bubble pre {
            max-height: none;
            border: 2px solid rgba(102, 126, 234, 0.3);
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.2);
        }
        
        .message-bubble code {
            background: rgba(0, 0, 0, 0.2);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
        }
        
        .message.user .message-bubble code {
            background: rgba(255, 255, 255, 0.2);
        }
        
        /* Message Actions */
        .message-actions {
            display: flex;
            gap: 8px;
            padding: 0 10px;
            opacity: 0;
            transition: opacity var(--transition-speed);
        }
        
        .message:hover .message-actions {
            opacity: 1;
        }
        
        .message-action {
            padding: 6px 10px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid var(--glass-border);
            border-radius: 8px;
            font-size: 12px;
            cursor: pointer;
            transition: all var(--transition-speed) var(--transition-timing);
            color: var(--text-secondary);
        }
        
        .message-action:hover {
            background: var(--primary-gradient);
            color: white;
            transform: scale(1.05);
        }
        
        /* Typing Indicator */
        .typing-indicator {
            display: none;
            align-items: center;
            gap: 15px;
            padding: 20px 30px;
            animation: fadeIn 0.3s;
        }
        
        .typing-indicator.active {
            display: flex;
        }
        
        .typing-dots {
            display: flex;
            gap: 5px;
        }
        
        .typing-dot {
            width: 10px;
            height: 10px;
            background: var(--primary-gradient);
            border-radius: 50%;
            animation: typingAnimation 1.4s infinite;
        }
        
        .typing-dot:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-dot:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typingAnimation {
            0%, 60%, 100% {
                transform: translateY(0) scale(1);
                opacity: 0.7;
            }
            30% {
                transform: translateY(-15px) scale(1.2);
                opacity: 1;
            }
        }
        
        /* ========== INPUT SECTION ========== */
        .input-section {
            padding: 20px 30px 30px;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border-top: 1px solid var(--glass-border);
            position: relative;
        }
        
        /* Advanced Features Bar */
        .advanced-features {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            border: 1px solid var(--glass-border);
        }
        
        .feature-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 15px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            font-size: 13px;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all var(--transition-speed) var(--transition-timing);
            border: 1px solid transparent;
        }
        
        .feature-item:hover {
            background: var(--primary-gradient);
            color: white;
            transform: scale(1.05);
        }
        
        .feature-item.active {
            background: var(--primary-gradient);
            color: white;
            border-color: rgba(255, 255, 255, 0.3);
        }
        
        .feature-item#featureAdvancedCoding.active {
            background: var(--danger-gradient);
            animation: pulse 2s ease-in-out infinite;
            font-weight: 600;
        }
        
        /* Token Usage */
        .token-usage {
            flex: 1;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .token-bar {
            flex: 1;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }
        
        .token-fill {
            height: 100%;
            background: var(--primary-gradient);
            border-radius: 10px;
            transition: width 0.5s ease;
            position: relative;
            overflow: hidden;
        }
        
        .token-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            bottom: 0;
            right: 0;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(255, 255, 255, 0.3),
                transparent
            );
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .token-text {
            font-size: 12px;
            color: var(--text-secondary);
            white-space: nowrap;
        }
        
        /* Input Container */
        .input-container {
            display: flex;
            gap: 15px;
            align-items: flex-end;
        }
        
        .input-wrapper {
            flex: 1;
            position: relative;
        }
        
        .input-box {
            width: 100%;
            min-height: 50px;
            max-height: 200px;
            padding: 15px 60px 15px 20px;
            background: rgba(255, 255, 255, 0.1);
            border: 2px solid var(--glass-border);
            border-radius: 20px;
            color: var(--text-primary);
            font-size: 15px;
            resize: none;
            transition: all var(--transition-speed) var(--transition-timing);
            font-family: 'Inter', sans-serif;
            line-height: 1.6;
        }
        
        .input-box:focus {
            outline: none;
            border-color: #667eea;
            background: rgba(255, 255, 255, 0.15);
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
        }
        
        .input-box::placeholder {
            color: var(--text-muted);
        }
        
        /* Input Actions */
        .input-actions {
            position: absolute;
            right: 10px;
            bottom: 10px;
            display: flex;
            gap: 5px;
        }
        
        .input-action-btn {
            width: 35px;
            height: 35px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid var(--glass-border);
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all var(--transition-speed) var(--transition-timing);
        }
        
        .input-action-btn:hover {
            background: var(--primary-gradient);
            color: white;
            transform: scale(1.1);
        }
        
        /* Send Button */
        .send-button {
            padding: 15px 30px;
            background: var(--primary-gradient);
            color: white;
            border: none;
            border-radius: 20px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all var(--transition-speed) var(--transition-timing);
            display: flex;
            align-items: center;
            gap: 10px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        }
        
        .send-button::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: translate(-50%, -50%);
            transition: width 0.4s, height 0.4s;
        }
        
        .send-button:hover::before {
            width: 300px;
            height: 300px;
        }
        
        .send-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        }
        
        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        /* File Upload Area */
        .file-upload-area {
            display: none;
            margin-bottom: 15px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            border: 2px dashed var(--glass-border);
            border-radius: 15px;
            text-align: center;
            position: relative;
            transition: all var(--transition-speed) var(--transition-timing);
        }
        
        .file-upload-area.active {
            display: block;
            animation: slideDown 0.3s ease;
        }
        
        .file-upload-area.dragover {
            background: rgba(102, 126, 234, 0.1);
            border-color: #667eea;
        }
        
        @keyframes slideDown {
            from {
                opacity: 0;
                max-height: 0;
            }
            to {
                opacity: 1;
                max-height: 300px;
            }
        }
        
        .file-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
        }
        
        .file-item {
            padding: 10px 15px;
            background: var(--primary-gradient);
            color: white;
            border-radius: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            animation: fadeIn 0.3s;
            position: relative;
        }
        
        .file-item .remove-file {
            cursor: pointer;
            opacity: 0.8;
            transition: opacity var(--transition-speed);
        }
        
        .file-item .remove-file:hover {
            opacity: 1;
        }
        
        /* ========== MODALS ========== */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
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
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .modal-content {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 25px;
            padding: 40px;
            max-width: 600px;
            width: 100%;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: var(--glass-shadow);
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
        
        .modal-header {
            margin-bottom: 30px;
        }
        
        .modal-title {
            font-size: 24px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 10px;
        }
        
        .modal-subtitle {
            color: var(--text-secondary);
            font-size: 14px;
        }
        
        .modal-body {
            margin-bottom: 30px;
        }
        
        .modal-actions {
            display: flex;
            gap: 15px;
        }
        
        .modal-btn {
            flex: 1;
            padding: 15px;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all var(--transition-speed) var(--transition-timing);
            text-align: center;
        }
        
        .modal-btn-primary {
            background: var(--primary-gradient);
            color: white;
            border: none;
        }
        
        .modal-btn-secondary {
            background: transparent;
            color: var(--text-primary);
            border: 2px solid var(--glass-border);
        }
        
        .modal-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.2);
        }
        
        /* ========== NOTIFICATIONS ========== */
        .notification-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 2000;
            pointer-events: none;
        }
        
        .notification {
            padding: 20px 25px;
            border-radius: 15px;
            color: white;
            font-size: 14px;
            font-weight: 500;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
            animation: slideInRight 0.3s ease-out;
            backdrop-filter: blur(10px);
            pointer-events: auto;
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
            background: var(--success-gradient);
        }
        
        .notification.error {
            background: var(--danger-gradient);
        }
        
        .notification.warning {
            background: var(--warning-gradient);
        }
        
        .notification.info {
            background: var(--info-gradient);
        }
        
        .notification-icon {
            font-size: 20px;
        }
        
        .notification-close {
            margin-left: auto;
            cursor: pointer;
            opacity: 0.8;
            transition: opacity var(--transition-speed);
        }
        
        .notification-close:hover {
            opacity: 1;
        }
        
        /* ========== LOADING STATES ========== */
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(5px);
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
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            border-radius: 50%;
            border: 4px solid transparent;
            border-top-color: white;
            animation: spin 1s linear infinite;
        }
        
        .loading-spinner::after {
            animation-delay: 0.5s;
            opacity: 0.5;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* ========== RESPONSIVE DESIGN ========== */
        @media (max-width: 768px) {
            .sidebar {
                position: fixed;
                z-index: 200;
                box-shadow: 5px 0 20px rgba(0, 0, 0, 0.3);
            }
            
            .message-content-wrapper {
                max-width: 85%;
            }
            
            .advanced-features {
                flex-wrap: wrap;
            }
            
            .mode-selector {
                display: none;
            }
        }
        
        @media (max-width: 480px) {
            .header-right {
                display: none;
            }
            
            .input-actions {
                bottom: 15px;
                right: 15px;
            }
            
            .messages-area {
                padding: 20px 15px;
            }
        }
        
        /* ========== ANIMATIONS & EFFECTS ========== */
        .ripple {
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.5);
            transform: scale(0);
            animation: ripple 0.6s ease-out;
            pointer-events: none;
        }
        
        @keyframes ripple {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }
        
        .glow {
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
            animation: glow 2s ease-in-out infinite alternate;
        }
        
        @keyframes glow {
            from { box-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
            to { box-shadow: 0 0 30px rgba(102, 126, 234, 0.8); }
        }
    </style>
</head>
<body>
    <!-- Background Effects -->
    <div class="background-container">
        <div class="animated-gradient"></div>
        <canvas class="particle-canvas" id="particleCanvas"></canvas>
    </div>
    
    <!-- Loading Overlay -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-spinner"></div>
    </div>
    
    <!-- Notification Container -->
    <div class="notification-container" id="notificationContainer"></div>
    
    <!-- Main App Container -->
    <div class="app-container">
        <!-- Sidebar -->
        <aside class="sidebar" id="sidebar">
            <div class="sidebar-header">
                <div class="logo-container">
                    <div class="logo-icon">
                        <i class="fas fa-brain"></i>
                    </div>
                    <div class="logo-text">
                        <h1>Jack's AI Ultra</h1>
                        <p>Next Generation Intelligence</p>
                    </div>
                </div>
            </div>
            
            <div class="chat-sessions" id="chatSessions">
                <!-- Sessions will be dynamically loaded here -->
            </div>
            
            <div class="sidebar-actions">
                <button class="btn-new-chat" id="btnNewChat">
                    <i class="fas fa-plus"></i> New Chat
                </button>
            </div>
        </aside>
        
        <!-- Main Content -->
        <main class="main-content">
            <!-- Chat Header -->
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
                    <button class="header-btn" id="btnTheme">
                        <i class="fas fa-moon"></i>
                        <span class="tooltip">Toggle Theme</span>
                    </button>
                    <button class="header-btn" id="btnExport">
                        <i class="fas fa-download"></i>
                        <span class="tooltip">Export Chat</span>
                    </button>
                    <button class="header-btn" id="btnSettings">
                        <i class="fas fa-cog"></i>
                        <span class="tooltip">Settings</span>
                    </button>
                </div>
            </header>
            
            <!-- Messages Area -->
            <div class="messages-area" id="messagesArea">
                <!-- Messages will be dynamically loaded here -->
            </div>
            
            <!-- Typing Indicator -->
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
            
            <!-- Input Section -->
            <div class="input-section">
                <!-- Advanced Features Bar -->
                <div class="advanced-features">
                    <div class="feature-item" id="featureEnhance">
                        <i class="fas fa-magic"></i>
                        <span>Enhance</span>
                    </div>
                    <div class="feature-item" id="featureFiles">
                        <i class="fas fa-paperclip"></i>
                        <span>Files</span>
                    </div>
                    <div class="feature-item" id="featureVoice">
                        <i class="fas fa-microphone"></i>
                        <span>Voice</span>
                    </div>
                    <div class="feature-item" id="featureWeb">
                        <i class="fas fa-globe"></i>
                        <span>Web Search</span>
                    </div>
                    <div class="feature-item" id="featureAdvancedCoding">
                        <i class="fas fa-code-branch"></i>
                        <span>Advanced Code</span>
                    </div>
                    
                    <div class="token-usage">
                        <div class="token-bar">
                            <div class="token-fill" id="tokenFill" style="width: 0%"></div>
                        </div>
                        <span class="token-text" id="tokenText">0 / 1M tokens</span>
                    </div>
                </div>
                
                <!-- File Upload Area -->
                <div class="file-upload-area" id="fileUploadArea">
                    <i class="fas fa-cloud-upload-alt" style="font-size: 40px; color: var(--text-muted); margin-bottom: 10px;"></i>
                    <p style="color: var(--text-secondary); margin-bottom: 10px;">Drop files here or click to browse</p>
                    <p style="font-size: 12px; color: var(--text-muted);">Supports images, documents, spreadsheets, code, and more</p>
                    <div class="file-list" id="fileList"></div>
                </div>
                
                <!-- Input Container -->
                <div class="input-container">
                    <div class="input-wrapper">
                        <textarea 
                            class="input-box" 
                            id="messageInput" 
                            placeholder="Ask me anything... (Shift+Enter for new line)"
                            rows="1"
                        ></textarea>
                        <div class="input-actions">
                            <button class="input-action-btn" id="btnEmoji">
                                <i class="fas fa-smile"></i>
                            </button>
                            <button class="input-action-btn" id="btnMarkdown">
                                <i class="fab fa-markdown"></i>
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
    
    <!-- Settings Modal -->
    <div class="modal-overlay" id="settingsModal">
        <div class="modal-content">
            <div class="modal-header">
                <h2 class="modal-title">⚙️ Settings</h2>
                <p class="modal-subtitle">Customize your AI experience</p>
            </div>
            
            <div class="modal-body">
                <!-- Settings content will be dynamically loaded -->
            </div>
            
            <div class="modal-actions">
                <button class="modal-btn modal-btn-secondary" onclick="closeModal('settingsModal')">
                    Cancel
                </button>
                <button class="modal-btn modal-btn-primary" onclick="saveSettings()">
                    Save Settings
                </button>
            </div>
        </div>
    </div>
    
    <!-- Hidden file input -->
    <input type="file" id="fileInput" multiple style="display: none;">
    
    <!-- Scripts -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-javascript.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.3.0/marked.min.js"></script>
    
    <script>
        // ========== APPLICATION STATE ==========
        const AppState = {
            sessionId: null,
            currentMode: 'balanced',
            messages: [],
            tokenUsage: 0,
            maxTokens: 1000000,
            attachedFiles: [],
            isTyping: false,
            isDarkTheme: true,
            advancedCoding: false,
            settings: {
                autoEnhance: false,
                streamResponses: true,
                saveHistory: true,
                notifications: true,
                soundEffects: false
            },
            webSocket: null,
            voiceRecognition: null
        };
        
        // ========== INITIALIZATION ==========
        document.addEventListener('DOMContentLoaded', function() {
            initializeApp();
            setupEventListeners();
            setupWebSocket();
            createParticleEffect();
            loadTheme();
            loadSettings();
        });
        
        function initializeApp() {
            // Check for existing session or create new
            AppState.sessionId = localStorage.getItem('sessionId');
            if (!AppState.sessionId) {
                createNewSession();
            } else {
                loadSession();
            }
            
            // Initialize components
            updateTokenDisplay();
            addWelcomeMessage();
        }
        
        function createNewSession() {
            AppState.sessionId = generateUUID();
            localStorage.setItem('sessionId', AppState.sessionId);
            AppState.messages = [];
            AppState.tokenUsage = 0;
            
            // Send to server
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
        
        // ========== EVENT LISTENERS ==========
        function setupEventListeners() {
            // Menu toggle
            document.getElementById('menuToggle').addEventListener('click', toggleSidebar);
            
            // New chat
            document.getElementById('btnNewChat').addEventListener('click', startNewChat);
            
            // Mode selection
            document.querySelectorAll('.mode-btn').forEach(btn => {
                btn.addEventListener('click', () => changeMode(btn.dataset.mode));
            });
            
            // Theme toggle
            document.getElementById('btnTheme').addEventListener('click', toggleTheme);
            
            // Export chat
            document.getElementById('btnExport').addEventListener('click', exportChat);
            
            // Settings
            document.getElementById('btnSettings').addEventListener('click', openSettings);
            
            // Message input
            const messageInput = document.getElementById('messageInput');
            messageInput.addEventListener('input', autoResizeTextarea);
            messageInput.addEventListener('keydown', handleInputKeydown);
            
            // Send button
            document.getElementById('sendButton').addEventListener('click', sendMessage);
            
            // Advanced features
            document.getElementById('featureEnhance').addEventListener('click', toggleEnhancement);
            document.getElementById('featureFiles').addEventListener('click', toggleFileUpload);
            document.getElementById('featureVoice').addEventListener('click', toggleVoiceInput);
            document.getElementById('featureWeb').addEventListener('click', toggleWebSearch);
            document.getElementById('featureAdvancedCoding').addEventListener('click', toggleAdvancedCoding);
            
            // File upload
            setupFileUpload();
            
            // Input actions
            document.getElementById('btnEmoji').addEventListener('click', insertEmoji);
            document.getElementById('btnMarkdown').addEventListener('click', toggleMarkdownHelp);
        }
        
        // ========== WEBSOCKET CONNECTION ==========
        function setupWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            AppState.webSocket = new WebSocket(wsUrl);
            
            AppState.webSocket.onopen = () => {
                console.log('WebSocket connected');
                AppState.webSocket.send(JSON.stringify({
                    type: 'join',
                    sessionId: AppState.sessionId
                }));
            };
            
            AppState.webSocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            AppState.webSocket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            AppState.webSocket.onclose = () => {
                console.log('WebSocket disconnected');
                // Attempt to reconnect after 3 seconds
                setTimeout(setupWebSocket, 3000);
            };
        }
        
        function handleWebSocketMessage(data) {
            switch(data.type) {
                case 'stream':
                    handleStreamedResponse(data);
                    break;
                case 'complete':
                    handleCompleteResponse(data);
                    break;
                case 'error':
                    handleErrorResponse(data);
                    break;
                case 'status':
                    updateStatus(data);
                    break;
            }
        }
        
        // ========== CORE FUNCTIONS ==========
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message && AppState.attachedFiles.length === 0) return;
            
            // Clear input
            input.value = '';
            autoResizeTextarea();
            
            // Disable send button
            document.getElementById('sendButton').disabled = true;
            
            // Add user message to UI
            addMessage('user', message);
            
            // Show typing indicator
            showTypingIndicator();
            
            // Prepare form data
            const formData = new FormData();
            formData.append('message', message);
            formData.append('sessionId', AppState.sessionId);
            formData.append('mode', AppState.currentMode);
            formData.append('advancedCoding', AppState.advancedCoding ? 'true' : 'false');
            
            // Add files if any
            AppState.attachedFiles.forEach(file => {
                formData.append('files', file);
            });
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    body: formData
                });
                
                if (AppState.settings.streamResponses && response.body) {
                    handleStreamingResponse(response.body);
                } else {
                    const data = await response.json();
                    handleResponse(data);
                }
            } catch (error) {
                console.error('Error sending message:', error);
                hideTypingIndicator();
                showNotification('Failed to send message', 'error');
                document.getElementById('sendButton').disabled = false;
            }
            
            // Clear attached files
            clearFiles();
        }
        
        async function handleStreamingResponse(stream) {
            const reader = stream.getReader();
            const decoder = new TextDecoder();
            let assistantMessage = '';
            let messageElement = null;
            
            try {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6);
                            if (data === '[DONE]') {
                                hideTypingIndicator();
                                document.getElementById('sendButton').disabled = false;
                            } else {
                                try {
                                    const parsed = JSON.parse(data);
                                    if (parsed.content) {
                                        assistantMessage += parsed.content;
                                        
                                        if (!messageElement) {
                                            hideTypingIndicator();
                                            messageElement = addMessage('assistant', assistantMessage, false);
                                        } else {
                                            updateMessage(messageElement, assistantMessage);
                                        }
                                    }
                                    
                                    if (parsed.tokenUsage) {
                                        AppState.tokenUsage = parsed.tokenUsage;
                                        updateTokenDisplay();
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
                hideTypingIndicator();
                showNotification('Stream interrupted', 'error');
            }
        }
        
        function handleResponse(data) {
            hideTypingIndicator();
            
            if (data.success) {
                addMessage('assistant', data.response);
                AppState.tokenUsage = data.tokenUsage || AppState.tokenUsage;
                updateTokenDisplay();
                
                // Check token usage
                if (AppState.tokenUsage > AppState.maxTokens * 0.8) {
                    showNotification('Approaching token limit', 'warning');
                }
            } else {
                showNotification(data.error || 'Failed to get response', 'error');
            }
            
            document.getElementById('sendButton').disabled = false;
        }
        
        // ========== UI FUNCTIONS ==========
        function addMessage(role, content, save = true) {
            const messagesArea = document.getElementById('messagesArea');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            
            const time = new Date().toLocaleTimeString([], { 
                hour: '2-digit', 
                minute: '2-digit' 
            });
            
            // Process content for markdown and code blocks
            const processedContent = processMessageContent(content);
            
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
                        <button class="message-action" onclick="copyMessage(this)">
                            <i class="fas fa-copy"></i> Copy
                        </button>
                        <button class="message-action" onclick="editMessage(this)">
                            <i class="fas fa-edit"></i> Edit
                        </button>
                        <button class="message-action" onclick="regenerateMessage(this)">
                            <i class="fas fa-redo"></i> Regenerate
                        </button>
                    </div>
                </div>
            `;
            
            messagesArea.appendChild(messageDiv);
            messagesArea.scrollTop = messagesArea.scrollHeight;
            
            // Add ripple effect
            addRippleEffect(messageDiv);
            
            // Save to state
            if (save) {
                AppState.messages.push({ role, content, timestamp: new Date() });
                saveSession();
            }
            
            return messageDiv;
        }
        
        function updateMessage(element, content) {
            const bubble = element.querySelector('.message-bubble');
            if (bubble) {
                bubble.innerHTML = processMessageContent(content);
                // Re-highlight code blocks
                element.querySelectorAll('pre code').forEach(block => {
                    Prism.highlightElement(block);
                });
            }
        }
        
        function processMessageContent(content) {
            // Convert markdown to HTML
            let processed = marked.parse(content);
            
            // Add syntax highlighting to code blocks
            processed = processed.replace(/<pre><code class="language-(\\w+)">/g, (match, lang) => {
                return `<pre><code class="language-${lang}">`;
            });
            
            return processed;
        }
        
        function addWelcomeMessage() {
            let welcome = `# Welcome to Jack's AI Ultra! 🚀
            
I'm your advanced AI assistant powered by cutting-edge technology. I can help you with:

- 💡 **Complex problem solving** - From mathematics to business strategy
- 🎨 **Creative projects** - Writing, design ideas, and artistic concepts
- 💻 **Programming** - Code generation, debugging, and optimization
- 📊 **Data analysis** - Process spreadsheets, visualize data, and insights
- 🔬 **Research** - Deep dives into any topic with comprehensive analysis
- 📝 **Document processing** - Analyze PDFs, Word docs, and more

Select a mode above to optimize my responses for your needs, or just start chatting!`;

            if (AppState.advancedCoding) {
                welcome += `\n\n⚡ **ADVANCED CODING MODE IS ACTIVE!** ⚡
                
I will provide:
- Complete, production-ready implementations
- Enterprise-grade code with all best practices
- Comprehensive documentation and tests
- No shortcuts or placeholders - everything implemented
- Multiple approaches and full project structures
- Even simple requests will receive professional solutions!`;
            }

            welcome += `\n\nHow can I assist you today?`;
            
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
                `${formatNumber(AppState.tokenUsage)} / ${formatNumber(AppState.maxTokens)} tokens`;
            
            // Change color based on usage
            const fill = document.getElementById('tokenFill');
            if (percentage > 80) {
                fill.style.background = 'var(--danger-gradient)';
            } else if (percentage > 60) {
                fill.style.background = 'var(--warning-gradient)';
            } else {
                fill.style.background = 'var(--primary-gradient)';
            }
        }
        
        function showNotification(message, type = 'info') {
            if (!AppState.settings.notifications) return;
            
            const container = document.getElementById('notificationContainer');
            const notification = document.createElement('div');
            notification.className = `notification ${type}`;
            
            const icons = {
                success: 'check-circle',
                error: 'exclamation-circle',
                warning: 'exclamation-triangle',
                info: 'info-circle'
            };
            
            notification.innerHTML = `
                <i class="fas fa-${icons[type]} notification-icon"></i>
                <span>${message}</span>
                <i class="fas fa-times notification-close" onclick="this.parentElement.remove()"></i>
            `;
            
            container.appendChild(notification);
            
            // Auto remove after 5 seconds
            setTimeout(() => {
                notification.style.animation = 'slideOutRight 0.3s ease-out';
                setTimeout(() => notification.remove(), 300);
            }, 5000);
            
            // Play sound effect if enabled
            if (AppState.settings.soundEffects) {
                playNotificationSound(type);
            }
        }
        
        // ========== MODE MANAGEMENT ==========
        function changeMode(mode) {
            AppState.currentMode = mode;
            
            // Update UI
            document.querySelectorAll('.mode-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.mode === mode);
            });
            
            // Show notification
            const modeNames = {
                balanced: 'Balanced Mode',
                creative: 'Creative Mode',
                precise: 'Precise Mode',
                code: 'Code Mode',
                research: 'Research Mode'
            };
            
            showNotification(`Switched to ${modeNames[mode]}`, 'success');
            
            // Special notification if advanced coding is active
            if (AppState.advancedCoding && mode === 'code') {
                showNotification(
                    '💻 Code Mode + Advanced Coding = Maximum code generation power!',
                    'info'
                );
            } else if (AppState.advancedCoding && mode !== 'code') {
                showNotification(
                    'Note: Advanced Coding works best in Code Mode',
                    'warning'
                );
            }
        }
        
        // ========== FILE HANDLING ==========
        function setupFileUpload() {
            const fileInput = document.getElementById('fileInput');
            const uploadArea = document.getElementById('fileUploadArea');
            
            fileInput.addEventListener('change', handleFileSelect);
            
            // Drag and drop
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
                if (file.size > 200 * 1024 * 1024) {
                    showNotification(`File ${file.name} is too large (max 200MB)`, 'error');
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
        }
        
        function toggleFileUpload() {
            const uploadArea = document.getElementById('fileUploadArea');
            uploadArea.classList.toggle('active');
            document.getElementById('featureFiles').classList.toggle('active');
        }
        
        // ========== ADVANCED FEATURES ==========
        function toggleEnhancement() {
            AppState.settings.autoEnhance = !AppState.settings.autoEnhance;
            document.getElementById('featureEnhance').classList.toggle('active');
            showNotification(
                AppState.settings.autoEnhance ? 'Auto-enhancement enabled' : 'Auto-enhancement disabled',
                'info'
            );
        }
        
        function toggleVoiceInput() {
            if (!('webkitSpeechRecognition' in window)) {
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
            const recognition = new webkitSpeechRecognition();
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
            document.getElementById('featureVoice').classList.add('active');
            showNotification('Voice recording started', 'info');
        }
        
        function stopVoiceRecording() {
            AppState.voiceRecognition.stop();
            AppState.voiceRecognition.isRecording = false;
            document.getElementById('featureVoice').classList.remove('active');
            showNotification('Voice recording stopped', 'info');
        }
        
        function toggleWebSearch() {
            const isActive = document.getElementById('featureWeb').classList.toggle('active');
            showNotification(
                isActive ? 'Web search enabled for this message' : 'Web search disabled',
                'info'
            );
        }
        
        function toggleAdvancedCoding() {
            AppState.advancedCoding = !AppState.advancedCoding;
            document.getElementById('featureAdvancedCoding').classList.toggle('active');
            document.body.classList.toggle('advanced-code-active', AppState.advancedCoding);
            
            if (AppState.advancedCoding) {
                showNotification(
                    '🚀 ADVANCED CODING MODE ACTIVATED! All code will be complete, production-ready, and comprehensive. No shortcuts!',
                    'warning'
                );
                
                // Auto-switch to code mode if not already
                if (AppState.currentMode !== 'code') {
                    changeMode('code');
                    showNotification('Automatically switched to Code Mode for optimal results', 'info');
                }
            } else {
                showNotification('Advanced coding mode disabled', 'info');
            }
        }
        
        // ========== THEME MANAGEMENT ==========
        function toggleTheme() {
            AppState.isDarkTheme = !AppState.isDarkTheme;
            document.documentElement.setAttribute('data-theme', AppState.isDarkTheme ? 'dark' : 'light');
            
            const themeIcon = document.querySelector('#btnTheme i');
            themeIcon.className = AppState.isDarkTheme ? 'fas fa-moon' : 'fas fa-sun';
            
            localStorage.setItem('theme', AppState.isDarkTheme ? 'dark' : 'light');
            showNotification(`Switched to ${AppState.isDarkTheme ? 'dark' : 'light'} theme`, 'success');
        }
        
        function loadTheme() {
            const savedTheme = localStorage.getItem('theme');
            AppState.isDarkTheme = savedTheme !== 'light';
            document.documentElement.setAttribute('data-theme', AppState.isDarkTheme ? 'dark' : 'light');
            
            const themeIcon = document.querySelector('#btnTheme i');
            if (themeIcon) {
                themeIcon.className = AppState.isDarkTheme ? 'fas fa-moon' : 'fas fa-sun';
            }
        }
        
        // ========== UTILITY FUNCTIONS ==========
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
            const format = await selectExportFormat();
            
            let content = '';
            const timestamp = new Date().toISOString().split('T')[0];
            
            if (format === 'json') {
                content = JSON.stringify({
                    sessionId: AppState.sessionId,
                    mode: AppState.currentMode,
                    messages: AppState.messages,
                    tokenUsage: AppState.tokenUsage,
                    exportDate: new Date().toISOString()
                }, null, 2);
            } else if (format === 'markdown') {
                content = `# Jack's AI Ultra Chat Export\\n\\n`;
                content += `**Date:** ${new Date().toLocaleDateString()}\\n`;
                content += `**Session:** ${AppState.sessionId}\\n`;
                content += `**Mode:** ${AppState.currentMode}\\n\\n`;
                content += `---\\n\\n`;
                
                AppState.messages.forEach(msg => {
                    content += `### ${msg.role === 'user' ? '👤 You' : '🤖 AI'}\\n`;
                    content += `*${new Date(msg.timestamp).toLocaleString()}*\\n\\n`;
                    content += `${msg.content}\\n\\n`;
                    content += `---\\n\\n`;
                });
            } else {
                content = `Jack's AI Ultra Chat Export\\n`;
                content += `Date: ${new Date().toLocaleDateString()}\\n`;
                content += `Session: ${AppState.sessionId}\\n\\n`;
                
                AppState.messages.forEach(msg => {
                    content += `[${new Date(msg.timestamp).toLocaleTimeString()}] `;
                    content += `${msg.role === 'user' ? 'You' : 'AI'}: `;
                    content += `${msg.content}\\n\\n`;
                });
            }
            
            const blob = new Blob([content], { 
                type: format === 'json' ? 'application/json' : 'text/plain' 
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `chat-export-${timestamp}.${format === 'json' ? 'json' : format === 'markdown' ? 'md' : 'txt'}`;
            a.click();
            URL.revokeObjectURL(url);
            
            showNotification('Chat exported successfully', 'success');
        }
        
        function selectExportFormat() {
            return new Promise((resolve) => {
                // For simplicity, defaulting to markdown
                // In a real app, you'd show a modal to select format
                resolve('markdown');
            });
        }
        
        function copyMessage(button) {
            const bubble = button.closest('.message').querySelector('.message-bubble');
            const text = bubble.textContent;
            
            navigator.clipboard.writeText(text).then(() => {
                showNotification('Message copied to clipboard', 'success');
            }).catch(err => {
                console.error('Failed to copy:', err);
                showNotification('Failed to copy message', 'error');
            });
        }
        
        function editMessage(button) {
            const message = button.closest('.message');
            const bubble = message.querySelector('.message-bubble');
            const originalContent = bubble.textContent;
            
            // Create editable textarea
            const textarea = document.createElement('textarea');
            textarea.className = 'message-edit-textarea';
            textarea.value = originalContent;
            textarea.style.width = '100%';
            textarea.style.minHeight = '100px';
            
            bubble.innerHTML = '';
            bubble.appendChild(textarea);
            
            // Add save/cancel buttons
            const actions = document.createElement('div');
            actions.innerHTML = `
                <button onclick="saveEdit(this, '${originalContent}')">Save</button>
                <button onclick="cancelEdit(this, '${originalContent}')">Cancel</button>
            `;
            bubble.appendChild(actions);
            
            textarea.focus();
        }
        
        async function regenerateMessage(button) {
            const message = button.closest('.message');
            const previousMessage = message.previousElementSibling;
            
            if (!previousMessage || !previousMessage.classList.contains('user')) {
                showNotification('No user message to regenerate from', 'error');
                return;
            }
            
            // Remove current assistant message
            message.remove();
            
            // Resend the previous user message
            const userContent = previousMessage.querySelector('.message-bubble').textContent;
            
            showTypingIndicator();
            
            const formData = new FormData();
            formData.append('message', userContent);
            formData.append('sessionId', AppState.sessionId);
            formData.append('mode', AppState.currentMode);
            formData.append('regenerate', 'true');
            formData.append('advancedCoding', AppState.advancedCoding ? 'true' : 'false');
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                handleResponse(data);
            } catch (error) {
                console.error('Error regenerating message:', error);
                hideTypingIndicator();
                showNotification('Failed to regenerate message', 'error');
            }
        }
        
        function openSettings() {
            document.getElementById('settingsModal').classList.add('active');
            loadSettingsUI();
        }
        
        function closeModal(modalId) {
            document.getElementById(modalId).classList.remove('active');
        }
        
        function loadSettingsUI() {
            const modalBody = document.querySelector('#settingsModal .modal-body');
            modalBody.innerHTML = `
                <div class="settings-section">
                    <h3>Response Settings</h3>
                    <label>
                        <input type="checkbox" ${AppState.settings.autoEnhance ? 'checked' : ''} 
                               onchange="AppState.settings.autoEnhance = this.checked">
                        Auto-enhance prompts
                    </label>
                    <label>
                        <input type="checkbox" ${AppState.settings.streamResponses ? 'checked' : ''} 
                               onchange="AppState.settings.streamResponses = this.checked">
                        Stream responses
                    </label>
                </div>
                
                <div class="settings-section">
                    <h3>Interface Settings</h3>
                    <label>
                        <input type="checkbox" ${AppState.settings.notifications ? 'checked' : ''} 
                               onchange="AppState.settings.notifications = this.checked">
                        Show notifications
                    </label>
                    <label>
                        <input type="checkbox" ${AppState.settings.soundEffects ? 'checked' : ''} 
                               onchange="AppState.settings.soundEffects = this.checked">
                        Sound effects
                    </label>
                </div>
                
                <div class="settings-section">
                    <h3>Data Settings</h3>
                    <label>
                        <input type="checkbox" ${AppState.settings.saveHistory ? 'checked' : ''} 
                               onchange="AppState.settings.saveHistory = this.checked">
                        Save chat history
                    </label>
                    <button onclick="clearAllData()">Clear All Data</button>
                </div>
            `;
        }
        
        function saveSettings() {
            localStorage.setItem('settings', JSON.stringify(AppState.settings));
            closeModal('settingsModal');
            showNotification('Settings saved', 'success');
        }
        
        function loadSettings() {
            const saved = localStorage.getItem('settings');
            if (saved) {
                AppState.settings = { ...AppState.settings, ...JSON.parse(saved) };
            }
        }
        
        function clearAllData() {
            if (confirm('Clear all chat history and settings? This cannot be undone.')) {
                localStorage.clear();
                AppState.messages = [];
                AppState.tokenUsage = 0;
                createNewSession();
                document.getElementById('messagesArea').innerHTML = '';
                addWelcomeMessage();
                showNotification('All data cleared', 'success');
                closeModal('settingsModal');
            }
        }
        
        async function saveSession() {
            if (!AppState.settings.saveHistory) return;
            
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
        
        function insertEmoji() {
            const emojis = ['😀', '👍', '❤️', '🚀', '🎉', '🤔', '👏', '🔥', '✨', '💡'];
            const emoji = emojis[Math.floor(Math.random() * emojis.length)];
            
            const input = document.getElementById('messageInput');
            const start = input.selectionStart;
            const end = input.selectionEnd;
            const text = input.value;
            
            input.value = text.substring(0, start) + emoji + text.substring(end);
            input.selectionStart = input.selectionEnd = start + emoji.length;
            input.focus();
        }
        
        function toggleMarkdownHelp() {
            showNotification('Markdown support: **bold**, *italic*, `code`, [links](url)', 'info');
        }
        
        function addRippleEffect(element) {
            element.addEventListener('click', function(e) {
                const ripple = document.createElement('span');
                ripple.className = 'ripple';
                
                const rect = this.getBoundingClientRect();
                const size = Math.max(rect.width, rect.height);
                const x = e.clientX - rect.left - size / 2;
                const y = e.clientY - rect.top - size / 2;
                
                ripple.style.width = ripple.style.height = size + 'px';
                ripple.style.left = x + 'px';
                ripple.style.top = y + 'px';
                
                this.appendChild(ripple);
                
                setTimeout(() => ripple.remove(), 600);
            });
        }
        
        function playNotificationSound(type) {
            // Implement sound effects if needed
            // For now, just a placeholder
        }
        
        // ========== PARTICLE EFFECT ==========
        function createParticleEffect() {
            const canvas = document.getElementById('particleCanvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            
            const particles = [];
            const particleCount = 100;
            
            class Particle {
                constructor() {
                    this.x = Math.random() * canvas.width;
                    this.y = Math.random() * canvas.height;
                    this.vx = (Math.random() - 0.5) * 0.5;
                    this.vy = (Math.random() - 0.5) * 0.5;
                    this.radius = Math.random() * 2 + 1;
                    this.opacity = Math.random() * 0.5 + 0.2;
                }
                
                update() {
                    this.x += this.vx;
                    this.y += this.vy;
                    
                    if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
                    if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
                }
                
                draw() {
                    ctx.beginPath();
                    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(255, 255, 255, ${this.opacity})`;
                    ctx.fill();
                }
            }
            
            for (let i = 0; i < particleCount; i++) {
                particles.push(new Particle());
            }
            
            function animate() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                particles.forEach(particle => {
                    particle.update();
                    particle.draw();
                });
                
                // Draw connections
                particles.forEach((p1, i) => {
                    particles.slice(i + 1).forEach(p2 => {
                        const distance = Math.hypot(p1.x - p2.x, p1.y - p2.y);
                        if (distance < 100) {
                            ctx.beginPath();
                            ctx.moveTo(p1.x, p1.y);
                            ctx.lineTo(p2.x, p2.y);
                            ctx.strokeStyle = `rgba(255, 255, 255, ${0.1 * (1 - distance / 100)})`;
                            ctx.stroke();
                        }
                    });
                });
                
                requestAnimationFrame(animate);
            }
            
            animate();
            
            window.addEventListener('resize', () => {
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;
            });
        }
    </script>
</body>
</html>
"""

# ============= FLASK ROUTES =============

@app.route('/')
def index():
    """Serve the main application"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/session/create', methods=['POST'])
@limiter.limit("10 per minute")
def create_session():
    """Create a new chat session"""
    try:
        data = request.json
        session_id = data.get('sessionId')
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
        print(f"Error creating session: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/session/<session_id>', methods=['GET'])
@cache.cached(timeout=60)
def get_session(session_id):
    """Get session data"""
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
        print(f"Error getting session: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/session/save', methods=['POST'])
@limiter.limit("30 per minute")
def save_session():
    """Save session data"""
    try:
        data = request.json
        session_id = data.get('sessionId')
        
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        # Update session with new data
        # This is simplified - in production, you'd properly update the session
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error saving session: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
@limiter.limit("30 per minute")
async def chat():
    """Handle chat messages with streaming support"""
    try:
        message = request.form.get('message', '')
        session_id = request.form.get('sessionId')
        mode = ChatMode(request.form.get('mode', 'balanced'))
        files = request.files.getlist('files')
        regenerate = request.form.get('regenerate', 'false') == 'true'
        
        # Get or create session
        session = session_manager.get_session(session_id)
        if not session:
            session = session_manager.create_session(mode=mode)
            session.id = session_id
        
        # Process files if any
        file_contents = []
        if files:
            for file in files:
                content, img_data, metadata = await file_processor.process_file(file)
                file_contents.append(content)
        
        # Enhance prompt if enabled
        enhanced_message = message
        advanced_coding = request.form.get('advancedCoding', 'false') == 'true'
        if not regenerate:  # Don't enhance on regeneration
            enhanced_message = prompt_engineer.enhance_prompt(
                message, 
                mode,
                strategies=['clarity', 'structure', 'context'] if mode != ChatMode.CODE else ['clarity'],
                advanced_coding=advanced_coding
            )
        
        # Combine message with file contents
        full_prompt = enhanced_message
        if file_contents:
            full_prompt += "\n\n--- Attached Files ---\n" + "\n\n".join(file_contents)
        
        # Build conversation context
        advanced_coding = request.form.get('advancedCoding', 'false') == 'true'
        system_prompt = prompt_engineer.create_system_prompt(mode, advanced_coding=advanced_coding)
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add recent conversation history (last 10 messages)
        for msg in session.messages[-10:]:
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        # Add current message
        messages.append({"role": "user", "content": full_prompt})
        
        # Get AI client
        client, api_key = client_manager.get_client("gemini")
        if not client:
            return jsonify({
                'success': False,
                'error': 'No AI service available. Please try again later.'
            }), 503
        
        # Make API call with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Determine model based on mode
                model = "gemini-2.5-pro" if mode in [ChatMode.RESEARCH, ChatMode.PRECISE] else "gemini-2.5-flash"
                
                # Determine max tokens based on provider and mode
                max_tokens = 8192  # Default for non-Gemini models
                if api_key.provider == 'gemini':
                    max_tokens = 60000  # 60k tokens for all Gemini models
                
                # Check if advanced coding mode is enabled
                advanced_coding = request.form.get('advancedCoding', 'false') == 'true'
                
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.9 if mode == ChatMode.CREATIVE else 0.7 if mode == ChatMode.BALANCED else 0.3,
                    stream=request.form.get('stream', 'false') == 'true'
                )
                
                # Handle streaming response
                if request.form.get('stream', 'false') == 'true':
                    def generate():
                        assistant_message = ""
                        for chunk in response:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                assistant_message += content
                                yield f"data: {json.dumps({'content': content})}\n\n"
                        
                        # Save complete message
                        user_msg = Message(
                            role=MessageRole.USER,
                            content=message,
                            timestamp=datetime.now(timezone.utc),
                            tokens=len(message) // 4
                        )
                        assistant_msg = Message(
                            role=MessageRole.ASSISTANT,
                            content=assistant_message,
                            timestamp=datetime.now(timezone.utc),
                            tokens=len(assistant_message) // 4
                        )
                        
                        session.add_message(user_msg)
                        session.add_message(assistant_msg)
                        session_manager.update_session(session)
                        
                        yield f"data: {json.dumps({'tokenUsage': session.total_tokens})}\n\n"
                        yield "data: [DONE]\n\n"
                    
                    return Response(stream_with_context(generate()), 
                                  mimetype='text/event-stream')
                
                # Non-streaming response
                ai_response = response.choices[0].message.content
                
                # Update session
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
                
                session.add_message(user_msg)
                session.add_message(assistant_msg)
                session_manager.update_session(session)
                
                # Mark API key as successful
                key_manager.mark_key_success(api_key)
                
                return jsonify({
                    'success': True,
                    'response': ai_response,
                    'tokenUsage': session.total_tokens,
                    'mode': mode.value
                })
                
            except Exception as api_error:
                print(f"API attempt {attempt + 1} failed: {api_error}")
                key_manager.mark_key_failure(api_key)
                
                if attempt < max_retries - 1:
                    # Try with a different key
                    client, api_key = client_manager.get_client("gemini")
                    if not client:
                        break
                else:
                    raise api_error
        
        return jsonify({
            'success': False,
            'error': 'Failed to get AI response after multiple attempts'
        }), 503
        
    except Exception as e:
        print(f"Chat error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/enhance', methods=['POST'])
@limiter.limit("20 per minute")
async def enhance_prompt():
    """Enhance a user's prompt"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        mode = ChatMode(data.get('mode', 'balanced'))
        strategies = data.get('strategies', ['clarity', 'structure', 'context'])
        advanced_coding = data.get('advancedCoding', False)
        
        if len(prompt) < 10:
            return jsonify({
                'success': False,
                'error': 'Prompt too short to enhance'
            }), 400
        
        enhanced = prompt_engineer.enhance_prompt(prompt, mode, strategies, advanced_coding)
        
        # If significantly different, use AI to further enhance
        if len(enhanced) > len(prompt) * 1.2:
            client, api_key = client_manager.get_client("gemini")
            if client:
                try:
                    system_content = "You are a prompt enhancement specialist. Improve the clarity and effectiveness of prompts while preserving intent."
                    if advanced_coding:
                        system_content += " The user wants COMPREHENSIVE, PRODUCTION-READY code implementations with no shortcuts."
                    
                    response = client.chat.completions.create(
                        model="gemini-2.5-flash",
                        messages=[
                            {
                                "role": "system",
                                "content": system_content
                            },
                            {
                                "role": "user",
                                "content": f"Enhance this prompt for better AI understanding:\n{enhanced}"
                            }
                        ],
                        max_tokens=1000,
                        temperature=0.7
                    )
                    enhanced = response.choices[0].message.content
                    key_manager.mark_key_success(api_key)
                except Exception as e:
                    print(f"AI enhancement failed: {e}")
                    key_manager.mark_key_failure(api_key)
        
        return jsonify({
            'success': True,
            'original': prompt,
            'enhanced': enhanced
        })
        
    except Exception as e:
        print(f"Enhancement error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
@limiter.limit("10 per minute")
async def analyze_content():
    """Analyze uploaded content with AI"""
    try:
        files = request.files.getlist('files')
        analysis_type = request.form.get('type', 'general')
        
        if not files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400
        
        analyses = []
        
        for file in files:
            content, img_data, metadata = await file_processor.process_file(file)
            
            # Prepare analysis prompt based on type
            if analysis_type == 'sentiment':
                prompt = f"Analyze the sentiment of this content:\n{content}"
            elif analysis_type == 'summary':
                prompt = f"Provide a comprehensive summary of this content:\n{content}"
            elif analysis_type == 'insights':
                prompt = f"Extract key insights and patterns from this content:\n{content}"
            else:
                prompt = f"Analyze this content and provide detailed observations:\n{content}"
            
            # Get AI analysis
            client, api_key = client_manager.get_client("gemini")
            if not client:
                continue
            
            try:
                response = client.chat.completions.create(
                    model="gemini-2.5-pro",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert analyst. Provide detailed, insightful analysis."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=2048,
                    temperature=0.5
                )
                
                analysis = response.choices[0].message.content
                key_manager.mark_key_success(api_key)
                
                analyses.append({
                    'filename': file.filename,
                    'metadata': metadata,
                    'analysis': analysis
                })
                
            except Exception as e:
                print(f"Analysis failed for {file.filename}: {e}")
                key_manager.mark_key_failure(api_key)
                analyses.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'analyses': analyses
        })
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/export/<format>', methods=['GET'])
@limiter.limit("10 per minute")
def export_session(format):
    """Export session in various formats"""
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
            content = f"# Jack's AI Ultra Chat Export\n\n"
            content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            content += f"**Session ID:** {session.id}\n"
            content += f"**Mode:** {session.mode.value}\n"
            content += f"**Total Tokens:** {session.total_tokens:,}\n\n"
            content += "---\n\n"
            
            for msg in session.messages:
                icon = "👤" if msg.role == MessageRole.USER else "🤖"
                content += f"### {icon} {msg.role.value.title()}\n"
                content += f"*{msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
                content += f"{msg.content}\n\n"
                content += "---\n\n"
            
            return Response(content, mimetype='text/markdown', headers={
                'Content-Disposition': f'attachment; filename=chat_export_{timestamp}.md'
            })
            
        elif format == 'html':
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Chat Export - {session.id}</title>
                <style>
                    body {{ font-family: 'Inter', sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                    .message {{ margin: 20px 0; padding: 15px; border-radius: 10px; }}
                    .user {{ background: #667eea; color: white; }}
                    .assistant {{ background: #f0f0f0; }}
                    .timestamp {{ font-size: 12px; opacity: 0.7; }}
                </style>
            </head>
            <body>
                <h1>Jack's AI Ultra Chat Export</h1>
                <p>Session: {session.id}</p>
                <p>Mode: {session.mode.value}</p>
                <p>Tokens: {session.total_tokens:,}</p>
                <hr>
            """
            
            for msg in session.messages:
                html_content += f"""
                <div class="message {msg.role.value}">
                    <div class="timestamp">{msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</div>
                    <div>{msg.content}</div>
                </div>
                """
            
            html_content += "</body></html>"
            
            return Response(html_content, mimetype='text/html', headers={
                'Content-Disposition': f'attachment; filename=chat_export_{timestamp}.html'
            })
            
        else:
            return jsonify({'success': False, 'error': 'Invalid format'}), 400
            
    except Exception as e:
        print(f"Export error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
@cache.cached(timeout=300)
def get_available_models():
    """Get list of available AI models"""
    models = []
    
    # Check which providers have active keys
    for provider in ['gemini', 'openai', 'anthropic']:
        if key_manager.provider_keys.get(provider):
            if provider == 'gemini':
                models.extend([
                    {'id': 'gemini-2.5-pro', 'name': 'Gemini 2.5 Pro', 'provider': 'Google'},
                    {'id': 'gemini-2.5-flash', 'name': 'Gemini 2.5 Flash', 'provider': 'Google'}
                ])
            elif provider == 'openai':
                models.extend([
                    {'id': 'gpt-4-turbo', 'name': 'GPT-4 Turbo', 'provider': 'OpenAI'},
                    {'id': 'gpt-4', 'name': 'GPT-4', 'provider': 'OpenAI'}
                ])
            elif provider == 'anthropic':
                models.extend([
                    {'id': 'claude-3-opus', 'name': 'Claude 3 Opus', 'provider': 'Anthropic'},
                    {'id': 'claude-3-sonnet', 'name': 'Claude 3 Sonnet', 'provider': 'Anthropic'}
                ])
    
    return jsonify({
        'success': True,
        'models': models
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get usage statistics"""
    try:
        total_sessions = len(session_manager.sessions)
        total_messages = sum(len(s.messages) for s in session_manager.sessions.values())
        total_tokens = sum(s.total_tokens for s in session_manager.sessions.values())
        
        # Get API key stats
        api_stats = []
        for key_id, key in key_manager.keys.items():
            api_stats.append({
                'provider': key.provider,
                'model': key.model,
                'usage_today': key.usage_today,
                'daily_limit': key.daily_limit,
                'failures': key.failures,
                'is_active': key.is_active
            })
        
        return jsonify({
            'success': True,
            'stats': {
                'sessions': total_sessions,
                'messages': total_messages,
                'tokens': total_tokens,
                'api_keys': api_stats
            }
        })
        
    except Exception as e:
        print(f"Stats error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'version': '5.0.0'
    })

# ============= WEBSOCKET SUPPORT =============

@app.route('/ws')
def websocket_endpoint():
    """WebSocket endpoint for real-time features"""
    # Note: This would require flask-socketio or similar for full implementation
    # Placeholder for WebSocket support
    return jsonify({'error': 'WebSocket support requires additional setup'}), 501

# ============= ERROR HANDLERS =============

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============= MAIN EXECUTION =============

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    # In production, use a proper WSGI server
    if os.environ.get('ENVIRONMENT') == 'production':
        from waitress import serve
        print(f"Starting Jack's AI Ultra on port {port} (Production Mode)")
        serve(app, host='0.0.0.0', port=port, threads=10)
    else:
        print(f"Starting Jack's AI Ultra on port {port} (Development Mode)")
        app.run(host='0.0.0.0', port=port, debug=True)
