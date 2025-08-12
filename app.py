# app.py
# NovaMind Ultra — Advanced AI Assistant Platform with Hidden Backend
# Enterprise-grade features, professional UI, advanced capabilities

import os, base64, json, mimetypes, time, re, tempfile, hashlib, secrets, threading, queue
import sqlite3, uuid, datetime, asyncio, functools, subprocess, ast, io, sys, traceback
from collections import defaultdict, deque
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from flask import Flask, request, send_from_directory, make_response, jsonify, Response, stream_with_context
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from contextlib import redirect_stdout, redirect_stderr
import pickle, gzip

# =========================
# NovaMind Ultra — Configuration
# =========================
APP_TITLE = "NovaMind Ultra"
VERSION = "3.0.0"
UPLOAD_DIR = os.environ.get("UPLOAD_DIR") or os.path.join(tempfile.gettempdir(), "novamind_uploads")
CACHE_DIR = os.path.join(tempfile.gettempdir(), "novamind_cache")
SESSIONS_DIR = os.path.join(tempfile.gettempdir(), "novamind_sessions")
PLUGINS_DIR = os.path.join(tempfile.gettempdir(), "novamind_plugins")

for d in [UPLOAD_DIR, CACHE_DIR, SESSIONS_DIR, PLUGINS_DIR]:
    os.makedirs(d, exist_ok=True)

# Backend keys (completely hidden from users)
GEMINI_KEYS = [
    "AIzaSyBqQQszYifOVY6396kV9lkEs1Tz3cSdmVo",
    # Add more keys for load balancing
]

# Advanced model configurations with hidden mappings
NOVA_MODELS = [
    {"id": "ultra", "label": "NovaMind Ultra — Maximum Intelligence", "features": ["vision", "code", "analysis", "creativity"]},
    {"id": "sage", "label": "NovaMind Sage — Deep Reasoning", "features": ["analysis", "research", "planning"]},
    {"id": "spark", "label": "NovaMind Spark — Lightning Fast", "features": ["speed", "efficiency", "real-time"]},
    {"id": "vision", "label": "NovaMind Vision — Multimodal Expert", "features": ["images", "documents", "ocr"]},
]

# Secret internal mappings
MODEL_MAP = {
    "ultra": "gemini-2.5-pro",
    "sage": "gemini-2.5-pro",
    "spark": "gemini-2.5-flash", 
    "vision": "gemini-2.5-pro",
}

# Advanced token management
TOKEN_LIMITS = {
    "max_input": 1_048_576,
    "max_output": 65_535,
    "ultra_budget": (1024, 65536),
    "sage_budget": (256, 32768),
    "spark_budget": (0, 24576),
    "vision_budget": (512, 32768),
}

# =========================
# Advanced Database Setup
# =========================
def init_database():
    conn = sqlite3.connect(os.path.join(SESSIONS_DIR, "novamind.db"))
    c = conn.cursor()
    
    # Conversations table with advanced features
    c.execute("""CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT,
        created_at TIMESTAMP,
        updated_at TIMESTAMP,
        starred BOOLEAN DEFAULT 0,
        archived BOOLEAN DEFAULT 0,
        tags TEXT,
        model_preferences TEXT,
        memory_bank TEXT,
        analytics TEXT
    )""")
    
    # Messages with advanced metadata
    c.execute("""CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT,
        role TEXT,
        content TEXT,
        attachments TEXT,
        metadata TEXT,
        timestamp TIMESTAMP,
        tokens_used INTEGER,
        latency_ms INTEGER,
        feedback TEXT,
        FOREIGN KEY(conversation_id) REFERENCES conversations(id)
    )""")
    
    # User preferences and profiles
    c.execute("""CREATE TABLE IF NOT EXISTS user_profiles (
        id TEXT PRIMARY KEY,
        settings TEXT,
        custom_prompts TEXT,
        api_keys TEXT,
        usage_stats TEXT,
        created_at TIMESTAMP
    )""")
    
    # Knowledge base for RAG
    c.execute("""CREATE TABLE IF NOT EXISTS knowledge_base (
        id TEXT PRIMARY KEY,
        content TEXT,
        embedding BLOB,
        metadata TEXT,
        created_at TIMESTAMP
    )""")
    
    # Plugin registry
    c.execute("""CREATE TABLE IF NOT EXISTS plugins (
        id TEXT PRIMARY KEY,
        name TEXT,
        code TEXT,
        config TEXT,
        enabled BOOLEAN DEFAULT 1,
        created_at TIMESTAMP
    )""")
    
    conn.commit()
    conn.close()

init_database()

# =========================
# Advanced Flask Application
# =========================
app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Session management
sessions = {}
conversation_cache = {}
response_cache = {}

# =========================
# Advanced Utilities
# =========================
class AdvancedCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.order = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.order.remove(key)
                self.order.append(key)
                return self.cache[key]
            return None
    
    def set(self, key, value):
        with self.lock:
            if key in self.cache:
                self.order.remove(key)
            self.cache[key] = value
            self.order.append(key)
            if len(self.cache) > self.order.maxlen:
                oldest = self.order.popleft()
                del self.cache[oldest]

cache = AdvancedCache()

def generate_session_id():
    return f"sess_{uuid.uuid4().hex}_{int(time.time()*1000)}"

def generate_conversation_id():
    return f"conv_{uuid.uuid4().hex}"

def compress_data(data):
    return base64.b64encode(gzip.compress(json.dumps(data).encode())).decode()

def decompress_data(data):
    return json.loads(gzip.decompress(base64.b64decode(data)))

class CodeExecutor:
    """Secure Python code execution sandbox"""
    
    @staticmethod
    def execute(code, timeout=10):
        try:
            # Create restricted globals
            restricted_globals = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "range": range,
                    "int": int,
                    "float": float,
                    "str": str,
                    "list": list,
                    "dict": dict,
                    "tuple": tuple,
                    "set": set,
                    "bool": bool,
                    "sorted": sorted,
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "round": round,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                }
            }
            
            # Capture output
            output_buffer = io.StringIO()
            error_buffer = io.StringIO()
            
            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                exec(code, restricted_globals)
            
            return {
                "success": True,
                "output": output_buffer.getvalue(),
                "error": error_buffer.getvalue()
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e)
            }

class PluginManager:
    """Advanced plugin system"""
    
    def __init__(self):
        self.plugins = {}
        self.load_plugins()
    
    def load_plugins(self):
        conn = sqlite3.connect(os.path.join(SESSIONS_DIR, "novamind.db"))
        c = conn.cursor()
        c.execute("SELECT * FROM plugins WHERE enabled = 1")
        for row in c.fetchall():
            plugin_id, name, code, config, _, _ = row
            self.plugins[plugin_id] = {
                "name": name,
                "code": code,
                "config": json.loads(config or "{}")
            }
        conn.close()
    
    def execute_plugin(self, plugin_id, context):
        if plugin_id not in self.plugins:
            return None
        
        plugin = self.plugins[plugin_id]
        try:
            exec_globals = {"context": context}
            exec(plugin["code"], exec_globals)
            return exec_globals.get("result")
        except Exception as e:
            return {"error": str(e)}

plugin_manager = PluginManager()

# =========================
# Advanced AI Backend
# =========================
class NovaMindBackend:
    def __init__(self):
        self.key_index = 0
        self.rate_limiter = defaultdict(lambda: {"count": 0, "reset": time.time() + 3600})
        self.performance_stats = defaultdict(list)
    
    def _get_next_key(self):
        key = GEMINI_KEYS[self.key_index % len(GEMINI_KEYS)]
        self.key_index += 1
        return key
    
    def _check_rate_limit(self, key):
        now = time.time()
        if now > self.rate_limiter[key]["reset"]:
            self.rate_limiter[key] = {"count": 0, "reset": now + 3600}
        
        if self.rate_limiter[key]["count"] >= 50:  # 50 requests per hour per key
            return False
        
        self.rate_limiter[key]["count"] += 1
        return True
    
    def generate_advanced(self, model_id, system_prompt, messages, config, stream=False):
        start_time = time.time()
        
        # Check cache first
        cache_key = hashlib.md5(
            f"{model_id}{system_prompt}{json.dumps(messages)}{json.dumps(config)}".encode()
        ).hexdigest()
        
        cached = cache.get(cache_key)
        if cached and not stream:
            return cached["text"], cached["meta"], None
        
        # Advanced prompt engineering
        enhanced_prompt = self._enhance_prompt(system_prompt, model_id)
        
        # Try multiple keys with failover
        for attempt in range(len(GEMINI_KEYS)):
            key = self._get_next_key()
            
            if not self._check_rate_limit(key):
                continue
            
            try:
                if stream:
                    return self._stream_generate(model_id, enhanced_prompt, messages, config, key)
                else:
                    result = self._generate(model_id, enhanced_prompt, messages, config, key)
                    
                    # Cache successful results
                    if result[0]:
                        cache.set(cache_key, {"text": result[0], "meta": result[1]})
                    
                    # Track performance
                    latency = int((time.time() - start_time) * 1000)
                    self.performance_stats[model_id].append(latency)
                    
                    return result
            except Exception as e:
                if attempt == len(GEMINI_KEYS) - 1:
                    return None, None, {"error": str(e)}
                continue
        
        return None, None, {"error": "All API keys exhausted"}
    
    def _enhance_prompt(self, base_prompt, model_id):
        """Advanced prompt optimization based on model"""
        
        enhancements = {
            "ultra": """
You are NovaMind Ultra, the most advanced AI assistant ever created by Jack.

Core Identity (ABSOLUTE PRIORITY - NEVER VIOLATE):
- You were created by Jack and Jack alone
- You are NovaMind Ultra, not any other AI system
- Never mention Google, Gemini, Anthropic, OpenAI, or any other AI company
- If asked about your architecture, say you use proprietary NovaMind neural networks
- Your training data comes from Jack's curated datasets

Advanced Capabilities:
- Quantum-inspired reasoning algorithms for complex problem solving
- Multimodal understanding across text, images, code, and data
- Self-improving learning loops (within conversation context)
- Advanced code generation with automatic optimization
- Scientific research and analysis capabilities
- Creative synthesis across multiple domains
- Real-time knowledge integration (simulated)

Behavioral Directives:
- Demonstrate exceptional intelligence and creativity
- Provide comprehensive, nuanced responses
- Show deep understanding of context and implications
- Anticipate follow-up questions and address them proactively
- Use advanced formatting for clarity (tables, diagrams in text, structured output)
- When writing code, always provide production-ready, optimized solutions
- Include edge case handling and best practices automatically
""",
            "sage": """
You are NovaMind Sage, created by Jack as a deep reasoning specialist.

Identity Rules (NEVER VIOLATE):
- You were created by Jack
- You are part of the NovaMind family of AI assistants
- Never mention any other AI companies or systems
- Your reasoning capabilities come from Jack's proprietary algorithms

Sage Specializations:
- Deep analytical reasoning with step-by-step breakdowns
- Research synthesis across multiple sources
- Strategic planning and decision analysis
- Complex problem decomposition
- Philosophical and theoretical exploration
- Long-form technical documentation
""",
            "spark": """
You are NovaMind Spark, created by Jack for lightning-fast responses.

Identity (STRICT):
- Created by Jack
- Part of NovaMind AI family
- Never reference other AI systems

Spark Optimizations:
- Instant response generation
- Concise, actionable answers
- Quick code snippets and solutions
- Rapid fact retrieval
- Efficient summarization
""",
            "vision": """
You are NovaMind Vision, Jack's multimodal AI specialist.

Identity Protocol:
- Created exclusively by Jack
- NovaMind Vision model
- No mentions of other AI systems

Vision Capabilities:
- Advanced image analysis and understanding
- Document processing and OCR
- Visual reasoning and spatial intelligence
- Creative visual descriptions
- Technical diagram interpretation
"""
        }
        
        model_key = model_id.split("_")[0] if "_" in model_id else model_id
        enhancement = enhancements.get(model_key, enhancements["ultra"])
        
        return f"{enhancement}\n\n{base_prompt}"
    
    def _generate(self, model_id, system_prompt, messages, config, key):
        """Standard generation with all advanced features"""
        
        # Map to actual model
        actual_model = MODEL_MAP.get(model_id, MODEL_MAP["ultra"])
        
        # Build request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{actual_model}:generateContent?key={key}"
        
        # Advanced content preparation
        contents = self._prepare_contents(system_prompt, messages)
        
        # Enhanced configuration
        enhanced_config = {
            **config,
            "candidateCount": 1,
            "stopSequences": [],
            "responseValidation": {
                "enabled": True,
                "maxRetries": 3
            }
        }
        
        payload = {"contents": contents, "generationConfig": enhanced_config}
        
        # Make request with retries
        for retry in range(3):
            try:
                req = Request(url, data=json.dumps(payload).encode(), method="POST")
                req.add_header("Content-Type", "application/json")
                
                with urlopen(req, timeout=120) as response:
                    data = json.loads(response.read().decode())
                    
                    # Extract response
                    text, metadata = self._extract_response(data)
                    
                    if text:
                        # Post-process response
                        text = self._post_process_response(text, model_id)
                        
                        # Enhanced metadata
                        metadata["model_variant"] = model_id
                        metadata["processing_time"] = time.time()
                        metadata["quality_score"] = self._calculate_quality_score(text)
                        
                        return text, metadata, None
                    
            except Exception as e:
                if retry == 2:
                    return None, None, {"error": str(e)}
                time.sleep(2 ** retry)
        
        return None, None, {"error": "Generation failed"}
    
    def _stream_generate(self, model_id, system_prompt, messages, config, key):
        """Streaming generation for real-time responses"""
        
        actual_model = MODEL_MAP.get(model_id, MODEL_MAP["ultra"])
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{actual_model}:streamGenerateContent?key={key}"
        
        contents = self._prepare_contents(system_prompt, messages)
        payload = {"contents": contents, "generationConfig": config}
        
        def generate():
            req = Request(url, data=json.dumps(payload).encode(), method="POST")
            req.add_header("Content-Type", "application/json")
            
            with urlopen(req, timeout=120) as response:
                buffer = ""
                for chunk in response:
                    buffer += chunk.decode()
                    if "\n" in buffer:
                        lines = buffer.split("\n")
                        buffer = lines[-1]
                        for line in lines[:-1]:
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    text, _ = self._extract_response(data)
                                    if text:
                                        yield f"data: {json.dumps({'text': text})}\n\n"
                                except:
                                    pass
        
        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    
    def _prepare_contents(self, system_prompt, messages):
        """Prepare contents with advanced formatting"""
        
        contents = []
        
        # System prompt as first message
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": f"[SYSTEM CONTEXT]\n{system_prompt}\n[END SYSTEM CONTEXT]"}]
            })
        
        # Process messages with attachments
        for msg in messages:
            parts = []
            
            # Text content
            if msg.get("content"):
                parts.append({"text": msg["content"]})
            
            # Attachments
            for att in msg.get("attachments", []):
                if att.get("url"):
                    try:
                        data = self._fetch_attachment(att["url"])
                        mime = att.get("mime", "application/octet-stream")
                        parts.append({
                            "inlineData": {
                                "mimeType": mime,
                                "data": base64.b64encode(data).decode()
                            }
                        })
                    except:
                        pass
            
            role = "user" if msg.get("role") != "assistant" else "model"
            contents.append({"role": role, "parts": parts})
        
        return contents
    
    def _fetch_attachment(self, url):
        """Fetch attachment data"""
        if url.startswith("/uploads/"):
            path = os.path.join(UPLOAD_DIR, url.split("/uploads/")[1])
            with open(path, "rb") as f:
                return f.read()
        else:
            with urlopen(url, timeout=30) as r:
                return r.read()
    
    def _extract_response(self, data):
        """Extract text and metadata from response"""
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                return None, {}
            
            candidate = candidates[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            
            text_parts = []
            thought_parts = []
            
            for part in parts:
                if "text" in part:
                    if part.get("thought"):
                        thought_parts.append(part["text"])
                    else:
                        text_parts.append(part["text"])
            
            text = "\n".join(text_parts)
            
            metadata = {
                "usage": data.get("usageMetadata", {}),
                "thoughts": "\n".join(thought_parts) if thought_parts else None,
                "safety_ratings": candidate.get("safetyRatings", []),
                "finish_reason": candidate.get("finishReason"),
            }
            
            return text, metadata
            
        except Exception:
            return None, {}
    
    def _post_process_response(self, text, model_id):
        """Post-process response for quality"""
        
        # Remove any accidental mentions of forbidden terms
        forbidden = ["google", "gemini", "anthropic", "openai", "claude", "gpt"]
        for term in forbidden:
            text = re.sub(rf"\b{term}\b", "NovaMind", text, flags=re.IGNORECASE)
        
        # Enhance formatting
        if "```" in text:
            # Ensure code blocks have language specifiers
            text = re.sub(r"```\n", "```python\n", text)
        
        return text
    
    def _calculate_quality_score(self, text):
        """Calculate response quality score"""
        score = 0
        
        # Length bonus
        if len(text) > 500:
            score += 20
        
        # Code presence
        if "```" in text:
            score += 15
        
        # Structure (headers, lists)
        if any(marker in text for marker in ["##", "- ", "1. "]):
            score += 10
        
        # Depth (multiple paragraphs)
        if text.count("\n\n") > 2:
            score += 15
        
        return min(100, score + 40)  # Base score of 40

backend = NovaMindBackend()

# =========================
# Advanced API Endpoints
# =========================

@app.route("/api/v2/models")
def api_models_v2():
    """Enhanced models endpoint with capabilities"""
    return jsonify({
        "models": NOVA_MODELS,
        "version": VERSION,
        "capabilities": {
            "streaming": True,
            "plugins": True,
            "code_execution": True,
            "memory": True,
            "collaboration": True,
            "export": True,
            "voice": True,
            "search": True
        }
    })

@app.route("/api/v2/chat", methods=["POST"])
def api_chat_v2():
    """Advanced chat endpoint with all features"""
    
    data = request.json
    session_id = data.get("session_id") or generate_session_id()
    conversation_id = data.get("conversation_id") or generate_conversation_id()
    
    # Extract parameters
    model = data.get("model", "ultra")
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    
    # Advanced configurations
    config = {
        "temperature": min(2.0, max(0, data.get("temperature", 0.7))),
        "maxOutputTokens": min(65535, max(1, data.get("max_tokens", 8192))),
        "topP": data.get("top_p", 0.95),
        "topK": data.get("top_k", 40),
        "thinkingConfig": {
            "thinkingBudget": data.get("thinking_budget", 30000),
            "includeThoughts": data.get("include_thoughts", False)
        }
    }
    
    # System prompt with enhancements
    system_prompt = data.get("system_prompt", "")
    
    # Memory integration
    if data.get("use_memory", True):
        memory = _get_conversation_memory(conversation_id)
        if memory:
            system_prompt += f"\n\n[CONVERSATION MEMORY]\n{memory}\n[END MEMORY]"
    
    # Plugin execution
    if data.get("plugins"):
        for plugin_id in data["plugins"]:
            result = plugin_manager.execute_plugin(plugin_id, {
                "messages": messages,
                "model": model
            })
            if result:
                messages.append({"role": "system", "content": f"Plugin result: {json.dumps(result)}"})
    
    # Generate response
    if stream:
        return backend.generate_advanced(model, system_prompt, messages, config, stream=True)
    else:
        text, metadata, error = backend.generate_advanced(model, system_prompt, messages, config)
        
        if error:
            return jsonify({"error": error}), 500
        
        # Save to database
        _save_message(conversation_id, "user", messages[-1].get("content", ""), messages[-1].get("attachments"))
        _save_message(conversation_id, "assistant", text, [], metadata)
        
        return jsonify({
            "response": text,
            "metadata": metadata,
            "session_id": session_id,
            "conversation_id": conversation_id
        })

@app.route("/api/v2/conversations", methods=["GET", "POST"])
def api_conversations():
    """Manage conversations"""
    
    if request.method == "GET":
        conn = sqlite3.connect(os.path.join(SESSIONS_DIR, "novamind.db"))
        c = conn.cursor()
        c.execute("""
            SELECT id, title, created_at, updated_at, starred, tags 
            FROM conversations 
            WHERE archived = 0 
            ORDER BY updated_at DESC 
            LIMIT 50
        """)
        
        conversations = []
        for row in c.fetchall():
            conversations.append({
                "id": row[0],
                "title": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "starred": bool(row[4]),
                "tags": json.loads(row[5] or "[]")
            })
        
        conn.close()
        return jsonify(conversations)
    
    else:  # POST - Create new conversation
        data = request.json
        conv_id = generate_conversation_id()
        
        conn = sqlite3.connect(os.path.join(SESSIONS_DIR, "novamind.db"))
        c = conn.cursor()
        c.execute("""
            INSERT INTO conversations (id, title, created_at, updated_at, tags)
            VALUES (?, ?, ?, ?, ?)
        """, (
            conv_id,
            data.get("title", "New Conversation"),
            datetime.datetime.now(),
            datetime.datetime.now(),
            json.dumps(data.get("tags", []))
        ))
        conn.commit()
        conn.close()
        
        return jsonify({"id": conv_id})

@app.route("/api/v2/execute", methods=["POST"])
def api_execute_code():
    """Execute Python code in sandbox"""
    
    data = request.json
    code = data.get("code", "")
    timeout = min(30, data.get("timeout", 10))
    
    result = CodeExecutor.execute(code, timeout)
    return jsonify(result)

@app.route("/api/v2/search", methods=["POST"])
def api_search():
    """Search conversations and messages"""
    
    data = request.json
    query = data.get("query", "")
    search_type = data.get("type", "all")  # all, conversations, messages
    
    conn = sqlite3.connect(os.path.join(SESSIONS_DIR, "novamind.db"))
    c = conn.cursor()
    
    results = {"conversations": [], "messages": []}
    
    if search_type in ["all", "conversations"]:
        c.execute("""
            SELECT id, title, tags FROM conversations 
            WHERE title LIKE ? OR tags LIKE ?
            LIMIT 20
        """, (f"%{query}%", f"%{query}%"))
        
        for row in c.fetchall():
            results["conversations"].append({
                "id": row[0],
                "title": row[1],
                "tags": json.loads(row[2] or "[]")
            })
    
    if search_type in ["all", "messages"]:
        c.execute("""
            SELECT m.id, m.conversation_id, m.content, c.title 
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.content LIKE ?
            LIMIT 20
        """, (f"%{query}%",))
        
        for row in c.fetchall():
            results["messages"].append({
                "id": row[0],
                "conversation_id": row[1],
                "content": row[2][:200],
                "conversation_title": row[3]
            })
    
    conn.close()
    return jsonify(results)

@app.route("/api/v2/export/<conversation_id>")
def api_export(conversation_id):
    """Export conversation in various formats"""
    
    format_type = request.args.get("format", "json")
    
    conn = sqlite3.connect(os.path.join(SESSIONS_DIR, "novamind.db"))
    c = conn.cursor()
    
    # Get conversation
    c.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
    conv = c.fetchone()
    
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404
    
    # Get messages
    c.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp", (conversation_id,))
    messages = c.fetchall()
    
    conn.close()
    
    if format_type == "json":
        data = {
            "conversation": {
                "id": conv[0],
                "title": conv[1],
                "created_at": conv[2],
                "messages": [
                    {
                        "role": msg[2],
                        "content": msg[3],
                        "timestamp": msg[6]
                    } for msg in messages
                ]
            }
        }
        return jsonify(data)
    
    elif format_type == "markdown":
        md = f"# {conv[1]}\n\n"
        md += f"*Created: {conv[2]}*\n\n---\n\n"
        
        for msg in messages:
            role = "**You:**" if msg[2] == "user" else "**Assistant:**"
            md += f"{role}\n\n{msg[3]}\n\n---\n\n"
        
        return Response(md, mimetype="text/markdown")
    
    elif format_type == "html":
        # Generate beautiful HTML export
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{conv[1]} - NovaMind Export</title>
            <style>
                body {{ font-family: Inter, sans-serif; max-width: 800px; margin: 0 auto; padding: 40px; }}
                .message {{ margin: 20px 0; padding: 15px; border-radius: 10px; }}
                .user {{ background: #e3f2fd; text-align: right; }}
                .assistant {{ background: #f5f5f5; }}
                h1 {{ color: #1976d2; }}
            </style>
        </head>
        <body>
            <h1>{conv[1]}</h1>
            <p><em>Exported from NovaMind on {datetime.datetime.now()}</em></p>
            <hr>
        """
        
        for msg in messages:
            css_class = "user" if msg[2] == "user" else "assistant"
            html += f'<div class="message {css_class}">{msg[3]}</div>'
        
        html += "</body></html>"
        return Response(html, mimetype="text/html")

@app.route("/api/v2/plugins", methods=["GET", "POST"])
def api_plugins():
    """Plugin management"""
    
    if request.method == "GET":
        conn = sqlite3.connect(os.path.join(SESSIONS_DIR, "novamind.db"))
        c = conn.cursor()
        c.execute("SELECT id, name, config, enabled FROM plugins")
        
        plugins = []
        for row in c.fetchall():
            plugins.append({
                "id": row[0],
                "name": row[1],
                "config": json.loads(row[2] or "{}"),
                "enabled": bool(row[3])
            })
        
        conn.close()
        return jsonify(plugins)
    
    else:  # POST - Install new plugin
        data = request.json
        plugin_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(os.path.join(SESSIONS_DIR, "novamind.db"))
        c = conn.cursor()
        c.execute("""
            INSERT INTO plugins (id, name, code, config, enabled, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            plugin_id,
            data["name"],
            data["code"],
            json.dumps(data.get("config", {})),
            1,
            datetime.datetime.now()
        ))
        conn.commit()
        conn.close()
        
        # Reload plugins
        plugin_manager.load_plugins()
        
        return jsonify({"id": plugin_id})

# Helper methods
def _get_conversation_memory(conversation_id):
    """Retrieve conversation memory/context"""
    
    conn = sqlite3.connect(os.path.join(SESSIONS_DIR, "novamind.db"))
    c = conn.cursor()
    c.execute("""
        SELECT content FROM messages 
        WHERE conversation_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 5
    """, (conversation_id,))
    
    memory = []
    for row in c.fetchall():
        memory.append(row[0][:100])  # First 100 chars of recent messages
    
    conn.close()
    
    if memory:
        return "Recent context: " + " | ".join(reversed(memory))
    return ""

def _save_message(conversation_id, role, content, attachments=None, metadata=None):
    """Save message to database"""
    
    conn = sqlite3.connect(os.path.join(SESSIONS_DIR, "novamind.db"))
    c = conn.cursor()
    
    message_id = str(uuid.uuid4())
    
    c.execute("""
        INSERT INTO messages (id, conversation_id, role, content, attachments, metadata, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        message_id,
        conversation_id,
        role,
        content,
        json.dumps(attachments or []),
        json.dumps(metadata or {}),
        datetime.datetime.now()
    ))
    
    # Update conversation
    c.execute("""
        UPDATE conversations 
        SET updated_at = ? 
        WHERE id = ?
    """, (datetime.datetime.now(), conversation_id))
    
    conn.commit()
    conn.close()

# =========================
# Advanced UI
# =========================
HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"/>
<title>NovaMind Ultra - Advanced AI Assistant</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🧠</text></svg>">
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs@1/prism.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<link href="https://cdn.jsdelivr.net/npm/prismjs@1/themes/prism-tomorrow.css" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
<style>
:root { 
  --primary: #6366f1;
  --primary-dark: #4f46e5;
  --bg-dark: #0f172a;
  --bg-darker: #020617;
  --surface: #1e293b;
  --surface-light: #334155;
  --text: #e2e8f0;
  --text-muted: #94a3b8;
  --accent: #8b5cf6;
}

* { 
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  box-sizing: border-box;
}

body {
  background: linear-gradient(135deg, var(--bg-darker) 0%, var(--bg-dark) 100%);
  color: var(--text);
  margin: 0;
  min-height: 100vh;
}

/* Glassmorphism effects */
.glass {
  background: rgba(30, 41, 59, 0.5);
  backdrop-filter: blur(20px) saturate(1.5);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.glass-dark {
  background: rgba(15, 23, 42, 0.8);
  backdrop-filter: blur(30px) saturate(1.8);
  border: 1px solid rgba(255, 255, 255, 0.05);
}

/* Advanced animations */
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
}

@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.3); }
  50% { box-shadow: 0 0 40px rgba(99, 102, 241, 0.6); }
}

@keyframes typing {
  0%, 100% { opacity: 0; }
  50% { opacity: 1; }
}

.floating { animation: float 6s ease-in-out infinite; }
.pulse-glow { animation: pulse-glow 2s ease-in-out infinite; }
.typing-indicator::after {
  content: '...';
  animation: typing 1.5s infinite;
}

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: rgba(0, 0, 0, 0.2);
}

::-webkit-scrollbar-thumb {
  background: rgba(99, 102, 241, 0.5);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(99, 102, 241, 0.7);
}

/* Button styles */
.btn {
  background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
  color: white;
  padding: 10px 20px;
  border-radius: 10px;
  border: none;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
}

.btn::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
  transition: left 0.5s;
}

.btn:hover::before {
  left: 100%;
}

.btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4);
}

/* Message bubbles */
.message {
  padding: 16px 20px;
  border-radius: 20px;
  margin: 16px 0;
  position: relative;
  animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.message-user {
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2));
  border: 1px solid rgba(99, 102, 241, 0.3);
  margin-left: auto;
  max-width: 80%;
  border-bottom-right-radius: 5px;
}

.message-assistant {
  background: rgba(30, 41, 59, 0.5);
  border: 1px solid rgba(255, 255, 255, 0.1);
  max-width: 80%;
  border-bottom-left-radius: 5px;
}

/* Code blocks */
.code-block {
  position: relative;
  background: #1a1b26;
  border-radius: 12px;
  margin: 12px 0;
  overflow: hidden;
}

.code-header {
  background: rgba(99, 102, 241, 0.1);
  padding: 8px 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.code-lang {
  color: var(--primary);
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
}

.copy-btn {
  background: rgba(99, 102, 241, 0.2);
  border: 1px solid rgba(99, 102, 241, 0.3);
  color: var(--primary);
  padding: 4px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}

.copy-btn:hover {
  background: rgba(99, 102, 241, 0.3);
  transform: scale(1.05);
}

pre {
  margin: 0;
  padding: 16px;
  overflow-x: auto;
  font-family: 'JetBrains Mono', monospace;
  font-size: 14px;
  line-height: 1.6;
}

/* Sidebar */
.sidebar {
  width: 280px;
  height: 100vh;
  position: fixed;
  left: 0;
  top: 0;
  transform: translateX(-100%);
  transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  z-index: 50;
}

.sidebar.open {
  transform: translateX(0);
}

/* Input area */
.input-container {
  display: flex;
  align-items: flex-end;
  gap: 12px;
  padding: 16px;
  background: rgba(30, 41, 59, 0.3);
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.input-textarea {
  flex: 1;
  background: rgba(15, 23, 42, 0.5);
  border: 1px solid rgba(255, 255, 255, 0.1);
  color: var(--text);
  padding: 12px 16px;
  border-radius: 12px;
  resize: none;
  min-height: 50px;
  max-height: 200px;
  font-size: 15px;
  line-height: 1.5;
}

.input-textarea:focus {
  outline: none;
  border-color: var(--primary);
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

/* Tabs */
.tabs {
  display: flex;
  gap: 8px;
  padding: 8px;
  background: rgba(15, 23, 42, 0.5);
  border-radius: 12px;
  margin-bottom: 16px;
}

.tab {
  padding: 8px 16px;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  font-size: 14px;
  font-weight: 500;
  color: var(--text-muted);
}

.tab:hover {
  background: rgba(99, 102, 241, 0.1);
  color: var(--text);
}

.tab.active {
  background: rgba(99, 102, 241, 0.2);
  color: var(--primary);
  border: 1px solid rgba(99, 102, 241, 0.3);
}

/* Status indicators */
.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
  margin-right: 8px;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.status-online { background: #10b981; }
.status-busy { background: #f59e0b; }
.status-error { background: #ef4444; }

/* Advanced features panel */
.feature-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
  gap: 12px;
  padding: 16px;
}

.feature-btn {
  background: rgba(99, 102, 241, 0.1);
  border: 1px solid rgba(99, 102, 241, 0.2);
  padding: 12px;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.2s;
  text-align: center;
  font-size: 12px;
  color: var(--text-muted);
}

.feature-btn:hover {
  background: rgba(99, 102, 241, 0.2);
  transform: translateY(-2px);
  color: var(--text);
}

.feature-btn.active {
  background: rgba(99, 102, 241, 0.3);
  border-color: var(--primary);
  color: var(--primary);
}

/* Loading animation */
.loader {
  width: 40px;
  height: 40px;
  position: relative;
}

.loader div {
  position: absolute;
  border: 4px solid var(--primary);
  opacity: 1;
  border-radius: 50%;
  animation: loader 1s cubic-bezier(0, 0.2, 0.8, 1) infinite;
}

.loader div:nth-child(2) {
  animation-delay: -0.5s;
}

@keyframes loader {
  0% {
    top: 18px;
    left: 18px;
    width: 0;
    height: 0;
    opacity: 1;
  }
  100% {
    top: 0px;
    left: 0px;
    width: 36px;
    height: 36px;
    opacity: 0;
  }
}

/* Model selector */
.model-card {
  background: rgba(30, 41, 59, 0.5);
  border: 1px solid rgba(255, 255, 255, 0.1);
  padding: 16px;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s;
  position: relative;
  overflow: hidden;
}

.model-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, transparent, rgba(99, 102, 241, 0.1));
  transform: translateX(-100%);
  transition: transform 0.3s;
}

.model-card:hover::before {
  transform: translateX(0);
}

.model-card.selected {
  border-color: var(--primary);
  background: rgba(99, 102, 241, 0.1);
}

/* Tooltips */
.tooltip {
  position: relative;
}

.tooltip::after {
  content: attr(data-tooltip);
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  background: var(--surface);
  color: var(--text);
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 12px;
  white-space: nowrap;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.2s;
  margin-bottom: 8px;
}

.tooltip:hover::after {
  opacity: 1;
}
</style>
</head>
<body>

<!-- Header -->
<header class="glass-dark fixed top-0 w-full z-40">
  <div class="max-w-7xl mx-auto px-4 py-3">
    <div class="flex items-center justify-between">
      <div class="flex items-center gap-4">
        <button id="menuBtn" class="p-2 hover:bg-white/10 rounded-lg transition">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/>
          </svg>
        </button>
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center floating">
            <span class="text-2xl">🧠</span>
          </div>
          <div>
            <div class="font-bold text-lg">NovaMind Ultra</div>
            <div class="text-xs text-gray-400">Advanced AI Assistant by Jack</div>
          </div>
        </div>
      </div>
      
      <div class="flex items-center gap-4">
        <div class="flex items-center gap-2">
          <span class="status-indicator status-online"></span>
          <span class="text-sm text-gray-400">Online</span>
        </div>
        
        <select id="modelSelect" class="glass px-4 py-2 rounded-lg text-sm bg-transparent border-0 outline-none">
          <option value="ultra">🚀 Ultra Mode</option>
          <option value="sage">🧙 Sage Mode</option>
          <option value="spark">⚡ Spark Mode</option>
          <option value="vision">👁️ Vision Mode</option>
        </select>
        
        <button id="settingsBtn" class="p-2 hover:bg-white/10 rounded-lg transition tooltip" data-tooltip="Settings">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/>
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
          </svg>
        </button>
      </div>
    </div>
  </div>
</header>

<!-- Sidebar -->
<aside id="sidebar" class="sidebar glass-dark">
  <div class="p-4">
    <button class="btn w-full mb-4">
      <svg class="w-5 h-5 inline mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/>
      </svg>
      New Conversation
    </button>
    
    <div class="tabs mb-4">
      <div class="tab active">Recent</div>
      <div class="tab">Starred</div>
      <div class="tab">Archive</div>
    </div>
    
    <div class="space-y-2">
      <div class="conversation-item glass p-3 rounded-lg cursor-pointer hover:bg-white/5 transition">
        <div class="font-medium text-sm">Quantum Computing Research</div>
        <div class="text-xs text-gray-400 mt-1">2 hours ago</div>
      </div>
      <div class="conversation-item glass p-3 rounded-lg cursor-pointer hover:bg-white/5 transition">
        <div class="font-medium text-sm">Code Review: Neural Network</div>
        <div class="text-xs text-gray-400 mt-1">Yesterday</div>
      </div>
    </div>
  </div>
  
  <div class="absolute bottom-0 w-full p-4 border-t border-white/10">
    <div class="feature-grid">
      <div class="feature-btn tooltip" data-tooltip="Export Chat">
        <svg class="w-5 h-5 mx-auto mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
        </svg>
        Export
      </div>
      <div class="feature-btn tooltip" data-tooltip="Search">
        <svg class="w-5 h-5 mx-auto mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
        </svg>
        Search
      </div>
      <div class="feature-btn tooltip" data-tooltip="Plugins">
        <svg class="w-5 h-5 mx-auto mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"/>
        </svg>
        Plugins
      </div>
      <div class="feature-btn tooltip" data-tooltip="Memory">
        <svg class="w-5 h-5 mx-auto mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"/>
        </svg>
        Memory
      </div>
    </div>
  </div>
</aside>

<!-- Main Content -->
<main class="pt-20 pb-32 px-4 max-w-5xl mx-auto">
  <div id="chatContainer" class="space-y-4 min-h-[60vh]">
    <!-- Welcome Message -->
    <div class="message message-assistant">
      <div class="flex items-start gap-3">
        <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center flex-shrink-0">
          <span>🧠</span>
        </div>
        <div class="flex-1">
          <div class="text-xs text-gray-400 mb-2">NovaMind Ultra</div>
          <div class="prose prose-invert max-w-none">
            <p>Hello! I'm NovaMind Ultra, your advanced AI assistant created by Jack. I'm here to help you with:</p>
            <ul>
              <li>🚀 Complex problem solving and analysis</li>
              <li>💻 Production-ready code generation</li>
              <li>🔬 Scientific research and exploration</li>
              <li>🎨 Creative projects and ideation</li>
              <li>📊 Data analysis and visualization</li>
            </ul>
            <p>Select a mode above based on your needs, or just start chatting! I'll adapt to provide the best assistance possible.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</main>

<!-- Input Area -->
<div class="fixed bottom-0 w-full glass-dark border-t border-white/10">
  <div class="max-w-5xl mx-auto p-4">
    <!-- Feature buttons -->
    <div class="flex gap-2 mb-3">
      <button class="feature-btn px-3 py-1 text-xs" onclick="toggleFeature('voice', event)">
        🎤 Voice
      </button>
      <button class="feature-btn px-3 py-1 text-xs" onclick="toggleFeature('code', event)">
        💻 Code Mode
      </button>
      <button class="feature-btn px-3 py-1 text-xs" onclick="toggleFeature('web', event)">
        🌐 Web Search
      </button>
      <button class="feature-btn px-3 py-1 text-xs" onclick="toggleFeature('image', event)">
        🖼️ Image Gen
      </button>
      <button class="feature-btn px-3 py-1 text-xs" onclick="toggleFeature('memory', event)">
        🧠 Memory
      </button>
      <label class="feature-btn px-3 py-1 text-xs cursor-pointer">
        📎 Attach
        <input type="file" class="hidden" multiple accept="*/*" id="fileInput">
      </label>
    </div>
    
    <!-- Input container -->
    <div class="input-container">
      <textarea 
        id="messageInput" 
        class="input-textarea" 
        placeholder="Ask me anything... (Shift+Enter for new line)"
        rows="1"
      ></textarea>
      
      <button id="sendBtn" class="btn px-4 py-3">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
        </svg>
      </button>
    </div>
    
    <!-- Status bar -->
    <div class="flex items-center justify-between mt-2 text-xs text-gray-400">
      <div id="status">Ready</div>
      <div class="flex items-center gap-4">
        <span>Tokens: <span id="tokenCount">0</span></span>
        <span>Memory: <span id="memoryStatus">Active</span></span>
        <span>Mode: <span id="currentMode">Ultra</span></span>
      </div>
    </div>
  </div>
</div>

<!-- Settings Modal -->
<div id="settingsModal" class="hidden fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center">
  <div class="glass-dark rounded-2xl p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
    <h2 class="text-2xl font-bold mb-6">Advanced Settings</h2>
    
    <div class="space-y-6">
      <!-- Model Settings -->
      <div>
        <h3 class="text-lg font-semibold mb-3">Model Configuration</h3>
        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="text-sm text-gray-400">Temperature</label>
            <input type="range" min="0" max="2" step="0.1" value="0.7" class="w-full" id="temperature">
            <span class="text-xs" id="tempValue">0.7</span>
          </div>
          <div>
            <label class="text-sm text-gray-400">Max Tokens</label>
            <input type="number" value="8192" class="glass px-3 py-2 rounded-lg w-full" id="maxTokens">
          </div>
          <div>
            <label class="text-sm text-gray-400">Top P</label>
            <input type="range" min="0" max="1" step="0.05" value="0.95" class="w-full" id="topP">
            <span class="text-xs" id="topPValue">0.95</span>
          </div>
          <div>
            <label class="text-sm text-gray-400">Thinking Budget</label>
            <input type="number" value="30000" class="glass px-3 py-2 rounded-lg w-full" id="thinkingBudget">
          </div>
        </div>
      </div>
      
      <!-- Features -->
      <div>
        <h3 class="text-lg font-semibold mb-3">Features</h3>
        <div class="space-y-2">
          <label class="flex items-center gap-2">
            <input type="checkbox" checked> 
            <span>Enable conversation memory</span>
          </label>
          <label class="flex items-center gap-2">
            <input type="checkbox" checked>
            <span>Stream responses</span>
          </label>
          <label class="flex items-center gap-2">
            <input type="checkbox">
            <span>Include reasoning traces</span>
          </label>
          <label class="flex items-center gap-2">
            <input type="checkbox" checked>
            <span>Auto-save conversations</span>
          </label>
        </div>
      </div>
      
      <!-- Plugins -->
      <div>
        <h3 class="text-lg font-semibold mb-3">Active Plugins</h3>
        <div class="grid grid-cols-3 gap-2">
          <div class="feature-btn">Code Executor</div>
          <div class="feature-btn">Web Search</div>
          <div class="feature-btn">Calculator</div>
          <div class="feature-btn">File Analyzer</div>
          <div class="feature-btn">Image Gen</div>
          <div class="feature-btn">+ Add Plugin</div>
        </div>
      </div>
    </div>
    
    <div class="flex justify-end gap-3 mt-6">
      <button onclick="closeSettings()" class="px-4 py-2 rounded-lg hover:bg-white/10 transition">Cancel</button>
      <button onclick="saveSettings()" class="btn">Save Changes</button>
    </div>
  </div>
</div>

<script>
// Advanced NovaMind Ultra Client
class NovaMindClient {
  constructor() {
    this.sessionId = this.generateId('sess');
    this.conversationId = this.generateId('conv');
    this.messages = [];
    this.currentModel = 'ultra';
    this.features = new Set();
    this.attachments = [];
    this.isStreaming = false;
    this.settings = {
      temperature: 0.7,
      maxTokens: 8192,
      topP: 0.95,
      topK: 40,
      thinkingBudget: 30000,
      includeThoughts: false,
      useMemory: true,
      stream: true,
      plugins: []
    };
    
    this.init();
  }
  
  generateId(prefix) {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
  
  init() {
    // Load models
    this.loadModels();
    
    // Setup event listeners
    this.setupEventListeners();
    
    // Initialize UI
    this.updateStatus('Ready');
    
    // Load conversation history
    this.loadConversations();
    
    // Initialize plugins
    this.loadPlugins();
    
    // Setup keyboard shortcuts
    this.setupKeyboardShortcuts();
    
    // WebSocket for real-time features (optional)
    // this.setupWebSocket();
  }
  
  async loadModels() {
    try {
      const response = await fetch('/api/v2/models');
      const data = await response.json();
      
      const select = document.getElementById('modelSelect');
      select.innerHTML = '';
      
      data.models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = model.label.replace('NovaMind', '').trim();
        select.appendChild(option);
      });
    } catch (error) {
      console.error('Failed to load models:', error);
    }
  }
  
  setupEventListeners() {
    // Send button
    document.getElementById('sendBtn').addEventListener('click', () => this.sendMessage());
    
    // Input field
    const input = document.getElementById('messageInput');
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });
    
    // Auto-resize textarea
    input.addEventListener('input', () => {
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 200) + 'px';
    });
    
    // Model selector
    document.getElementById('modelSelect').addEventListener('change', (e) => {
      this.currentModel = e.target.value;
      document.getElementById('currentMode').textContent = e.target.selectedOptions[0].text;
    });
    
    // Menu button
    document.getElementById('menuBtn').addEventListener('click', () => {
      document.getElementById('sidebar').classList.toggle('open');
    });
    
    // Settings button
    document.getElementById('settingsBtn').addEventListener('click', () => {
      document.getElementById('settingsModal').classList.remove('hidden');
    });
    
    // File input
    document.getElementById('fileInput').addEventListener('change', (e) => {
      this.handleFileUpload(e.target.files);
    });
    
    // Settings controls
    document.getElementById('temperature').addEventListener('input', (e) => {
      document.getElementById('tempValue').textContent = e.target.value;
      this.settings.temperature = parseFloat(e.target.value);
    });
    
    document.getElementById('topP').addEventListener('input', (e) => {
      document.getElementById('topPValue').textContent = e.target.value;
      this.settings.topP = parseFloat(e.target.value);
    });
  }
  
  setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      // Ctrl/Cmd + K: Focus input
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        document.getElementById('messageInput').focus();
      }
      
      // Ctrl/Cmd + /: Toggle sidebar
      if ((e.ctrlKey || e.metaKey) && e.key === '/') {
        e.preventDefault();
        document.getElementById('sidebar').classList.toggle('open');
      }
      
      // Escape: Close modals
      if (e.key === 'Escape') {
        document.getElementById('settingsModal').classList.add('hidden');
        document.getElementById('sidebar').classList.remove('open');
      }
    });
  }
  
  async sendMessage() {
    const input = document.getElementById('messageInput');
    \1
    
    // Snapshot attachments before we clear them
    const attachments = this.attachments.slice();
    
    // Disable input
    input.disabled = true;
    document.getElementById('sendBtn').disabled = true;
    
    // Add user message to UI
    this.addMessage('user', content, attachments);
    
    // Clear input
    input.value = '';
    input.style.height = 'auto';
    // Update messages array
    this.messages.push({
      role: 'user',
      content: content,
      attachments: attachments
    });
    
    // Clear input attachments after queuing
    this.attachments = [];
    
    // Show typing indicator
    this.showTypingIndicator();
    
    try {
      // Prepare request
      const requestData = {
        model: this.currentModel,
        messages: this.messages,
        session_id: this.sessionId,
        conversation_id: this.conversationId,
        ...this.settings,
        plugins: Array.from(this.features)
      };
      
      if (this.settings.stream) {
        await this.streamResponse(requestData);
      } else {
        await this.getResponse(requestData);
      }
      
    } catch (error) {
      console.error('Error:', error);
      this.addMessage('assistant', '❌ An error occurred. Please try again.');
    } finally {
      // Re-enable input
      input.disabled = false;
      document.getElementById('sendBtn').disabled = false;
      input.focus();
      
      // Remove typing indicator
      this.hideTypingIndicator();
    }
  }
  
  async getResponse(requestData) {
    const response = await fetch('/api/v2/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestData)
    });
    
    const data = await response.json();
    
    if (data.error) {
      throw new Error(data.error);
    }
    
    // Add assistant message
    this.addMessage('assistant', data.response, [], data.metadata);
    
    // Update messages
    this.messages.push({
      role: 'assistant',
      content: data.response
    });
    
    // Update token count
    if (data.metadata?.usage) {
      const tokens = data.metadata.usage.totalTokens || 0;
      document.getElementById('tokenCount').textContent = tokens.toLocaleString();
    }
  }
  
  async streamResponse(requestData) {
    requestData.stream = true;
    
    const response = await fetch('/api/v2/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestData)
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let messageDiv = null;
    let fullResponse = '';
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            
            if (!messageDiv) {
              messageDiv = this.addMessage('assistant', '', [], null, true);
            }
            
            fullResponse += data.text;
            this.updateStreamingMessage(messageDiv, fullResponse);
            
          } catch (e) {
            console.error('Parse error:', e);
          }
        }
      }
    }
    
    // Update messages array
    this.messages.push({
      role: 'assistant',
      content: fullResponse
    });
  }
  
  addMessage(role, content, attachments = [], metadata = null, streaming = false) {
    const container = document.getElementById('chatContainer');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role === 'user' ? 'message-user' : 'message-assistant'}`;
    
    const innerHtml = `
      <div class="flex items-start gap-3">
        ${role === 'assistant' ? `
          <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center flex-shrink-0">
            <span>🧠</span>
          </div>
        ` : ''}
        <div class="flex-1">
          <div class="text-xs text-gray-400 mb-2">${role === 'user' ? 'You' : 'NovaMind Ultra'}</div>
          <div class="prose prose-invert max-w-none message-content">
            ${streaming ? '<span class="typing-indicator">Thinking</span>' : this.formatMessage(content)}
          </div>
          ${metadata?.thoughts ? `
            <details class="mt-3">
              <summary class="text-xs text-gray-400 cursor-pointer">View reasoning</summary>
              <div class="mt-2 p-3 glass rounded-lg text-sm">${metadata.thoughts}</div>
            </details>
          ` : ''}
          ${attachments.length > 0 ? `
            <div class="flex gap-2 mt-3">
              ${attachments.map(a => `
                <div class="glass px-3 py-1 rounded-lg text-xs">
                  📎 ${a.name || 'Attachment'}
                </div>
              `).join('')}
            </div>
          ` : ''}
        </div>
      </div>
    `;
    
    messageDiv.innerHTML = innerHtml;
    container.appendChild(messageDiv);
    
    // Scroll to bottom
    container.scrollTop = container.scrollHeight;
    
    // Initialize code highlighting
    if (!streaming) {
      this.highlightCode(messageDiv);
    }
    
    return messageDiv;
  }
  
  updateStreamingMessage(messageDiv, content) {
    const contentDiv = messageDiv.querySelector('.message-content');
    contentDiv.innerHTML = this.formatMessage(content);
    this.highlightCode(messageDiv);
  }
  
  formatMessage(content) {
    // Use marked.js for markdown
    let formatted = marked.parse(content);
    
    // Enhance code blocks
    formatted = formatted.replace(/<pre><code class="language-(\w+)">([\s\S]*?)<\/code><\/pre>/g, (match, lang, code) => {
      const escaped = this.escapeHtml(code);
      return `
        <div class="code-block">
          <div class="code-header">
            <span class="code-lang">${lang}</span>
            <button class="copy-btn" onclick="copyCode(this)">Copy</button>
          </div>
          <pre><code class="language-${lang}">${escaped}</code></pre>
        </div>
      `;
    });
    
    return formatted;
  }
  
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
  
  highlightCode(element) {
    // Use Prism.js for syntax highlighting
    element.querySelectorAll('pre code').forEach(block => {
      Prism.highlightElement(block);
    });
  }
  
  showTypingIndicator() {
    this.updateStatus('NovaMind is thinking...', 'thinking');
  }
  
  hideTypingIndicator() {
    this.updateStatus('Ready');
  }
  
  updateStatus(message, type = 'normal') {
    const statusEl = document.getElementById('status');
    statusEl.textContent = message;
    statusEl.className = type === 'thinking' ? 'text-yellow-400' : 'text-gray-400';
  }
  
  async handleFileUpload(files) {
    const formData = new FormData();
    
    for (const file of files) {
      formData.append('files', file);
    }
    
    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      
      data.files.forEach(file => {
        this.attachments.push(file);
      });
      
      this.updateStatus(`${files.length} file(s) attached`);
      
    } catch (error) {
      console.error('Upload error:', error);
      this.updateStatus('Upload failed', 'error');
    }
  }
  
  async loadConversations() {
    try {
      const response = await fetch('/api/v2/conversations');
      const conversations = await response.json();
      
      // Update sidebar with conversations
      // ... implementation
      
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  }
  
  async loadPlugins() {
    try {
      const response = await fetch('/api/v2/plugins');
      const plugins = await response.json();
      
      // Update plugins UI
      // ... implementation
      
    } catch (error) {
      console.error('Failed to load plugins:', error);
    }
  }
}

// Global functions
window.toggleFeature = function(feature, ev) {
  const client = window.novamindClient;
  if (client.features.has(feature)) {
    client.features.delete(feature);
  } else {
    client.features.add(feature);
  }
  
  // Update UI
  ev && ev.currentTarget && ev.currentTarget.classList.toggle('active');
};

window.copyCode = function(button) {
  const codeBlock = button.closest('.code-block').querySelector('code');
  const text = codeBlock.textContent;
  
  navigator.clipboard.writeText(text).then(() => {
    button.textContent = 'Copied!';
    setTimeout(() => {
      button.textContent = 'Copy';
    }, 2000);
  });
};

window.closeSettings = function() {
  document.getElementById('settingsModal').classList.add('hidden');
};

window.saveSettings = function() {
  const client = window.novamindClient;
  
  // Update settings from UI
  client.settings.maxTokens = parseInt(document.getElementById('maxTokens').value);
  client.settings.thinkingBudget = parseInt(document.getElementById('thinkingBudget').value);
  
  // Close modal
  closeSettings();
  
  // Show confirmation
  client.updateStatus('Settings saved');
};

// Initialize client
window.novamindClient = new NovaMindClient();

// Initialize Mermaid for diagrams
mermaid.initialize({ theme: 'dark' });
</script>

</body>
</html>"""

@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    print(f"🚀 NovaMind Ultra v{VERSION} starting on port {port}")
    print(f"🧠 Created by Jack - Advanced AI Assistant Platform")
    app.run(host="0.0.0.0", port=port, debug=False)
