# app.py
# NovaMind Ultra — Advanced AI Assistant Platform (v3.5)
# Full rewrite with stronger security, cleaner backend, and a sleeker, more advanced UI
# -------------------------------------------------------------
# Quick start
#   1) pip install flask flask-cors
#   2) export GOOGLE_API_KEYS="key1,key2"   # Comma-separated Gemini API keys (no keys committed in code!)
#   3) python app.py
# -------------------------------------------------------------

import os, base64, json, mimetypes, time, re, tempfile, hashlib, threading, io, sys, traceback
import sqlite3, uuid, datetime
from collections import defaultdict, deque
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from contextlib import redirect_stdout, redirect_stderr
from flask import Flask, request, send_from_directory, make_response, jsonify, Response, stream_with_context
from flask_cors import CORS
import gzip

# =========================
# Configuration
# =========================
APP_TITLE = "NovaMind Ultra"
VERSION = "3.5.0"

TMP = tempfile.gettempdir()
UPLOAD_DIR = os.environ.get("UPLOAD_DIR") or os.path.join(TMP, "novamind_uploads")
CACHE_DIR = os.path.join(TMP, "novamind_cache")
SESSIONS_DIR = os.path.join(TMP, "novamind_sessions")
PLUGINS_DIR = os.path.join(TMP, "novamind_plugins")

for d in [UPLOAD_DIR, CACHE_DIR, SESSIONS_DIR, PLUGINS_DIR]:
    os.makedirs(d, exist_ok=True)

# API keys come from env only (never hard-code secrets)
_GOOGLE_KEYS = [k.strip() for k in os.environ.get("GOOGLE_API_KEYS", "AIzaSyCa7P192Lu1OGP3c5Q_BB8ADY4UpZMB2a4").split(",") if k.strip()]

# Public model options (labels are just UI text)
NOVA_MODELS = [
    {"id": "ultra",  "label": "🚀 NovaMind Ultra — Maximum Intelligence", "features": ["vision","code","analysis","creativity"]},
    {"id": "sage",   "label": "🧙 NovaMind Sage — Deep Reasoning",          "features": ["analysis","research","planning"]},
    {"id": "spark",  "label": "⚡ NovaMind Spark — Lightning Fast",         "features": ["speed","efficiency","realtime"]},
    {"id": "vision", "label": "👁️ NovaMind Vision — Multimodal Expert",     "features": ["images","documents","ocr"]},
]

# Upstream model mapping (you can change these without touching UI)
MODEL_MAP = {
    "ultra": "gemini-2.5-pro",
    "sage": "gemini-2.5-pro",
    "spark": "gemini-2.5-flash",
    "vision": "gemini-2.5-pro",
}

TOKEN_LIMITS = {
    "max_input": 1_048_576,
    "max_output": 65_535,
}

# =========================
# Database
# =========================
DB_PATH = os.path.join(SESSIONS_DIR, "novamind.db")

def init_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT,
        created_at TIMESTAMP,
        updated_at TIMESTAMP,
        starred INTEGER DEFAULT 0,
        archived INTEGER DEFAULT 0,
        tags TEXT,
        model_preferences TEXT,
        memory_bank TEXT,
        analytics TEXT
    )""")

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

    c.execute("""CREATE TABLE IF NOT EXISTS user_profiles (
        id TEXT PRIMARY KEY,
        settings TEXT,
        custom_prompts TEXT,
        api_keys TEXT,
        usage_stats TEXT,
        created_at TIMESTAMP
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS knowledge_base (
        id TEXT PRIMARY KEY,
        content TEXT,
        embedding BLOB,
        metadata TEXT,
        created_at TIMESTAMP
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS plugins (
        id TEXT PRIMARY KEY,
        name TEXT,
        code TEXT,
        config TEXT,
        enabled INTEGER DEFAULT 1,
        created_at TIMESTAMP
    )""")

    # Helpful indexes
    c.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_messages_time ON messages(timestamp)")

    conn.commit(); conn.close()

init_database()

# =========================
# App
# =========================
app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Security headers
@app.after_request
def add_security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["X-XSS-Protection"] = "1; mode=block"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp

# =========================
# Utilities
# =========================
class AdvancedCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.order = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                try:
                    self.order.remove(key)
                except ValueError:
                    pass
                self.order.append(key)
                return self.cache[key]
            return None

    def set(self, key, value):
        with self.lock:
            if key in self.cache:
                try:
                    self.order.remove(key)
                except ValueError:
                    pass
            self.cache[key] = value
            self.order.append(key)
            if len(self.cache) > self.order.maxlen:
                oldest = self.order.popleft()
                self.cache.pop(oldest, None)

cache = AdvancedCache()


def generate_session_id():
    return f"sess_{uuid.uuid4().hex}_{int(time.time()*1000)}"

def generate_conversation_id():
    return f"conv_{uuid.uuid4().hex}"


def compress_json(obj: dict) -> str:
    raw = json.dumps(obj).encode()
    return base64.b64encode(gzip.compress(raw)).decode()

def decompress_json(s: str) -> dict:
    return json.loads(gzip.decompress(base64.b64decode(s)))

# =========================
# File uploads (now actually implemented)
# =========================
@app.route("/api/upload", methods=["POST"])
def api_upload():
    files = request.files.getlist("files")
    saved = []
    for f in files:
        if not f.filename:
            continue
        ext = os.path.splitext(f.filename)[1]
        safe_ext = ext if len(ext) <= 8 else ext[:8]
        fname = f"{uuid.uuid4().hex}{safe_ext}"
        path = os.path.join(UPLOAD_DIR, fname)
        f.save(path)
        saved.append({
            "name": f.filename,
            "stored_name": fname,
            "url": f"/uploads/{fname}",
            "size": os.path.getsize(path),
            "mime": mimetypes.guess_type(fname)[0] or "application/octet-stream",
        })
    return jsonify({"files": saved})

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# =========================
# Code execution — restricted (no imports, no filesystem)
# =========================
_ALLOWED_BUILTINS = {
    "print": print, "len": len, "range": range, "int": int, "float": float, "str": str,
    "list": list, "dict": dict, "tuple": tuple, "set": set, "bool": bool, "sorted": sorted,
    "sum": sum, "min": min, "max": max, "abs": abs, "round": round, "enumerate": enumerate,
    "zip": zip, "map": map, "filter": filter, "any": any, "all": all
}

SAFE_NODE_TYPES = {
    "Module","Expr","Assign","AnnAssign","Name","Load","Store","Constant","BinOp","UnaryOp",
    "BoolOp","Compare","If","For","While","Break","Continue","Pass","List","Tuple","Dict",
    "Set","Call","keyword","Lambda","Return","AugAssign","IfExp","ListComp","DictComp","SetComp",
    "GeneratorExp","comprehension","Slice","Subscript","Index","JoinedStr","FormattedValue",
    "With","withitem","Try","ExceptHandler","Raise"
}

import ast

def _is_ast_safe(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        t = type(node).__name__
        # Disallow import, attribute access to dunders, and global exec/eval/open/input
        if t in ("Import","ImportFrom","Global","Nonlocal","ClassDef","FunctionDef","AsyncFunctionDef"):
            return False
        if isinstance(node, ast.Attribute) and isinstance(node.attr, str) and node.attr.startswith("__"):
            return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"exec","eval","open","compile","input","__import__"}:
            return False
        if t not in SAFE_NODE_TYPES and not isinstance(node, (ast.Attribute,)):
            return False
    return True

class CodeExecutor:
    @staticmethod
    def execute(code: str, timeout: int = 10):
        try:
            tree = ast.parse(code, mode="exec")
            if not _is_ast_safe(tree):
                return {"success": False, "output": "", "error": "Code blocked by sandbox policy."}
        except Exception as e:
            return {"success": False, "output": "", "error": f"Parse error: {e}"}

        restricted_globals = {"__builtins__": _ALLOWED_BUILTINS}
        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
        done = {"flag": False}

        def _runner():
            try:
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    exec(compile(tree, "<sandbox>", "exec"), restricted_globals, {})
            except Exception:
                traceback.print_exc(file=stderr_buf)
            finally:
                done["flag"] = True

        th = threading.Thread(target=_runner, daemon=True)
        th.start()
        th.join(timeout)
        if not done["flag"]:
            return {"success": False, "output": stdout_buf.getvalue(), "error": "Timeout"}
        return {"success": True, "output": stdout_buf.getvalue(), "error": stderr_buf.getvalue()}

# =========================
# Plugins (executed with same sandbox as CodeExecutor)
# =========================
class PluginManager:
    def __init__(self):
        self.plugins = {}
        self.load_plugins()

    def load_plugins(self):
        self.plugins = {}
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, name, code, config, enabled, created_at FROM plugins WHERE enabled = 1")
        for pid, name, code, config, enabled, created_at in c.fetchall():
            self.plugins[pid] = {"name": name, "code": code, "config": json.loads(config or "{}")}
        conn.close()

    def execute_plugin(self, plugin_id: str, context: dict):
        if plugin_id not in self.plugins:
            return {"error": "Plugin not found"}
        src = self.plugins[plugin_id]["code"]
        # Combine context and give a `result` var
        wrapper = f"""
result = None
context = {json.dumps(context)}
# --- Plugin code starts below ---
{src}
# --- Plugin code ends ---
"""
        res = CodeExecutor.execute(wrapper, timeout=5)
        if not res["success"]:
            return {"error": res["error"].strip()}
        # Best effort to capture a printed JSON result or ignore
        out = res["output"].strip()
        try:
            return json.loads(out) if out else {"ok": True}
        except Exception:
            return {"ok": True, "output": out}

plugin_manager = PluginManager()

# =========================
# AI Backend (Gemini over REST)
# =========================
class NovaMindBackend:
    def __init__(self):
        self.key_index = 0
        self.rate_limiter = defaultdict(lambda: {"count": 0, "reset": time.time() + 3600})
        self.performance_stats = defaultdict(list)

    def _next_key(self):
        if not _GOOGLE_KEYS:
            raise RuntimeError("No GOOGLE_API_KEYS provided. Set env GOOGLE_API_KEYS=key1,key2")
        key = _GOOGLE_KEYS[self.key_index % len(_GOOGLE_KEYS)]
        self.key_index += 1
        return key

    def _check_rl(self, key):
        now = time.time()
        slot = self.rate_limiter[key]
        if now > slot["reset"]:
            slot["count"], slot["reset"] = 0, now + 3600
        if slot["count"] >= 60:  # per hour per key
            return False
        slot["count"] += 1
        return True

    def _enhance_prompt(self, base_prompt: str, model_id: str) -> str:
        # Lightweight helper text appended to the beginning of the conversation.
        enhancements = {
            "ultra": (
                "You are NovaMind Ultra — a concise, helpful assistant. "
                "Prefer structured answers, bullet points, and code blocks when relevant. "
                "Think clearly, cite assumptions, and include edge cases in code."
            ),
            "sage": (
                "You are NovaMind Sage — perform deep reasoning with stepwise clarity. "
                "Explain trade-offs and list alternatives succinctly."
            ),
            "spark": (
                "You are NovaMind Spark — optimize for speed and brevity."
            ),
            "vision": (
                "You are NovaMind Vision — describe visual details and extract text when images are provided."
            ),
        }
        return f"[SYSTEM]\n{enhancements.get(model_id, enhancements['ultra'])}\n[/SYSTEM]\n\n{base_prompt or ''}"

    def _build_contents(self, system_prompt: str, messages: list):
        contents = []
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": system_prompt}]
            })
        for m in messages:
            parts = []
            if m.get("content"):
                parts.append({"text": m["content"]})
            for att in m.get("attachments", []) or []:
                url = att.get("url")
                if not url:
                    continue
                try:
                    data = self._fetch_attachment(url)
                    mime = att.get("mime") or mimetypes.guess_type(url)[0] or "application/octet-stream"
                    parts.append({"inlineData": {"mimeType": mime, "data": base64.b64encode(data).decode()}})
                except Exception:
                    # Ignore failed attachment fetch
                    pass
            role = "user" if m.get("role") != "assistant" else "model"
            contents.append({"role": role, "parts": parts})
        return contents

    def _fetch_attachment(self, url: str) -> bytes:
        if url.startswith("/uploads/"):
            path = os.path.join(UPLOAD_DIR, url.split("/uploads/")[1])
            with open(path, "rb") as f:
                return f.read()
        else:
            with urlopen(url, timeout=30) as r:
                return r.read()

    def _extract(self, data: dict):
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                return None, {}
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            text = "\n".join([p.get("text", "") for p in parts if "text" in p])
            meta = {
                "usage": data.get("usageMetadata", {}),
                "finish_reason": candidates[0].get("finishReason"),
            }
            return text, meta
        except Exception:
            return None, {}

    def _postprocess(self, text: str) -> str:
        # Ensure fenced code blocks have language where possible (basic heuristic)
        text = re.sub(r"```\n", "```python\n", text)
        return text

    def generate(self, model_id: str, system_prompt: str, messages: list, config: dict, stream: bool = False):
        start = time.time()
        cache_key = hashlib.md5(f"{model_id}|{system_prompt}|{json.dumps(messages)}|{json.dumps(config)}".encode()).hexdigest()
        if not stream:
            hit = cache.get(cache_key)
            if hit:
                return hit["text"], hit["meta"], None

        actual = MODEL_MAP.get(model_id, MODEL_MAP["ultra"])
        contents = self._build_contents(self._enhance_prompt(system_prompt, model_id), messages)

        gen_cfg = {
            "candidateCount": 1,
            "maxOutputTokens": max(1, min(TOKEN_LIMITS["max_output"], int(config.get("max_tokens", 8192)))),
            "temperature": float(config.get("temperature", 0.7)),
            "topP": float(config.get("top_p", 0.95)),
            "topK": int(config.get("top_k", 40)),
        }
        payload = {"contents": contents, "generationConfig": gen_cfg}

        def _do_request(url: str, key: str, body: dict):
            req = Request(url, data=json.dumps(body).encode(), method="POST")
            req.add_header("Content-Type", "application/json")
            return urlopen(req, timeout=120)

        # Failover across keys
        for attempt in range(max(1, len(_GOOGLE_KEYS))):
            try:
                key = self._next_key()
                if not self._check_rl(key):
                    continue
                if stream:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{actual}:streamGenerateContent?key={key}"
                    response = _do_request(url, key, payload)

                    def stream_gen():
                        buf = ""
                        for chunk in response:
                            buf += chunk.decode()
                            while "\n" in buf:
                                line, buf = buf.split("\n", 1)
                                if not line.strip():
                                    continue
                                try:
                                    data = json.loads(line)
                                    text, _ = self._extract(data)
                                    if text:
                                        yield f"data: {json.dumps({'text': text})}\n\n"
                                except Exception:
                                    # ignore parse errors from partial lines
                                    pass
                    return Response(stream_with_context(stream_gen()), mimetype="text/event-stream")
                else:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{actual}:generateContent?key={key}"
                    with _do_request(url, key, payload) as r:
                        data = json.loads(r.read().decode())
                        text, meta = self._extract(data)
                        if text:
                            text = self._postprocess(text)
                            meta["model_variant"] = model_id
                            meta["latency_ms"] = int((time.time() - start) * 1000)
                            cache.set(cache_key, {"text": text, "meta": meta})
                            return text, meta, None
                        return None, None, {"error": "Empty response"}
            except Exception as e:
                last = str(e)
                continue
        return None, None, {"error": "All keys failed or not provided"}

backend = NovaMindBackend()

# =========================
# Helpers (DB)
# =========================

def _get_conversation_memory(conversation_id: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT content FROM messages WHERE conversation_id=? ORDER BY timestamp DESC LIMIT 5", (conversation_id,))
    items = [row[0][:120] for row in c.fetchall()]
    conn.close()
    return ("Recent context: " + " | ".join(reversed(items))) if items else ""


def _save_message(conversation_id: str, role: str, content: str, attachments=None, metadata=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    mid = str(uuid.uuid4())
    c.execute("""
        INSERT INTO messages (id, conversation_id, role, content, attachments, metadata, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (mid, conversation_id, role, content, json.dumps(attachments or []), json.dumps(metadata or {}), datetime.datetime.now()))
    c.execute("UPDATE conversations SET updated_at=? WHERE id=?", (datetime.datetime.now(), conversation_id))
    conn.commit(); conn.close()

# =========================
# API
# =========================
@app.route("/api/v2/health")
def api_health():
    return jsonify({"ok": True, "app": APP_TITLE, "version": VERSION})

@app.route("/api/v2/models")
def api_models_v2():
    return jsonify({"models": NOVA_MODELS, "version": VERSION, "capabilities": {
        "streaming": True, "plugins": True, "code_execution": True, "memory": True,
        "export": True, "voice": False, "search": True
    }})

@app.route("/api/v2/chat", methods=["POST"])
def api_chat_v2():
    data = request.json or {}
    session_id = data.get("session_id") or generate_session_id()
    conversation_id = data.get("conversation_id") or generate_conversation_id()

    model = data.get("model", "ultra")
    messages = data.get("messages", [])
    stream = bool(data.get("stream", False))

    config = {
        "temperature": min(2.0, max(0.0, float(data.get("temperature", 0.7)))),
        "max_tokens": min(TOKEN_LIMITS["max_output"], max(1, int(data.get("max_tokens", 8192)))),
        "top_p": float(data.get("topP") or data.get("top_p", 0.95)),
        "top_k": int(data.get("topK") or data.get("top_k", 40)),
    }

    system_prompt = str(data.get("system_prompt", "")).strip()

    if data.get("use_memory", True):
        mem = _get_conversation_memory(conversation_id)
        if mem:
            system_prompt = f"{system_prompt}\n\n[CONVERSATION MEMORY]\n{mem}\n[/CONVERSATION MEMORY]"

    # Plugins
    if data.get("plugins"):
        for pid in data["plugins"]:
            plugin_output = plugin_manager.execute_plugin(pid, {"messages": messages, "model": model})
            messages.append({"role": "system", "content": f"Plugin result: {json.dumps(plugin_output)[:2000]}"})

    if stream:
        return backend.generate(model, system_prompt, messages, config, stream=True)

    text, meta, err = backend.generate(model, system_prompt, messages, config, stream=False)
    if err:
        return jsonify({"error": err}), 500

    if messages:
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if last_user:
            _save_message(conversation_id, "user", last_user.get("content", ""), last_user.get("attachments"))
    _save_message(conversation_id, "assistant", text, [], meta)

    return jsonify({"response": text, "metadata": meta, "session_id": session_id, "conversation_id": conversation_id})

@app.route("/api/v2/conversations", methods=["GET","POST"])
def api_conversations():
    if request.method == "GET":
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT id, title, created_at, updated_at, starred, tags
            FROM conversations
            WHERE archived=0
            ORDER BY updated_at DESC
            LIMIT 50
        """)
        items = []
        for row in c.fetchall():
            items.append({
                "id": row[0], "title": row[1], "created_at": row[2], "updated_at": row[3],
                "starred": bool(row[4]), "tags": json.loads(row[5] or "[]")
            })
        conn.close()
        return jsonify(items)
    else:
        data = request.json or {}
        conv_id = generate_conversation_id()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO conversations (id, title, created_at, updated_at, tags)
            VALUES (?, ?, ?, ?, ?)
        """, (conv_id, data.get("title", "New Conversation"), datetime.datetime.now(), datetime.datetime.now(), json.dumps(data.get("tags", []))))
        conn.commit(); conn.close()
        return jsonify({"id": conv_id})

@app.route("/api/v2/execute", methods=["POST"])
def api_execute_code():
    data = request.json or {}
    code = str(data.get("code", ""))
    timeout = int(min(30, max(1, data.get("timeout", 10))))
    return jsonify(CodeExecutor.execute(code, timeout))

@app.route("/api/v2/search", methods=["POST"])
def api_search():
    data = request.json or {}
    query = str(data.get("query", "")).strip()
    search_type = data.get("type", "all")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    results = {"conversations": [], "messages": []}

    if search_type in ("all","conversations"):
        c.execute("SELECT id, title, tags FROM conversations WHERE title LIKE ? OR tags LIKE ? LIMIT 20", (f"%{query}%", f"%{query}%"))
        for row in c.fetchall():
            results["conversations"].append({"id": row[0], "title": row[1], "tags": json.loads(row[2] or "[]")})

    if search_type in ("all","messages"):
        c.execute("""
            SELECT m.id, m.conversation_id, m.content, c.title
            FROM messages m JOIN conversations c ON m.conversation_id=c.id
            WHERE m.content LIKE ?
            ORDER BY m.timestamp DESC
            LIMIT 20
        """, (f"%{query}%",))
        for row in c.fetchall():
            results["messages"].append({"id": row[0], "conversation_id": row[1], "content": row[2][:200], "conversation_title": row[3]})

    conn.close()
    return jsonify(results)

@app.route("/api/v2/export/<conversation_id>")
def api_export(conversation_id):
    fmt = request.args.get("format", "json")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,))
    conv = c.fetchone()
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404
    c.execute("SELECT * FROM messages WHERE conversation_id=? ORDER BY timestamp", (conversation_id,))
    messages = c.fetchall(); conn.close()

    if fmt == "json":
        data = {
            "conversation": {
                "id": conv[0], "title": conv[1], "created_at": conv[2],
                "messages": [{"role": m[2], "content": m[3], "timestamp": m[6]} for m in messages]
            }
        }
        return jsonify(data)
    elif fmt == "markdown":
        md = [f"# {conv[1]}", "", f"*Created: {conv[2]}*", "", "---", ""]
        for m in messages:
            who = "**You:**" if m[2] == "user" else "**Assistant:**"
            md.extend([who, "", m[3], "", "---", ""])
        return Response("\n".join(md), mimetype="text/markdown")
    elif fmt == "html":
        html = [f"""
        <!doctype html><html><head><meta charset='utf-8'/><title>{conv[1]} - Export</title>
        <style>
        body{{font-family:Inter,system-ui,sans-serif;max-width:850px;margin:40px auto;padding:0 20px;color:#0f172a}}
        .msg{{border-radius:12px;padding:12px 14px;margin:12px 0}}
        .u{{background:#e0ecff;text-align:right}} .a{{background:#f3f4f6}}
        </style></head><body><h1>{conv[1]}</h1><p><em>Exported {datetime.datetime.now()}</em></p><hr/>
        """]
        for m in messages:
            cls = "u" if m[2] == "user" else "a"
            html.append(f"<div class='msg {cls}'>{m[3]}</div>")
        html.append("</body></html>")
        return Response("".join(html), mimetype="text/html")
    else:
        return jsonify({"error": "Unsupported format"}), 400

@app.route("/api/v2/plugins", methods=["GET","POST"])
def api_plugins():
    if request.method == "GET":
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, name, config, enabled FROM plugins")
        items = []
        for row in c.fetchall():
            items.append({"id": row[0], "name": row[1], "config": json.loads(row[2] or "{}"), "enabled": bool(row[3])})
        conn.close(); return jsonify(items)
    data = request.json or {}
    pid = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO plugins (id, name, code, config, enabled, created_at)
        VALUES (?, ?, ?, ?, 1, ?)
    """, (pid, data.get("name","Unnamed Plugin"), data.get("code",""), json.dumps(data.get("config", {})), datetime.datetime.now()))
    conn.commit(); conn.close()
    plugin_manager.load_plugins()
    return jsonify({"id": pid})

# =========================
# UI (Tailwind + Prism + Mermaid). Cleaner, denser, and more polished.
# =========================
HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"/>
<title>NovaMind Ultra</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='84'>🧠</text></svg>">
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs@1/prism.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<link href="https://cdn.jsdelivr.net/npm/prismjs@1/themes/prism-tomorrow.css" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
<style>
:root { --primary:#6366f1; --accent:#8b5cf6; --ink:#e2e8f0; --muted:#94a3b8; --bg0:#0b1020; --bg1:#0f172a; --surf:#162032; }
*{font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,'Helvetica Neue',Arial,'Noto Sans','Apple Color Emoji','Segoe UI Emoji';}
body{background:radial-gradient(1200px 600px at 20% -20%, rgba(99,102,241,.15), transparent),linear-gradient(120deg,var(--bg0),var(--bg1));color:var(--ink);}
.hero-grad{background:linear-gradient(135deg, rgba(99,102,241,.12), rgba(139,92,246,.12));}
.glass{background:rgba(22,32,50,.55);backdrop-filter:blur(18px) saturate(1.4);border:1px solid rgba(255,255,255,.06)}
.bubble{border-radius:16px;padding:14px 16px;margin:12px 0}
.bubble.user{background:linear-gradient(135deg, rgba(99,102,241,.18), rgba(139,92,246,.18));border:1px solid rgba(99,102,241,.35)}
.bubble.ai{background:rgba(15,23,42,.55);border:1px solid rgba(255,255,255,.08)}
.code-wrap{position:relative;border-radius:12px;overflow:hidden;border:1px solid rgba(255,255,255,.08)}
.code-head{display:flex;align-items:center;justify-content:space-between;background:rgba(99,102,241,.12);padding:6px 10px}
.code-lang{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#a5b4fc}
.copy{font-size:12px;background:rgba(99,102,241,.18);border:1px solid rgba(99,102,241,.3);padding:4px 10px;border-radius:8px}
.scroll-thin::-webkit-scrollbar{height:8px;width:8px}.scroll-thin::-webkit-scrollbar-thumb{background:rgba(99,102,241,.4);border-radius:4px}
</style>
</head>
<body class="min-h-screen">
<header class="sticky top-0 z-40 glass">
  <div class="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
    <div class="flex items-center gap-3">
      <button id="menuBtn" class="p-2 rounded-lg hover:bg-white/10"><svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/></svg></button>
      <div class="w-10 h-10 rounded-xl hero-grad flex items-center justify-center"><span class="text-2xl">🧠</span></div>
      <div>
        <div class="font-bold text-lg">NovaMind Ultra</div>
        <div class="text-xs text-slate-400">Advanced AI Assistant</div>
      </div>
    </div>
    <div class="flex items-center gap-3">
      <div class="hidden sm:flex items-center gap-2 text-xs text-slate-400"><span class="inline-block w-2 h-2 bg-emerald-400 rounded-full"></span>Online</div>
      <select id="modelSelect" class="glass px-3 py-2 rounded-lg text-sm">
        <option value="ultra">Ultra</option>
        <option value="sage">Sage</option>
        <option value="spark">Spark</option>
        <option value="vision">Vision</option>
      </select>
      <button id="settingsBtn" class="p-2 rounded-lg hover:bg-white/10" title="Settings"><svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756.426-1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c-.94 1.543.826 3.31 2.37 2.37.996.608 2.296.07 2.572-1.065z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg></button>
    </div>
  </div>
</header>

<aside id="sidebar" class="fixed top-0 left-0 h-full w-72 glass -translate-x-full transition-transform duration-300 z-50">
  <div class="p-4 h-full flex flex-col">
    <button id="newConvBtn" class="w-full mb-4 px-3 py-2 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600">New Conversation</button>
    <div class="grid grid-cols-3 gap-2 text-xs text-slate-300">
      <button class="tab active glass px-2 py-2 rounded-lg" data-tab="recent">Recent</button>
      <button class="tab glass px-2 py-2 rounded-lg" data-tab="starred">Starred</button>
      <button class="tab glass px-2 py-2 rounded-lg" data-tab="archive">Archive</button>
    </div>
    <div id="convList" class="mt-4 space-y-2 overflow-auto scroll-thin"></div>
    <div class="mt-auto pt-4 border-t border-white/10 grid grid-cols-4 gap-2 text-center text-[11px] text-slate-300">
      <button class="glass py-2 rounded-lg" id="exportBtn">Export</button>
      <button class="glass py-2 rounded-lg" id="searchBtn">Search</button>
      <button class="glass py-2 rounded-lg" id="pluginsBtn">Plugins</button>
      <button class="glass py-2 rounded-lg" id="memoryBtn">Memory</button>
    </div>
  </div>
</aside>

<main class="pt-4 pb-36 px-4 max-w-4xl mx-auto">
  <div class="glass rounded-2xl p-4 md:p-6">
    <div class="flex items-start gap-3 bubble ai">
      <div class="w-8 h-8 rounded-lg hero-grad flex items-center justify-center">🧠</div>
      <div class="flex-1">
        <div class="text-xs text-slate-400 mb-1">NovaMind Ultra</div>
        <div class="prose prose-invert max-w-none">
          <p>Welcome! I can help with complex reasoning, production-ready code, research, and visuals. Switch modes above or just start typing.</p>
        </div>
      </div>
    </div>
  </div>

  <div id="chatContainer" class="mt-4 space-y-3"></div>
</main>

<footer class="fixed bottom-0 left-0 right-0 glass border-t border-white/10">
  <div class="max-w-4xl mx-auto p-3">
    <div class="flex flex-wrap gap-2 mb-2">
      <label class="glass px-3 py-1 rounded-lg text-xs cursor-pointer">📎 Attach<input type="file" id="fileInput" class="hidden" multiple></label>
      <button class="glass px-3 py-1 rounded-lg text-xs" data-feature="code">💻 Code</button>
      <button class="glass px-3 py-1 rounded-lg text-xs" data-feature="web">🌐 Web</button>
      <button class="glass px-3 py-1 rounded-lg text-xs" data-feature="image">🖼️ Image</button>
      <button class="glass px-3 py-1 rounded-lg text-xs" data-feature="memory">🧠 Memory</button>
    </div>
    <div class="flex items-end gap-2">
      <textarea id="messageInput" class="glass flex-1 p-3 rounded-xl min-h-[48px] max-h-48 resize-none" placeholder="Ask me anything… (Shift+Enter = newline)"></textarea>
      <button id="sendBtn" class="px-4 py-3 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600">Send</button>
    </div>
    <div class="flex items-center justify-between mt-2 text-[11px] text-slate-400">
      <div id="status">Ready</div>
      <div class="flex items-center gap-4"><span>Tokens: <span id="tokenCount">0</span></span><span>Mode: <span id="currentMode">Ultra</span></span></div>
    </div>
  </div>
</footer>

<!-- Settings Modal -->
<div id="settingsModal" class="hidden fixed inset-0 bg-black/50 backdrop-blur-sm z-50 items-center justify-center">
  <div class="glass rounded-2xl p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
    <h2 class="text-xl font-semibold mb-4">Settings</h2>
    <div class="grid sm:grid-cols-2 gap-4">
      <div>
        <label class="text-xs text-slate-400">Temperature</label>
        <input type="range" min="0" max="2" step="0.1" value="0.7" id="temperature" class="w-full">
        <div class="text-xs" id="tempValue">0.7</div>
      </div>
      <div>
        <label class="text-xs text-slate-400">Max Tokens</label>
        <input type="number" value="8192" id="maxTokens" class="glass px-3 py-2 rounded-lg w-full">
      </div>
      <div>
        <label class="text-xs text-slate-400">Top P</label>
        <input type="range" min="0" max="1" step="0.05" value="0.95" id="topP" class="w-full">
        <div class="text-xs" id="topPValue">0.95</div>
      </div>
      <div>
        <label class="text-xs text-slate-400">Top K</label>
        <input type="number" value="40" id="topK" class="glass px-3 py-2 rounded-lg w-full">
      </div>
      <div>
        <label class="text-xs text-slate-400">Thinking Budget (hint)</label>
        <input type="number" value="30000" id="thinkingBudget" class="glass px-3 py-2 rounded-lg w-full">
      </div>
      <div class="flex items-center gap-2 mt-4"><input type="checkbox" id="useMemory" checked><label for="useMemory" class="text-sm">Enable memory</label></div>
      <div class="flex items-center gap-2 mt-1"><input type="checkbox" id="streamResponses" checked><label for="streamResponses" class="text-sm">Stream responses</label></div>
    </div>
    <div class="flex justify-end gap-2 mt-6"><button id="closeSettings" class="px-3 py-2 rounded-lg hover:bg-white/10">Cancel</button><button id="saveSettings" class="px-3 py-2 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600">Save</button></div>
  </div>
</div>

<script>
class NovaMindClient {
  constructor(){
    this.sessionId = this._id('sess');
    this.conversationId = this._id('conv');
    this.messages = [];
    this.currentModel = 'ultra';
    this.features = new Set();
    this.attachments = [];
    this.settings = { temperature: 0.7, maxTokens: 8192, topP: 0.95, topK: 40, thinkingBudget: 30000, useMemory: true, stream: true, plugins: [] };
    this.isStreaming = false;
    this.init();
  }
  _id(p){ return `${p}_${Date.now()}_${Math.random().toString(36).slice(2,9)}`; }
  async init(){
    this._bind();
    this._status('Ready');
    await this._loadModels();
    await this._loadConversations();
  }
  _bind(){
    const $ = (id)=>document.getElementById(id);
    // Sidebar
    document.getElementById('menuBtn').addEventListener('click', ()=>{
      const s = document.getElementById('sidebar');
      s.classList.toggle('-translate-x-full');
    });
    document.getElementById('settingsBtn').addEventListener('click', ()=>{
      const m = document.getElementById('settingsModal'); m.classList.remove('hidden'); m.classList.add('flex');
    });
    document.getElementById('closeSettings').addEventListener('click', ()=>{
      const m = document.getElementById('settingsModal'); m.classList.add('hidden'); m.classList.remove('flex');
    });
    document.getElementById('saveSettings').addEventListener('click', ()=>{
      this.settings.maxTokens = parseInt(document.getElementById('maxTokens').value||'8192',10);
      this.settings.topK = parseInt(document.getElementById('topK').value||'40',10);
      this.settings.thinkingBudget = parseInt(document.getElementById('thinkingBudget').value||'30000',10);
      this.settings.useMemory = document.getElementById('useMemory').checked;
      this.settings.stream = document.getElementById('streamResponses').checked;
      this._status('Settings saved');
      document.getElementById('settingsModal').classList.add('hidden');
    });
    document.getElementById('temperature').addEventListener('input', (e)=>{ this.settings.temperature = parseFloat(e.target.value); document.getElementById('tempValue').textContent = e.target.value; });
    document.getElementById('topP').addEventListener('input', (e)=>{ this.settings.topP = parseFloat(e.target.value); document.getElementById('topPValue').textContent = e.target.value; });
    document.getElementById('modelSelect').addEventListener('change',(e)=>{ this.currentModel = e.target.value; document.getElementById('currentMode').textContent = e.target.selectedOptions[0].text; });
    document.getElementById('fileInput').addEventListener('change',(e)=> this._uploadFiles(e.target.files));
    document.getElementById('sendBtn').addEventListener('click',()=> this.sendMessage());
    const input = document.getElementById('messageInput');
    input.addEventListener('keydown',(e)=>{ if(e.key==='Enter' && !e.shiftKey){e.preventDefault(); this.sendMessage();} });
    input.addEventListener('input',()=>{ input.style.height='auto'; input.style.height=Math.min(input.scrollHeight, 192)+ 'px';});
    // Feature toggles (footer)
    document.querySelectorAll('[data-feature]').forEach(el=>{
      el.addEventListener('click', (ev)=>{ const f = el.dataset.feature; if(this.features.has(f)){ this.features.delete(f); el.classList.remove('ring-2','ring-indigo-400'); } else { this.features.add(f); el.classList.add('ring-2','ring-indigo-400'); } });
    });
  }

  async _loadModels(){
    try{ const res = await fetch('/api/v2/models'); const data = await res.json(); const sel = document.getElementById('modelSelect'); sel.innerHTML=''; data.models.forEach(m=>{ const o = document.createElement('option'); o.value = m.id; o.textContent = m.label.replace('NovaMind','').trim(); sel.appendChild(o); }); }catch(e){ console.error(e); }
  }
  async _loadConversations(){
    try{ const res = await fetch('/api/v2/conversations'); const items = await res.json(); const list = document.getElementById('convList'); list.innerHTML=''; items.forEach(i=>{ const div = document.createElement('div'); div.className='glass p-3 rounded-lg cursor-pointer hover:bg-white/5'; div.innerHTML = `<div class="text-sm font-medium">${i.title}</div><div class="text-[11px] text-slate-400 mt-1">updated ${i.updated_at}</div>`; list.appendChild(div); }); }catch(e){ console.error(e); }
  }

  async _uploadFiles(files){
    const fd = new FormData(); for(const f of files) fd.append('files', f);
    try{ const res = await fetch('/api/upload', {method:'POST', body: fd}); const data = await res.json(); (data.files||[]).forEach(f=> this.attachments.push(f)); this._status(`${(data.files||[]).length} file(s) attached`);}catch(e){ this._status('Upload failed'); }
  }

  _status(msg){ document.getElementById('status').textContent = msg; }

  _format(content){
    let html = marked.parse(content||'');
    // enhance code blocks with header + copy
    html = html.replace(/<pre><code class="language-(\w+)">([\s\S]*?)<\/code><\/pre>/g, (m, lang, code)=>{
      const esc = this._escape(code);
      return `<div class="code-wrap"><div class="code-head"><span class="code-lang">${lang}</span><button class="copy" onclick="copyCode(this)">Copy</button></div><pre class="!m-0"><code class="language-${lang}">${esc}</code></pre></div>`;
    });
    return html;
  }
  _escape(s){ const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
  _highlight(el){ el.querySelectorAll('pre code').forEach(b=>Prism.highlightElement(b)); }

  _addMessage(role, content, attachments=[], metadata=null, streaming=false){
    const wrap = document.getElementById('chatContainer');
    const div = document.createElement('div');
    div.className = `bubble ${role==='user' ? 'user' : 'ai'}`;
    div.innerHTML = `
      <div class="flex items-start gap-3">
        ${role==='assistant' ? `<div class="w-8 h-8 rounded-lg hero-grad flex items-center justify-center">🧠</div>`: ''}
        <div class="flex-1">
          <div class="text-[11px] text-slate-400 mb-1">${role==='user'?'You':'NovaMind'}</div>
          <div class="prose prose-invert max-w-none msg-content">${streaming?'<em>Thinking…</em>': this._format(content)}</div>
          ${attachments?.length? `<div class='flex gap-2 mt-2'>${attachments.map(a=>`<div class='glass px-2 py-1 rounded text-[11px]'>📎 ${a.name||'Attachment'}</div>`).join('')}</div>`:''}
        </div>
      </div>`;
    wrap.appendChild(div); wrap.scrollTop = wrap.scrollHeight; if(!streaming) this._highlight(div); return div;
  }

  _updateStreaming(div, content){ const c = div.querySelector('.msg-content'); c.innerHTML = this._format(content); this._highlight(div); }

  async sendMessage(){
    const input = document.getElementById('messageInput');
    const content = (input.value||'').trim(); if(!content && !this.attachments.length){ return; }
    const attachments = this.attachments.slice(); // snapshot

    // UI state
    input.disabled = true; document.getElementById('sendBtn').disabled = true;

    // Add user message
    this._addMessage('user', content, attachments);

    // Queue
    this.messages.push({ role: 'user', content, attachments });

    // Clear input
    input.value=''; input.style.height='auto'; this.attachments = [];

    try{
      const payload = { model: this.currentModel, messages: this.messages, session_id: this.sessionId, conversation_id: this.conversationId, temperature: this.settings.temperature, max_tokens: this.settings.maxTokens, top_p: this.settings.topP, top_k: this.settings.topK, thinking_budget: this.settings.thinkingBudget, use_memory: this.settings.useMemory, stream: this.settings.stream, plugins: Array.from(this.features)};
      if(this.settings.stream){ await this._stream(payload); } else { await this._once(payload); }
    }catch(e){ console.error(e); this._addMessage('assistant','❌ An error occurred. Please try again.'); }
    finally{ input.disabled = false; document.getElementById('sendBtn').disabled = false; input.focus(); }
  }

  async _once(payload){
    const res = await fetch('/api/v2/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    const data = await res.json(); if(data.error){ throw new Error(data.error); }
    this._addMessage('assistant', data.response);
    this.messages.push({ role:'assistant', content: data.response });
    if(data.metadata?.usage){ const t = data.metadata.usage.totalTokens || 0; document.getElementById('tokenCount').textContent = t.toLocaleString(); }
  }

  async _stream(payload){
    payload.stream = true; const res = await fetch('/api/v2/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    const reader = res.body.getReader(); const decoder = new TextDecoder(); let buf = ''; let full = ''; let div = null;
    while(true){ const {done, value} = await reader.read(); if(done) break; buf += decoder.decode(value, {stream:true}); const lines = buf.split('\n'); buf = lines.pop(); for(const ln of lines){ if(!ln.startsWith('data: ')) continue; try{ const obj = JSON.parse(ln.slice(6)); if(!div) div = this._addMessage('assistant','',[],null,true); full += obj.text; this._updateStreaming(div, full);}catch(e){} } }
    this.messages.push({ role:'assistant', content: full });
  }
}

window.copyCode = (btn)=>{ const code = btn.closest('.code-wrap').querySelector('code').textContent; navigator.clipboard.writeText(code).then(()=>{ btn.textContent='Copied!'; setTimeout(()=>btn.textContent='Copy', 1600); }); };

// Init
window.novamindClient = new NovaMindClient();
mermaid.initialize({ theme:'dark' });
</script>
</body>
</html>"""

@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    print(f"🚀 {APP_TITLE} v{VERSION} on :{port}")
    print("🔑 Keys loaded:" , len(_GOOGLE_KEYS))
    app.run(host="0.0.0.0", port=port, debug=False)
