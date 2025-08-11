import os, base64, json, mimetypes, time, re, tempfile
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from flask import Flask, request, send_from_directory, make_response, jsonify, Response

# ---------------------------
# Config
# ---------------------------
APP_TITLE = "All-in-One AI Chat — OpenAI • Claude • Gemini"
UPLOAD_DIR = os.environ.get("UPLOAD_DIR") or os.path.join(tempfile.gettempdir(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# API keys
OPENAI_KEY_SERVER = os.environ.get("OPENAI_API_KEY_SERVER")
ANTHROPIC_KEY_SERVER = os.environ.get("ANTHROPIC_API_KEY_SERVER")
GEMINI_KEYS = [k for k in [os.environ.get("GEMINI_KEY_1"), os.environ.get("GEMINI_KEY_2")] if k]

# Optional Google Custom Search
GOOGLE_SEARCH_KEY = os.environ.get("GOOGLE_SEARCH_KEY")
GOOGLE_SEARCH_CX  = os.environ.get("GOOGLE_SEARCH_CX")

# Model lists are configurable via env; otherwise use solid defaults
def split_models(s, fallback):
    if s and s.strip():
        return [x.strip() for x in s.split(",") if x.strip()]
    return fallback

DEFAULT_OPENAI_MODELS = [
    # keep these broad & stable; override via OPENAI_MODELS for bleeding-edge
    "o3", "o4-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4o"
]
DEFAULT_ANTHROPIC_MODELS = [
    "claude-3.7-sonnet", "claude-3.7-haiku", "claude-3-opus"
]
DEFAULT_GOOGLE_MODELS = [
    "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-flash-8b"
]

OPENAI_MODELS    = split_models(os.environ.get("OPENAI_MODELS"),    DEFAULT_OPENAI_MODELS)
ANTHROPIC_MODELS = split_models(os.environ.get("ANTHROPIC_MODELS"), DEFAULT_ANTHROPIC_MODELS)
GOOGLE_MODELS    = split_models(os.environ.get("GOOGLE_MODELS"),    DEFAULT_GOOGLE_MODELS)

# ---------------------------
# Flask
# ---------------------------
app = Flask(__name__, static_folder=None)

def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp

@app.after_request
def _after(resp):
    return cors(resp)

@app.route("/health")
def health():
    return "ok"

# ---------------------------
# Helpers
# ---------------------------
def http_json(method, url, headers=None, data=None, timeout=60):
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = Request(url, data=body, method=method.upper())
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8")), None
    except HTTPError as e:
        try:
            err = e.read().decode("utf-8")
        except Exception:
            err = str(e)
        return None, {"status": e.code, "error": err}
    except URLError as e:
        return None, {"status": 0, "error": str(e)}

def fetch_bytes(url, max_mb=20):
    with urlopen(url) as r:
        b = r.read()
        if len(b) > max_mb * 1024 * 1024:
            raise ValueError("File too large")
        return b

def guess_mime(name_or_bytes):
    if isinstance(name_or_bytes, bytes):
        b = name_or_bytes[:16]
        if b.startswith(b"\x89PNG"): return "image/png"
        if b.startswith(b"\xff\xd8"): return "image/jpeg"
        if b[:4] == b"GIF8": return "image/gif"
        if b[:4] == b"%PDF": return "application/pdf"
    mt, _ = mimetypes.guess_type(str(name_or_bytes))
    return mt or "application/octet-stream"

def google_search_top(q, n=3):
    if not (GOOGLE_SEARCH_KEY and GOOGLE_SEARCH_CX):
        return []
    base = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_SEARCH_KEY, "cx": GOOGLE_SEARCH_CX, "q": q}
    url = f"{base}?{urlencode(params)}"
    try:
        with urlopen(url, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
            items = data.get("items", [])[:n]
            out = []
            for it in items:
                out.append({
                    "title": it.get("title"),
                    "snippet": it.get("snippet"),
                    "link": it.get("link")
                })
            return out
    except Exception:
        return []

def build_search_context(results):
    if not results: return ""
    lines = ["[Search context]\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n{r['snippet']}\n{r['link']}\n")
    return "\n".join(lines)

def normalize_messages_for_last_user_images(messages, attachments):
    msgs = list(messages)
    if attachments:
        for i in range(len(msgs)-1, -1, -1):
            if msgs[i].get("role") == "user":
                extras = msgs[i].setdefault("attachments", [])
                extras.extend(attachments)
                break
    return msgs

# ---------------------------
# Providers
# ---------------------------
def openai_chat(model, system_prompt, messages, key):
    """Try Responses API first, fall back to Chat Completions."""
    api = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {key}"}

    def to_parts(turn):
        parts = []
        text = turn.get("content") or ""
        if text:
            parts.append({"type": "input_text", "text": text})
        for att in turn.get("attachments", []):
            url = att.get("url")
            mime = att.get("mime") or guess_mime(url)
            if url and mime and str(mime).startswith("image/"):
                parts.append({"type": "input_image", "image_url": url})
        return parts

    input_list = []
    if system_prompt and system_prompt.strip():
        input_list.append({"role": "system", "content":[{"type":"text","text": system_prompt}]})
    for m in messages:
        role = m.get("role","user")
        if role == "user":
            input_list.append({"role":"user","content": to_parts(m)})
        elif role == "assistant":
            input_list.append({"role":"assistant","content":[{"type":"output_text","text": m.get("content","")}]})
        else:
            input_list.append({"role":"user","content":[{"type":"input_text","text": m.get("content","")}]})

    payload = {"model": model, "input": input_list, "max_output_tokens": 1024}
    data, err = http_json("POST", api, headers, payload, timeout=120)
    if not err and data:
        if "output_text" in data and data["output_text"]:
            return data["output_text"], {"provider":"openai","model":model}, None
        # Some responses use "output" blocks
        try:
            blocks = data.get("output", []) or data.get("response", {}).get("output", [])
            texts = []
            for b in blocks:
                for c in b.get("content", []):
                    if c.get("type") in ("output_text","text"):
                        texts.append(c.get("text",""))
            if texts:
                return "\n".join(texts), {"provider":"openai","model":model}, None
        except Exception:
            pass

    # Fallback
    api2 = "https://api.openai.com/v1/chat/completions"
    msg_list = []
    if system_prompt and system_prompt.strip():
        msg_list.append({"role":"system","content":system_prompt})
    for m in messages:
        role = m.get("role","user")
        content_text = m.get("content","")
        parts = []
        if content_text:
            parts.append({"type":"text","text":content_text})
        for att in m.get("attachments", []):
            url = att.get("url")
            mime = att.get("mime") or guess_mime(url)
            if url and mime and str(mime).startswith("image/"):
                parts.append({"type":"image_url","image_url":{"url": url}})
        msg_list.append({"role": role, "content": parts if parts else content_text})

    payload2 = {"model": model, "messages": msg_list, "temperature": 0.2}
    data2, err2 = http_json("POST", api2, headers, payload2, timeout=120)
    if err2:
        return None, None, err2
    try:
        txt = data2["choices"][0]["message"]["content"]
    except Exception:
        txt = json.dumps(data2)
    return txt, {"provider":"openai","model":model}, None

def anthropic_chat(model, system_prompt, messages, key):
    api = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}

    def to_content(turn):
        parts = []
        txt = turn.get("content") or ""
        if txt:
            parts.append({"type":"text","text": txt})
        for att in turn.get("attachments", []):
            url = att.get("url")
            if not url: 
                continue
            try:
                b = fetch_bytes(url)
            except Exception:
                continue
            mime = att.get("mime") or guess_mime(url) or "image/png"
            parts.append({
                "type":"image",
                "source":{
                    "type":"base64",
                    "media_type": mime,
                    "data": base64.b64encode(b).decode("utf-8")
                }
            })
        return parts

    msgs = []
    sys = system_prompt.strip() if system_prompt and system_prompt.strip() else None
    for m in messages:
        if m.get("role") == "assistant":
            continue
        msgs.append({"role":"user", "content": to_content(m)})

    payload = {"model": model, "max_tokens": 1024, "messages": msgs}
    if sys: 
        payload["system"] = sys

    data, err = http_json("POST", api, headers, payload, timeout=120)
    if err: 
        return None, None, err
    try:
        txts = []
        for c in data.get("content", []):
            if c.get("type") == "text":
                txts.append(c.get("text",""))
        return "\n".join(txts), {"provider":"anthropic","model":model}, None
    except Exception:
        return json.dumps(data), {"provider":"anthropic","model":model}, None

def gemini_chat(model, system_prompt, messages, key):
    base = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

    def to_parts(turn):
        parts = []
        txt = turn.get("content") or ""
        if txt:
            parts.append({"text": txt})
        for att in turn.get("attachments", []):
            url = att.get("url")
            if not url: 
                continue
            try:
                b = fetch_bytes(url)
            except Exception:
                continue
            mime = att.get("mime") or guess_mime(url) or "image/png"
            parts.append({"inlineData":{"mimeType": mime, "data": base64.b64encode(b).decode("utf-8")}})
        return parts

    contents = []
    if system_prompt and system_prompt.strip():
        contents.append({"role":"user","parts":[{"text": f"[System]\n{system_prompt}"}]})
    for m in messages:
        role = m.get("role","user")
        contents.append({"role": "user" if role!="assistant" else "model", "parts": to_parts(m)})

    payload = {"contents": contents, "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024}}
    data, err = http_json("POST", base, {}, payload, timeout=120)
    if err: 
        return None, None, err
    try:
        parts = data["candidates"][0]["content"]["parts"]
        texts = []
        for p in parts:
            if "text" in p:
                texts.append(p["text"])
        return "\n".join(texts), {"provider":"gemini","model":model}, None
    except Exception:
        return json.dumps(data), {"provider":"gemini","model":model}, None

# Gemini key round-robin
_gemini_rr = 0
def pick_gemini_key():
    global _gemini_rr
    if GEMINI_KEYS:
        key = GEMINI_KEYS[_gemini_rr % len(GEMINI_KEYS)]
        _gemini_rr += 1
        return key
    return None

# ---------------------------
# API Routes
# ---------------------------
@app.route("/api/models")
def api_models():
    """Return provider -> models map and a friendly label for UI."""
    out = {
        "openai":    [{"id": m, "label": m} for m in OPENAI_MODELS],
        "anthropic": [{"id": m, "label": m} for m in ANTHROPIC_MODELS],
        "google":    [{"id": m, "label": m} for m in GOOGLE_MODELS],
    }
    return jsonify(out)

@app.route("/api/upload", methods=["POST", "OPTIONS"])
def upload():
    if request.method == "OPTIONS":
        return cors(make_response(("", 204)))
    files = request.files.getlist("files")
    saved = []
    for f in files:
        name = f.filename
        ext = os.path.splitext(name)[1].lower()
        ts = int(time.time()*1000)
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", name) or f"file{ts}{ext}"
        path = os.path.join(UPLOAD_DIR, f"{ts}_{safe}")
        f.save(path)
        url = f"/uploads/{os.path.basename(path)}"
        saved.append({"name": name, "url": url, "mime": guess_mime(name), "size": os.path.getsize(path)})
    return jsonify({"files": saved})

@app.route("/uploads/<path:fname>")
def serve_upload(fname):
    return send_from_directory(UPLOAD_DIR, fname, as_attachment=False)

@app.route("/api/chat", methods=["POST", "OPTIONS"])
def api_chat():
    if request.method == "OPTIONS":
        return cors(make_response(("", 204)))
    data = request.get_json(force=True, silent=True) or {}
    provider = (data.get("provider") or "openai").lower()
    model = data.get("model") or ""
    system_prompt = data.get("system_prompt") or ""
    messages = data.get("messages") or []
    attachments = data.get("attachments") or []
    use_search = bool(data.get("use_search"))

    messages = normalize_messages_for_last_user_images(messages, attachments)

    # Optional: prepend Google search context from the last user turn
    if use_search and GOOGLE_SEARCH_KEY and GOOGLE_SEARCH_CX:
        last_user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_text = m.get("content") or ""
                break
        if last_user_text:
            results = google_search_top(last_user_text, n=3)
            context = build_search_context(results)
            if context:
                system_prompt = (system_prompt + "\n\n" + context).strip()

    try:
        if provider == "openai":
            key = OPENAI_KEY_SERVER
            if not key:
                return jsonify({"error":"OpenAI API key not configured on server"}), 400
            out, meta, err = openai_chat(model, system_prompt, messages, key)
        elif provider == "anthropic":
            key = ANTHROPIC_KEY_SERVER
            if not key:
                return jsonify({"error":"Anthropic API key not configured on server"}), 400
            out, meta, err = anthropic_chat(model, system_prompt, messages, key)
        elif provider == "google":
            key = pick_gemini_key()
            if not key:
                return jsonify({"error":"Gemini API key not configured on server"}), 400
            out, meta, err = gemini_chat(model, system_prompt, messages, key)
        else:
            return jsonify({"error":"Unknown provider"}), 400

        if err:
            # Try alternate Gemini key if available
            if provider == "google" and len(GEMINI_KEYS) > 1:
                alt = GEMINI_KEYS[1] if GEMINI_KEYS[0] == key else GEMINI_KEYS[0]
                out2, meta2, err2 = gemini_chat(model, system_prompt, messages, alt)
                if not err2:
                    return jsonify({"output": out2, "meta": meta2})
            return jsonify({"error": err}), 502

        return jsonify({"output": out, "meta": meta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json(force=True, silent=True) or {}
    q = (data.get("q","") or "").strip()
    if not q:
        return jsonify({"results":[]})
    res = google_search_top(q, n=5)
    return jsonify({"results": res})

# ---------------------------
# HTML (Mobile-first, polished)
# ---------------------------
HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"/>
<title>{{APP_TITLE}}</title>
<link rel="icon" href="data:,">
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --primary: #6366f1;
    --surface: #0b1220;
    --card: #0f172a;
    --muted: #94a3b8;
  }
  * { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  body { background: radial-gradient(1200px 800px at 10% 10%, rgba(99,102,241,.12), transparent 40%), #0b1220; color: #e5e7eb; }
  .glass { background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(16px); border: 1px solid rgba(255,255,255,0.06); }
  .btn { transition: .2s ease; }
  .btn-primary { background: linear-gradient(135deg, #6366f1, #7c3aed); }
  .btn-primary:hover { filter: brightness(1.05); transform: translateY(-1px); }
  .btn-ghost { background: transparent; border: 1px solid rgba(255,255,255,0.1); }
  .btn-ghost:hover { background: rgba(255,255,255,0.06); }
  .bubble { border: 1px solid rgba(255,255,255,0.08); }
  .bubble-user { background: linear-gradient(135deg, rgba(99,102,241,.18), rgba(124,58,237,.18)); border-radius: 18px 18px 4px 18px; }
  .bubble-ai { background: rgba(255,255,255,0.04); border-radius: 18px 18px 18px 4px; }
  .model-badge { font-size: 12px; padding: 4px 8px; border: 1px solid rgba(255,255,255,0.1); border-radius: 9999px; color: #c7d2fe; }
  .input { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); }
  .markdown { color: #e5e7eb; }
  .markdown pre { background: rgba(0,0,0,0.5); padding: 12px; border-radius: 12px; overflow: auto; }
  .markdown code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
  .chip { background: rgba(99,102,241,.15); border: 1px solid rgba(99,102,241,.35); }
  @media (min-width:1024px){
    #sidebar { position: sticky; top: 20px; height: calc(100vh - 40px); }
  }
</style>
</head>
<body>
  <!-- Top Bar (mobile-first) -->
  <header class="sticky top-0 z-40 glass">
    <div class="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600"></div>
        <div class="text-sm lg:text-base font-semibold">{{APP_TITLE}}</div>
      </div>
      <div class="hidden sm:flex items-center gap-2">
        <span id="status" class="text-xs text-indigo-300/80">Ready</span>
      </div>
      <button id="menuBtn" class="sm:hidden btn btn-ghost px-3 py-2 rounded-lg">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7h16M4 12h16M4 17h16"/></svg>
      </button>
    </div>
  </header>

  <div class="max-w-7xl mx-auto px-3 sm:px-4 py-4 grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-6">
    <!-- Sidebar -->
    <aside id="sidebar" class="lg:col-span-4 xl:col-span-3 space-y-4 sm:space-y-6 glass rounded-2xl p-4 sm:p-6 hidden sm:block">
      <div class="space-y-4">
        <div class="flex items-center justify-between">
          <div class="text-base sm:text-lg font-semibold">Settings</div>
          <button id="newChat" class="btn btn-ghost px-3 py-2 rounded-lg text-sm">New Chat</button>
        </div>

        <div>
          <label class="text-xs text-slate-400">Provider</label>
          <div class="grid grid-cols-3 mt-2 gap-2">
            <button data-provider="openai" class="provider btn btn-ghost rounded-lg py-2">OpenAI</button>
            <button data-provider="anthropic" class="provider btn btn-ghost rounded-lg py-2">Claude</button>
            <button data-provider="google" class="provider btn btn-ghost rounded-lg py-2">Gemini</button>
          </div>
        </div>

        <div>
          <label class="text-xs text-slate-400">Model</label>
          <select id="model" class="input w-full mt-2 px-3 py-2 rounded-lg"></select>
        </div>

        <div class="flex items-center justify-between">
          <label class="flex items-center gap-2 text-sm">
            <input id="toggleSearch" type="checkbox" class="w-4 h-4">
            Use Google Search
          </label>
          <span class="model-badge" id="providerTag">openai</span>
        </div>

        <div>
          <label class="text-xs text-slate-400">System Prompt</label>
          <textarea id="systemPrompt" rows="5" class="input w-full mt-2 px-3 py-2 rounded-lg" placeholder="Set custom behavior..."></textarea>
          <div class="mt-2 flex items-center gap-2">
            <button id="resetSystem" class="btn btn-ghost px-3 py-2 rounded-lg text-sm">Reset</button>
          </div>
        </div>

        <div>
          <div class="text-sm font-semibold mb-2">Conversations</div>
          <div id="chatList" class="space-y-2 max-h-[35vh] overflow-y-auto pr-1"></div>
          <div class="mt-3 flex gap-2">
            <button id="exportChats" class="btn btn-ghost px-3 py-2 rounded-lg text-sm">Export</button>
            <label class="btn btn-ghost px-3 py-2 rounded-lg text-sm cursor-pointer">
              Import<input type="file" id="importFile" class="hidden" accept=".json"/>
            </label>
            <button id="clearChats" class="btn btn-ghost px-3 py-2 rounded-lg text-sm text-red-300">Clear</button>
          </div>
        </div>
      </div>
    </aside>

    <!-- Chat -->
    <main class="lg:col-span-8 xl:col-span-9">
      <div id="chat" class="glass rounded-2xl p-3 sm:p-4 md:p-6 h-[70vh] sm:h-[72vh] overflow-y-auto"></div>

      <!-- Composer -->
      <div class="mt-3 sm:mt-4 glass rounded-2xl p-3 sm:p-4">
        <div id="dropzone" class="border-2 border-dashed border-white/10 rounded-xl p-3 sm:p-4 text-center text-sm text-slate-400">
          Drag & drop files here or
          <label class="underline cursor-pointer">browse<input id="fileInput" type="file" class="hidden" multiple accept="image/*,.pdf,.txt,.md,.doc,.docx,.csv,.json,.xml"/></label>
        </div>
        <div id="fileChips" class="flex gap-2 flex-wrap mt-2"></div>
        <div class="flex items-end gap-2 sm:gap-3 mt-3">
          <textarea id="prompt" rows="3" class="input flex-1 px-3 py-3 rounded-xl" placeholder="Message the AI..."></textarea>
          <button id="sendBtn" class="btn btn-primary px-4 sm:px-5 py-3 rounded-xl font-medium">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/></svg>
          </button>
        </div>
        <div class="flex items-center justify-between mt-2">
          <span id="statusMobile" class="sm:hidden text-xs text-indigo-300/80">Ready</span>
          <span class="hidden sm:inline text-xs text-slate-500">Enter to send • Shift+Enter newline</span>
        </div>
      </div>
    </main>
  </div>

<script>
const DEFAULT_SYSTEM = `You are a helpful, precise, and friendly assistant. Answer clearly, format with markdown when useful, and ask for missing details only when necessary.`;

const $ = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));
const chatEl = $("#chat");
const modelEl = $("#model");
const toggleSearchEl = $("#toggleSearch");
const systemPromptEl = $("#systemPrompt");
const resetSystemEl = $("#resetSystem");
const promptEl = $("#prompt");
const sendBtn = $("#sendBtn");
const fileInput = $("#fileInput");
const fileChips = $("#fileChips");
const chatListEl = $("#chatList");
const newChatEl = $("#newChat");
const clearChatsEl = $("#clearChats");
const exportChatsEl = $("#exportChats");
const importFileEl = $("#importFile");
const statusEl = $("#status");
const statusMobileEl = $("#statusMobile");
const menuBtn = $("#menuBtn");
const sidebar = $("#sidebar");
const providerTag = $("#providerTag");

let filesToSend = [];
let currentProvider = "openai";
let MODELS = {openai:[], anthropic:[], google:[]};
let state = {
  chats: [],
  activeId: null,
  useSearch: JSON.parse(localStorage.getItem("use_search") || "false"),
  systemPrompt: localStorage.getItem("system_prompt") || DEFAULT_SYSTEM
};

function uid(){ return Math.random().toString(36).slice(2) + Date.now().toString(36); }
function saveState(){
  localStorage.setItem("chats_v4", JSON.stringify(state.chats));
  localStorage.setItem("active_chat", state.activeId || "");
  localStorage.setItem("use_search", JSON.stringify(state.useSearch));
  localStorage.setItem("system_prompt", state.systemPrompt);
}
function loadState(){
  try { state.chats = JSON.parse(localStorage.getItem("chats_v4") || "[]"); } catch { state.chats=[]; }
  state.activeId = localStorage.getItem("active_chat") || (state.chats[0]?.id || null);
  renderChatList();
  if (!state.activeId) newChat();
  else renderActive();
}

function setStatus(msg, type="normal"){
  const el1 = statusEl, el2 = statusMobileEl;
  [el1, el2].forEach(el=>{
    if(!el) return;
    el.textContent = msg;
    el.className = type==="loading" ? "text-xs text-yellow-300/90" :
                   type==="error"   ? "text-xs text-red-300/90" :
                                      "text-xs text-indigo-300/80";
  });
  if (type !== "loading"){
    setTimeout(()=> setStatus("Ready","normal"), 2500);
  }
}

async function fetchModels(){
  try{
    const res = await fetch("/api/models");
    MODELS = await res.json();
  }catch(e){
    MODELS = {
      openai:    [{id:"o3",label:"o3"}, {id:"gpt-4.1",label:"gpt-4.1"}],
      anthropic: [{id:"claude-3.7-sonnet",label:"claude-3.7-sonnet"}],
      google:    [{id:"gemini-1.5-pro",label:"gemini-1.5-pro"}]
    };
  }
}

function fillModelSelect(provider){
  modelEl.innerHTML = "";
  (MODELS[provider]||[]).forEach(m=>{
    const opt = document.createElement("option");
    opt.value = m.id; opt.textContent = m.label || m.id;
    modelEl.appendChild(opt);
  });
}

function newChat(){
  const id = uid();
  const model = modelEl.value || (MODELS[currentProvider]?.[0]?.id || "");
  const chat = { id, title: "New conversation", provider: currentProvider, model, messages: [] };
  state.chats.unshift(chat);
  state.activeId = id;
  saveState(); renderChatList(); renderActive();
}

function renderChatList(){
  chatListEl.innerHTML = "";
  state.chats.forEach(ch => {
    const div = document.createElement("div");
    div.className = "flex items-center justify-between p-2 rounded-xl hover:bg-white/5 cursor-pointer";
    if (ch.id === state.activeId) div.classList.add("bg-white/10");
    div.innerHTML = `
      <div class="flex-1 truncate text-sm">${ch.title}</div>
      <div class="flex gap-1">
        <button class="p-1.5 hover:bg-white/10 rounded-lg rename" title="Rename">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"/></svg>
        </button>
        <button class="p-1.5 hover:bg-white/10 rounded-lg del text-red-300" title="Delete">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
        </button>
      </div>`;
    div.onclick = (e) => { 
      if (e.target.closest(".rename")||e.target.closest(".del")) return; 
      state.activeId = ch.id; saveState(); renderActive();
    };
    div.querySelector(".rename").onclick = (e) => {
      e.stopPropagation();
      const t = prompt("Rename conversation", ch.title);
      if (t) { ch.title = t; saveState(); renderChatList(); }
    };
    div.querySelector(".del").onclick = (e) => {
      e.stopPropagation();
      if (!confirm("Delete this conversation?")) return;
      state.chats = state.chats.filter(c => c.id !== ch.id);
      if (state.activeId === ch.id) state.activeId = state.chats[0]?.id || null;
      saveState(); renderChatList(); renderActive();
    };
    chatListEl.appendChild(div);
  });
}

function renderActive(){
  const ch = state.chats.find(c => c.id === state.activeId);
  if (!ch) return;
  currentProvider = ch.provider || "openai";
  providerTag.textContent = currentProvider;
  fillModelSelect(currentProvider);
  modelEl.value = ch.model || modelEl.value;
  chatEl.innerHTML = "";
  ch.messages.forEach(m => addBubble(m.role, m.content, m.attachments || []));
  chatEl.scrollTop = chatEl.scrollHeight;
}

function setProvidersUI(){
  $$(".provider").forEach(btn=>{
    btn.classList.toggle("bg-white/10", btn.dataset.provider===currentProvider);
    btn.onclick = ()=>{
      currentProvider = btn.dataset.provider;
      const ch = state.chats.find(c => c.id === state.activeId);
      if (ch){ ch.provider = currentProvider; ch.model = ""; saveState(); }
      providerTag.textContent = currentProvider;
      fillModelSelect(currentProvider);
      const first = MODELS[currentProvider]?.[0]?.id || "";
      modelEl.value = first;
    };
  });
}

function addBubble(role, text, atts=[]){
  const wrap = document.createElement("div");
  wrap.className = "mb-3 flex " + (role==="user" ? "justify-end" : "justify-start");

  const bubble = document.createElement("div");
  bubble.className = "bubble " + (role==="user" ? "bubble-user" : "bubble-ai") + " rounded-2xl p-3 sm:p-4 max-w-[90%] sm:max-w-[80%]";
  
  const who = document.createElement("div");
  who.className = "text-[10px] uppercase tracking-wide text-slate-400 mb-1";
  who.textContent = role === "user" ? "You" : "Assistant";
  bubble.appendChild(who);

  if (text) {
    const textDiv = document.createElement("div");
    textDiv.className = "markdown text-sm leading-relaxed";
    textDiv.innerHTML = marked.parse(text);
    bubble.appendChild(textDiv);
  }
  
  if (atts && atts.length){
    const grid = document.createElement("div");
    grid.className = "grid grid-cols-2 gap-2 mt-3";
    atts.forEach(a => {
      if ((a.mime||"").startsWith("image/")){
        const img = document.createElement("img");
        img.src = a.url;
        img.alt = a.name || "image";
        img.className = "rounded-lg max-h-48 object-cover cursor-pointer hover:opacity-90 transition-opacity";
        img.onclick = () => window.open(a.url, "_blank");
        grid.appendChild(img);
      } else {
        const link = document.createElement("a");
        link.href = a.url;
        link.textContent = a.name || a.url;
        link.target = "_blank";
        link.className = "chip rounded-lg px-3 py-1.5 text-xs hover:bg-white/10 transition-colors";
        grid.appendChild(link);
      }
    });
    bubble.appendChild(grid);
  }
  wrap.appendChild(bubble);
  chatEl.appendChild(wrap);
}

function scrollToBottom(){
  chatEl.scrollTo({ top: chatEl.scrollHeight, behavior: "smooth" });
}

promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendBtn.click(); }
});

// Mobile menu
menuBtn?.addEventListener("click", ()=>{
  const visible = sidebar.classList.contains("block");
  sidebar.classList.toggle("block", !visible);
  sidebar.classList.toggle("hidden", visible);
});

// Toggle search & system prompt persistence
toggleSearchEl.checked = state.useSearch;
toggleSearchEl.addEventListener("change", () => { state.useSearch = toggleSearchEl.checked; saveState(); });
systemPromptEl.value = state.systemPrompt;
systemPromptEl.addEventListener("input", () => { state.systemPrompt = systemPromptEl.value; saveState(); });
resetSystemEl.onclick = () => { systemPromptEl.value = DEFAULT_SYSTEM; state.systemPrompt = DEFAULT_SYSTEM; saveState(); };

// Chats
newChatEl?.addEventListener("click", newChat);
clearChatsEl?.addEventListener("click", ()=>{
  if (!confirm("Delete all conversations?")) return;
  state.chats = []; state.activeId = null; saveState(); newChat();
});
exportChatsEl?.addEventListener("click", ()=>{
  const blob = new Blob([JSON.stringify(state.chats, null, 2)], {type:"application/json"});
  const u = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = u; a.download = "ai-chats.json"; a.click(); URL.revokeObjectURL(u);
});
importFileEl?.addEventListener("change", (e)=>{
  const f = e.target.files[0]; if(!f) return;
  const reader = new FileReader();
  reader.onload = ()=>{
    try {
      const arr = JSON.parse(reader.result);
      if (Array.isArray(arr)) { state.chats = arr.concat(state.chats); saveState(); renderChatList(); renderActive(); setStatus("Conversations imported","normal"); }
      else setStatus("Invalid file format","error");
    } catch { setStatus("Invalid JSON file","error"); }
  };
  reader.readAsText(f);
});

// File uploads (click + drag/drop)
fileInput.addEventListener("change", async ()=>{
  if (!fileInput.files.length) return;
  await uploadFiles(fileInput.files);
  fileInput.value = "";
});

const dropzone = document.getElementById("dropzone");
dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("ring-2","ring-indigo-400/50"); });
dropzone.addEventListener("dragleave", e => { dropzone.classList.remove("ring-2","ring-indigo-400/50"); });
dropzone.addEventListener("drop", async e => {
  e.preventDefault();
  dropzone.classList.remove("ring-2","ring-indigo-400/50");
  if (e.dataTransfer.files?.length) await uploadFiles(e.dataTransfer.files);
});

async function uploadFiles(fileList){
  const fd = new FormData();
  for (const f of fileList) fd.append("files", f);
  setStatus("Uploading files...","loading");
  try {
    const res = await fetch("/api/upload", { method:"POST", body: fd });
    const data = await res.json();
    (data.files||[]).forEach(f => {
      filesToSend.push(f);
      const chip = document.createElement("div");
      chip.className = "chip rounded-lg px-3 py-1.5 flex items-center gap-2 text-xs";
      chip.innerHTML = `<span class="truncate max-w-[150px]">${f.name || f.url}</span><button class="opacity-70 hover:opacity-100">×</button>`;
      chip.querySelector("button").onclick = () => {
        filesToSend = filesToSend.filter(x => x.url !== f.url);
        chip.remove();
      };
      fileChips.appendChild(chip);
    });
    setStatus("Files uploaded","normal");
  } catch(e) {
    console.error(e); setStatus("Upload failed","error");
  }
}

sendBtn.addEventListener("click", async ()=>{
  const txt = promptEl.value.trim();
  const ch = state.chats.find(c => c.id === state.activeId);
  if (!ch) return;
  if (!txt && filesToSend.length===0) return;

  const model = modelEl.value;
  ch.provider = currentProvider;
  ch.model = model;
  state.systemPrompt = systemPromptEl.value;
  saveState();

  const atts = filesToSend.slice();
  ch.messages.push({role:"user", content: txt, attachments: atts});
  if (ch.messages.length === 1 && txt) {
    ch.title = txt.slice(0, 60) + (txt.length > 60 ? "..." : "");
    renderChatList();
  }
  filesToSend = []; fileChips.innerHTML = ""; promptEl.value = "";
  addBubble("user", txt, atts); scrollToBottom();

  setStatus("Thinking...","loading");
  try {
    const res = await fetch("/api/chat", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({
        provider: currentProvider,
        model,
        system_prompt: state.systemPrompt,
        messages: ch.messages,
        attachments: [],
        use_search: toggleSearchEl.checked
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(typeof data.error==="string" ? data.error : (data.error?.error || "Request failed"));
    const out = (data.output || "").trim();
    ch.messages.push({role:"assistant", content: out});
    saveState();
    addBubble("assistant", out);
    scrollToBottom();
    setStatus("Ready","normal");
  } catch(err) {
    console.error(err);
    addBubble("assistant", "⚠️ " + (err.message || "Something went wrong. Please try again."));
    setStatus("Error","error");
  }
});

// init
(async ()=>{
  await fetchModels();
  fillModelSelect("openai");
  setProvidersUI();
  loadState();
})();
</script>
</body>
</html>"""

@app.route("/")
def index():
    html = HTML_TEMPLATE.replace("{{APP_TITLE}}", APP_TITLE)
    return Response(html, mimetype="text/html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)