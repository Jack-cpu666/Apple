import os, io, base64, json, mimetypes, time, re, tempfile
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from flask import Flask, request, send_from_directory, make_response, jsonify, Response

APP_TITLE = "All-in-One AI Chat (OpenAI • Claude • Gemini)"
UPLOAD_DIR = os.environ.get("UPLOAD_DIR")
if not UPLOAD_DIR:
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), “uploads”)

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(**name**, static_folder=None)

def cors(resp):
resp.headers[“Access-Control-Allow-Origin”] = “*”
resp.headers[“Access-Control-Allow-Headers”] = “Content-Type, Authorization”
resp.headers[“Access-Control-Allow-Methods”] = “GET, POST, OPTIONS”
return resp

@app.after_request
def _after(resp):
return cors(resp)

@app.route(”/health”)
def health():
return “ok”

def read_env(name, default=None):
v = os.environ.get(name, default)
return v if (v is not None and v.strip() != “”) else None

OPENAI_KEY_SERVER = read_env(“OPENAI_API_KEY_SERVER”)
ANTHROPIC_KEY_SERVER = read_env(“ANTHROPIC_API_KEY_SERVER”)
GEMINI_KEYS = [k for k in [read_env(“GEMINI_KEY_1”), read_env(“GEMINI_KEY_2”)] if k]
GOOGLE_SEARCH_KEY = read_env(“GOOGLE_SEARCH_KEY”)
GOOGLE_SEARCH_CX  = read_env(“GOOGLE_SEARCH_CX”)

def http_json(method, url, headers=None, data=None, timeout=60):
body = json.dumps(data).encode(“utf-8”) if data is not None else None
req = Request(url, data=body, method=method.upper())
req.add_header(“Content-Type”, “application/json”)
if headers:
for k, v in headers.items():
req.add_header(k, v)
try:
with urlopen(req, timeout=timeout) as r:
return json.loads(r.read().decode(“utf-8”)), None
except HTTPError as e:
try:
err = e.read().decode(“utf-8”)
except Exception:
err = str(e)
return None, {“status”: e.code, “error”: err}
except URLError as e:
return None, {“status”: 0, “error”: str(e)}

def fetch_bytes(url, max_mb=20):
with urlopen(url) as r:
b = r.read()
if len(b) > max_mb * 1024 * 1024:
raise ValueError(“File too large”)
return b

def guess_mime(name_or_bytes):
if isinstance(name_or_bytes, bytes):
b = name_or_bytes[:16]
if b.startswith(b”\x89PNG”): return “image/png”
if b.startswith(b”\xff\xd8”): return “image/jpeg”
if b[:4] == b”GIF8”: return “image/gif”
if b[:4] == b”%PDF”: return “application/pdf”
mt, _ = mimetypes.guess_type(str(name_or_bytes))
return mt or “application/octet-stream”

def google_search_top(q, n=3):
if not (GOOGLE_SEARCH_KEY and GOOGLE_SEARCH_CX):
return []
base = “https://www.googleapis.com/customsearch/v1”
params = {“key”: GOOGLE_SEARCH_KEY, “cx”: GOOGLE_SEARCH_CX, “q”: q}
url = f”{base}?{urlencode(params)}”
try:
with urlopen(url, timeout=15) as r:
data = json.loads(r.read().decode(“utf-8”))
items = data.get(“items”, [])[:n]
out = []
for it in items:
out.append({
“title”: it.get(“title”),
“snippet”: it.get(“snippet”),
“link”: it.get(“link”)
})
return out
except Exception:
return []

def build_search_context(results):
if not results: return “”
lines = [”[Search context]\n”]
for i, r in enumerate(results, 1):
lines.append(f”{i}. {r[‘title’]}\n{r[‘snippet’]}\n{r[‘link’]}\n”)
return “\n”.join(lines)

_gemini_rr = 0
def pick_gemini_key():
global _gemini_rr
if GEMINI_KEYS:
key = GEMINI_KEYS[_gemini_rr % len(GEMINI_KEYS)]
_gemini_rr += 1
return key
return None

def normalize_messages_for_last_user_images(messages, attachments):
msgs = list(messages)
if attachments:
for i in range(len(msgs)-1, -1, -1):
if msgs[i].get(“role”) == “user”:
extras = msgs[i].setdefault(“attachments”, [])
extras.extend(attachments)
break
return msgs

def openai_chat(model, system_prompt, messages, key):
api = “https://api.openai.com/v1/responses”
headers = {“Authorization”: f”Bearer {key}”}

```
def to_parts(turn):
    parts = []
    text = turn.get("content") or ""
    if text:
        parts.append({"type": "input_text", "text": text})
    for att in turn.get("attachments", []):
        url = att.get("url")
        mime = att.get("mime") or guess_mime(url)
        if url and mime.startswith("image/"):
            parts.append({"type": "input_image", "image_url": url})
    return parts

input_list = []
if system_prompt and system_prompt.strip():
    input_list.append({"role": "system", "content":[{"type":"text","text": system_prompt}]})
for m in messages:
    role = m.get("role","user")
    if role == "user":
        content = to_parts(m)
        input_list.append({"role":"user","content":content})
    elif role == "assistant":
        input_list.append({"role":"assistant","content":[{"type":"output_text","text": m.get("content","")}]})
    else:
        input_list.append({"role":"user","content":[{"type":"input_text","text": m.get("content","")}]})

payload = {"model": model, "input": input_list, "max_output_tokens": 1024}
data, err = http_json("POST", api, headers, payload, timeout=120)
if not err and data:
    if "output_text" in data and data["output_text"]:
        return data["output_text"], {"provider":"openai","model":model}, None
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
        if url and mime.startswith("image/"):
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
```

def anthropic_chat(model, system_prompt, messages, key):
api = “https://api.anthropic.com/v1/messages”
headers = {“x-api-key”: key, “anthropic-version”: “2023-06-01”}

```
def to_content(turn):
    parts = []
    txt = turn.get("content") or ""
    if txt:
        parts.append({"type":"text","text": txt})
    for att in turn.get("attachments", []):
        url = att.get("url")
        if not url: continue
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
if sys: payload["system"] = sys

data, err = http_json("POST", api, headers, payload, timeout=120)
if err: return None, None, err
try:
    txts = []
    for c in data["content"]:
        if c.get("type") == "text":
            txts.append(c.get("text",""))
    return "\n".join(txts), {"provider":"anthropic","model":model}, None
except Exception:
    return json.dumps(data), {"provider":"anthropic","model":model}, None
```

def gemini_chat(model, system_prompt, messages, key):
base = f”https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}”

```
def to_parts(turn):
    parts = []
    txt = turn.get("content") or ""
    if txt:
        parts.append({"text": txt})
    for att in turn.get("attachments", []):
        url = att.get("url")
        if not url: continue
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
if err: return None, None, err
try:
    parts = data["candidates"][0]["content"]["parts"]
    texts = []
    for p in parts:
        if "text" in p:
            texts.append(p["text"])
    return "\n".join(texts), {"provider":"gemini","model":model}, None
except Exception:
    return json.dumps(data), {"provider":"gemini","model":model}, None
```

@app.route(”/api/upload”, methods=[“POST”, “OPTIONS”])
def upload():
if request.method == “OPTIONS”:
return cors(make_response((””, 204)))
files = request.files.getlist(“files”)
saved = []
for f in files:
name = f.filename
ext = os.path.splitext(name)[1].lower()
ts = int(time.time()*1000)
safe = re.sub(r”[^a-zA-Z0-9_.-]”, “*”, name) or f”file{ts}{ext}”
path = os.path.join(UPLOAD_DIR, f”{ts}*{safe}”)
f.save(path)
url = f”/uploads/{os.path.basename(path)}”
saved.append({“name”: name, “url”: url, “mime”: guess_mime(name), “size”: os.path.getsize(path)})
return jsonify({“files”: saved})

@app.route(”/uploads/<path:fname>”)
def serve_upload(fname):
return send_from_directory(UPLOAD_DIR, fname, as_attachment=False)

@app.route(”/api/chat”, methods=[“POST”, “OPTIONS”])
def api_chat():
if request.method == “OPTIONS”:
return cors(make_response((””, 204)))
data = request.get_json(force=True, silent=True) or {}
provider = (data.get(“provider”) or “openai”).lower()
model = data.get(“model”) or “”
system_prompt = data.get(“system_prompt”) or “”
messages = data.get(“messages”) or []
attachments = data.get(“attachments”) or []
use_search = bool(data.get(“use_search”))

```
messages = normalize_messages_for_last_user_images(messages, attachments)

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
        if provider == "google" and len(GEMINI_KEYS) > 1:
            alt = GEMINI_KEYS[1] if GEMINI_KEYS[0] == key else GEMINI_KEYS[0]
            out2, meta2, err2 = gemini_chat(model, system_prompt, messages, alt)
            if not err2:
                return jsonify({"output": out2, "meta": meta2})
        return jsonify({"error": err}), 502

    return jsonify({"output": out, "meta": meta})
except Exception as e:
    return jsonify({"error": str(e)}), 500
```

@app.route(”/api/search”, methods=[“POST”])
def api_search():
data = request.get_json(force=True, silent=True) or {}
q = data.get(“q”,””).strip()
if not q:
return jsonify({“results”:[]})
res = google_search_top(q, n=5)
return jsonify({“results”: res})

HTML_TEMPLATE = “””<!doctype html>

<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"/>
<title>{{APP_TITLE}}</title>
<link rel="icon" href="data:,">
<script src="https://cdn.tailwindcss.com"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  * { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
  :root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --secondary: #ec4899;
    --accent: #8b5cf6;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --dark: #0f172a;
    --darker: #020617;
    --light: #f8fafc;
    --muted: #64748b;
  }
  body { 
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #e2e8f0;
    min-height: 100vh;
  }
  .glass {
    background: rgba(30, 41, 59, 0.5);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.05);
  }
  .glass-dark {
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.05);
  }
  .gradient-bg {
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
  }
  .btn-primary {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    color: white;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px 0 rgba(99, 102, 241, 0.3);
  }
  .btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.4);
  }
  .btn-secondary {
    background: transparent;
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
    transition: all 0.3s ease;
  }
  .btn-secondary:hover {
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(255, 255, 255, 0.2);
  }
  .bubble-user {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 20px 20px 4px 20px;
  }
  .bubble-assistant {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px 20px 20px 4px;
  }
  .model-card {
    transition: all 0.3s ease;
    cursor: pointer;
  }
  .model-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.3);
  }
  .model-card.active {
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
  }
  .chat-item {
    transition: all 0.3s ease;
    cursor: pointer;
  }
  .chat-item:hover {
    background: rgba(255, 255, 255, 0.05);
  }
  .chat-item.active {
    background: rgba(99, 102, 241, 0.1);
    border-left: 3px solid var(--primary);
  }
  textarea {
    resize: none;
    scrollbar-width: thin;
    scrollbar-color: #4b5563 transparent;
  }
  textarea::-webkit-scrollbar {
    width: 6px;
  }
  textarea::-webkit-scrollbar-thumb {
    background: #4b5563;
    border-radius: 3px;
  }
  .scrollbar-thin {
    scrollbar-width: thin;
    scrollbar-color: #4b5563 transparent;
  }
  .scrollbar-thin::-webkit-scrollbar {
    width: 6px;
  }
  .scrollbar-thin::-webkit-scrollbar-thumb {
    background: #4b5563;
    border-radius: 3px;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .fade-in {
    animation: fadeIn 0.3s ease-out;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  .animate-pulse {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  }
  .file-chip {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.3);
    transition: all 0.3s ease;
  }
  .file-chip:hover {
    background: rgba(99, 102, 241, 0.2);
  }
  @media (max-width: 768px) {
    .mobile-menu {
      transform: translateX(-100%);
      transition: transform 0.3s ease;
    }
    .mobile-menu.active {
      transform: translateX(0);
    }
  }
</style>
</head>
<body>
  <div id="banner" class="gradient-bg text-white text-center py-3 px-4 relative">
    <span class="text-sm font-medium">✨ Free for now! Soon $10/month for unlimited access to all AI models</span>
    <button onclick="document.getElementById('banner').style.display='none'" class="absolute right-4 top-1/2 -translate-y-1/2 text-white/80 hover:text-white">
      <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
      </svg>
    </button>
  </div>

  <div class="container mx-auto px-4 py-6 max-w-7xl">
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
      <button id="mobileMenuBtn" class="lg:hidden fixed top-6 left-4 z-50 p-3 glass rounded-xl">
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/>
        </svg>
      </button>

```
  <aside id="sidebar" class="lg:col-span-3 space-y-6 mobile-menu fixed lg:relative inset-y-0 left-0 z-40 w-80 lg:w-auto p-6 lg:p-0 glass-dark lg:bg-transparent">
    <div class="glass rounded-2xl p-6">
      <div class="flex items-center justify-between mb-4">
        <h2 class="text-xl font-semibold">Conversations</h2>
        <button id="newChat" class="btn-primary px-4 py-2 rounded-xl text-sm font-medium">
          New Chat
        </button>
      </div>
      <div id="chatList" class="space-y-2 max-h-[40vh] overflow-y-auto scrollbar-thin pr-2"></div>
      <div class="mt-4 flex flex-wrap gap-2">
        <button id="exportChats" class="btn-secondary px-3 py-1.5 rounded-lg text-sm">Export</button>
        <label class="btn-secondary px-3 py-1.5 rounded-lg text-sm cursor-pointer">
          Import<input type="file" id="importFile" class="hidden" accept=".json"/>
        </label>
        <button id="clearChats" class="btn-secondary px-3 py-1.5 rounded-lg text-sm text-red-400 hover:text-red-300">Clear All</button>
      </div>
    </div>

    <div class="glass rounded-2xl p-6">
      <h2 class="text-xl font-semibold mb-4">Select AI Model</h2>
      <div class="grid grid-cols-1 gap-3">
        <div class="model-card glass rounded-xl p-4" data-provider="openai">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-lg bg-gradient-to-br from-green-400 to-green-600 flex items-center justify-center text-white font-bold">O</div>
            <div>
              <h3 class="font-semibold">OpenAI</h3>
              <p class="text-xs text-gray-400">GPT-4.1 & o3</p>
            </div>
          </div>
        </div>
        <div class="model-card glass rounded-xl p-4" data-provider="anthropic">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-400 to-orange-600 flex items-center justify-center text-white font-bold">C</div>
            <div>
              <h3 class="font-semibold">Claude</h3>
              <p class="text-xs text-gray-400">Opus 4.1 & Sonnet 4</p>
            </div>
          </div>
        </div>
        <div class="model-card glass rounded-xl p-4" data-provider="google">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-400 to-blue-600 flex items-center justify-center text-white font-bold">G</div>
            <div>
              <h3 class="font-semibold">Gemini</h3>
              <p class="text-xs text-gray-400">1.5 Pro & Flash</p>
            </div>
          </div>
        </div>
      </div>
      <select id="model" class="w-full mt-4 bg-transparent border border-gray-600 rounded-xl px-4 py-2.5 focus:border-primary focus:outline-none"></select>
      <div class="mt-4 flex items-center gap-3">
        <input id="toggleSearch" type="checkbox" class="w-4 h-4 rounded border-gray-600 text-primary focus:ring-primary focus:ring-offset-0"/>
        <label for="toggleSearch" class="text-sm">Enable Google Search</label>
      </div>
    </div>

    <div class="glass rounded-2xl p-6">
      <h2 class="text-xl font-semibold mb-4">System Instructions</h2>
      <textarea id="systemPrompt" rows="6" class="w-full bg-transparent border border-gray-600 rounded-xl px-4 py-3 text-sm focus:border-primary focus:outline-none scrollbar-thin" placeholder="Set custom instructions for the AI..."></textarea>
      <button id="resetSystem" class="btn-secondary px-3 py-1.5 rounded-lg text-sm mt-3">Reset to Default</button>
    </div>
  </aside>

  <main class="lg:col-span-9 space-y-6">
    <div id="chat" class="glass rounded-2xl p-6 h-[70vh] overflow-y-auto scrollbar-thin"></div>

    <div class="glass rounded-2xl p-6">
      <div class="flex items-start gap-4">
        <div class="flex-1">
          <textarea id="prompt" rows="3" placeholder="Type your message here..." class="w-full bg-transparent border border-gray-600 rounded-xl px-4 py-3 focus:border-primary focus:outline-none scrollbar-thin"></textarea>
          <div class="flex items-center gap-3 mt-3">
            <label class="btn-secondary px-4 py-2 rounded-xl text-sm cursor-pointer">
              <svg class="w-5 h-5 inline mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"/>
              </svg>
              Attach Files
              <input id="fileInput" type="file" class="hidden" multiple accept="image/*,.pdf,.txt,.md,.doc,.docx,.csv,.json,.xml"/>
            </label>
            <div id="fileChips" class="flex gap-2 flex-wrap"></div>
          </div>
        </div>
        <button id="sendBtn" class="btn-primary px-6 py-3 rounded-xl font-medium">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
          </svg>
        </button>
      </div>
      <div class="flex items-center justify-between mt-4">
        <span id="status" class="text-sm text-gray-400">Ready</span>
        <span class="text-xs text-gray-500">Press Enter to send, Shift+Enter for new line</span>
      </div>
    </div>
  </main>
</div>
```

  </div>

<script>
const DEFAULT_SYSTEM = `You are a helpful, creative, and smart assistant. Follow the user's instructions carefully and provide detailed, accurate responses.`;

const PRESETS = {
  openai: [
    { id:"o3", label:"o3 - Advanced Reasoning"},
    { id:"o4-mini", label:"o4-mini - Fast Reasoning"},
    { id:"gpt-4.1", label:"GPT-4.1 - Most Capable"},
    { id:"gpt-4.1-mini", label:"GPT-4.1 Mini - Fast & Cheap"}
  ],
  anthropic: [
    { id:"claude-opus-4-1-20250805", label:"Claude Opus 4.1 - Most Powerful"},
    { id:"claude-sonnet-4-20250514", label:"Claude Sonnet 4 - Balanced"}
  ],
  google: [
    { id:"gemini-1.5-pro", label:"Gemini 1.5 Pro - Advanced"},
    { id:"gemini-1.5-flash", label:"Gemini 1.5 Flash - Fast"}
  ]
};

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
const mobileMenuBtn = $("#mobileMenuBtn");
const sidebar = $("#sidebar");

let filesToSend = [];
let currentProvider = "openai";
let state = {
  chats: [],
  activeId: null,
  useSearch: JSON.parse(localStorage.getItem("use_search") || "false"),
  systemPrompt: localStorage.getItem("system_prompt") || DEFAULT_SYSTEM
};

function uid(){ return Math.random().toString(36).slice(2) + Date.now().toString(36); }

function saveState(){
  localStorage.setItem("chats_v3", JSON.stringify(state.chats));
  localStorage.setItem("active_chat", state.activeId || "");
  localStorage.setItem("use_search", JSON.stringify(state.useSearch));
  localStorage.setItem("system_prompt", state.systemPrompt);
}

function loadState(){
  try { state.chats = JSON.parse(localStorage.getItem("chats_v3") || "[]"); } catch { state.chats=[]; }
  state.activeId = localStorage.getItem("active_chat") || (state.chats[0]?.id || null);
  renderChatList();
  if (!state.activeId) newChat();
  else renderActive();
}

function newChat(){
  const id = uid();
  const model = modelEl.value;
  const chat = { id, title: "New conversation", provider: currentProvider, model, messages: [] };
  state.chats.unshift(chat);
  state.activeId = id;
  saveState();
  renderChatList();
  renderActive();
}

function renderChatList(){
  chatListEl.innerHTML = "";
  state.chats.forEach(ch => {
    const div = document.createElement("div");
    div.className = "chat-item rounded-xl p-3 flex items-center justify-between";
    if (ch.id === state.activeId) div.classList.add("active");
    div.innerHTML = `
      <div class="flex-1 truncate pr-2">${ch.title}</div>
      <div class="flex gap-1">
        <button class="p-1.5 hover:bg-white/10 rounded-lg rename">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"/>
          </svg>
        </button>
        <button class="p-1.5 hover:bg-white/10 rounded-lg del text-red-400">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
          </svg>
        </button>
      </div>`;
    div.onclick = (e) => { 
      if (e.target.closest(".rename")||e.target.closest(".del")) return; 
      state.activeId = ch.id; 
      saveState(); 
      renderActive();
      if (window.innerWidth < 1024) sidebar.classList.remove("active");
    };
    div.querySelector(".rename").onclick = (e) => {
      e.stopPropagation();
      const t = prompt("Rename conversation", ch.title);
      if (t) {
        ch.title = t;
        saveState(); 
        renderChatList();
      }
    };
    div.querySelector(".del").onclick = (e) => {
      e.stopPropagation();
      if (!confirm("Delete this conversation?")) return;
      state.chats = state.chats.filter(c => c.id !== ch.id);
      if (state.activeId === ch.id) state.activeId = state.chats[0]?.id || null;
      saveState(); 
      renderChatList(); 
      renderActive();
    };
    chatListEl.appendChild(div);
  });
}

function renderActive(){
  const ch = state.chats.find(c => c.id === state.activeId);
  if (!ch) return;
  currentProvider = ch.provider || "openai";
  updateModelSelection();
  modelEl.value = ch.model || modelEl.value;
  chatEl.innerHTML = "";
  ch.messages.forEach(m => addBubble(m.role, m.content, m.attachments || []));
  scrollToBottom();
}

function updateModelSelection(){
  $$(".model-card").forEach(card => {
    card.classList.toggle("active", card.dataset.provider === currentProvider);
  });
  modelEl.innerHTML = "";
  PRESETS[currentProvider].forEach(m => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.label;
    modelEl.appendChild(opt);
  });
}

$$(".model-card").forEach(card => {
  card.onclick = () => {
    currentProvider = card.dataset.provider;
    updateModelSelection();
    const ch = state.chats.find(c => c.id === state.activeId);
    if (ch) {
      ch.provider = currentProvider;
      ch.model = modelEl.value;
      saveState();
    }
  };
});

function addBubble(role, text, atts=[]){
  const div = document.createElement("div");
  div.className = "mb-4 fade-in flex " + (role==="user" ? "justify-end" : "justify-start");
  const bubble = document.createElement("div");
  bubble.className = "max-w-[80%] " + (role==="user"?"bubble-user":"bubble-assistant") + " p-4";
  
  if (text) {
    const textDiv = document.createElement("div");
    textDiv.className = "whitespace-pre-wrap";
    textDiv.textContent = text;
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
        link.className = "file-chip rounded-lg p-3 text-sm hover:bg-white/10 transition-colors";
        grid.appendChild(link);
      }
    });
    bubble.appendChild(grid);
  }
  
  div.appendChild(bubble);
  chatEl.appendChild(div);
}

function scrollToBottom(){
  chatEl.scrollTo({ top: chatEl.scrollHeight, behavior: "smooth" });
}

promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});

toggleSearchEl.checked = state.useSearch;
toggleSearchEl.addEventListener("change", () => {
  state.useSearch = toggleSearchEl.checked;
  saveState();
});

systemPromptEl.value = state.systemPrompt;
systemPromptEl.addEventListener("input", () => { 
  state.systemPrompt = systemPromptEl.value; 
  saveState(); 
});

resetSystemEl.onclick = () => {
  systemPromptEl.value = DEFAULT_SYSTEM;
  state.systemPrompt = DEFAULT_SYSTEM;
  saveState();
};

newChatEl.onclick = newChat;
clearChatsEl.onclick = () => {
  if (!confirm("Delete all conversations?")) return;
  state.chats = [];
  state.activeId = null;
  saveState();
  newChat();
};

exportChatsEl.onclick = () => {
  const blob = new Blob([JSON.stringify(state.chats, null, 2)], {type:"application/json"});
  const u = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = u;
  a.download = "ai-chats.json";
  a.click();
  URL.revokeObjectURL(u);
};

importFileEl.onchange = (e) => {
  const f = e.target.files[0];
  if(!f) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const arr = JSON.parse(reader.result);
      if (Array.isArray(arr)) {
        state.chats = arr.concat(state.chats);
        saveState();
        renderChatList();
        renderActive();
        setStatus("Conversations imported", "success");
      } else setStatus("Invalid file format", "error");
    } catch { 
      setStatus("Invalid JSON file", "error"); 
    }
  };
  reader.readAsText(f);
};

fileInput.onchange = async (e) => {
  if (!fileInput.files.length) return;
  const fd = new FormData();
  for (const f of fileInput.files) fd.append("files", f);
  setStatus("Uploading files...", "loading");
  try {
    const res = await fetch("/api/upload", { method:"POST", body: fd });
    const data = await res.json();
    (data.files||[]).forEach(f => {
      filesToSend.push(f);
      const chip = document.createElement("div");
      chip.className = "file-chip rounded-lg px-3 py-1.5 flex items-center gap-2 text-sm";
      chip.innerHTML = `
        <span class="truncate max-w-[150px]">${f.name || f.url}</span>
        <button class="text-gray-400 hover:text-white">×</button>`;
      chip.querySelector("button").onclick = () => {
        filesToSend = filesToSend.filter(x => x.url !== f.url);
        chip.remove();
      };
      fileChips.appendChild(chip);
    });
    setStatus("Files uploaded", "success");
  } catch(e) {
    console.error(e);
    setStatus("Upload failed", "error");
  } finally {
    fileInput.value = "";
  }
};

sendBtn.onclick = async () => {
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
    ch.title = txt.slice(0, 50) + (txt.length > 50 ? "..." : "");
    renderChatList();
  }
  
  filesToSend = [];
  fileChips.innerHTML = "";
  promptEl.value = "";
  addBubble("user", txt, atts);
  scrollToBottom();

  setStatus("AI is thinking...", "loading");
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
    if (!res.ok) throw new Error(data.error || "Request failed");
    const out = (data.output || "").trim();
    ch.messages.push({role:"assistant", content: out});
    saveState();
    addBubble("assistant", out);
    scrollToBottom();
    setStatus("Ready");
  } catch(err) {
    console.error(err);
    addBubble("assistant", "⚠️ " + (err.message || "Something went wrong. Please try again."));
    setStatus("Error occurred", "error");
  }
};

function setStatus(msg, type = "normal"){
  statusEl.textContent = msg;
  statusEl.className = "text-sm ";
  if (type === "loading") statusEl.className += "text-yellow-400 animate-pulse";
  else if (type === "error") statusEl.className += "text-red-400";
  else if (type === "success") statusEl.className += "text-green-400";
  else statusEl.className += "text-gray-400";
  
  if (type !== "loading") {
    setTimeout(() => {
      statusEl.textContent = "Ready";
      statusEl.className = "text-sm text-gray-400";
    }, 3000);
  }
}

mobileMenuBtn.onclick = () => {
  sidebar.classList.toggle("active");
};

document.addEventListener("click", (e) => {
  if (window.innerWidth < 1024 && !sidebar.contains(e.target) && !mobileMenuBtn.contains(e.target)) {
    sidebar.classList.remove("active");
  }
});

updateModelSelection();
loadState();
</script>

</body>
</html>"""

@app.route(”/”)
def index():
html = HTML_TEMPLATE.replace(”{{APP_TITLE}}”, APP_TITLE)
return Response(html, mimetype=“text/html”)

if **name** == “**main**”:
port = int(os.environ.get(“PORT”, “10000”))
app.run(host=“0.0.0.0”, port=port)