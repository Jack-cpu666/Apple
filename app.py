import os, io, base64, json, mimetypes, time, re, tempfile
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from flask import Flask, request, send_from_directory, make_response, jsonify, Response

APP_TITLE = "All-in-One AI Chat (OpenAI • Claude • Gemini)"
UPLOAD_DIR = os.environ.get("UPLOAD_DIR")
if not UPLOAD_DIR:
    UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

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

def read_env(name, default=None):
    v = os.environ.get(name, default)
    return v if (v is not None and v.strip() != "") else None

PRO_PASSWORD = read_env("PRO_PASSWORD")
OPENAI_KEY_SERVER = read_env("OPENAI_API_KEY_SERVER")
ANTHROPIC_KEY_SERVER = read_env("ANTHROPIC_API_KEY_SERVER")
GEMINI_KEYS = [k for k in [read_env("GEMINI_KEY_1"), read_env("GEMINI_KEY_2")] if k]
GOOGLE_SEARCH_KEY = read_env("GOOGLE_SEARCH_KEY")
GOOGLE_SEARCH_CX  = read_env("GOOGLE_SEARCH_CX")

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

def pick_openai_key(user_key, pro_password):
    if user_key: return user_key
    if OPENAI_KEY_SERVER and PRO_PASSWORD and pro_password and pro_password == PRO_PASSWORD:
        return OPENAI_KEY_SERVER
    return None

def pick_anthropic_key(user_key, pro_password):
    if user_key: return user_key
    if ANTHROPIC_KEY_SERVER and PRO_PASSWORD and pro_password and pro_password == PRO_PASSWORD:
        return ANTHROPIC_KEY_SERVER
    return None

_gemini_rr = 0
def pick_gemini_key(user_key):
    global _gemini_rr
    if user_key: return user_key
    if GEMINI_KEYS:
        key = GEMINI_KEYS[_gemini_rr % len(GEMINI_KEYS)]
        _gemini_rr += 1
        return key
    return None

def normalize_messages_for_last_user_images(messages, attachments):
    msgs = list(messages)
    if attachments:
        for i in range(len(msgs)-1, -1, -1):
            if msgs[i].get("role") == "user":
                extras = msgs[i].setdefault("attachments", [])
                extras.extend(attachments)
                break
    return msgs

def openai_chat(model, system_prompt, messages, key):
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

def gemini_chat(model, system_prompt, messages, key):
    base = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

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

    keys = data.get("keys") or {}
    openai_key_user = keys.get("openai")
    anthropic_key_user = keys.get("anthropic")
    gemini_key_user = keys.get("gemini")
    pro_password = keys.get("pro_password")

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
            key = pick_openai_key(openai_key_user, pro_password)
            if not key:
                return jsonify({"error":"Missing OpenAI key. Add your key in the UI or unlock server key with the password."}), 400
            out, meta, err = openai_chat(model, system_prompt, messages, key)
        elif provider == "anthropic":
            key = pick_anthropic_key(anthropic_key_user, pro_password)
            if not key:
                return jsonify({"error":"Missing Anthropic key. Add your key in the UI or unlock server key with the password."}), 400
            out, meta, err = anthropic_chat(model, system_prompt, messages, key)
        elif provider == "google":
            key = pick_gemini_key(gemini_key_user)
            if not key:
                return jsonify({"error":"Missing Gemini key. Add your key in the UI, or set GEMINI_KEY_1/2 server-side."}), 400
            out, meta, err = gemini_chat(model, system_prompt, messages, key)
        else:
            return jsonify({"error":"Unknown provider"}), 400

        if err:
            if provider == "google" and not gemini_key_user and len(GEMINI_KEYS) > 1:
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
    q = data.get("q","").strip()
    if not q:
        return jsonify({"results":[]})
    res = google_search_top(q, n=5)
    return jsonify({"results": res})

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"/>
<title>{{APP_TITLE}}</title>
<link rel="icon" href="data:,">
<script src="https://cdn.tailwindcss.com"></script>
<style>
  @keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }
  @keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-20px); }
  }
  @keyframes pulse-glow {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.8; transform: scale(1.05); }
  }
  @keyframes slide-up {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
  }
  @keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
  }
  
  :root {
    --bg-primary: #0a0118;
    --bg-secondary: #1a0f2e;
    --panel: rgba(26, 15, 46, 0.4);
    --ink: #f0e6ff;
    --ink-dim: #b794f4;
    --accent: #9f7aea;
    --accent-bright: #b794f4;
    --chip: rgba(159, 122, 234, 0.2);
    --border: rgba(159, 122, 234, 0.3);
    --glow: rgba(159, 122, 234, 0.5);
  }
  
  * {
    box-sizing: border-box;
  }
  
  body {
    background: var(--bg-primary);
    color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    overflow-x: hidden;
    position: relative;
  }
  
  body::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
      radial-gradient(circle at 20% 50%, rgba(159, 122, 234, 0.15) 0%, transparent 50%),
      radial-gradient(circle at 80% 80%, rgba(183, 148, 244, 0.1) 0%, transparent 50%),
      radial-gradient(circle at 40% 20%, rgba(139, 92, 246, 0.1) 0%, transparent 50%);
    pointer-events: none;
    z-index: 1;
  }
  
  .floating-orb {
    position: fixed;
    width: 300px;
    height: 300px;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.3;
    pointer-events: none;
    animation: float 20s ease-in-out infinite;
    z-index: 0;
  }
  
  .orb-1 {
    background: linear-gradient(135deg, #9f7aea, #805ad5);
    top: -150px;
    left: -150px;
    animation-delay: 0s;
  }
  
  .orb-2 {
    background: linear-gradient(135deg, #b794f4, #9f7aea);
    bottom: -150px;
    right: -150px;
    animation-delay: 10s;
  }
  
  .orb-3 {
    background: linear-gradient(135deg, #805ad5, #6b46c1);
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    animation-delay: 5s;
  }
  
  .glass {
    backdrop-filter: blur(20px) saturate(180%);
    background: rgba(26, 15, 46, 0.6);
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
  }
  
  .glass::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(159, 122, 234, 0.1), transparent);
    animation: shimmer 3s infinite;
  }
  
  .card {
    box-shadow: 
      0 0 30px rgba(159, 122, 234, 0.2),
      0 10px 40px rgba(0, 0, 0, 0.3),
      inset 0 1px 0 rgba(255, 255, 255, 0.1);
    transition: all 0.3s ease;
    position: relative;
    z-index: 2;
  }
  
  .card:hover {
    box-shadow: 
      0 0 40px rgba(159, 122, 234, 0.3),
      0 15px 50px rgba(0, 0, 0, 0.4),
      inset 0 1px 0 rgba(255, 255, 255, 0.15);
    transform: translateY(-2px);
  }
  
  .scrollbar::-webkit-scrollbar {
    width: 10px;
    height: 10px;
  }
  
  .scrollbar::-webkit-scrollbar-track {
    background: rgba(26, 15, 46, 0.4);
    border-radius: 10px;
  }
  
  .scrollbar::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--accent), var(--accent-bright));
    border-radius: 10px;
    border: 2px solid rgba(26, 15, 46, 0.4);
  }
  
  .scrollbar::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, var(--accent-bright), var(--accent));
  }
  
  .bubble-user {
    background: linear-gradient(135deg, rgba(159, 122, 234, 0.3), rgba(139, 92, 246, 0.25));
    border: 1px solid rgba(183, 148, 244, 0.4);
    box-shadow: 
      0 5px 20px rgba(159, 122, 234, 0.2),
      inset 0 1px 0 rgba(255, 255, 255, 0.1);
    animation: slide-up 0.3s ease;
  }
  
  .bubble-assistant {
    background: linear-gradient(135deg, rgba(26, 15, 46, 0.8), rgba(44, 25, 84, 0.6));
    border: 1px solid rgba(159, 122, 234, 0.25);
    box-shadow: 
      0 5px 20px rgba(0, 0, 0, 0.3),
      inset 0 1px 0 rgba(255, 255, 255, 0.05);
    animation: slide-up 0.3s ease;
  }
  
  #composer-wrapper {
    position: sticky;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 10;
    background: linear-gradient(to top, var(--bg-primary), transparent);
    padding-top: 20px;
  }
  
  #chat {
    scroll-padding-bottom: 1rem;
  }
  
  textarea {
    resize: none;
    line-height: 1.5;
    background: rgba(26, 15, 46, 0.5);
    border: 1px solid var(--border);
    color: var(--ink);
    transition: all 0.3s ease;
  }
  
  textarea:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 20px rgba(159, 122, 234, 0.3);
    background: rgba(26, 15, 46, 0.7);
  }
  
  .btn {
    background: linear-gradient(135deg, var(--accent), var(--accent-bright));
    border: 1px solid var(--accent-bright);
    color: white;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 
      0 4px 15px rgba(159, 122, 234, 0.4),
      inset 0 1px 0 rgba(255, 255, 255, 0.2);
    position: relative;
    overflow: hidden;
  }
  
  .btn::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    background: rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
  }
  
  .btn:hover::before {
    width: 300px;
    height: 300px;
  }
  
  .btn:hover {
    transform: translateY(-2px);
    box-shadow: 
      0 6px 20px rgba(159, 122, 234, 0.5),
      inset 0 1px 0 rgba(255, 255, 255, 0.3);
  }
  
  .btn:active {
    transform: translateY(0);
  }
  
  .tag {
    background: var(--chip);
    border: 1px solid var(--border);
    color: var(--accent-bright);
    transition: all 0.3s ease;
  }
  
  .tag:hover {
    background: rgba(159, 122, 234, 0.3);
    box-shadow: 0 0 10px rgba(159, 122, 234, 0.3);
  }
  
  .pill {
    background: rgba(26, 15, 46, 0.6);
    border: 1px solid var(--border);
    color: var(--ink-dim);
    transition: all 0.3s ease;
    font-weight: 500;
  }
  
  .pill:hover {
    background: rgba(159, 122, 234, 0.2);
    border-color: var(--accent);
    color: var(--ink);
    box-shadow: 0 0 15px rgba(159, 122, 234, 0.3);
  }
  
  .mono {
    font-family: 'Fira Code', 'SF Mono', Monaco, 'Inconsolata', 'Fira Mono', monospace;
    font-size: 0.9em;
  }
  
  input[type="text"], input[type="password"], select {
    background: rgba(26, 15, 46, 0.5);
    border: 1px solid var(--border);
    color: var(--ink);
    transition: all 0.3s ease;
  }
  
  input[type="text"]:focus, input[type="password"]:focus, select:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 20px rgba(159, 122, 234, 0.3);
    background: rgba(26, 15, 46, 0.7);
  }
  
  .chat-item {
    transition: all 0.3s ease;
    cursor: pointer;
    position: relative;
    overflow: hidden;
  }
  
  .chat-item::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(159, 122, 234, 0.2), transparent);
    transition: left 0.5s;
  }
  
  .chat-item:hover::before {
    left: 100%;
  }
  
  .chat-item:hover {
    background: rgba(159, 122, 234, 0.1);
    border-color: var(--accent);
  }
  
  .loading-dots {
    display: inline-flex;
    gap: 4px;
  }
  
  .loading-dots span {
    width: 8px;
    height: 8px;
    background: var(--accent);
    border-radius: 50%;
    animation: pulse-glow 1.4s ease-in-out infinite;
  }
  
  .loading-dots span:nth-child(2) {
    animation-delay: 0.2s;
  }
  
  .loading-dots span:nth-child(3) {
    animation-delay: 0.4s;
  }
  
  .gradient-text {
    background: linear-gradient(135deg, var(--accent), var(--accent-bright), var(--ink));
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradient 3s ease infinite;
  }
  
  .status-ready {
    color: #68d391;
    border-color: #68d391;
    background: rgba(104, 211, 145, 0.1);
  }
  
  .status-busy {
    color: var(--accent-bright);
    border-color: var(--accent-bright);
    background: rgba(183, 148, 244, 0.1);
  }
  
  .status-error {
    color: #fc8181;
    border-color: #fc8181;
    background: rgba(252, 129, 129, 0.1);
  }
  
  @media (max-width: 1024px) {
    .floating-orb {
      width: 200px;
      height: 200px;
    }
  }
</style>
</head>
<body class="min-h-screen">
  <div class="floating-orb orb-1"></div>
  <div class="floating-orb orb-2"></div>
  <div class="floating-orb orb-3"></div>
  
  <div class="grid grid-cols-12 gap-4 max-w-7xl mx-auto p-4 relative z-10">
    <aside class="col-span-12 lg:col-span-3 space-y-4">
      <div class="glass card rounded-2xl p-4">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-xl font-bold gradient-text">Chats</h2>
          <button id="newChat" class="px-4 py-2 rounded-xl btn text-sm">✨ New</button>
        </div>
        <div id="chatList" class="space-y-2 max-h-[55vh] overflow-y-auto scrollbar"></div>
        <div class="mt-4 flex gap-2 flex-wrap">
          <button id="exportChats" class="px-3 py-1.5 rounded-xl pill text-sm">📥 Export</button>
          <label class="px-3 py-1.5 rounded-xl pill cursor-pointer text-sm">
            📤 Import<input type="file" id="importFile" class="hidden" accept=".json"/>
          </label>
          <button id="clearChats" class="px-3 py-1.5 rounded-xl pill text-sm">🗑️ Clear</button>
        </div>
      </div>

      <div class="glass card rounded-2xl p-4">
        <h2 class="text-xl font-bold gradient-text mb-4">Models</h2>
        <label class="block text-sm font-medium mb-2 text-ink-dim">Provider</label>
        <select id="provider" class="w-full rounded-xl p-2.5 mb-4">
          <option value="openai">🤖 OpenAI</option>
          <option value="anthropic">🧠 Claude</option>
          <option value="google">✨ Gemini</option>
        </select>
        <label class="block text-sm font-medium mb-2 text-ink-dim">Model</label>
        <select id="model" class="w-full rounded-xl p-2.5"></select>
        <div class="mt-4">
          <input id="customModel" placeholder="Custom model id (optional)" class="w-full rounded-xl p-2.5 text-sm" />
          <p class="text-xs text-ink-dim mt-2 opacity-70">Override with any model id</p>
        </div>
        <div class="mt-4 flex items-center gap-3">
          <input id="toggleSearch" type="checkbox" class="w-4 h-4 rounded accent-accent"/>
          <label class="text-sm cursor-pointer">🔍 Enable Google Search</label>
        </div>
      </div>

      <div class="glass card rounded-2xl p-4">
        <h2 class="text-xl font-bold gradient-text mb-4">Keys & Access</h2>
        <p class="text-xs text-ink-dim mb-4 opacity-70">Keys are saved locally. Server keys require password.</p>
        <div class="space-y-3">
          <input id="openaiKey" class="w-full rounded-xl p-2.5 mono" placeholder="🔑 OpenAI API key" type="password"/>
          <input id="anthropicKey" class="w-full rounded-xl p-2.5 mono" placeholder="🔑 Anthropic API key" type="password"/>
          <input id="geminiKey" class="w-full rounded-xl p-2.5 mono" placeholder="🔑 Gemini API key" type="password"/>
          <input id="proPassword" class="w-full rounded-xl p-2.5 mono" placeholder="🔐 Server key password" type="password"/>
        </div>
        <div class="mt-4 flex gap-2">
          <button id="saveKeys" class="px-4 py-2 rounded-xl pill text-sm">💾 Save</button>
          <button id="clearKeys" class="px-4 py-2 rounded-xl pill text-sm">🗑️ Clear</button>
        </div>
      </div>

      <div class="glass card rounded-2xl p-4">
        <h2 class="text-xl font-bold gradient-text mb-4">System Prompt</h2>
        <textarea id="systemPrompt" rows="6" class="w-full rounded-2xl p-3 text-sm mono scrollbar"></textarea>
        <button id="resetSystem" class="px-4 py-2 rounded-xl pill mt-3 text-sm">🔄 Reset Default</button>
      </div>
    </aside>

    <main class="col-span-12 lg:col-span-9">
      <div id="chat" class="glass card rounded-2xl p-6 h-[75vh] overflow-y-auto scrollbar"></div>

      <div id="composer-wrapper">
        <div class="glass card rounded-2xl p-4">
          <div class="flex items-center gap-3 flex-wrap mb-3">
            <label class="px-4 py-2 rounded-xl pill cursor-pointer text-sm font-medium hover:scale-105 transition-transform">
              📎 Upload
              <input id="fileInput" type="file" class="hidden" multiple accept="image/*,.pdf,.txt,.md,.doc,.docx,.csv,.json,.xml"/>
            </label>
            <div id="fileChips" class="flex gap-2 flex-wrap flex-1"></div>
            <div class="flex items-center gap-3">
              <span class="tag rounded-xl px-3 py-1.5 text-sm font-medium status-ready" id="status">Ready</span>
              <button id="sendBtn" class="px-6 py-2.5 rounded-xl btn font-semibold">Send ✨</button>
            </div>
          </div>
          <textarea id="prompt" rows="1" placeholder="Write your message..." class="w-full rounded-2xl p-4 text-base scrollbar"></textarea>
        </div>
      </div>
    </main>
  </div>

<script>
const DEFAULT_SYSTEM = \`You are a careful, helpful assistant. Follow the user's instructions exactly, ask for missing context only when essential, cite concrete dates when clarifying time, and keep answers concise unless deeply technical. When images are attached, describe what you see before analyzing. Prefer bullet lists and short paragraphs for readability.\`;

const PRESETS = {
  openai: [
    { id:"o3", label:"o3 (reasoning, premium)"},
    { id:"o4-mini", label:"o4-mini (cheap reasoning)"},
    { id:"gpt-4.1", label:"GPT-4.1 (flagship)"},
    { id:"gpt-4.1-mini", label:"GPT-4.1 mini (cheap)"}
  ],
  anthropic: [
    { id:"claude-opus-4-1-20250805", label:"Claude Opus 4.1"},
    { id:"claude-sonnet-4-20250514", label:"Claude Sonnet 4"}
  ],
  google: [
    { id:"gemini-1.5-pro", label:"Gemini 1.5 Pro"},
    { id:"gemini-1.5-flash", label:"Gemini 1.5 Flash"}
  ]
};

const $ = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));

const chatEl = $("#chat");
const providerEl = $("#provider");
const modelEl = $("#model");
const customModelEl = $("#customModel");
const toggleSearchEl = $("#toggleSearch");
const openaiKeyEl = $("#openaiKey");
const anthropicKeyEl = $("#anthropicKey");
const geminiKeyEl = $("#geminiKey");
const proPasswordEl = $("#proPassword");
const saveKeysEl = $("#saveKeys");
const clearKeysEl = $("#clearKeys");
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

let filesToSend = [];
let state = {
  chats: [],
  activeId: null,
  keys: {
    openai: localStorage.getItem("openai_key") || "",
    anthropic: localStorage.getItem("anthropic_key") || "",
    gemini: localStorage.getItem("gemini_key") || "",
    pro_password: localStorage.getItem("pro_password") || ""
  },
  useSearch: JSON.parse(localStorage.getItem("use_search") || "false"),
  systemPrompt: localStorage.getItem("system_prompt") || DEFAULT_SYSTEM
};

function uid(){ return Math.random().toString(36).slice(2) + Date.now().toString(36); }

function saveState(){
  localStorage.setItem("chats_v2", JSON.stringify(state.chats));
  localStorage.setItem("active_chat", state.activeId || "");
  localStorage.setItem("use_search", JSON.stringify(state.useSearch));
  localStorage.setItem("system_prompt", state.systemPrompt);
}

function loadState(){
  try { state.chats = JSON.parse(localStorage.getItem("chats_v2") || "[]"); } catch { state.chats=[]; }
  state.activeId = localStorage.getItem("active_chat") || (state.chats[0]?.id || null);
  renderChatList();
  if (!state.activeId) newChat();
  else renderActive();
}

function newChat(){
  const id = uid();
  const provider = providerEl.value;
  const model = (customModelEl.value.trim() || modelEl.value);
  const chat = { id, title: "New chat", provider, model, messages: [] };
  state.chats.unshift(chat);
  state.activeId = id;
  saveState();
  renderChatList();
  renderActive();
}

function renderChatList(){
  chatListEl.innerHTML = "";
  state.chats.forEach(ch => {
    const btn = document.createElement("div");
    btn.className = "chat-item p-3 rounded-xl border border-transparent hover:border-accent cursor-pointer flex items-center justify-between transition-all";
    btn.innerHTML = \`<span class="truncate font-medium">\${ch.title}</span>
      <div class="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
        <button class="text-xs tag px-2 py-1 rounded-lg rename">✏️</button>
        <button class="text-xs tag px-2 py-1 rounded-lg del">🗑️</button>
      </div>\`;
    btn.classList.add("group");
    btn.onclick = (e) => { if (e.target.closest(".rename")||e.target.closest(".del")) return; state.activeId = ch.id; saveState(); renderActive(); };
    btn.querySelector(".rename").onclick = (e) => {
      e.stopPropagation();
      const t = prompt("Rename chat", ch.title) || ch.title;
      ch.title = t;
      saveState(); renderChatList();
    };
    btn.querySelector(".del").onclick = (e) => {
      e.stopPropagation();
      if (!confirm("Delete this chat?")) return;
      state.chats = state.chats.filter(c => c.id !== ch.id);
      if (state.activeId === ch.id) state.activeId = state.chats[0]?.id || null;
      saveState(); renderChatList(); renderActive();
    };
    if (ch.id === state.activeId) {
      btn.style.background = "rgba(159, 122, 234, 0.15)";
      btn.style.borderColor = "var(--accent)";
    }
    chatListEl.appendChild(btn);
  });
}

function renderActive(){
  const ch = state.chats.find(c => c.id === state.activeId);
  if (!ch) return;
  providerEl.value = ch.provider || "openai";
  setModelOptions();
  modelEl.value = ch.model || modelEl.value;
  customModelEl.value = "";
  chatEl.innerHTML = "";
  ch.messages.forEach(m => addBubble(m.role, m.content, m.attachments || []));
  scrollToBottom();
}

function setModelOptions(){
  const p = providerEl.value;
  modelEl.innerHTML = "";
  PRESETS[p].forEach(m => {
    const opt = document.createElement("option");
    opt.value = m.id; opt.textContent = m.label;
    modelEl.appendChild(opt);
  });
}

function addBubble(role, text, atts=[]){
  const wrap = document.createElement("div");
  wrap.className = "mb-4";
  const b = document.createElement("div");
  b.className = "rounded-2xl p-4 max-w-[85%] " + (role==="user"?"bubble-user ml-auto":"bubble-assistant");
  
  const header = document.createElement("div");
  header.className = "flex items-center gap-2 mb-2 text-sm opacity-70";
  header.innerHTML = \`<span class="font-medium">\${role === "user" ? "You" : "AI"}</span>\`;
  b.appendChild(header);
  
  if (text) {
    const content = document.createElement("div");
    content.className = "whitespace-pre-wrap leading-relaxed";
    content.innerHTML = sanitize(text);
    b.appendChild(content);
  }
  
  if (atts && atts.length){
    const grid = document.createElement("div");
    grid.className = "grid grid-cols-3 gap-3 mt-3";
    atts.forEach(a => {
      if ((a.mime||"").startsWith("image/")){
        const img = document.createElement("img");
        img.src = a.url; img.alt = a.name || "image";
        img.className = "rounded-xl border border-border shadow-lg hover:scale-105 transition-transform cursor-pointer";
        img.onclick = () => window.open(a.url, '_blank');
        grid.appendChild(img);
      } else {
        const link = document.createElement("a");
        link.href = a.url; link.textContent = a.name || a.url; link.target = "_blank";
        link.className = "text-accent-bright hover:text-accent underline text-sm font-medium";
        grid.appendChild(link);
      }
    });
    b.appendChild(grid);
  }
  wrap.appendChild(b);
  chatEl.appendChild(wrap);
}

function sanitize(s){
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

function scrollToBottom(){
  chatEl.scrollTo({ top: chatEl.scrollHeight, behavior: "smooth" });
}

function growTextarea(){
  promptEl.style.height = "auto";
  const max = 240;
  promptEl.style.height = Math.min(promptEl.scrollHeight, max) + "px";
}

promptEl.addEventListener("input", growTextarea);
window.addEventListener("resize", growTextarea);
setTimeout(growTextarea, 0);

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
systemPromptEl.addEventListener("input", () => { state.systemPrompt = systemPromptEl.value; saveState(); });
resetSystemEl.onclick = () => {
  systemPromptEl.value = DEFAULT_SYSTEM;
  state.systemPrompt = DEFAULT_SYSTEM; saveState();
};

providerEl.addEventListener("change", setModelOptions);
setModelOptions();

function syncKeysUI(){
  openaiKeyEl.value = state.keys.openai;
  anthropicKeyEl.value = state.keys.anthropic;
  geminiKeyEl.value = state.keys.gemini;
  proPasswordEl.value = state.keys.pro_password;
}
syncKeysUI();

saveKeysEl.onclick = () => {
  state.keys.openai = openaiKeyEl.value.trim();
  state.keys.anthropic = anthropicKeyEl.value.trim();
  state.keys.gemini = geminiKeyEl.value.trim();
  state.keys.pro_password = proPasswordEl.value.trim();
  localStorage.setItem("openai_key", state.keys.openai);
  localStorage.setItem("anthropic_key", state.keys.anthropic);
  localStorage.setItem("gemini_key", state.keys.gemini);
  localStorage.setItem("pro_password", state.keys.pro_password);
  status("Keys saved ✅", "ready");
};
clearKeysEl.onclick = () => {
  ["openai_key","anthropic_key","gemini_key","pro_password"].forEach(k => localStorage.removeItem(k));
  state.keys = { openai:"",anthropic:"",gemini:"",pro_password:"" };
  syncKeysUI();
  status("Keys cleared 🗑️", "ready");
};

newChatEl.onclick = newChat;
clearChatsEl.onclick = () => {
  if (!confirm("Clear ALL chats?")) return;
  state.chats = []; state.activeId = null; saveState(); newChat();
};
exportChatsEl.onclick = () => {
  const blob = new Blob([JSON.stringify(state.chats, null, 2)], {type:"application/json"});
  const u = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = u; a.download="chats.json"; a.click();
  URL.revokeObjectURL(u);
};
importFileEl.onchange = (e) => {
  const f = e.target.files[0]; if(!f) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const arr = JSON.parse(reader.result);
      if (Array.isArray(arr)) {
        state.chats = arr.concat(state.chats);
        saveState(); renderChatList(); renderActive();
        status("Imported chats ✅", "ready");
      } else status("Invalid file ❌", "error");
    } catch { status("Invalid JSON ❌", "error"); }
  };
  reader.readAsText(f);
};

fileInput.onchange = async (e) => {
  if (!fileInput.files.length) return;
  const fd = new FormData();
  for (const f of fileInput.files) fd.append("files", f);
  setBusy(true, "Uploading...");
  try {
    const res = await fetch("/api/upload", { method:"POST", body: fd });
    const data = await res.json();
    (data.files||[]).forEach(addChip);
    status("Files uploaded ✅", "ready");
  } catch(e) {
    console.error(e); 
    status("Upload failed ❌", "error");
  } finally {
    setBusy(false);
    fileInput.value = "";
  }
};

function addChip(f){
  filesToSend.push(f);
  const chip = document.createElement("div");
  chip.className = "tag rounded-xl px-3 py-1.5 flex items-center gap-2 hover:scale-105 transition-transform";
  chip.innerHTML = \`<span class="text-xs truncate max-w-[160px] font-medium">\${f.name || f.url}</span>
    <button class="text-sm hover:text-red-400 transition-colors">×</button>\`;
  chip.querySelector("button").onclick = () => {
    filesToSend = filesToSend.filter(x => x.url !== f.url);
    chip.remove();
  };
  fileChips.appendChild(chip);
}

sendBtn.onclick = async () => {
  const txt = promptEl.value.trim();
  const ch = state.chats.find(c => c.id === state.activeId);
  if (!ch) return;
  if (!txt && filesToSend.length===0) return;
  const provider = providerEl.value;
  const model = (customModelEl.value.trim() || modelEl.value);

  ch.provider = provider; ch.model = model;
  state.systemPrompt = systemPromptEl.value;
  saveState();

  const atts = filesToSend.slice();
  ch.messages.push({role:"user", content: txt, attachments: atts});
  filesToSend = []; fileChips.innerHTML = "";
  promptEl.value=""; growTextarea();
  addBubble("user", txt, atts); scrollToBottom();

  setBusy(true, \`<div class="loading-dots"><span></span><span></span><span></span></div> Thinking...\`);
  try {
    const res = await fetch("/api/chat", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({
        provider,
        model,
        system_prompt: state.systemPrompt,
        messages: ch.messages,
        attachments: [],
        use_search: toggleSearchEl.checked,
        keys: {
          openai: state.keys.openai || undefined,
          anthropic: state.keys.anthropic || undefined,
          gemini: state.keys.gemini || undefined,
          pro_password: state.keys.pro_password || undefined
        }
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error((data && (data.error?.error || data.error)) || "Request failed");
    const out = (data.output || "").trim();
    ch.messages.push({role:"assistant", content: out});
    saveState();
    addBubble("assistant", out); scrollToBottom();
    status("Ready", "ready");
  } catch(err) {
    console.error(err);
    addBubble("assistant", "⚠️ " + (err.message || "Error"));
    status("Error occurred", "error");
  } finally {
    setBusy(false);
  }
};

function status(msg, type = "ready"){ 
  statusEl.textContent = msg;
  statusEl.className = \`tag rounded-xl px-3 py-1.5 text-sm font-medium status-\${type}\`;
  if (type !== "busy") {
    setTimeout(() => {
      statusEl.textContent = "Ready";
      statusEl.className = "tag rounded-xl px-3 py-1.5 text-sm font-medium status-ready";
    }, 3000);
  }
}

function setBusy(isBusy, msg){ 
  sendBtn.disabled = !!isBusy; 
  if (isBusy) {
    statusEl.innerHTML = msg || "Working...";
    statusEl.className = "tag rounded-xl px-3 py-1.5 text-sm font-medium status-busy";
  } else {
    statusEl.textContent = "Ready";
    statusEl.className = "tag rounded-xl px-3 py-1.5 text-sm font-medium status-ready";
  }
}

loadState();
document.addEventListener("DOMContentLoaded", () => { setModelOptions(); });
setTimeout(growTextarea, 0);
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