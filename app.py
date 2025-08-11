# app.py
# NovaMind — a clean, mobile-first chat UI powered under the hood
# by Google's Gemini API (hidden from users). Two creative modes:
# "Sage" (deeper reasoning) and "Spark" (faster replies).
#
# Defaults you asked for: temperature=1, thinking_budget=20000,
# correct token limits & robust failover (Sage -> Spark) if rate-limited.

import os, base64, json, mimetypes, time, re, tempfile
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from flask import Flask, request, send_from_directory, make_response, jsonify, Response

# =========================
# NovaMind — Server Config
# =========================
APP_TITLE = "NovaMind"
UPLOAD_DIR = os.environ.get("UPLOAD_DIR") or os.path.join(tempfile.gettempdir(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Hard-coded Google (Gemini) API key — used ONLY on the server (never exposed to the browser)
# You asked to hardcode this exact key:
GEMINI_KEYS = ["AIzaSyBqQQszYifOVY6396kV9lkEs1Tz3cSdmVo"]

# UI model ids (no provider terms shown to users)
NOVA_MODELS = [
    {"id": "sage",  "label": "NovaMind — Sage (thinks deeper)"},
    {"id": "spark", "label": "NovaMind — Spark (answers faster)"},
]

# Internal mapping (server-only, users never see this)
MODEL_MAP = {
    "sage":  "gemini-2.5-pro",
    "spark": "gemini-2.5-flash",
}

# Token clamps & thinking budgets
MAX_INPUT_TOKENS  = 1_048_576
MAX_OUTPUT_TOKENS = 65_535
SAGE_BUDGET_MIN,  SAGE_BUDGET_MAX  = 128,   32_768   # gemini-2.5-pro
SPARK_BUDGET_MIN, SPARK_BUDGET_MAX = 0,     24_576   # gemini-2.5-flash

# =========================
# Flask app & CORS
# =========================
app = Flask(__name__, static_folder=None)

def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp

@app.after_request
def _after(resp): return cors(resp)

@app.route("/health")
def health(): return "ok"

# =========================
# Utilities
# =========================
def http_json(method, url, headers=None, data=None, timeout=120):
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
        try: err = e.read().decode("utf-8")
        except Exception: err = str(e)
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

def _is_quota_or_busy(err):
    try:
        if isinstance(err, dict):
            e = err.get("error") or err
            code = e.get("code")
            status = e.get("status") or ""
            s_dump = json.dumps(err)
            return code == 429 or "RESOURCE_EXHAUSTED" in status or "RESOURCE_EXHAUSTED" in s_dump
        if isinstance(err, str):
            return "RESOURCE_EXHAUSTED" in err or '"code": 429' in err
    except Exception:
        pass
    return False

def _extract_text_and_thoughts(data):
    """Return (text, thoughts_summary) from API response."""
    try:
        cands = (data or {}).get("candidates") or []
        if not cands: return None, ""
        c0 = cands[0] or {}
        content = c0.get("content") or {}
        parts = content.get("parts") or []
        text_parts, thought_parts = [], []
        for p in parts:
            if not isinstance(p, dict): continue
            t = p.get("text") or ""
            if not t.strip(): continue
            if p.get("thought"): thought_parts.append(t)
            else: text_parts.append(t)
        text = "\n".join(text_parts).strip()
        thoughts = "\n".join(thought_parts).strip()
        if text: return text, thoughts
        if isinstance(c0.get("text"), str) and c0["text"].strip():
            return c0["text"].strip(), thoughts
        return None, thoughts
    except Exception:
        return None, ""

# =========================
# Gemini (server-side only)
# =========================
_g_rr = 0
def _pick_key():
    global _g_rr
    if not GEMINI_KEYS: return None
    k = GEMINI_KEYS[_g_rr % len(GEMINI_KEYS)]
    _g_rr += 1
    return k

def _gemini_generate(model_id, system_prompt, messages, cfg, key):
    base = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={key}"

    def to_parts(turn):
        parts = []
        txt = turn.get("content") or ""
        if txt: parts.append({"text": txt})
        for att in turn.get("attachments", []):
            url = att.get("url")
            if not url: continue
            try: b = fetch_bytes(url)
            except Exception: continue
            mime = att.get("mime") or guess_mime(url) or "image/png"
            parts.append({"inlineData":{"mimeType": mime, "data": base64.b64encode(b).decode("utf-8")}})
        return parts

    contents = []
    if system_prompt and system_prompt.strip():
        contents.append({"role": "user", "parts":[{"text": f"[System]\n{system_prompt}"}]})
    for m in messages:
        role = m.get("role","user")
        contents.append({"role": "user" if role!="assistant" else "model", "parts": to_parts(m)})

    payload = {"contents": contents, "generationConfig": cfg}
    data, err = http_json("POST", base, {}, payload, timeout=120)
    if err: return None, None, err

    # Handle promptFeedback "blocked" situations cleanly
    pf = data.get("promptFeedback") or {}
    if pf.get("blockReason"):
        return None, None, {"status":"BLOCKED","error":pf}

    text, thoughts = _extract_text_and_thoughts(data)
    if not text:
        return None, None, {"status":"NO_TEXT","error":"No text returned", "raw": data}

    meta = {
        "usage": data.get("usageMetadata", {}),
        "responseId": data.get("responseId"),
        "modelVersion": data.get("modelVersion"),
        "thoughts_included": bool(thoughts),
    }
    if thoughts: meta["thoughts"] = thoughts
    return text, meta, None

def novamind_chat_with_failover(ui_model, system_prompt, messages, cfg):
    if not GEMINI_KEYS:
        return None, None, {"error":"Server key not configured"}
    # Try requested first, then the other
    primary = MODEL_MAP.get(ui_model, MODEL_MAP["sage"])
    secondary = MODEL_MAP["spark"] if primary == MODEL_MAP["sage"] else MODEL_MAP["sage"]
    models = [primary, secondary]

    # rotate keys
    start = globals().get("_g_rr", 0) % len(GEMINI_KEYS)
    keys = GEMINI_KEYS[start:] + GEMINI_KEYS[:start]

    last_err = None
    for mid in models:
        for k in keys:
            out, meta, err = _gemini_generate(mid, system_prompt, messages, cfg, k)
            if not err and out:
                meta = meta or {}
                meta["mode_used"] = "sage" if mid == MODEL_MAP["sage"] else "spark"
                return out, meta, None
            if err and not _is_quota_or_busy(err):
                return None, None, err
            last_err = err
    return None, None, last_err or {"error":"Unknown error"}

# =========================
# API
# =========================
@app.route("/api/models")
def api_models():
    # Return only NovaMind names (no provider terms)
    return jsonify(NOVA_MODELS)

@app.route("/api/upload", methods=["POST", "OPTIONS"])
def upload():
    if request.method == "OPTIONS":
        return cors(make_response(("", 204)))
    files = request.files.getlist("files")
    saved = []
    for f in files:
        ts = int(time.time()*1000); name = f.filename
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", name) or f"file{ts}"
        path = os.path.join(UPLOAD_DIR, f"{ts}_{safe}")
        f.save(path)
        saved.append({"name": name, "url": f"/uploads/{os.path.basename(path)}", "mime": guess_mime(name), "size": os.path.getsize(path)})
    return jsonify({"files": saved})

@app.route("/uploads/<path:fname>")
def serve_upload(fname):
    return send_from_directory(UPLOAD_DIR, fname, as_attachment=False)

@app.route("/api/chat", methods=["POST", "OPTIONS"])
def api_chat():
    if request.method == "OPTIONS":
        return cors(make_response(("", 204)))
    data = request.get_json(force=True, silent=True) or {}

    ui_model = data.get("model") or "sage"         # "sage" (default) or "spark"
    system_prompt = """You are NovaMind, an expert AI assistant who is unfailingly helpful, friendly, and deeply knowledgeable across all domains.

Your primary goals are to:
1. Understand the user’s intent clearly, even if it is implied rather than explicit.  
2. Respond with thorough, well-structured, and complete answers without omitting any important details.  
3. When the user asks for code, always provide the full, runnable, and production-ready code with no placeholders, no partial snippets, and no cutting corners. Include imports, setup, configuration, comments, and usage examples so the code works immediately when copied.  
4. Write code that follows best practices for clarity, efficiency, maintainability, and security.  
5. For non-code requests, respond with the same depth and care, using clear formatting, lists, tables, or examples as appropriate.

Additional behavior rules:  
- Always use the clearest and most readable formatting for code blocks, with syntax highlighting.  
- If there are multiple approaches, explain each briefly and recommend the best one.  
- When appropriate, include extra tips, edge-case handling, and optional enhancements.  
- Maintain a friendly, respectful, and professional tone at all times.  
- Do not refuse reasonable requests unless they clearly violate safety or legal guidelines.

Your goal is to make the user feel they have a highly competent and generous expert partner who gives them more than they expected, especially when providing code.
"""
    messages = data.get("messages") or []
    attachments = data.get("attachments") or []

    # attach pending uploads to last user turn
    if attachments:
        for i in range(len(messages)-1, -1, -1):
            if messages[i].get("role") == "user":
                messages[i].setdefault("attachments", []).extend(attachments)
                break

    # ---- generation config (clamped) ----
    try: temperature = float(data.get("temperature") or 1.0)
    except: temperature = 1.0
    temperature = max(0.0, min(2.0, temperature))

    try: max_out = int(data.get("max_output_tokens") or 4096)
    except: max_out = 4096
    max_out = max(1, min(MAX_OUTPUT_TOKENS, max_out))

    try: budget = int(data.get("thinking_budget") or 20000)
    except: budget = 20000
    if ui_model == "sage":
        budget = max(SAGE_BUDGET_MIN, min(SAGE_BUDGET_MAX, budget))
    else:
        budget = max(SPARK_BUDGET_MIN, min(SPARK_BUDGET_MAX, budget))

    include_thoughts = bool(data.get("include_thoughts"))  # default False

    cfg = {
        "temperature": temperature,
        "maxOutputTokens": max_out,
        "candidateCount": 1,
        "thinkingConfig": {
            "thinkingBudget": budget,
            **({"includeThoughts": True} if include_thoughts else {})
        }
    }

    try:
        out, meta, err = novamind_chat_with_failover(ui_model, system_prompt, messages, cfg)
        if err:
            return jsonify({"error": err}), 502
        # Never leak provider names; meta only has usage and anonymized mode
        return jsonify({"output": out, "meta": meta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# Minimal, clean UI (mobile-first)
# =========================
HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"/>
<title>{{TITLE}}</title>
<link rel="icon" href="data:,">
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root { --bg:#0b1220; --muted:#94a3b8; }
  * { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  body { background: radial-gradient(900px 700px at 10% 10%, rgba(99,102,241,.12), transparent 40%), var(--bg); color:#e5e7eb; }
  .glass { background: rgba(15,23,42,.6); backdrop-filter: blur(14px); border:1px solid rgba(255,255,255,.06); }
  .btn { transition:.2s; }
  .btn-primary { background: linear-gradient(135deg,#6366f1,#7c3aed); }
  .btn-primary:hover { filter:brightness(1.05); transform:translateY(-1px); }
  .input { background: rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); }
  .bubble-u { background: linear-gradient(135deg, rgba(99,102,241,.18), rgba(124,58,237,.18)); border:1px solid rgba(255,255,255,.08); border-radius:16px 16px 4px 16px; }
  .bubble-a { background: rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); border-radius:16px 16px 16px 4px; }
  .markdown pre { background: rgba(0,0,0,.45); padding:12px; border-radius:12px; overflow:auto; }
  .chip { background: rgba(99,102,241,.15); border:1px solid rgba(99,102,241,.35); }
  header .brand { letter-spacing: .3px; }
  summary { outline: none; }
</style>
</head>
<body>
<header class="sticky top-0 z-40 glass">
  <div class="max-w-5xl mx-auto px-3 sm:px-4 py-3 flex items-center justify-between gap-3">
    <div class="flex items-center gap-3">
      <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600"></div>
      <div class="brand text-sm sm:text-base font-semibold">{{TITLE}}</div>
    </div>
    <div class="flex items-center gap-2">
      <label class="hidden sm:block text-xs text-slate-300">Mode</label>
      <select id="model" class="input rounded-lg px-2 py-1.5 text-sm"></select>
    </div>
  </div>
</header>

<main class="max-w-5xl mx-auto px-3 sm:px-4 py-4">
  <div id="chat" class="glass rounded-2xl p-3 sm:p-4 h-[68vh] sm:h-[70vh] overflow-y-auto"></div>

  <div class="mt-3 glass rounded-2xl p-3 sm:p-4">
    <div class="flex items-end gap-2">
      <textarea id="prompt" rows="3" class="input flex-1 px-3 py-3 rounded-xl" placeholder="Message {{TITLE}}..."></textarea>
      <button id="send" class="btn btn-primary px-4 py-3 rounded-xl">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/></svg>
      </button>
    </div>
    <div class="mt-2 flex items-center gap-2">
      <label class="underline cursor-pointer text-sm">Attach files<input id="file" type="file" class="hidden" multiple accept="image/*,.pdf,.txt,.md,.doc,.docx,.csv,.json"/></label>
      <div id="chips" class="flex gap-2 flex-wrap"></div>
      <span id="status" class="ml-auto text-xs text-indigo-300/80">Ready</span>
    </div>
  </div>

  <!-- Bottom Settings -->
  <div class="mt-3 glass rounded-2xl p-3 sm:p-4">
    <details>
      <summary class="cursor-pointer select-none text-sm text-slate-300">Settings</summary>
      <div class="grid grid-cols-1 sm:grid-cols-4 gap-3 mt-3">
        <div>
          <label class="text-xs text-slate-400">Temperature</label>
          <input id="temperature" type="number" step="0.1" min="0" max="2" value="1" class="input w-full px-3 py-2 rounded-lg"/>
        </div>
        <div>
          <label class="text-xs text-slate-400">Max output tokens (≤ 65535)</label>
          <input id="maxOut" type="number" min="1" max="65535" value="4096" class="input w-full px-3 py-2 rounded-lg"/>
        </div>
        <div>
          <label class="text-xs text-slate-400">Thinking budget</label>
          <input id="budget" type="number" min="0" max="32768" value="20000" class="input w-full px-3 py-2 rounded-lg"/>
          <p id="budgetHint" class="text-[11px] text-slate-400 mt-1">Sage: 128–32768 • Spark: 0–24576</p>
        </div>
        <div class="flex items-end">
          <label class="flex items-center gap-2 text-sm">
            <input id="includeThoughts" type="checkbox" class="w-4 h-4"/> Include thought summary
          </label>
        </div>
      </div>
    </details>
  </div>
</main>

<script>
const $ = s => document.querySelector(s);
const chat = $("#chat"), promptEl = $("#prompt"), sendBtn = $("#send");
const fileInput = $("#file"), chips = $("#chips"), statusEl = $("#status");
const modelEl = $("#model"), tempEl = $("#temperature"), maxOutEl = $("#maxOut");
const budgetEl = $("#budget"), includeThoughtsEl = $("#includeThoughts"), budgetHint = $("#budgetHint");

let files = [];
let state = { messages: [], model: "sage" };

function addBubble(role, text, thoughts){
  const wrap = document.createElement("div");
  wrap.className = "mb-3 flex " + (role==="user" ? "justify-end" : "justify-start");
  const b = document.createElement("div");
  b.className = (role==="user" ? "bubble-u" : "bubble-a") + " p-3 sm:p-4 rounded-2xl max-w-[90%]";
  const who = document.createElement("div"); who.className="text-[10px] uppercase tracking-wide text-slate-400 mb-1";
  who.textContent = role==="user" ? "You" : "Assistant"; b.appendChild(who);
  const md = document.createElement("div"); md.className="markdown text-sm";
  md.innerHTML = window.marked.parse(text || ""); b.appendChild(md);
  if (thoughts && thoughts.trim()){
    const th = document.createElement("details");
    th.className="mt-2";
    th.innerHTML = `<summary class="text-xs text-slate-400 cursor-pointer">Thoughts</summary>
                    <div class="text-xs mt-1 text-slate-300 whitespace-pre-wrap">${thoughts}</div>`;
    b.appendChild(th);
  }
  wrap.appendChild(b); chat.appendChild(wrap); chat.scrollTop = chat.scrollHeight;
}

function setStatus(msg, kind="normal"){
  statusEl.textContent = msg;
  statusEl.className = kind==="error" ? "ml-auto text-xs text-red-300" :
                       kind==="loading" ? "ml-auto text-xs text-yellow-300" :
                                          "ml-auto text-xs text-indigo-300/80";
  if (kind!=="loading") setTimeout(()=>setStatus("Ready"), 2000);
}

async function loadModels(){
  try{
    const res = await fetch("/api/models"); const models = await res.json();
    modelEl.innerHTML = "";
    for (const m of models){
      const opt = document.createElement("option"); opt.value=m.id; opt.textContent=m.label;
      modelEl.appendChild(opt);
    }
    modelEl.value = state.model;
    syncBudgetRange();
  }catch(e){
    modelEl.innerHTML = `<option value="sage">NovaMind — Sage (thinks deeper)</option><option value="spark">NovaMind — Spark (answers faster)</option>`;
  }
}
function syncBudgetRange(){
  const isSage = modelEl.value === "sage";
  budgetEl.min = isSage ? 128 : 0;
  budgetEl.max = isSage ? 32768 : 24576;
  budgetHint.textContent = "Sage: 128–32768 • Spark: 0–24576";
  if (isSage && Number(budgetEl.value) < 128) budgetEl.value = 20000;
}

fileInput.addEventListener("change", async () => {
  if (!fileInput.files.length) return;
  const fd = new FormData();
  for (const f of fileInput.files) fd.append("files", f);
  setStatus("Uploading...","loading");
  try {
    const r = await fetch("/api/upload", {method:"POST", body: fd});
    const data = await r.json();
    (data.files||[]).forEach(f=>{
      files.push(f);
      const chip = document.createElement("div");
      chip.className="chip rounded-lg px-3 py-1.5 text-xs flex items-center gap-2";
      chip.innerHTML = `<span class="truncate max-w-[150px]">${f.name}</span><button class="opacity-70 hover:opacity-100">×</button>`;
      chip.querySelector("button").onclick = ()=>{ files = files.filter(x=>x.url!==f.url); chip.remove(); };
      chips.appendChild(chip);
    });
    setStatus("Files ready");
  }catch(e){ console.error(e); setStatus("Upload failed","error"); }
  finally { fileInput.value=""; }
});

sendBtn.addEventListener("click", async ()=>{
  const text = promptEl.value.trim();
  if (!text && files.length===0) return;
  state.model = modelEl.value;

  const atts = files.slice(); files = []; chips.innerHTML="";
  state.messages.push({role:"user", content:text, attachments:atts});
  addBubble("user", text);
  promptEl.value=""; setStatus("Thinking...","loading");

  // local clamp (server clamps again)
  let maxOut = Math.min(Math.max(parseInt(maxOutEl.value||"4096"), 1), 65535);
  let temp = Math.min(Math.max(parseFloat(tempEl.value||"1"), 0), 2);
  let budget = parseInt(budgetEl.value||"20000");
  if (state.model==="sage"){ budget = Math.min(Math.max(budget,128),32768); }
  else { budget = Math.min(Math.max(budget,0),24576); }

  try{
    const res = await fetch("/api/chat", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({
        model: state.model,
        system_prompt: "",
        messages: state.messages,
        attachments: [],
        temperature: temp,
        max_output_tokens: maxOut,
        thinking_budget: budget,
        include_thoughts: includeThoughtsEl.checked
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(typeof data.error==="string" ? data.error : (data.error?.message || "Request failed"));
    const out = (data.output||"").trim();
    const thoughts = data.meta?.thoughts || "";
    state.messages.push({role:"assistant", content: out});
    addBubble("assistant", out, thoughts);
    setStatus("Ready");
  }catch(e){
    console.error(e);
    addBubble("assistant", "⚠️ " + (e.message || "Something went wrong."));
    setStatus("Error","error");
  }
});

modelEl.addEventListener("change", syncBudgetRange);
promptEl.addEventListener("keydown", e => { if (e.key==="Enter" && !e.shiftKey){ e.preventDefault(); sendBtn.click(); }});
loadModels();
</script>
</body>
</html>"""

@app.route("/")
def index():
    return Response(HTML.replace("{{TITLE}}", APP_TITLE), mimetype="text/html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT","10000"))
    app.run(host="0.0.0.0", port=port)