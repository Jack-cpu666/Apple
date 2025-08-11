# app.py
import os, base64, json, mimetypes, time, re, tempfile
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from flask import Flask, request, send_from_directory, make_response, jsonify, Response

APP_TITLE = "Gemini Chat — 2.5 Pro & 2.5 Flash"
UPLOAD_DIR = os.environ.get("UPLOAD_DIR") or os.path.join(tempfile.gettempdir(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----- Keys (Gemini only) -----
GEMINI_KEYS = [k for k in [os.environ.get("GEMINI_KEY_1"), os.environ.get("GEMINI_KEY_2")] if k]

# ----- Models (fixed & simple) -----
GEMINI_MODELS = [
    {"id": "gemini-2.5-pro",   "label": "Gemini 2.5 Pro (reasoning)"},
    {"id": "gemini-2.5-flash", "label": "Gemini 2.5 Flash (fast)"},
]

# Official limits (Vertex/Gemini API docs)
MAX_INPUT_TOKENS     = 1_048_576  # 1M context window
MAX_OUTPUT_TOKENS    = 65_535     # default hard max
# Thinking budgets by model (docs)
PRO_BUDGET_MIN,  PRO_BUDGET_MAX  = 128,   32_768
FLASH_BUDGET_MIN, FLASH_BUDGET_MAX = 0,     24_576

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

# ---------- Utils ----------
def http_json(method, url, headers=None, data=None, timeout=120):
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = Request(url, data=body, method=method.upper())
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items(): req.add_header(k, v)
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

# ---------- Gemini-only client ----------
_g_rr = 0
def _pick_key():
    global _g_rr
    if not GEMINI_KEYS: return None
    k = GEMINI_KEYS[_g_rr % len(GEMINI_KEYS)]
    _g_rr += 1
    return k

def _is_quota429(err):
    try:
        if isinstance(err, dict):
            e = err.get("error") or err
            return e.get("status") == "RESOURCE_EXHAUSTED" or e.get("code") == 429 or "RESOURCE_EXHAUSTED" in json.dumps(err)
        if isinstance(err, str):
            return "RESOURCE_EXHAUSTED" in err or '"code": 429' in err
    except Exception:
        pass
    return False

def _extract_text_and_thoughts(data):
    """Return (text, thoughts_summary) where thoughts_summary may be ''."""
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
        # very rare fallback
        if isinstance(c0.get("text"), str) and c0["text"].strip():
            return c0["text"].strip(), thoughts
        return None, thoughts
    except Exception:
        return None, ""

def _gemini_generate(model, system_prompt, messages, cfg, key):
    """One attempt with a specific model/key."""
    base = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

    def to_parts(turn):
        parts = []
        txt = turn.get("content") or ""
        if txt: parts.append({"text": txt})
        for att in turn.get("attachments", []):
            url = att.get("url"); 
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
    text, thoughts = _extract_text_and_thoughts(data)
    if not text:
        return None, None, {"status":"NO_TEXT","error":"Gemini returned no text","raw":data}
    meta = {
        "provider": "gemini",
        "model": model,
        "usage": data.get("usageMetadata", {}),
        "modelVersion": data.get("modelVersion"),
        "responseId": data.get("responseId"),
        "thoughts_included": bool(thoughts),
    }
    if thoughts: meta["thoughts"] = thoughts
    return text, meta, None

def gemini_chat_with_failover(model, system_prompt, messages, cfg):
    """Try model/key combos: requested model first, then the other; rotate keys."""
    if not GEMINI_KEYS:
        return None, None, {"error":"Gemini API key not configured"}
    # model order: requested -> the other
    models = [model] if model else []
    for m in ["gemini-2.5-pro","gemini-2.5-flash"]:
        if m not in models: models.append(m)

    # key order: start from round-robin index
    start = globals().get("_g_rr", 0) % len(GEMINI_KEYS)
    keys = GEMINI_KEYS[start:] + GEMINI_KEYS[:start]

    last_err = None
    for m in models:
        for k in keys:
            out, meta, err = _gemini_generate(m, system_prompt, messages, cfg, k)
            if not err and out:
                meta = meta or {}; meta["model_used"] = m; meta["key_index"] = keys.index(k)
                return out, meta, None
            if err and not _is_quota429(err):
                return None, None, err
            last_err = err
    return None, None, last_err or {"error":"Unknown error"}

# ---------- API ----------
@app.route("/api/models")
def api_models():
    return jsonify(GEMINI_MODELS)

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

    model = data.get("model") or "gemini-2.5-pro"
    system_prompt = data.get("system_prompt") or ""
    messages = data.get("messages") or []
    attachments = data.get("attachments") or []
    # normalize: attach pending uploads to last user turn
    if attachments:
        for i in range(len(messages)-1, -1, -1):
            if messages[i].get("role") == "user":
                messages[i].setdefault("attachments", []).extend(attachments)
                break

    # ---- generation config (with clamps) ----
    temperature = float(data.get("temperature") or 1.0)
    # output tokens clamp
    try: max_out = int(data.get("max_output_tokens") or 4096)
    except: max_out = 4096
    max_out = max(1, min(MAX_OUTPUT_TOKENS, max_out))

    # thinking budget clamp by model
    try: budget = int(data.get("thinking_budget") or 20000)
    except: budget = 20000
    if model == "gemini-2.5-pro":
        budget = max(PRO_BUDGET_MIN, min(PRO_BUDGET_MAX, budget))
    else:
        budget = max(FLASH_BUDGET_MIN, min(FLASH_BUDGET_MAX, budget))

    include_thoughts = bool(data.get("include_thoughts"))  # default false

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
        out, meta, err = gemini_chat_with_failover(model, system_prompt, messages, cfg)
        if err:
            return jsonify({"error": err}), 502
        return jsonify({"output": out, "meta": meta})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- UI (minimal: top model selector, bottom settings) ----------
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
  :root { --bg:#0b1220; --card:#0f172a; --muted:#94a3b8; --ring:rgba(99,102,241,.5); }
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
  .settings { position: sticky; bottom: 0; }
</style>
</head>
<body>
<header class="sticky top-0 z-40 glass">
  <div class="max-w-5xl mx-auto px-3 sm:px-4 py-3 flex items-center justify-between gap-3">
    <div class="flex items-center gap-3">
      <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600"></div>
      <div class="text-sm sm:text-base font-semibold">{{TITLE}}</div>
    </div>
    <div class="flex items-center gap-2">
      <label class="hidden sm:block text-xs text-slate-300">Model</label>
      <select id="model" class="input rounded-lg px-2 py-1.5 text-sm"></select>
    </div>
  </div>
</header>

<main class="max-w-5xl mx-auto px-3 sm:px-4 py-4">
  <div id="chat" class="glass rounded-2xl p-3 sm:p-4 h-[68vh] sm:h-[70vh] overflow-y-auto"></div>

  <div class="mt-3 glass rounded-2xl p-3 sm:p-4">
    <div class="flex items-end gap-2">
      <textarea id="prompt" rows="3" class="input flex-1 px-3 py-3 rounded-xl" placeholder="Message Gemini..."></textarea>
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
  <div class="settings mt-3 glass rounded-2xl p-3 sm:p-4">
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
          <p id="budgetHint" class="text-[11px] text-slate-400 mt-1">Pro: 128–32768 • Flash: 0–24576</p>
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
let state = { messages: [], model: "gemini-2.5-pro" };

function addBubble(role, text, thoughts){
  const wrap = document.createElement("div");
  wrap.className = "mb-3 flex " + (role==="user" ? "justify-end" : "justify-start");
  const b = document.createElement("div");
  b.className = (role==="user" ? "bubble-u" : "bubble-a") + " p-3 sm:p-4 rounded-2xl max-w-[90%]";
  const who = document.createElement("div"); who.className="text-[10px] uppercase tracking-wide text-slate-400 mb-1";
  who.textContent = role==="user" ? "You" : "Assistant"; b.appendChild(who);
  const md = document.createElement("div"); md.className="markdown text-sm";
  md.innerHTML = marked.parse(text || ""); b.appendChild(md);
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
    modelEl.innerHTML = `<option value="gemini-2.5-pro">Gemini 2.5 Pro</option><option value="gemini-2.5-flash">Gemini 2.5 Flash</option>`;
  }
}
function syncBudgetRange(){
  const isPro = modelEl.value === "gemini-2.5-pro";
  budgetEl.min = isPro ? 128 : 0;
  budgetEl.max = isPro ? 32768 : 24576;
  budgetHint.textContent = "Pro: 128–32768 • Flash: 0–24576";
  if (isPro && Number(budgetEl.value) < 128) budgetEl.value = 20000;
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
  // push user turn
  const atts = files.slice(); files = []; chips.innerHTML="";
  state.messages.push({role:"user", content:text, attachments:atts});
  addBubble("user", text);
  promptEl.value=""; setStatus("Thinking...","loading");

  // clamp outputs & budget client-side (server clamps again)
  let maxOut = Math.min(Math.max(parseInt(maxOutEl.value||"4096"), 1), 65535);
  let temp = Math.min(Math.max(parseFloat(tempEl.value||"1"), 0), 2);
  let budget = parseInt(budgetEl.value||"20000");
  if (state.model==="gemini-2.5-pro"){ budget = Math.min(Math.max(budget,128),32768); }
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