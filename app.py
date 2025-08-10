# app.py
# One-file multi-model chat + Veo 3 video app for Render.com
# Providers: OpenAI (GPT-5, GPT-5-mini), Anthropic (Opus 4.1, Sonnet 4), Google (Gemini 2.5 Pro/Flash), Veo 3
# Keys: 
#  - Gemini uses SERVER-SIDE FAILOVER with GEMINI_KEY_1 / GEMINI_KEY_2 (no password required)
#  - OpenAI & Anthropic can use SERVER KEYS if the correct PRO_PASSWORD is supplied in the UI.
#  - Users may still supply their own keys client-side (saved to localStorage). 
#
# SECURITY: Do NOT hardcode secrets in source. Put them in Render Environment Variables.
#   PRO_PASSWORD            (e.g., AEaeAEae123@)
#   OPENAI_API_KEY_SERVER   (server OpenAI key)
#   ANTHROPIC_API_KEY_SERVER(server Anthropic key)
#   GEMINI_KEY_1            (first Gemini key)
#   GEMINI_KEY_2            (second Gemini key)

import os, json, base64, time, io, re
from datetime import datetime
from flask import Flask, request, jsonify, Response, send_file, make_response
import requests

app = Flask(__name__)

# ----------------------------
# Model catalog & limits (adjust if providers change)
# ----------------------------
MODEL_LIMITS = {
    # OpenAI (Responses API)
    "openai": {
        "gpt-5": {"context": 400_000, "max_output": 128_000, "display": "GPT-5 (expensive)"},
        "gpt-5-mini": {"context": 400_000, "max_output": 128_000, "display": "GPT-5 mini (cheap)"},
    },
    # Anthropic
    "anthropic": {
        "claude-opus-4-1-20250805": {"context": 200_000, "max_output": 32_000, "display": "Claude Opus 4.1"},
        "claude-sonnet-4-20250514": {"context": 200_000, "max_output": 64_000, "display": "Claude Sonnet 4"},
    },
    # Google Gemini
    "gemini": {
        "gemini-2.5-pro": {"context": 1_048_576, "max_output": 65_536, "display": "Gemini 2.5 Pro"},
        "gemini-2.5-flash": {"context": 1_048_576, "max_output": 65_536, "display": "Gemini 2.5 Flash"},
    },
}

VEO3_MODEL = "veo-3.0-generate-preview"  # Gemini API video model (preview LRO)

# ----------------------------
# Utilities
# ----------------------------

def _safe_int(v, default):
    try:
        return int(v)
    except Exception:
        return default


def _strip_data_url(data_url):
    """Convert data URL -> (mime, base64_str)"""
    if not data_url:
        return None, None
    m = re.match(r"^data:(.*?);base64,(.*)$", data_url)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _last_user_text(messages):
    for m in reversed(messages or []):
        if m.get("role") == "user":
            for p in reversed(m.get("content", [])):
                if p.get("type") in ("text", "input_text") and p.get("text"):
                    return p["text"]
    return ""


def _summarize_cse_results(items, limit=3):
    out = []
    for it in (items or [])[:limit]:
        title = (it.get("title") or "").strip()
        snippet = (it.get("snippet") or "").strip()
        link = it.get("link") or ""
        chunk = []
        if title:
            chunk.append(f"• {title}")
        if snippet:
            chunk.append(f"  {snippet}")
        if link:
            chunk.append(f"  {link}")
        out.append("\n".join(chunk))
    return ("Web search results (Google CSE):\n" + "\n\n".join(out)) if out else ""


# ----------------------------
# Web Search (Google CSE)
# ----------------------------

@app.post("/api/search")
def api_search():
    data = request.get_json(force=True, silent=True) or {}
    q = (data.get("q") or "").strip()
    key = (data.get("cse_key") or "").strip()
    cx = (data.get("cse_cx") or "").strip()
    if not q or not key or not cx:
        return jsonify({"ok": False, "error": "Missing q, cse_key, or cse_cx"}), 400
    try:
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": key, "cx": cx, "q": q},
            timeout=20,
        )
        r.raise_for_status()
        js = r.json()
        return jsonify({"ok": True, "items": js.get("items", []) or []})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ----------------------------
# Provider adapters
# ----------------------------

def call_openai(api_key, model, system_text, messages, max_output, temperature):
    """OpenAI Responses API (vision-capable via data URLs)."""
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    input_msgs = []
    if system_text:
        input_msgs.append({"role": "system", "content": [{"type": "input_text", "text": system_text}]})
    for m in messages or []:
        role = m.get("role", "user")
        parts = []
        for p in m.get("content", []):
            if p.get("type") in ("text", "input_text"):
                parts.append({"type": "input_text", "text": p.get("text", "")})
            elif p.get("type") in ("image", "input_image"):
                data_url = p.get("dataUrl") or p.get("image_url")
                if data_url:
                    parts.append({"type": "input_image", "image_url": data_url})
        if parts:
            input_msgs.append({"role": role, "content": parts})

    body = {
        "model": model,
        "input": input_msgs,
        "max_output_tokens": max_output,
        "temperature": float(temperature),
    }

    r = requests.post(url, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    js = r.json()

    text = None
    if "output_text" in js:
        text = js["output_text"]
    elif isinstance(js.get("response"), dict):
        text = js["response"].get("output_text")
    elif "content" in js:
        parts = js.get("content", []) or []
        buf = []
        for p in parts:
            if isinstance(p, dict) and p.get("type") in ("output_text", "text"):
                buf.append(p.get("text", ""))
        text = "\n".join(buf).strip()

    if not text:
        text = json.dumps(js)[:5000]

    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def call_anthropic(api_key, model, system_text, messages, max_output, temperature):
    """Anthropic Messages API (vision-capable)."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    anthro_msgs = []
    for m in messages or []:
        role = "user" if m.get("role") == "user" else "assistant"
        content = []
        for p in m.get("content", []):
            if p.get("type") in ("text", "input_text"):
                content.append({"type": "text", "text": p.get("text", "")})
            elif p.get("type") in ("image", "input_image"):
                data_url = p.get("dataUrl")
                if not data_url:
                    continue
                mime, b64 = _strip_data_url(data_url)
                if not mime or not b64:
                    continue
                content.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}})
        if content:
            anthro_msgs.append({"role": role, "content": content})

    body = {"model": model, "max_tokens": max_output, "temperature": float(temperature)}
    if system_text:
        body["system"] = system_text
    body["messages"] = anthro_msgs

    r = requests.post(url, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    js = r.json()
    text_chunks = []
    for c in js.get("content", []) or []:
        if c.get("type") == "text":
            text_chunks.append(c.get("text", ""))
    text = "\n".join(text_chunks).strip() or json.dumps(js)[:5000]
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def call_gemini(api_key, model, system_text, messages, max_output, temperature):
    """Gemini REST v1 models:generateContent (vision-capable via inline_data)."""
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key, "content-type": "application/json"}

    contents = []
    system_instruction = {"parts": [{"text": system_text}]} if system_text else None

    for m in messages or []:
        parts = []
        for p in m.get("content", []):
            if p.get("type") in ("text", "input_text"):
                parts.append({"text": p.get("text", "")})
            elif p.get("type") in ("image", "input_image"):
                data_url = p.get("dataUrl")
                if not data_url:
                    continue
                mime, b64 = _strip_data_url(data_url)
                if not mime or not b64:
                    continue
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})
        if parts:
            contents.append({"role": m.get("role", "user"), "parts": parts})

    body = {"contents": contents, "generationConfig": {"temperature": float(temperature), "maxOutputTokens": max_output}}
    if system_instruction:
        body["systemInstruction"] = system_instruction

    r = requests.post(url, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    js = r.json()

    text = ""
    for cand in js.get("candidates", []) or []:
        parts = cand.get("content", {}).get("parts", []) or []
        for p in parts:
            if "text" in p:
                text += p["text"]
    text = text.strip() or json.dumps(js)[:5000]
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


# ----------------------------
# Chat endpoint with server-key policy & Gemini failover
# ----------------------------

@app.post("/api/chat")
def api_chat():
    data = request.get_json(force=True, silent=True) or {}
    provider = data.get("provider")
    model = data.get("model")
    system_text = data.get("system") or ""
    messages = data.get("messages") or []
    temperature = data.get("temperature", 0.7)
    max_output = _safe_int(data.get("max_output"), 1024)
    user_keys = data.get("keys") or {}
    search_cfg = data.get("search") or {}
    pro_password = (data.get("pro_password") or "").strip()

    # Optional: Google CSE grounding
    if search_cfg.get("enabled"):
        q = _last_user_text(messages)
        cse_key, cse_cx = (search_cfg.get("cse_key") or ""), (search_cfg.get("cse_cx") or "")
        if q and cse_key and cse_cx:
            try:
                r = requests.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params={"key": cse_key, "cx": cse_cx, "q": q},
                    timeout=20,
                )
                r.raise_for_status()
                items = (r.json().get("items", []) or [])[:3]
                search_note = _summarize_cse_results(items, limit=3)
                if search_note:
                    system_text = (system_text + "\n\n" if system_text else "") + search_note
            except Exception:
                pass

    try:
        if provider == "openai":
            # Use server key only if password matches, else require user key
            server_key = os.environ.get("OPENAI_API_KEY_SERVER", "").strip()
            allow_server = bool(server_key) and pro_password == os.environ.get("PRO_PASSWORD", "").strip()
            api_key = (user_keys.get("openai") or "").strip() or (server_key if allow_server else "")
            if not api_key:
                return jsonify({"ok": False, "error": "Missing OpenAI API key"}), 400
            out = call_openai(api_key, model, system_text, messages, max_output, temperature)
            return jsonify({"ok": True, "message": out})

        elif provider == "anthropic":
            server_key = os.environ.get("ANTHROPIC_API_KEY_SERVER", "").strip()
            allow_server = bool(server_key) and pro_password == os.environ.get("PRO_PASSWORD", "").strip()
            api_key = (user_keys.get("anthropic") or "").strip() or (server_key if allow_server else "")
            if not api_key:
                return jsonify({"ok": False, "error": "Missing Anthropic API key"}), 400
            out = call_anthropic(api_key, model, system_text, messages, max_output, temperature)
            return jsonify({"ok": True, "message": out})

        elif provider == "gemini":
            # Gemini: server-side failover keys if user didn't provide one
            user_key = (user_keys.get("gemini") or "").strip()
            if user_key:
                out = call_gemini(user_key, model, system_text, messages, max_output, temperature)
                return jsonify({"ok": True, "message": out})

            k1 = os.environ.get("GEMINI_KEY_1", "").strip()
            k2 = os.environ.get("GEMINI_KEY_2", "").strip()
            if not k1 and not k2:
                return jsonify({"ok": False, "error": "No Gemini server keys configured"}), 400

            # Try k1 then k2 on quota/auth errors
            errors = []
            for key in [k1, k2]:
                if not key:
                    continue
                try:
                    out = call_gemini(key, model, system_text, messages, max_output, temperature)
                    return jsonify({"ok": True, "message": out, "used": "server"})
                except requests.HTTPError as he:
                    status = he.response.status_code if he.response else None
                    if status in (401, 403, 429, 500):
                        errors.append(f"{status}")
                        continue
                    else:
                        raise
            return jsonify({"ok": False, "error": f"Gemini failed on all server keys ({', '.join(errors) or 'no status'})"}), 502

        else:
            return jsonify({"ok": False, "error": "Unknown provider"}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ----------------------------
# Veo 3 video generation (preview LRO)
# ----------------------------

@app.post("/api/veo3/start")
def api_veo3_start():
    data = request.get_json(force=True, silent=True) or {}
    gemini_key = (data.get("gemini_key") or "").strip() or os.environ.get("GEMINI_KEY_1", "").strip()
    prompt = (data.get("prompt") or "").strip()
    aspect_ratio = (data.get("aspect_ratio") or "16:9").strip()
    negative_prompt = (data.get("negative_prompt") or "").strip()
    image_data_url = data.get("image_data_url")

    if not gemini_key or not prompt:
        return jsonify({"ok": False, "error": "gemini_key and prompt required"}), 400

    headers = {"x-goog-api-key": gemini_key, "content-type": "application/json"}
    url = f"https://generativelanguage.googleapis.com/v1/models/{VEO3_MODEL}:predictLongRunning"

    instances = [{"prompt": prompt}]
    if image_data_url:
        mime, b64 = _strip_data_url(image_data_url)
        if mime and b64:
            instances[0]["image"] = {"imageBytes": b64, "mimeType": mime}

    payload = {"instances": instances, "parameters": {"aspectRatio": aspect_ratio}}
    if negative_prompt:
        payload["parameters"]["negativePrompt"] = negative_prompt

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        js = r.json()
        op_name = js.get("name") or js.get("operation", {}).get("name")
        if not op_name:
            return jsonify({"ok": False, "error": "No operation name returned", "raw": js}), 500
        return jsonify({"ok": True, "operation": op_name})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/veo3/poll")
def api_veo3_poll():
    gemini_key = (request.args.get("gemini_key") or "").strip() or os.environ.get("GEMINI_KEY_1", "").strip()
    operation = (request.args.get("operation") or "").strip()
    if not gemini_key or not operation:
        return jsonify({"ok": False, "error": "gemini_key and operation required"}), 400

    headers = {"x-goog-api-key": gemini_key}
    url = f"https://generativelanguage.googleapis.com/v1/{operation}"
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        js = r.json()
        done = bool(js.get("done"))
        resp = js.get("response", {})
        video_info = None
        if done:
            gv = (resp.get("generatedVideos") or [])
            if gv:
                video_info = gv[0].get("video")
        return jsonify({"ok": True, "done": done, "video": video_info, "raw": (None if done else js)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/veo3/download")
def api_veo3_download():
    gemini_key = (request.args.get("gemini_key") or "").strip() or os.environ.get("GEMINI_KEY_1", "").strip()
    file_name = (request.args.get("file") or "").strip()
    if not gemini_key or not file_name:
        return jsonify({"ok": False, "error": "gemini_key and file required"}), 400

    headers = {"x-goog-api-key": gemini_key}
    url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}:download"
    try:
        r = requests.get(url, headers=headers, timeout=120)
        if r.status_code == 200 and (r.headers.get("Content-Type", "").startswith("video")):
            return Response(r.content, mimetype=r.headers.get("Content-Type"))
        else:
            return jsonify({"ok": True, "note": "Direct download not available via API; returning metadata.", "raw_headers": dict(r.headers), "status": r.status_code})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ----------------------------
# Frontend (single HTML)
# ----------------------------

INDEX_HTML_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Multi-Model Chat (GPT-5 / Claude / Gemini) + Veo 3</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://unpkg.com/lucide@latest"></script>
<style>
  :root { --composer-height: 140px; }
  html, body { height: 100%; }
  body { background: #0b0f17; color: #eef2f7; }
  .app-grid { display: grid; grid-template-columns: 320px 1fr; height: 100vh; overflow: hidden; }
  .sidebar { background: #0e1420; border-right: 1px solid rgba(255,255,255,0.05); }
  .chat-wrap { display: grid; grid-template-rows: auto 1fr auto; height: 100vh; }
  .chat-scroll { overflow-y: auto; padding-bottom: var(--composer-height); scroll-behavior: smooth; }
  .msg { max-width: 900px; }
  .msg p { white-space: pre-wrap; }
  .bubble { border-radius: 18px; padding: 12px 14px; line-height: 1.55; box-shadow: 0 4px 16px rgba(0,0,0,0.25); }
  .bubble-user { background: #1a2234; }
  .bubble-assistant { background: #121a2a; border: 1px solid rgba(255,255,255,0.06); }
  .composer { position: fixed; left: 320px; right: 0; bottom: 0; background: rgba(10,14,22,0.9); border-top: 1px solid rgba(255,255,255,0.08); backdrop-filter: blur(6px); }
  .textarea { border-radius: 14px; background: #0f1726; border: 1px solid rgba(255,255,255,0.08); width: 100%; padding: 12px 14px; min-height: 56px; max-height: 240px; overflow: auto; resize: none; }
  .pill { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08); }
  .tab-active { background: #0a1222; border-bottom: 2px solid #60a5fa; }
  .thumb { max-height: 160px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);} 
  .sidebar input, .sidebar select { outline: none; }
</style>
</head>
<body>
<div class="app-grid">
  <!-- Sidebar -->
  <aside class="sidebar p-4 flex flex-col gap-4">
    <div class="flex items-center gap-2">
      <svg class="text-blue-400" width="28" height="28"><use href="#icon-spark"/></svg>
      <div>
        <div class="font-semibold">AI Studio</div>
        <div class="text-xs text-slate-400">Chat • Tools • Video (Veo 3)</div>
      </div>
    </div>

    <div class="flex gap-2">
      <button id="newChatBtn" class="px-3 py-2 rounded-xl bg-blue-600 hover:bg-blue-500">New chat</button>
      <button id="clearAllBtn" class="px-3 py-2 rounded-xl pill">Clear all</button>
    </div>

    <div class="space-y-2">
      <div class="text-xs uppercase tracking-wide text-slate-400">Provider</div>
      <div class="grid grid-cols-3 gap-2" id="providerTabs">
        <button data-p="openai" class="py-2 rounded-xl pill">OpenAI</button>
        <button data-p="anthropic" class="py-2 rounded-xl pill">Claude</button>
        <button data-p="gemini" class="py-2 rounded-xl pill">Gemini</button>
      </div>

      <div class="mt-2">
        <label class="text-xs text-slate-400">Model</label>
        <select id="modelSelect" class="w-full mt-1 bg-[#0f1726] border border-slate-700 rounded-xl px-3 py-2"></select>
        <div id="limitInfo" class="text-[11px] text-slate-400 mt-1"></div>
      </div>

      <div class="mt-3">
        <details class="bg-[#0f1726] border border-slate-700 rounded-xl p-3" open>
          <summary class="cursor-pointer">Keys, Password & Tools</summary>
          <div class="space-y-2 mt-2">
            <div>
              <label class="text-xs text-slate-400">OpenAI key (optional)</label>
              <input id="keyOpenAI" class="w-full bg-[#0b1120] border border-slate-700 rounded-lg px-2 py-2" placeholder="sk-..." />
            </div>
            <div>
              <label class="text-xs text-slate-400">Anthropic key (optional)</label>
              <input id="keyAnthropic" class="w-full bg-[#0b1120] border border-slate-700 rounded-lg px-2 py-2" placeholder="anthropic-key..." />
            </div>
            <div>
              <label class="text-xs text-slate-400">Gemini key (optional)</label>
              <input id="keyGemini" class="w-full bg-[#0b1120] border border-slate-700 rounded-lg px-2 py-2" placeholder="AIza..." />
            </div>
            <div>
              <label class="text-xs text-slate-400">Pro password (for server OpenAI/Claude keys)</label>
              <input id="proPass" type="password" class="w-full bg-[#0b1120] border border-slate-700 rounded-lg px-2 py-2" placeholder="enter password" />
            </div>

            <div class="pt-2 border-t border-slate-800"></div>

            <div class="flex items-center gap-2">
              <input id="searchToggle" type="checkbox" class="accent-blue-500">
              <label for="searchToggle">Enable Google Search tool</label>
            </div>
            <div class="grid grid-cols-2 gap-2">
              <input id="cseKey" class="bg-[#0b1120] border border-slate-700 rounded-lg px-2 py-2" placeholder="CSE API key" />
              <input id="cseCx"  class="bg-[#0b1120] border border-slate-700 rounded-lg px-2 py-2" placeholder="CSE CX id" />
            </div>

            <div class="flex items-center gap-2 pt-2">
              <input id="rememberKeys" type="checkbox" class="accent-blue-500" checked>
              <label for="rememberKeys" class="text-sm">Remember keys on this device</label>
            </div>

            <button id="saveKeysBtn" class="w-full mt-1 py-2 rounded-xl bg-slate-700 hover:bg-slate-600">Save</button>
          </div>
        </details>
      </div>

      <div class="mt-3">
        <div class="text-xs uppercase tracking-wide text-slate-400 mb-2">Chats</div>
        <div id="chatList" class="space-y-1 overflow-auto" style="max-height: calc(100vh - 560px)"></div>
      </div>
    </div>
  </aside>

  <!-- Main -->
  <main class="chat-wrap">
    <div class="px-6 pt-4 flex items-center gap-3">
      <button data-tab="chat" class="tab-btn px-3 py-2 rounded-t-xl tab-active">Chat</button>
      <button data-tab="veo"  class="tab-btn px-3 py-2 rounded-t-xl">Veo 3 (video)</button>
    </div>

    <!-- Chat area -->
    <section id="tab-chat" class="px-6">
      <div id="messages" class="chat-scroll space-y-4"></div>
    </section>

    <!-- Veo area -->
    <section id="tab-veo" class="hidden px-6">
      <div class="space-y-4">
        <div class="grid md:grid-cols-[1fr_320px] gap-4">
          <div>
            <label class="text-xs text-slate-400">Prompt</label>
            <textarea id="veoPrompt" class="textarea mt-1" rows="4" placeholder="Cinematic shot of ... with audio cues ('\"dialogue\"', ambient SFX)"></textarea>
            <div class="grid grid-cols-3 gap-2 mt-2">
              <select id="veoAspect" class="bg-[#0f1726] border border-slate-700 rounded-xl px-3 py-2">
                <option value="16:9">16:9</option>
              </select>
              <input id="veoNeg" class="bg-[#0f1726] border border-slate-700 rounded-xl px-3 py-2" placeholder="negative prompt (optional)">
              <label class="flex items-center gap-2 text-sm"><input id="veoUseImage" type="checkbox" class="accent-blue-500"> Use starter image</label>
            </div>
            <div id="veoImageWrap" class="hidden mt-2">
              <input id="veoImage" type="file" accept="image/*">
            </div>
          </div>
          <div class="bg-[#0f1726] border border-slate-700 rounded-xl p-3">
            <div class="text-sm text-slate-400 mb-2">Gemini key</div>
            <input id="veoKey" class="w-full bg-[#0b1120] border border-slate-700 rounded-lg px-2 py-2" placeholder="AIza..." />
            <button id="veoStart" class="w-full mt-3 py-2 rounded-xl bg-blue-600 hover:bg-blue-500">Generate 8s video</button>
            <div id="veoStatus" class="text-sm text-slate-400 mt-2"></div>
            <video id="veoPlayer" class="w-full mt-3 rounded-xl border border-slate-700" controls></video>
          </div>
        </div>
      </div>
    </section>

    <!-- Composer -->
    <div class="composer p-4">
      <div class="max-w-[1100px] mx-auto">
        <div class="flex items-center gap-2 text-sm text-slate-400 mb-2">
          <span>Temperature</span>
          <input id="temp" type="range" min="0" max="2" step="0.1" value="0.7" class="w-40">
          <span>Max output</span>
          <input id="maxOut" type="number" class="w-28 bg-[#0f1726] border border-slate-700 rounded-lg px-2 py-1" value="2048">
          <span class="text-xs text-slate-500" id="limitBadge"></span>
        </div>

        <div class="bg-[#0f1726] border border-slate-700 rounded-2xl p-3">
          <div class="flex items-center gap-2 flex-wrap">
            <button id="attachBtn" class="px-3 py-1 rounded-xl pill flex items-center gap-1">
              <i data-lucide="paperclip" class="w-4 h-4"></i> Attach
            </button>
            <input id="fileInput" type="file" class="hidden" accept="image/*,.txt,.md,.pdf">
            <div id="attachPreview" class="flex gap-2 flex-wrap"></div>
          </div>
          <textarea id="inputBox" class="textarea mt-3" rows="2" placeholder="Ask anything... (Shift+Enter for newline)"></textarea>
          <div class="flex justify-between items-center mt-3">
            <input id="systemPrompt" class="w-[70%] bg-[#0b1120] border border-slate-700 rounded-xl px-3 py-2"
              placeholder="System prompt (optional). Default is helpful, smart, safe." />
            <button id="sendBtn" class="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500">Send</button>
          </div>
        </div>
      </div>
    </div>
  </main>
</div>

<!-- Icons -->
<svg style="display:none" xmlns="http://www.w3.org/2000/svg">
  <symbol id="icon-spark" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2l2.2 5.6L20 10l-5.6 2.2L12 18l-2.4-5.8L4 10l5.6-2.4L12 2z"/>
  </symbol>
</svg>

<script>
const DEFAULT_SYSTEM = "You are a helpful, precise, friendly assistant. Ask for clarifications only when needed. Always think step-by-step but keep answers concise unless the user requests detail. If a web search is provided in system context, use it to ground facts and cite urls inline.";
const PROVIDERS = {
  openai: { name: "OpenAI", models: { "gpt-5": "GPT-5 (expensive)", "gpt-5-mini": "GPT-5 mini (cheap)" } },
  anthropic: { name: "Claude", models: { "claude-opus-4-1-20250805": "Claude Opus 4.1", "claude-sonnet-4-20250514": "Claude Sonnet 4" } },
  gemini: { name: "Gemini", models: { "gemini-2.5-pro": "Gemini 2.5 Pro", "gemini-2.5-flash": "Gemini 2.5 Flash" } },
};
const MODEL_LIMITS = __MODEL_LIMITS_JSON__;

let current = { provider: "gemini", model: "gemini-2.5-pro", chatId: null, tab: "chat" };
let chats = []; // [{id, title, provider, model, messages, created}]
let attachments = []; // {name, mime, dataUrl}
let keys = { openai:"", anthropic:"", gemini:"" };
let tool = { searchEnabled: false, cse_key:"", cse_cx:"" };

function elm(id){ return document.getElementById(id); }

function loadStorage(){
  try{
    const savedChats = localStorage.getItem("mm_chats_v1");
    if (savedChats) chats = JSON.parse(savedChats);
    const savedKeys = localStorage.getItem("mm_keys_v1");
    if (savedKeys) { keys = {...keys, ...JSON.parse(savedKeys)}; elm("keyOpenAI").value = keys.openai||""; elm("keyAnthropic").value = keys.anthropic||""; elm("keyGemini").value = keys.gemini||""; elm("veoKey").value = keys.gemini||""; }
    const savedTool = localStorage.getItem("mm_tool_v1");
    if (savedTool) { tool = {...tool, ...JSON.parse(savedTool)}; elm("searchToggle").checked = !!tool.searchEnabled; elm("cseKey").value = tool.cse_key||""; elm("cseCx").value = tool.cse_cx||""; }
  }catch(e){}
}
function saveChats(){ localStorage.setItem("mm_chats_v1", JSON.stringify(chats)); }
function saveKeysIfAllowed(){ if (elm("rememberKeys").checked) localStorage.setItem("mm_keys_v1", JSON.stringify(keys)); }
function saveTool(){ localStorage.setItem("mm_tool_v1", JSON.stringify(tool)); }

function setProvider(p){
  current.provider = p;
  document.querySelectorAll("#providerTabs button").forEach(b=>{ if (b.dataset.p===p) b.classList.add("bg-blue-600"); else b.classList.remove("bg-blue-600"); });
  fillModelSelect(); updateLimitInfo();
}
function fillModelSelect(){
  const sel = elm("modelSelect"); sel.innerHTML = ""; const map = PROVIDERS[current.provider].models;
  for (const [val, label] of Object.entries(map)) { const opt = document.createElement("option"); opt.value = val; opt.textContent = label; sel.appendChild(opt); }
  if (current.provider==="openai") current.model = "gpt-5-mini";
  if (current.provider==="anthropic") current.model = "claude-sonnet-4-20250514";
  if (current.provider==="gemini") current.model = "gemini-2.5-pro";
  sel.value = current.model;
}
function updateLimitInfo(){
  const info = elm("limitInfo"); const badge = elm("limitBadge");
  const lim = (MODEL_LIMITS[current.provider]||{})[current.model];
  if (!lim){ info.textContent=""; badge.textContent=""; return; }
  info.textContent = `Context: ${lim.context.toLocaleString()} • Max output: ${lim.max_output.toLocaleString()} tokens`;
  badge.textContent = `${current.model}`;
  elm("maxOut").value = Math.min(parseInt(elm("maxOut").value,10)||2048, lim.max_output);
}
function renderChatList(){
  const list = elm("chatList"); list.innerHTML = "";
  chats.slice().reverse().forEach(ch=>{
    const div = document.createElement("div"); div.className = "p-2 rounded-xl hover:bg-slate-800 cursor-pointer flex items-center justify-between";
    div.innerHTML = `<div class="truncate"><div class="font-medium">${ch.title||"(untitled)"}</div><div class="text-xs text-slate-400">${ch.provider} • ${ch.model}</div></div><div class="flex gap-2"><button class="text-xs pill px-2" data-act="rename" data-id="${ch.id}">Rename</button><button class="text-xs pill px-2" data-act="del" data-id="${ch.id}">Del</button></div>`;
    div.addEventListener("click",(e)=>{
      const act = e.target?.dataset?.act;
      if (act==="del"){ const id=e.target.dataset.id; const idx=chats.findIndex(x=>x.id===id); if(idx>=0){ chats.splice(idx,1); saveChats(); renderChatList(); } e.stopPropagation(); return; }
      if (act==="rename"){ const id=e.target.dataset.id; const chx=chats.find(x=>x.id===id); if(!chx) return; const t=prompt("Rename chat", chx.title||""); if(t!==null){ chx.title=t.trim(); saveChats(); renderChatList(); } e.stopPropagation(); return; }
      openChat(ch.id);
    });
    list.appendChild(div);
  });
}
function openChat(id){ const ch=chats.find(c=>c.id===id); if(!ch) return; current.chatId=id; current.provider=ch.provider; current.model=ch.model; setProvider(ch.provider); elm("modelSelect").value=ch.model; renderMessages(ch.messages); }
function newChat(){ const id="c_"+Date.now(); const ch={id, title:"New chat", provider:current.provider, model:current.model, created:Date.now(), messages:[]}; chats.push(ch); saveChats(); current.chatId=id; renderChatList(); renderMessages(ch.messages); }
function ensureChat(){ if(!current.chatId) newChat(); return chats.find(c=>c.id===current.chatId); }
function renderMessages(msgs){ const wrap=elm("messages"); wrap.innerHTML=""; (msgs||[]).forEach(m=>{ const el=document.createElement("div"); el.className="msg"; const who=m.role==="user"?"You":"Assistant"; const bubbleCls=m.role==="user"?"bubble bubble-user":"bubble bubble-assistant"; let html=`<div class="${bubbleCls}"><div class="text-xs text-slate-400 mb-1">${who}</div>`; (m.content||[]).forEach(p=>{ if((p.type==="text"||p.type==="input_text")&&p.text){ html+=`<p>${escapeHtml(p.text)}</p>`; } else if((p.type==="image"||p.type==="input_image")&&p.dataUrl){ html+=`<img src="${p.dataUrl}" class="mt-2 thumb"/>`; } }); html+=`</div>`; el.innerHTML=html; wrap.appendChild(el); }); wrap.scrollTop = wrap.scrollHeight; }
function escapeHtml(s){ return (s||"").replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c])); }

// Init
document.addEventListener("DOMContentLoaded", ()=>{
  loadStorage();
  document.querySelectorAll("#providerTabs button").forEach(btn=> btn.addEventListener("click", ()=> setProvider(btn.dataset.p)));
  setProvider(current.provider);
  elm("modelSelect").addEventListener("change", ()=>{ current.model = elm("modelSelect").value; updateLimitInfo(); });
  elm("saveKeysBtn").addEventListener("click", ()=>{ keys.openai=elm("keyOpenAI").value.trim(); keys.anthropic=elm("keyAnthropic").value.trim(); keys.gemini=elm("keyGemini").value.trim(); tool.searchEnabled=elm("searchToggle").checked; tool.cse_key=elm("cseKey").value.trim(); tool.cse_cx=elm("cseCx").value.trim(); saveTool(); saveKeysIfAllowed(); if(!elm("veoKey").value) elm("veoKey").value = keys.gemini||""; });
  renderChatList();
  elm("newChatBtn").addEventListener("click", newChat);
  elm("clearAllBtn").addEventListener("click", ()=>{ if(confirm("Delete all chats?")){ chats=[]; saveChats(); renderChatList(); elm("messages").innerHTML=""; current.chatId=null; } });
  elm("attachBtn").addEventListener("click", ()=> elm("fileInput").click());
  elm("fileInput").addEventListener("change", async (e)=>{ const files = Array.from(e.target.files||[]); for(const f of files){ const dataUrl = await fileToDataURL(f); attachments.push({ name:f.name, mime:f.type||"application/octet-stream", dataUrl }); } renderAttachPreview(); e.target.value=""; });
  const input = elm("inputBox"); const resize = ()=>{ input.style.height="auto"; const h=Math.min(input.scrollHeight,240); input.style.height=h+"px"; document.documentElement.style.setProperty('--composer-height', (h+140-56)+"px"); }; input.addEventListener("input", resize); input.addEventListener("keydown", (e)=>{ if(e.key==="Enter"&&!e.shiftKey){ e.preventDefault(); sendMessage(); } }); resize();
  elm("sendBtn").addEventListener("click", sendMessage);
  elm("systemPrompt").placeholder = DEFAULT_SYSTEM;
  document.querySelectorAll(".tab-btn").forEach(b=> b.addEventListener("click", ()=>{ current.tab=b.dataset.tab; document.querySelectorAll(".tab-btn").forEach(x=>x.classList.remove("tab-active")); b.classList.add("tab-active"); elm("tab-chat").classList.toggle("hidden", current.tab!=="chat"); elm("tab-veo").classList.toggle("hidden", current.tab!=="veo"); }));
  elm("veoUseImage").addEventListener("change", e=> elm("veoImageWrap").classList.toggle("hidden", !e.target.checked));
  elm("veoStart").addEventListener("click", startVeo);
});

function renderAttachPreview(){ const box=elm("attachPreview"); box.innerHTML=""; attachments.forEach((a,idx)=>{ const chip=document.createElement("div"); chip.className="pill px-2 py-1 flex items-center gap-2"; chip.innerHTML=`<span class="truncate max-w-[160px]">${a.name}</span><button class="text-xs underline" data-i="${idx}">remove</button>`; chip.querySelector("button").addEventListener("click", ()=>{ attachments.splice(idx,1); renderAttachPreview(); }); box.appendChild(chip); }); }

async function sendMessage(){
  // Read current keys/password/tool state every send
  keys.openai = elm("keyOpenAI").value.trim();
  keys.anthropic = elm("keyAnthropic").value.trim();
  keys.gemini = elm("keyGemini").value.trim();
  const proPass = elm("proPass").value.trim();
  tool.searchEnabled = elm("searchToggle").checked; tool.cse_key=elm("cseKey").value.trim(); tool.cse_cx=elm("cseCx").value.trim(); if (elm("rememberKeys").checked){ saveKeysIfAllowed(); saveTool(); }

  const ch = ensureChat();
  const text = elm("inputBox").value.trim();
  const sys = elm("systemPrompt").value.trim() || DEFAULT_SYSTEM;
  const temp = parseFloat(elm("temp").value);
  const maxOut = parseInt(elm("maxOut").value,10)||2048;

  // Require a key for OpenAI/Anthropic if no pro password is provided
  if (current.provider==="openai" && !keys.openai && !proPass){ alert("Enter your OpenAI key or Pro password"); return; }
  if (current.provider==="anthropic" && !keys.anthropic && !proPass){ alert("Enter your Anthropic key or Pro password"); return; }
  // Gemini can fall back to server keys; no client key required

  const lim = (MODEL_LIMITS[current.provider]||{})[current.model];
  if (lim && maxOut > lim.max_output){ alert(`Max output exceeds model limit (${lim.max_output})`); return; }
  if (!text && attachments.length===0) return;

  const userMsg = { role:"user", content: [] };
  if (text) userMsg.content.push({type:"text", text});
  attachments.forEach(a=>{ if ((a.mime||"").startsWith("image/")) userMsg.content.push({type:"image", dataUrl:a.dataUrl}); else userMsg.content.push({type:"text", text:`[Attached: ${a.name}]`}); });
  ch.messages.push(userMsg); ch.provider=current.provider; ch.model=current.model; if(!ch.title && text) ch.title=text.slice(0,50); saveChats(); renderChatList(); renderMessages(ch.messages); attachments=[]; renderAttachPreview(); elm("inputBox").value="";

  const payload = { provider: current.provider, model: current.model, system: sys, temperature: temp, max_output: maxOut, messages: ch.messages, keys, search: { enabled: tool.searchEnabled, cse_key: tool.cse_key, cse_cx: tool.cse_cx }, pro_password: proPass };
  const res = await fetch("/api/chat", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload) });
  const js = await res.json();
  if (!js.ok){ ch.messages.push({role:"assistant", content:[{type:"text", text:`Error: ${js.error || "Unknown error"}`}]}); }
  else { ch.messages.push(js.message); }
  saveChats(); renderMessages(ch.messages);
}

async function startVeo(){
  const k = elm("veoKey").value.trim() || keys.gemini;
  if (!k) { alert("Enter Gemini key or save it in Keys"); return; }
  const prompt = elm("veoPrompt").value.trim(); if(!prompt){ alert("Enter a prompt"); return; }
  const aspect = elm("veoAspect").value; const neg = elm("veoNeg").value.trim();
  let imgData = null; if (elm("veoUseImage").checked){ const file = elm("veoImage").files?.[0]; if (file) imgData = await fileToDataURL(file); }
  setVeoStatus("Starting video job...");
  const res = await fetch("/api/veo3/start", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ gemini_key:k, prompt, aspect_ratio:aspect, negative_prompt:neg, image_data_url:imgData }) });
  const js = await res.json(); if(!js.ok){ setVeoStatus("Error: "+(js.error||"unknown")); return; }
  const op = js.operation; setVeoStatus("Generating... polling"); let tries=0; let videoFile=null;
  while(tries<40){ await sleep(8000); tries++; const r=await fetch(`/api/veo3/poll?gemini_key=${encodeURIComponent(k)}&operation=${encodeURIComponent(op)}`); const pj=await r.json(); if(!pj.ok){ setVeoStatus("Error: "+(pj.error||"unknown")); return; } if(pj.done){ setVeoStatus("Done."); videoFile=pj.video; break; } else { setVeoStatus("Still generating..."); } }
  if(!videoFile){ setVeoStatus("Timed out waiting for video."); return; }
  const player=elm("veoPlayer"); if(videoFile?.uri){ player.src = videoFile.uri; player.load(); player.play().catch(()=>{}); } else { setVeoStatus("Video ready but no direct URL; use console."); }
}

function setVeoStatus(s){ elm("veoStatus").textContent = s; }
function sleep(ms){ return new Promise(r=>setTimeout(r, ms)); }
function fileToDataURL(file){ return new Promise((res, rej)=>{ const reader=new FileReader(); reader.onload=()=>res(reader.result); reader.onerror=rej; reader.readAsDataURL(file); }); }
</script>
</body>
</html>
"""

# Build final HTML safely (avoid % formatting collisions)
INDEX_HTML = INDEX_HTML_TEMPLATE.replace("__MODEL_LIMITS_JSON__", json.dumps(MODEL_LIMITS))


@app.get("/")
def index():
    return make_response(INDEX_HTML, 200, {"Content-Type": "text/html; charset=utf-8"})


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "time": datetime.utcnow().isoformat()+"Z"})


# ---------------
# Run
# ---------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
