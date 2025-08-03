import os
from flask import Flask, render_template_string, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# Initialize Gemini-compatible client
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Simple Bootstrap-based chat page
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Gemini 2.5 Pro Chat</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link 
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" 
    rel="stylesheet"
  >
  <style>
    body { background: #f8f9fa; }
    #chat { max-height: 60vh; overflow-y: auto; }
    .msg.user   { text-align: right; }
    .msg.assist { text-align: left; }
    .bubble { 
      display: inline-block; 
      padding: .5rem 1rem; 
      border-radius: 1rem; 
      margin: .25rem 0; 
      max-width: 80%;
    }
    .user .bubble   { background: #0d6efd; color: #fff; }
    .assist .bubble { background: #e9ecef; color: #212529; }
  </style>
</head>
<body>
  <div class="container py-4">
    <h1 class="mb-4 text-center">Gemini 2.5 Pro Chat</h1>
    <div id="chat" class="border rounded bg-white p-3 mb-3"></div>
    <div class="input-group">
      <input id="input" type="text" class="form-control" placeholder="Type your message..." />
      <button id="send" class="btn btn-primary">Send</button>
    </div>
  </div>

  <script>
    const chatEl = document.getElementById('chat');
    const inputEl = document.getElementById('input');
    const sendEl  = document.getElementById('send');

    function append(role, text) {
      const div = document.createElement('div');
      div.className = 'msg ' + (role==='user'?'user':'assist');
      div.innerHTML = `<div class="bubble">${text}</div>`;
      chatEl.appendChild(div);
      chatEl.scrollTop = chatEl.scrollHeight;
    }

    sendEl.onclick = async () => {
      const msg = inputEl.value.trim();
      if (!msg) return;
      append('user', msg);
      inputEl.value = '';
      append('assist', '<em>…thinking…</em>');

      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({message: msg})
        });
        const json = await res.json();
        // remove the “thinking” bubble
        chatEl.lastChild.remove();
        if (json.reply) {
          append('assist', json.reply);
        } else {
          append('assist', '<em>Error: ' + (json.error||'unknown') + '</em>');
        }
      } catch (err) {
        chatEl.lastChild.remove();
        append('assist', `<em>Connection error</em>`);
      }
    };

    // send on Enter
    inputEl.addEventListener('keydown', e => {
      if (e.key === 'Enter') sendEl.click();
    });
  </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_msg = data.get("message","").strip()
    if not user_msg:
        return jsonify(error="No message provided"), 400

    try:
        res = client.chat.completions.create(
            model="gemini-2.5-pro",
            messages=[
                {"role":"system", "content":"You are a helpful assistant."},
                {"role":"user",   "content": user_msg}
            ],
            temperature=0.7,
            max_tokens=800
        )
        reply = res.choices[0].message.content.strip()
        return jsonify(reply=reply)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)