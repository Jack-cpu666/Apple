import os
import json
import base64
import mimetypes
import traceback
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, session
from openai import OpenAI
import secrets
import time

# --- App Initialization ---
app = Flask(__name__)
# Generate a secure secret key for sessions, essential for production
app.secret_key = secrets.token_hex(32)

# --- In-memory Conversation Storage ---
# Note: This will reset if your Render instance restarts. For persistence, consider a database.
conversations = {}

# --- API Key Management ---
# Rotates through up to 10 API keys if they are set in the environment
API_KEYS = [
    os.getenv(f"GEMINI_API_KEY_{i}") for i in range(1, 11)
]
# Add the primary key if it exists
if os.getenv("GEMINI_API_KEY"):
    API_KEYS.append(os.getenv("GEMINI_API_KEY"))

# Filter out None values in case not all keys are set
API_KEYS = [key for key in API_KEYS if key]

# --- Sanity Check for API Keys ---
if not API_KEYS:
    print("FATAL ERROR: No GEMINI_API_KEY environment variables found!")
    print("Please set at least one GEMINI_API_KEY in your Render environment settings.")
else:
    print(f"Successfully loaded {len(API_KEYS)} API key(s).")

# --- API Client Rotation ---
current_api_key_index = 0

def get_next_api_client(model_type="pro"):
    """
    Initializes and returns an OpenAI client configured for Gemini.
    It rotates through the available API keys if multiple are provided.
    
    This function contains the fix for the original traceback.
    """
    global current_api_key_index
    
    if not API_KEYS:
        raise Exception("No API keys configured. Cannot create an API client.")
    
    # Rotate to the next API key
    api_key = API_KEYS[current_api_key_index % len(API_KEYS)]
    current_api_key_index += 1
    
    # CORRECTED INITIALIZATION:
    # The 'proxies' argument is removed to prevent the TypeError. The OpenAI client
    # is initialized with only the necessary parameters for connecting to the Gemini API.
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta"
    )
    
    return client

# --- Helper Functions ---
def encode_image_to_base64(file_content):
    """Encodes image bytes into a base64 string for API transmission."""
    return base64.b64encode(file_content).decode('utf-8')

def estimate_tokens(text):
    """Provides a rough estimation of token count (approx. 4 chars/token)."""
    return len(text) // 4

# --- HTML & CSS & JavaScript Template ---
# This is the single-file template for the web interface.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jack's AI - Advanced AI Assistant</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary-color: #2D3748; --secondary-color: #4A5568; --accent-color: #3182CE;
            --background-color: #F7FAFC; --surface-color: #FFFFFF; --text-primary: #1A202C;
            --text-secondary: #718096; --border-color: #E2E8F0; --success-color: #48BB78;
            --warning-color: #ED8936; --error-color: #F56565; --user-bubble: #3182CE;
            --ai-bubble: #F7FAFC;
        }
        body {
            font-family: 'Inter', sans-serif; background-color: var(--background-color); color: var(--text-primary);
            line-height: 1.6; height: 100vh; overflow: hidden; margin: 0; padding: 0; box-sizing: border-box;
        }
        .app-container { display: flex; flex-direction: column; height: 100vh; }
        .header { background: var(--surface-color); border-bottom: 1px solid var(--border-color); padding: 1rem 1.5rem; display: flex; justify-content: space-between; align-items: center; }
        .logo { display: flex; align-items: center; gap: 0.75rem; font-size: 1.25rem; font-weight: 600; }
        .logo i { color: var(--accent-color); font-size: 1.5rem; }
        .header-actions { display: flex; gap: 1rem; }
        .btn { padding: 0.5rem 1rem; border-radius: 0.5rem; border: none; font-weight: 500; cursor: pointer; transition: all 0.2s; display: inline-flex; align-items: center; gap: 0.5rem; }
        .btn-secondary { background: transparent; color: var(--text-secondary); border: 1px solid var(--border-color); }
        .btn-secondary:hover { background: var(--background-color); color: var(--text-primary); }
        .chat-container { flex: 1; overflow-y: auto; padding: 2rem 1rem; }
        .chat-wrapper { max-width: 900px; width: 100%; margin: 0 auto; }
        .message { display: flex; gap: 1rem; animation: fadeIn 0.3s ease; margin-bottom: 1.5rem; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .message.user { flex-direction: row-reverse; }
        .message-avatar { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
        .user .message-avatar { background: var(--user-bubble); color: white; }
        .assistant .message-avatar { background: var(--border-color); color: var(--text-secondary); }
        .message-content { flex: 1; max-width: 80%; }
        .message-bubble { padding: 1rem 1.25rem; border-radius: 1rem; word-wrap: break-word; }
        .user .message-bubble { background: var(--user-bubble); color: white; border-bottom-right-radius: 0.25rem; }
        .assistant .message-bubble { background: var(--ai-bubble); color: var(--text-primary); border: 1px solid var(--border-color); border-bottom-left-radius: 0.25rem; }
        .message-bubble pre { background: rgba(0,0,0,0.05); padding: 0.75rem; border-radius: 0.5rem; overflow-x: auto; margin: 1rem 0; }
        .message-bubble code { font-family: 'Courier New', monospace; }
        .input-container { background: var(--surface-color); border-top: 1px solid var(--border-color); padding: 1rem 1.5rem; }
        .input-wrapper { max-width: 900px; margin: 0 auto; display: flex; gap: 1rem; align-items: flex-end; }
        .input-group { flex: 1; position: relative; }
        .input-textarea { width: 100%; min-height: 50px; max-height: 200px; padding: 0.75rem 3rem 0.75rem 1rem; border-radius: 0.75rem; border: 1px solid var(--border-color); resize: vertical; font-family: inherit; font-size: 1rem; line-height: 1.5; }
        .input-textarea:focus { outline: none; border-color: var(--accent-color); }
        .input-actions { position: absolute; right: 0.5rem; bottom: 0.5rem; display: flex; }
        .icon-btn { width: 32px; height: 32px; border-radius: 0.5rem; border: none; background: transparent; color: var(--text-secondary); cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.2s; }
        .icon-btn:hover { background: var(--background-color); color: var(--text-primary); }
        .send-btn { background: var(--accent-color); color: white; }
        .send-btn:hover { background: #2C5282; }
        .send-btn:disabled { background: var(--border-color); cursor: not-allowed; }
        #file-input { display: none; }
        .typing-indicator { display: flex; gap: 4px; padding: 1rem; }
        .typing-dot { width: 8px; height: 8px; background: var(--text-secondary); border-radius: 50%; animation: typing 1.4s infinite; }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing { 0%, 60%, 100% { transform: translateY(0); opacity: 0.5; } 30% { transform: translateY(-10px); opacity: 1; } }
    </style>
</head>
<body>
    <div class="app-container">
        <div class="header">
            <div class="logo"><i class="fas fa-robot"></i><span>Jack's AI</span></div>
            <div class="header-actions">
                <button class="btn btn-secondary" onclick="clearChat()"><i class="fas fa-broom"></i> New Chat</button>
            </div>
        </div>
        <div class="chat-container" id="chat-container">
            <div class="chat-wrapper" id="chat-wrapper">
                <div class="message assistant">
                    <div class="message-avatar"><i class="fas fa-robot"></i></div>
                    <div class="message-content">
                        <div class="message-bubble">
                            <p>Welcome to Jack's AI! How can I assist you today?</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="input-container">
            <div class="input-wrapper">
                <div class="input-group">
                    <textarea class="input-textarea" id="user-input" placeholder="Type your message..." rows="1"></textarea>
                    <div class="input-actions">
                        <button class="icon-btn send-btn" id="send-btn" onclick="sendMessage()" title="Send message"><i class="fas fa-paper-plane"></i></button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const chatWrapper = document.getElementById('chat-wrapper');
        const chatContainer = document.getElementById('chat-container');

        userInput.addEventListener('input', () => {
            userInput.style.height = 'auto';
            userInput.style.height = (userInput.scrollHeight) + 'px';
        });

        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;

            addMessageToUI('user', message);
            userInput.value = '';
            userInput.style.height = 'auto';
            sendBtn.disabled = true;
            showTypingIndicator();

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ message: message })
                });

                const data = await response.json();
                hideTypingIndicator();

                if (response.ok && data.reply) {
                    addMessageToUI('assistant', data.reply);
                } else {
                    addMessageToUI('assistant', data.error || 'Sorry, something went wrong.');
                }
            } catch (error) {
                hideTypingIndicator();
                addMessageToUI('assistant', 'Error connecting to the server. Please try again.');
                console.error('Fetch error:', error);
            } finally {
                sendBtn.disabled = false;
            }
        }

        function addMessageToUI(role, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            
            const avatarIcon = role === 'user' ? 'fa-user' : 'fa-robot';
            const formattedContent = content.replace(/\\n/g, '<br>').replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>');

            messageDiv.innerHTML = `
                <div class="message-avatar"><i class="fas ${avatarIcon}"></i></div>
                <div class="message-content">
                    <div class="message-bubble">${formattedContent}</div>
                </div>
            `;
            chatWrapper.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function showTypingIndicator() {
            const indicator = document.createElement('div');
            indicator.id = 'typing-indicator';
            indicator.className = 'message assistant';
            indicator.innerHTML = `
                <div class="message-avatar"><i class="fas fa-robot"></i></div>
                <div class="message-content">
                    <div class="typing-indicator">
                        <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
                    </div>
                </div>
            `;
            chatWrapper.appendChild(indicator);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function hideTypingIndicator() {
            const indicator = document.getElementById('typing-indicator');
            if (indicator) indicator.remove();
        }

        async function clearChat() {
            if (confirm('Are you sure you want to start a new chat?')) {
                await fetch('/clear_chat', { method: 'POST' });
                chatWrapper.innerHTML = '';
                addMessageToUI('assistant', 'Welcome back! How can I help you?');
            }
        }
        
        // Load conversation on page load
        async function loadConversation() {
             try {
                const response = await fetch('/load_conversation');
                const data = await response.json();
                if (data.messages && data.messages.length > 0) {
                    chatWrapper.innerHTML = ''; // Clear welcome message
                    data.messages.forEach(msg => addMessageToUI(msg.role, msg.content));
                }
            } catch (error) {
                console.error('Error loading conversation:', error);
            }
        }

        window.onload = loadConversation;

    </script>
</body>
</html>
"""

# --- Flask Routes ---

@app.route("/", methods=["GET"])
def index():
    """Serves the main HTML page."""
    if 'session_id' not in session:
        session['session_id'] = secrets.token_hex(16)
        conversations[session['session_id']] = {'messages': []}
    return render_template_string(HTML_TEMPLATE)

@app.route("/chat", methods=["POST"])
def chat():
    """Handles the chat request, communicates with Gemini, and returns a response."""
    try:
        data = request.get_json(force=True)
        message = data.get("message", "").strip()

        if not message:
            return jsonify({"error": "Message cannot be empty."}), 400

        # Ensure session exists
        session_id = session.get('session_id')
        if not session_id or session_id not in conversations:
            session_id = secrets.token_hex(16)
            session['session_id'] = session_id
            conversations[session_id] = {'messages': []}
        
        conversation_history = conversations[session_id]['messages']
        
        # Add user message to history
        conversation_history.append({"role": "user", "content": message})
        
        # Use a system prompt to guide the model's behavior
        system_prompt = "You are Jack's AI, a helpful and comprehensive assistant. You should provide detailed, well-structured, and complete answers. Never mention you are a model from Google. Be thorough."
        
        messages_for_api = [
            {"role": "system", "content": system_prompt}
        ] + conversation_history

        reply = None
        for attempt in range(len(API_KEYS)):
            try:
                client = get_next_api_client()
                
                # Note: The 'openai' package is used here as a wrapper for the Gemini API endpoint.
                # We need to map the roles to 'user' and 'model'.
                gemini_messages = []
                for msg in messages_for_api:
                    role = 'model' if msg['role'] == 'assistant' else 'user'
                    # The Gemini API expects a specific structure.
                    # This is a simplified approach; for complex cases, mapping needs to be more robust.
                    if msg['role'] != 'system': # Gemini API doesn't use a 'system' role in the same way
                         gemini_messages.append({"role": role, "parts": [{"text": msg['content']}]})

                # The endpoint for Gemini via the OpenAI-compatible wrapper is `chat.completions`
                response = client.chat.completions.create(
                    model="gemini-1.5-pro",
                    messages=messages_for_api, # The OpenAI wrapper handles the conversion
                    temperature=0.7,
                )
                
                if response.choices and response.choices[0].message:
                    reply = response.choices[0].message.content.strip()
                    break # Success, exit loop
                
            except Exception as api_error:
                print(f"API attempt {attempt + 1} failed: {str(api_error)}")
                traceback.print_exc()
                if attempt == len(API_KEYS) - 1: # If all keys failed
                    return jsonify({"error": "The AI service is currently unavailable. Please try again later."}), 503

        if reply:
            # Add assistant response to history
            conversation_history.append({"role": "assistant", "content": reply})
            return jsonify({"reply": reply})
        else:
            return jsonify({"error": "Failed to get a response from the AI."}), 500

    except Exception as e:
        print(f"An unexpected error occurred in /chat: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    """Clears the conversation history for the current session."""
    session_id = session.get('session_id')
    if session_id and session_id in conversations:
        conversations[session_id]['messages'] = []
    return jsonify({"success": True})

@app.route("/load_conversation", methods=["GET"])
def load_conversation():
    """Loads the conversation history for the current session."""
    session_id = session.get('session_id')
    if session_id and session_id in conversations:
        return jsonify(conversations[session_id])
    return jsonify({"messages": []})

# --- Main Execution ---
if __name__ == "__main__":
    # Get port from environment variable, which is required for Render deployment
    port = int(os.environ.get("PORT", 5000))
    # Run the app on 0.0.0.0 to be accessible externally
    app.run(host="0.0.0.0", port=port, debug=False)