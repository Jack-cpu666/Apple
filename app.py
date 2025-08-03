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

app = Flask(__name__)
# Generate a secure secret key for sessions
app.secret_key = secrets.token_hex(32)

# Store conversations in memory (dictionary with session IDs as keys)
conversations = {}

# API keys management - will rotate through these if one fails
API_KEYS = [
    os.getenv(f"GEMINI_API_KEY_{i}", os.getenv("GEMINI_API_KEY")) 
    for i in range(1, 11)
]
# Filter out None values in case not all 10 keys are set
API_KEYS = [key for key in API_KEYS if key]

# Keep track of which API key to use next
current_api_key_index = 0

def get_next_api_client(model_type="pro"):
    """
    Get the next available API client, rotating through API keys if one fails.
    model_type can be 'pro' for Gemini 2.5 Pro or 'flash' for Gemini Flash
    """
    global current_api_key_index
    
    if not API_KEYS:
        raise Exception("No API keys configured")
    
    # Get the next API key in rotation
    api_key = API_KEYS[current_api_key_index % len(API_KEYS)]
    current_api_key_index += 1
    
    # Create client with the selected API key
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    return client

def encode_image_to_base64(file_content):
    """Convert image bytes to base64 string for API consumption"""
    return base64.b64encode(file_content).decode('utf-8')

def estimate_tokens(text):
    """
    Rough estimation of tokens - approximately 4 characters per token
    This is a simplified estimation for display purposes
    """
    return len(text) // 4

# The main HTML template with modern, clean UI similar to Claude.ai
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jack's AI - Advanced AI Assistant</title>
    
    <!-- Google Fonts for clean typography -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Font Awesome for icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        /* CSS Reset and Base Styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            /* Color scheme similar to Claude.ai */
            --primary-color: #2D3748;
            --secondary-color: #4A5568;
            --accent-color: #3182CE;
            --background-color: #F7FAFC;
            --surface-color: #FFFFFF;
            --text-primary: #1A202C;
            --text-secondary: #718096;
            --border-color: #E2E8F0;
            --success-color: #48BB78;
            --warning-color: #ED8936;
            --error-color: #F56565;
            --user-bubble: #3182CE;
            --ai-bubble: #F7FAFC;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background-color: var(--background-color);
            color: var(--text-primary);
            line-height: 1.6;
            height: 100vh;
            overflow: hidden;
        }
        
        /* Main Layout Container */
        .app-container {
            display: flex;
            flex-direction: column;
            height: 100vh;
            max-width: 100%;
            margin: 0 auto;
        }
        
        /* Header Section */
        .header {
            background: var(--surface-color);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .logo i {
            color: var(--accent-color);
            font-size: 1.5rem;
        }
        
        .header-actions {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        
        .btn {
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            border: none;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.875rem;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .btn-secondary {
            background: transparent;
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
        }
        
        .btn-secondary:hover {
            background: var(--background-color);
            color: var(--text-primary);
        }
        
        .btn-primary {
            background: var(--accent-color);
            color: white;
        }
        
        .btn-primary:hover {
            background: #2C5282;
        }
        
        /* Token Usage Bar */
        .token-usage {
            padding: 0.75rem 1.5rem;
            background: var(--surface-color);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 1rem;
            font-size: 0.875rem;
            flex-shrink: 0;
        }
        
        .token-bar {
            flex: 1;
            height: 8px;
            background: var(--border-color);
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }
        
        .token-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--success-color), var(--accent-color));
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        
        .token-text {
            color: var(--text-secondary);
            min-width: 150px;
            text-align: right;
        }
        
        /* Chat Area */
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 2rem 1rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }
        
        .chat-wrapper {
            max-width: 900px;
            width: 100%;
            margin: 0 auto;
        }
        
        /* Message Bubbles */
        .message {
            display: flex;
            gap: 1rem;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .message.user {
            flex-direction: row-reverse;
        }
        
        .message-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            font-size: 1rem;
        }
        
        .user .message-avatar {
            background: var(--user-bubble);
            color: white;
        }
        
        .assistant .message-avatar {
            background: var(--border-color);
            color: var(--text-secondary);
        }
        
        .message-content {
            flex: 1;
            max-width: 70%;
        }
        
        .message-bubble {
            padding: 1rem 1.25rem;
            border-radius: 1rem;
            position: relative;
            word-wrap: break-word;
        }
        
        .user .message-bubble {
            background: var(--user-bubble);
            color: white;
            border-bottom-right-radius: 0.25rem;
        }
        
        .assistant .message-bubble {
            background: var(--ai-bubble);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-bottom-left-radius: 0.25rem;
        }
        
        .message-bubble pre {
            background: rgba(0, 0, 0, 0.05);
            padding: 0.75rem;
            border-radius: 0.5rem;
            overflow-x: auto;
            margin: 0.5rem 0;
        }
        
        .user .message-bubble pre {
            background: rgba(255, 255, 255, 0.1);
        }
        
        .message-bubble code {
            background: rgba(0, 0, 0, 0.05);
            padding: 0.125rem 0.25rem;
            border-radius: 0.25rem;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        
        .user .message-bubble code {
            background: rgba(255, 255, 255, 0.2);
        }
        
        /* File attachment display */
        .attachment {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(0, 0, 0, 0.05);
            padding: 0.5rem 0.75rem;
            border-radius: 0.5rem;
            margin-top: 0.5rem;
            font-size: 0.875rem;
        }
        
        .user .attachment {
            background: rgba(255, 255, 255, 0.2);
        }
        
        /* Prompt Improvement Modal */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        
        .modal.active {
            display: flex;
        }
        
        .modal-content {
            background: var(--surface-color);
            border-radius: 1rem;
            padding: 2rem;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
            animation: slideUp 0.3s ease;
        }
        
        @keyframes slideUp {
            from {
                transform: translateY(20px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
        
        .modal-header {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--text-primary);
        }
        
        .modal-body {
            margin-bottom: 1.5rem;
        }
        
        .improved-prompt {
            background: var(--background-color);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
            white-space: pre-wrap;
        }
        
        .modal-actions {
            display: flex;
            gap: 1rem;
            justify-content: flex-end;
        }
        
        /* Input Area */
        .input-container {
            background: var(--surface-color);
            border-top: 1px solid var(--border-color);
            padding: 1rem 1.5rem;
            flex-shrink: 0;
        }
        
        .input-wrapper {
            max-width: 900px;
            margin: 0 auto;
            display: flex;
            gap: 1rem;
            align-items: flex-end;
        }
        
        .input-group {
            flex: 1;
            position: relative;
        }
        
        .input-textarea {
            width: 100%;
            min-height: 50px;
            max-height: 200px;
            padding: 0.75rem 3rem 0.75rem 1rem;
            border: 1px solid var(--border-color);
            border-radius: 0.75rem;
            resize: vertical;
            font-family: inherit;
            font-size: 0.95rem;
            line-height: 1.5;
            transition: border-color 0.2s;
        }
        
        .input-textarea:focus {
            outline: none;
            border-color: var(--accent-color);
        }
        
        .input-actions {
            position: absolute;
            right: 0.5rem;
            bottom: 0.5rem;
            display: flex;
            gap: 0.5rem;
        }
        
        .icon-btn {
            width: 32px;
            height: 32px;
            border-radius: 0.5rem;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }
        
        .icon-btn:hover {
            background: var(--background-color);
            color: var(--text-primary);
        }
        
        .send-btn {
            background: var(--accent-color);
            color: white;
        }
        
        .send-btn:hover {
            background: #2C5282;
        }
        
        .send-btn:disabled {
            background: var(--border-color);
            color: var(--text-secondary);
            cursor: not-allowed;
        }
        
        /* File upload input (hidden) */
        #file-input {
            display: none;
        }
        
        /* Notice Banner */
        .notice-banner {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.75rem 1.5rem;
            text-align: center;
            font-size: 0.875rem;
            flex-shrink: 0;
        }
        
        .notice-banner strong {
            font-weight: 600;
        }
        
        /* Loading animation */
        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 1rem;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            background: var(--text-secondary);
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }
        
        .typing-dot:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-dot:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typing {
            0%, 60%, 100% {
                transform: translateY(0);
                opacity: 0.5;
            }
            30% {
                transform: translateY(-10px);
                opacity: 1;
            }
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .header {
                padding: 0.75rem 1rem;
            }
            
            .logo {
                font-size: 1.1rem;
            }
            
            .header-actions {
                gap: 0.5rem;
            }
            
            .btn {
                padding: 0.4rem 0.75rem;
                font-size: 0.8rem;
            }
            
            .chat-container {
                padding: 1rem 0.5rem;
            }
            
            .message-content {
                max-width: 85%;
            }
            
            .input-container {
                padding: 0.75rem 1rem;
            }
            
            .modal-content {
                width: 95%;
                padding: 1.5rem;
            }
        }
        
        /* Dark mode support (if user prefers dark mode) */
        @media (prefers-color-scheme: dark) {
            :root {
                --background-color: #1A202C;
                --surface-color: #2D3748;
                --text-primary: #F7FAFC;
                --text-secondary: #A0AEC0;
                --border-color: #4A5568;
                --ai-bubble: #2D3748;
            }
        }
        
        /* Custom scrollbar */
        .chat-container::-webkit-scrollbar {
            width: 8px;
        }
        
        .chat-container::-webkit-scrollbar-track {
            background: var(--background-color);
        }
        
        .chat-container::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 4px;
        }
        
        .chat-container::-webkit-scrollbar-thumb:hover {
            background: var(--text-secondary);
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Notice Banner -->
        <div class="notice-banner">
            <i class="fas fa-gift"></i> <strong>Temporarily Free!</strong> 
            Enjoy unlimited access now. Coming soon: $5/month for unlimited use.
        </div>
        
        <!-- Header -->
        <div class="header">
            <div class="logo">
                <i class="fas fa-robot"></i>
                <span>Jack's AI</span>
            </div>
            <div class="header-actions">
                <button class="btn btn-secondary" onclick="compactConversation()">
                    <i class="fas fa-compress"></i>
                    Compact Chat
                </button>
                <button class="btn btn-secondary" onclick="clearChat()">
                    <i class="fas fa-broom"></i>
                    New Chat
                </button>
            </div>
        </div>
        
        <!-- Token Usage Bar -->
        <div class="token-usage">
            <span style="color: var(--text-secondary);">
                <i class="fas fa-coins"></i> Token Usage
            </span>
            <div class="token-bar">
                <div class="token-fill" id="token-fill" style="width: 0%"></div>
            </div>
            <span class="token-text" id="token-text">0 / 125,000</span>
        </div>
        
        <!-- Chat Container -->
        <div class="chat-container" id="chat-container">
            <div class="chat-wrapper" id="chat-wrapper">
                <!-- Welcome Message -->
                <div class="message assistant">
                    <div class="message-avatar">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="message-content">
                        <div class="message-bubble">
                            <p>👋 Welcome to Jack's AI!</p>
                            <p>I'm powered by advanced AI with a 125,000 token context window. You can:</p>
                            <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                                <li>Ask me questions about any topic</li>
                                <li>Upload images, documents, and files for analysis</li>
                                <li>Get help with writing, coding, and problem-solving</li>
                                <li>Have natural conversations with improved prompts</li>
                            </ul>
                            <p>How can I assist you today?</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Input Container -->
        <div class="input-container">
            <div class="input-wrapper">
                <div class="input-group">
                    <textarea 
                        class="input-textarea" 
                        id="user-input"
                        placeholder="Type your message here... (Shift+Enter for new line)"
                        rows="1"
                    ></textarea>
                    <div class="input-actions">
                        <button class="icon-btn" onclick="document.getElementById('file-input').click()" title="Attach file">
                            <i class="fas fa-paperclip"></i>
                        </button>
                        <button class="icon-btn send-btn" id="send-btn" onclick="sendMessage()" title="Send message">
                            <i class="fas fa-paper-plane"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Hidden file input -->
        <input type="file" id="file-input" multiple accept="image/*,.pdf,.txt,.doc,.docx,.xls,.xlsx,.csv,.json,.xml,.html,.css,.js,.py,.java,.cpp,.c,.h,.md">
        
        <!-- Prompt Improvement Modal -->
        <div class="modal" id="prompt-modal">
            <div class="modal-content">
                <div class="modal-header">
                    <i class="fas fa-magic"></i> Improved Prompt Suggestion
                </div>
                <div class="modal-body">
                    <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                        I've analyzed your input and created an improved version that might get better results:
                    </p>
                    <div class="improved-prompt" id="improved-prompt-text"></div>
                    <p style="color: var(--text-secondary); margin-top: 1rem;">
                        Would you like to use this improved version or stick with your original prompt?
                    </p>
                </div>
                <div class="modal-actions">
                    <button class="btn btn-secondary" onclick="useOriginalPrompt()">
                        Use Original
                    </button>
                    <button class="btn btn-primary" onclick="useImprovedPrompt()">
                        Use Improved
                    </button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Global variables to track state
        let currentTokenUsage = 0;
        const maxTokens = 125000;
        let uploadedFiles = [];
        let currentUserPrompt = '';
        let improvedPrompt = '';
        let isProcessing = false;
        
        // Initialize the application
        document.addEventListener('DOMContentLoaded', function() {
            // Auto-resize textarea
            const textarea = document.getElementById('user-input');
            textarea.addEventListener('input', autoResize);
            
            // Handle Enter key for sending (Shift+Enter for new line)
            textarea.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
            
            // Handle file uploads
            document.getElementById('file-input').addEventListener('change', handleFileUpload);
            
            // Load existing conversation if any
            loadConversation();
        });
        
        // Auto-resize textarea as user types
        function autoResize() {
            const textarea = document.getElementById('user-input');
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
        }
        
        // Handle file uploads
        async function handleFileUpload(event) {
            const files = event.target.files;
            if (!files || files.length === 0) return;
            
            for (let file of files) {
                // Check file size (limit to 10MB)
                if (file.size > 10 * 1024 * 1024) {
                    alert(`File "${file.name}" is too large. Maximum size is 10MB.`);
                    continue;
                }
                
                // Read file content
                const reader = new FileReader();
                reader.onload = function(e) {
                    uploadedFiles.push({
                        name: file.name,
                        type: file.type,
                        size: file.size,
                        content: e.target.result
                    });
                    
                    // Show file indicator in input area
                    showFileIndicator(file.name);
                };
                
                // Read as data URL for images, text for others
                if (file.type.startsWith('image/')) {
                    reader.readAsDataURL(file);
                } else {
                    reader.readAsText(file);
                }
            }
            
            // Clear the file input for next use
            event.target.value = '';
        }
        
        // Show file indicator
        function showFileIndicator(filename) {
            const indicator = document.createElement('span');
            indicator.className = 'attachment';
            indicator.innerHTML = `<i class="fas fa-file"></i> ${filename}`;
            
            // Add to a temporary display area (you can enhance this)
            const textarea = document.getElementById('user-input');
            textarea.placeholder = `${uploadedFiles.length} file(s) attached. Type your message...`;
        }
        
        // Send message to backend
        async function sendMessage() {
            if (isProcessing) return;
            
            const textarea = document.getElementById('user-input');
            const message = textarea.value.trim();
            
            if (!message && uploadedFiles.length === 0) return;
            
            // Check token limit
            if (currentTokenUsage >= maxTokens * 0.95) {
                if (confirm('You are approaching the token limit. Would you like to compact the conversation?')) {
                    compactConversation();
                    return;
                }
            }
            
            isProcessing = true;
            currentUserPrompt = message;
            
            // Disable send button
            document.getElementById('send-btn').disabled = true;
            
            // Clear input
            textarea.value = '';
            textarea.style.height = 'auto';
            textarea.placeholder = 'Type your message here... (Shift+Enter for new line)';
            
            // Add user message to chat
            addMessage('user', message, uploadedFiles);
            
            // Show typing indicator
            showTypingIndicator();
            
            try {
                // Step 1: Get improved prompt
                const improvedResponse = await fetch('/improve_prompt', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        prompt: message,
                        files: uploadedFiles.map(f => ({
                            name: f.name,
                            type: f.type,
                            content: f.content
                        }))
                    })
                });
                
                const improvedData = await improvedResponse.json();
                
                if (improvedData.improved_prompt && improvedData.improved_prompt !== message) {
                    improvedPrompt = improvedData.improved_prompt;
                    hideTypingIndicator();
                    showPromptModal();
                } else {
                    // Use original prompt directly
                    await processWithMainAI(message);
                }
                
            } catch (error) {
                console.error('Error:', error);
                hideTypingIndicator();
                addMessage('assistant', 'Sorry, an error occurred. Please try again.');
            } finally {
                isProcessing = false;
                document.getElementById('send-btn').disabled = false;
                uploadedFiles = [];
            }
        }
        
        // Show prompt improvement modal
        function showPromptModal() {
            document.getElementById('improved-prompt-text').textContent = improvedPrompt;
            document.getElementById('prompt-modal').classList.add('active');
        }
        
        // Use original prompt
        async function useOriginalPrompt() {
            document.getElementById('prompt-modal').classList.remove('active');
            showTypingIndicator();
            await processWithMainAI(currentUserPrompt);
        }
        
        // Use improved prompt
        async function useImprovedPrompt() {
            document.getElementById('prompt-modal').classList.remove('active');
            showTypingIndicator();
            await processWithMainAI(improvedPrompt);
        }
        
        // Process with main AI (Gemini 2.5 Pro)
        async function processWithMainAI(prompt) {
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        message: prompt,
                        files: uploadedFiles.map(f => ({
                            name: f.name,
                            type: f.type,
                            content: f.content
                        }))
                    })
                });
                
                const data = await response.json();
                hideTypingIndicator();
                
                if (data.reply) {
                    addMessage('assistant', data.reply);
                    
                    // Update token usage
                    if (data.token_usage) {
                        updateTokenUsage(data.token_usage);
                    }
                } else {
                    addMessage('assistant', 'Sorry, I could not process your request. Please try again.');
                }
                
            } catch (error) {
                console.error('Error:', error);
                hideTypingIndicator();
                addMessage('assistant', 'Sorry, an error occurred. Please try again.');
            }
        }
        
        // Add message to chat
        function addMessage(role, content, files = []) {
            const chatWrapper = document.getElementById('chat-wrapper');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            
            // Create avatar
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.innerHTML = role === 'user' ? 
                '<i class="fas fa-user"></i>' : 
                '<i class="fas fa-robot"></i>';
            
            // Create message content
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            
            const bubble = document.createElement('div');
            bubble.className = 'message-bubble';
            
            // Process content for display (convert markdown, etc.)
            bubble.innerHTML = formatMessage(content);
            
            // Add file attachments if any
            if (files && files.length > 0) {
                files.forEach(file => {
                    const attachment = document.createElement('div');
                    attachment.className = 'attachment';
                    attachment.innerHTML = `<i class="fas fa-file"></i> ${file.name}`;
                    bubble.appendChild(attachment);
                });
            }
            
            messageContent.appendChild(bubble);
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageContent);
            chatWrapper.appendChild(messageDiv);
            
            // Scroll to bottom
            const chatContainer = document.getElementById('chat-container');
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        // Format message content (basic markdown support)
        function formatMessage(content) {
            // Remove em dashes for cleaner output
            content = content.replace(/—/g, '-');
            
            // Basic markdown conversion
            content = content
                .replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>')
                .replace(/`([^`]+)`/g, '<code>$1</code>')
                .replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\*([^*]+)\\*/g, '<em>$1</em>')
                .replace(/\\n/g, '<br>');
            
            return content;
        }
        
        // Show typing indicator
        function showTypingIndicator() {
            const chatWrapper = document.getElementById('chat-wrapper');
            const indicator = document.createElement('div');
            indicator.className = 'message assistant';
            indicator.id = 'typing-indicator';
            
            indicator.innerHTML = `
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="typing-indicator">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            `;
            
            chatWrapper.appendChild(indicator);
            
            // Scroll to bottom
            const chatContainer = document.getElementById('chat-container');
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        // Hide typing indicator
        function hideTypingIndicator() {
            const indicator = document.getElementById('typing-indicator');
            if (indicator) {
                indicator.remove();
            }
        }
        
        // Update token usage display
        function updateTokenUsage(usage) {
            currentTokenUsage = usage;
            const percentage = (usage / maxTokens) * 100;
            
            document.getElementById('token-fill').style.width = percentage + '%';
            document.getElementById('token-text').textContent = 
                `${usage.toLocaleString()} / ${maxTokens.toLocaleString()}`;
            
            // Change color based on usage
            const tokenFill = document.getElementById('token-fill');
            if (percentage > 90) {
                tokenFill.style.background = 'linear-gradient(90deg, #F56565, #ED8936)';
            } else if (percentage > 75) {
                tokenFill.style.background = 'linear-gradient(90deg, #ED8936, #F6AD55)';
            } else {
                tokenFill.style.background = 'linear-gradient(90deg, #48BB78, #3182CE)';
            }
        }
        
        // Clear chat
        async function clearChat() {
            if (confirm('Are you sure you want to start a new chat? This will clear the current conversation.')) {
                try {
                    await fetch('/clear_chat', {method: 'POST'});
                    
                    // Clear UI
                    const chatWrapper = document.getElementById('chat-wrapper');
                    chatWrapper.innerHTML = `
                        <div class="message assistant">
                            <div class="message-avatar">
                                <i class="fas fa-robot"></i>
                            </div>
                            <div class="message-content">
                                <div class="message-bubble">
                                    <p>👋 Welcome back! How can I help you today?</p>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // Reset token usage
                    currentTokenUsage = 0;
                    updateTokenUsage(0);
                    
                } catch (error) {
                    console.error('Error clearing chat:', error);
                }
            }
        }
        
        // Compact conversation
        async function compactConversation() {
            if (!confirm('This will summarize the conversation to save tokens. Continue?')) {
                return;
            }
            
            showTypingIndicator();
            
            try {
                const response = await fetch('/compact_conversation', {method: 'POST'});
                const data = await response.json();
                
                hideTypingIndicator();
                
                if (data.success) {
                    // Clear chat and show compacted version
                    const chatWrapper = document.getElementById('chat-wrapper');
                    chatWrapper.innerHTML = `
                        <div class="message assistant">
                            <div class="message-avatar">
                                <i class="fas fa-robot"></i>
                            </div>
                            <div class="message-content">
                                <div class="message-bubble">
                                    <p><strong>Conversation Compacted</strong></p>
                                    <p>${data.summary}</p>
                                    <p style="margin-top: 1rem;">The conversation has been compressed. You can continue from here.</p>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // Update token usage
                    if (data.token_usage) {
                        updateTokenUsage(data.token_usage);
                    }
                } else {
                    alert('Failed to compact conversation. Please try again.');
                }
                
            } catch (error) {
                console.error('Error compacting:', error);
                hideTypingIndicator();
                alert('An error occurred while compacting the conversation.');
            }
        }
        
        // Load existing conversation
        async function loadConversation() {
            try {
                const response = await fetch('/load_conversation');
                const data = await response.json();
                
                if (data.messages && data.messages.length > 0) {
                    // Clear welcome message
                    const chatWrapper = document.getElementById('chat-wrapper');
                    chatWrapper.innerHTML = '';
                    
                    // Add messages
                    data.messages.forEach(msg => {
                        addMessage(msg.role, msg.content);
                    });
                    
                    // Update token usage
                    if (data.token_usage) {
                        updateTokenUsage(data.token_usage);
                    }
                }
            } catch (error) {
                console.error('Error loading conversation:', error);
            }
        }
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    """Serve the main application page"""
    # Initialize session if not exists
    if 'session_id' not in session:
        session['session_id'] = secrets.token_hex(16)
        conversations[session['session_id']] = {
            'messages': [],
            'token_usage': 0
        }
    
    return render_template_string(HTML_TEMPLATE)

@app.route("/improve_prompt", methods=["POST"])
def improve_prompt():
    """
    Step 1: Use Gemini Flash to improve the user's prompt
    This makes the prompt more structured and effective
    """
    try:
        data = request.get_json(force=True)
        user_prompt = data.get("prompt", "").strip()
        files = data.get("files", [])
        
        if not user_prompt and not files:
            return jsonify({"improved_prompt": user_prompt})
        
        # Construct the improvement request
        improvement_system_prompt = """You are a prompt improvement specialist. Your task is to take user prompts and make them clearer, more specific, and more effective for AI responses.

Rules for improvement:
1. Keep the original intent and meaning
2. Add clarity and specificity where needed
3. Structure the prompt logically
4. Add context if it helps
5. Make it concise but comprehensive
6. If the prompt is already good, return it as is
7. Never add unnecessary complexity
8. Maintain a natural conversational tone

Simply return the improved prompt without any explanation or meta-commentary."""

        messages = [
            {"role": "system", "content": improvement_system_prompt},
            {"role": "user", "content": f"Improve this prompt: {user_prompt}"}
        ]
        
        # Add file context if present
        if files:
            file_context = "The user has also attached the following files: "
            file_context += ", ".join([f['name'] for f in files])
            messages.append({"role": "user", "content": file_context})
        
        # Try multiple API keys if needed
        for attempt in range(min(3, len(API_KEYS))):
            try:
                client = get_next_api_client(model_type="flash")
                response = client.chat.completions.create(
                    model="gemini-1.5-flash",  # Using Flash for prompt improvement
                    messages=messages,
                    temperature=0.3,  # Lower temperature for more consistent improvements
                    max_tokens=1000
                )
                
                if response.choices and response.choices[0].message:
                    improved = response.choices[0].message.content.strip()
                    return jsonify({"improved_prompt": improved})
                    
            except Exception as api_error:
                print(f"API attempt {attempt + 1} failed: {api_error}")
                continue
        
        # If all attempts fail, return original
        return jsonify({"improved_prompt": user_prompt})
        
    except Exception as e:
        print(f"Error in improve_prompt: {e}")
        return jsonify({"improved_prompt": data.get("prompt", "")})

@app.route("/chat", methods=["POST"])
def chat():
    """
    Step 2: Process the message with Gemini 2.5 Pro
    This is the main AI response generation
    """
    try:
        data = request.get_json(force=True)
        message = data.get("message", "").strip()
        files = data.get("files", [])
        
        if not message and not files:
            return jsonify({"error": "No message or files provided"}), 400
        
        # Get session
        session_id = session.get('session_id')
        if not session_id or session_id not in conversations:
            session_id = secrets.token_hex(16)
            session['session_id'] = session_id
            conversations[session_id] = {
                'messages': [],
                'token_usage': 0
            }
        
        conversation = conversations[session_id]
        
        # Check token limit
        if conversation['token_usage'] >= 125000:
            return jsonify({
                "error": "Token limit reached. Please start a new chat or compact the conversation.",
                "token_usage": conversation['token_usage']
            }), 400
        
        # Main system prompt for Gemini 2.5 Pro
        main_system_prompt = """You are Jack's AI, an advanced AI assistant with exceptional capabilities. You have a 125,000 token context window and can process images, documents, and complex queries.

CRITICAL INSTRUCTIONS:
1. NEVER cut corners or provide shortened responses
2. ALWAYS use as many tokens as necessary to provide complete, detailed answers
3. Write extensive code with full implementations - no placeholders or shortcuts
4. Explain everything clearly as if teaching a 12-year-old
5. Include detailed comments in all code
6. Provide step-by-step instructions for everything
7. When writing essays or long-form content, avoid em dashes (—) and write naturally
8. Be thorough, comprehensive, and detailed in every response
9. If asked for code, provide COMPLETE, WORKING implementations
10. Never mention being Gemini or Google - you are Jack's AI

Your responses should be:
- Extremely detailed and comprehensive
- Clear and easy to understand
- Complete with no shortcuts
- Well-structured and organized
- Natural and conversational when appropriate

Remember: Use as many tokens as needed. Length and completeness are valued over brevity."""

        # Build conversation history
        messages = [{"role": "system", "content": main_system_prompt}]
        
        # Add conversation history (last 10 messages to manage context)
        history_messages = conversation['messages'][-10:] if len(conversation['messages']) > 10 else conversation['messages']
        messages.extend(history_messages)
        
        # Process files if present
        user_content = message
        if files:
            for file in files:
                if file['type'].startswith('image/'):
                    # Handle image files
                    user_content += f"\n\n[User attached image: {file['name']}]"
                    # Note: For actual image processing, you'd need to use Gemini's vision capabilities
                else:
                    # Handle text files
                    user_content += f"\n\n[Content of {file['name']}]:\n{file.get('content', '')[:5000]}"  # Limit file content
        
        # Add current message
        messages.append({"role": "user", "content": user_content})
        
        # Try multiple API keys if needed
        reply = None
        for attempt in range(min(5, len(API_KEYS))):
            try:
                client = get_next_api_client(model_type="pro")
                response = client.chat.completions.create(
                    model="gemini-2.0-flash-exp",  # Using Gemini 2.5 Pro equivalent
                    messages=messages,
                    temperature=0.7,
                    max_tokens=60000  # Maximum tokens for detailed responses
                )
                
                if response.choices and response.choices[0].message:
                    reply = response.choices[0].message.content.strip()
                    
                    # Remove em dashes for cleaner output
                    reply = reply.replace("—", "-")
                    
                    # Update conversation
                    conversation['messages'].append({"role": "user", "content": message})
                    conversation['messages'].append({"role": "assistant", "content": reply})
                    
                    # Estimate token usage
                    new_tokens = estimate_tokens(message) + estimate_tokens(reply)
                    conversation['token_usage'] += new_tokens
                    
                    return jsonify({
                        "reply": reply,
                        "token_usage": conversation['token_usage']
                    })
                    
            except Exception as api_error:
                print(f"API attempt {attempt + 1} failed: {api_error}")
                time.sleep(1)  # Brief delay before retry
                continue
        
        # If all attempts fail
        return jsonify({"error": "Unable to process request. Please try again."}), 500
        
    except Exception as e:
        print(f"Error in chat: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/compact_conversation", methods=["POST"])
def compact_conversation():
    """
    Step 3: Use Gemini Flash to compact/summarize the conversation
    This helps manage token usage
    """
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in conversations:
            return jsonify({"error": "No conversation to compact"}), 400
        
        conversation = conversations[session_id]
        
        if len(conversation['messages']) < 4:
            return jsonify({"error": "Conversation too short to compact"}), 400
        
        # Build conversation text for summarization
        conv_text = "Conversation to summarize:\n\n"
        for msg in conversation['messages']:
            role = "User" if msg['role'] == 'user' else "Assistant"
            conv_text += f"{role}: {msg['content'][:500]}...\n\n"  # Limit each message
        
        # Compaction system prompt
        compact_system_prompt = """You are a conversation summarizer. Create a concise but comprehensive summary of the conversation that preserves all important information, context, and key points. 

The summary should:
1. Maintain all critical information
2. Preserve the context and flow
3. Keep important details and decisions
4. Be clear and well-organized
5. Be significantly shorter than the original

Provide only the summary without any meta-commentary."""

        messages = [
            {"role": "system", "content": compact_system_prompt},
            {"role": "user", "content": conv_text}
        ]
        
        # Try to compact
        for attempt in range(min(3, len(API_KEYS))):
            try:
                client = get_next_api_client(model_type="flash")
                response = client.chat.completions.create(
                    model="gemini-1.5-flash",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000
                )
                
                if response.choices and response.choices[0].message:
                    summary = response.choices[0].message.content.strip()
                    
                    # Reset conversation with summary
                    conversation['messages'] = [
                        {"role": "system", "content": f"Previous conversation summary: {summary}"}
                    ]
                    conversation['token_usage'] = estimate_tokens(summary)
                    
                    return jsonify({
                        "success": True,
                        "summary": summary,
                        "token_usage": conversation['token_usage']
                    })
                    
            except Exception as api_error:
                print(f"Compact attempt {attempt + 1} failed: {api_error}")
                continue
        
        return jsonify({"error": "Failed to compact conversation"}), 500
        
    except Exception as e:
        print(f"Error in compact_conversation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    """Clear the current conversation"""
    try:
        session_id = session.get('session_id')
        if session_id and session_id in conversations:
            conversations[session_id] = {
                'messages': [],
                'token_usage': 0
            }
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error in clear_chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/load_conversation", methods=["GET"])
def load_conversation():
    """Load existing conversation for the session"""
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in conversations:
            return jsonify({"messages": [], "token_usage": 0})
        
        conversation = conversations[session_id]
        return jsonify({
            "messages": conversation['messages'],
            "token_usage": conversation['token_usage']
        })
    except Exception as e:
        print(f"Error in load_conversation: {e}")
        return jsonify({"messages": [], "token_usage": 0})

if __name__ == "__main__":
    # Get port from environment variable (for Render deployment)
    port = int(os.environ.get("PORT", 5000))
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=port, debug=False)