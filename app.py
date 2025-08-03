"""
Jack's AI - Complete Web Application for Render.com
This application provides an AI assistant interface using Google Gemini models
with prompt optimization, conversation management, and file handling capabilities.
"""

import os
import json
import base64
import hashlib
import secrets
import mimetypes
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from io import BytesIO
import traceback
import time
import random

# Flask and related imports for web server functionality
from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

# Google Generative AI import for Gemini models
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Initialize Flask application with secret key for sessions
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)  # Generate a random secret key for session management
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Set maximum upload size to 100MB
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Security: prevent JavaScript access to session cookie
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Security: CSRF protection
CORS(app)  # Enable Cross-Origin Resource Sharing for API access

# In-memory storage for all application data
# This dictionary stores all conversations and user data in memory
MEMORY_STORAGE = {
    'conversations': {},  # Stores all chat conversations by session ID
    'active_sessions': {},  # Tracks active user sessions
    'file_cache': {},  # Temporary storage for uploaded files
    'api_key_usage': {},  # Tracks API key usage for load balancing
    'total_users': 0,  # Counter for total users
    'daily_stats': {}  # Daily usage statistics
}

# Configuration for API keys with load balancing
# Store your Gemini API keys in environment variables GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.
API_KEYS = []
for i in range(1, 11):  # Load up to 10 API keys from environment
    key = os.environ.get(f'GEMINI_API_KEY_{i}')
    if key:
        API_KEYS.append(key)
        MEMORY_STORAGE['api_key_usage'][key] = {
            'count': 0,
            'last_used': None,
            'failures': 0
        }

# Fallback to single API key if numbered keys not found
if not API_KEYS:
    default_key = os.environ.get('GEMINI_API_KEY')
    if default_key:
        API_KEYS.append(default_key)
        MEMORY_STORAGE['api_key_usage'][default_key] = {
            'count': 0,
            'last_used': None,
            'failures': 0
        }

# Model configuration constants
MAX_TOKENS = 125000  # Maximum context window size
TOKEN_WARNING_THRESHOLD = 100000  # Warn users when approaching token limit
ALLOWED_EXTENSIONS = {
    'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 
    'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx',
    'csv', 'json', 'xml', 'html', 'css', 'js', 'py',
    'java', 'cpp', 'c', 'h', 'md', 'yaml', 'yml'
}

@dataclass
class Message:
    """
    Represents a single message in a conversation.
    This class stores all information about a message including content, files, and metadata.
    """
    role: str  # 'user' or 'assistant'
    content: str  # The text content of the message
    timestamp: datetime = field(default_factory=datetime.now)  # When the message was created
    files: List[Dict] = field(default_factory=list)  # List of attached files
    tokens_used: int = 0  # Number of tokens this message consumed
    prompt_improved: bool = False  # Whether this message had its prompt improved
    original_prompt: Optional[str] = None  # Original prompt before improvement

@dataclass
class Conversation:
    """
    Represents a complete conversation thread.
    Manages the entire conversation history and metadata for a user session.
    """
    id: str  # Unique conversation identifier
    messages: List[Message] = field(default_factory=list)  # All messages in the conversation
    total_tokens: int = 0  # Total tokens used in this conversation
    created_at: datetime = field(default_factory=datetime.now)  # When conversation started
    last_activity: datetime = field(default_factory=datetime.now)  # Last activity time
    compacted: bool = False  # Whether conversation has been compacted
    title: Optional[str] = None  # Optional conversation title

class GeminiManager:
    """
    Manages all interactions with Google Gemini AI models.
    Handles API key rotation, failover, and model selection.
    """
    
    def __init__(self):
        """Initialize the Gemini manager with API keys and model configurations."""
        self.api_keys = API_KEYS
        self.current_key_index = 0
        self.models = {}  # Cache for initialized models
        self.initialize_models()
    
    def initialize_models(self):
        """
        Initialize all three Gemini models with the first available API key.
        Sets up prompt improver, main assistant, and conversation compactor.
        """
        if not self.api_keys:
            raise ValueError("No API keys found! Please set GEMINI_API_KEY environment variables.")
        
        # Configure the first API key
        genai.configure(api_key=self.api_keys[0])
        
        # Initialize the three models we'll use
        try:
            # Model 1: Prompt Improver (Flash model for speed)
            self.prompt_improver = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 2048
                }
            )
            
            # Model 2: Main Assistant (Pro model for quality)
            self.main_assistant = genai.GenerativeModel(
                model_name="gemini-1.5-pro",
                generation_config={
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192
                }
            )
            
            # Model 3: Conversation Compactor (Flash for summarization)
            self.compactor = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "top_k": 30,
                    "max_output_tokens": 4096
                }
            )
            
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            raise
    
    def get_next_api_key(self) -> str:
        """
        Rotate to the next available API key for load balancing.
        Implements round-robin selection with failure tracking.
        """
        if not self.api_keys:
            raise ValueError("No API keys available")
        
        # Try to find a working API key
        attempts = 0
        while attempts < len(self.api_keys):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            key = self.api_keys[self.current_key_index]
            
            # Check if this key has too many recent failures
            usage = MEMORY_STORAGE['api_key_usage'].get(key, {})
            if usage.get('failures', 0) < 5:  # Allow up to 5 failures before skipping
                return key
            
            attempts += 1
        
        # If all keys have failures, reset and use the first one
        for key in self.api_keys:
            MEMORY_STORAGE['api_key_usage'][key]['failures'] = 0
        
        return self.api_keys[0]
    
    def switch_api_key(self):
        """
        Switch to the next API key and reinitialize models.
        Called when current key fails or reaches rate limits.
        """
        try:
            new_key = self.get_next_api_key()
            genai.configure(api_key=new_key)
            self.initialize_models()
            
            # Update usage statistics
            MEMORY_STORAGE['api_key_usage'][new_key]['count'] += 1
            MEMORY_STORAGE['api_key_usage'][new_key]['last_used'] = datetime.now()
            
        except Exception as e:
            print(f"Error switching API key: {str(e)}")
            raise
    
    def improve_prompt(self, original_prompt: str) -> Tuple[str, bool]:
        """
        Use Gemini Flash to improve the user's prompt.
        Returns improved prompt and success status.
        """
        system_prompt = """You are a prompt optimization expert. Your task is to improve user prompts to be clearer, more specific, and more effective for an AI assistant.

        Rules for improvement:
        1. Maintain the original intent and meaning
        2. Add clarity and specificity where needed
        3. Structure the prompt logically
        4. Include relevant context if missing
        5. Make it concise but comprehensive
        6. Fix any grammar or spelling errors
        7. Add specific output format requests if beneficial

        Important: Return ONLY the improved prompt, nothing else. No explanations, no prefixes, just the improved prompt text."""
        
        try:
            # Attempt to improve the prompt with retry logic
            for attempt in range(3):  # Try up to 3 times with different keys
                try:
                    response = self.prompt_improver.generate_content(
                        f"{system_prompt}\n\nOriginal prompt: {original_prompt}",
                        safety_settings={
                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        }
                    )
                    
                    if response and response.text:
                        return response.text.strip(), True
                    
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < 2:  # Don't switch on last attempt
                        self.switch_api_key()
                    continue
            
            # If all attempts fail, return original
            return original_prompt, False
            
        except Exception as e:
            print(f"Error improving prompt: {str(e)}")
            return original_prompt, False
    
    def generate_response(self, prompt: str, files: List[Dict] = None) -> Tuple[str, int]:
        """
        Generate a response using Gemini Pro with full context.
        Returns the response text and token count.
        """
        system_prompt = """You are Jack's AI, an advanced AI assistant with extraordinary capabilities. You must ALWAYS:

        1. NEVER cut corners or provide shortcuts - write complete, thorough responses
        2. Use as many tokens as necessary to fully answer the question
        3. Provide extremely detailed explanations as if explaining to a 12-year-old
        4. Include comprehensive code examples when relevant - NEVER use placeholders
        5. Write out EVERYTHING in full - no "..." or "etc." or "and so on"
        6. Break down complex topics into simple, understandable steps
        7. Use examples, analogies, and illustrations liberally
        8. Double-check your work and provide complete solutions
        9. If writing code, include detailed comments explaining every line
        10. Structure responses with clear headings and organization

        Remember: You have a 125,000 token context window - USE IT! The user wants comprehensive, detailed answers. Quality and completeness are paramount."""
        
        try:
            # Prepare the full prompt with system instructions
            full_prompt = f"{system_prompt}\n\nUser request: {prompt}"
            
            # Add file context if files are provided
            if files:
                file_context = "\n\nAttached files:\n"
                for file in files:
                    file_context += f"- {file['name']} ({file['type']}): {file.get('description', 'No description')}\n"
                full_prompt += file_context
            
            # Generate response with retry logic
            for attempt in range(3):
                try:
                    response = self.main_assistant.generate_content(
                        full_prompt,
                        safety_settings={
                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        }
                    )
                    
                    if response and response.text:
                        # Estimate token count (rough approximation)
                        token_count = len(response.text.split()) * 1.3
                        return response.text, int(token_count)
                    
                except Exception as e:
                    print(f"Response generation attempt {attempt + 1} failed: {str(e)}")
                    if attempt < 2:
                        self.switch_api_key()
                    continue
            
            return "I apologize, but I'm having trouble generating a response right now. Please try again.", 0
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"An error occurred: {str(e)}", 0
    
    def compact_conversation(self, messages: List[Message]) -> str:
        """
        Compact a conversation using Gemini Flash to reduce token usage.
        Returns a summarized version of the conversation.
        """
        system_prompt = """You are a conversation summarizer. Your task is to compact a conversation while preserving ALL important information, context, and details.

        Rules for compaction:
        1. Preserve all factual information, data, and specific details
        2. Maintain the flow and context of the conversation
        3. Keep all code snippets, formulas, and technical details intact
        4. Summarize verbose explanations while keeping the meaning
        5. Combine related messages when possible
        6. Remove only truly redundant information
        7. Maintain chronological order and conversation structure

        Format your summary as a clear, structured document that can be used as context for continuing the conversation."""
        
        try:
            # Build conversation text
            conversation_text = "Conversation to compact:\n\n"
            for msg in messages:
                conversation_text += f"{msg.role.upper()}: {msg.content}\n"
                if msg.files:
                    conversation_text += f"[Attached files: {', '.join([f['name'] for f in msg.files])}]\n"
                conversation_text += "\n"
            
            # Generate compacted version
            response = self.compactor.generate_content(
                f"{system_prompt}\n\n{conversation_text}",
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            if response and response.text:
                return response.text
            
            return "Unable to compact conversation."
            
        except Exception as e:
            print(f"Error compacting conversation: {str(e)}")
            return "Error during compaction."

# Initialize Gemini Manager
gemini_manager = GeminiManager() if API_KEYS else None

def get_or_create_session() -> str:
    """
    Get existing session ID or create a new one.
    Manages user sessions without requiring login.
    """
    if 'session_id' not in session:
        # Create new session ID
        session['session_id'] = secrets.token_hex(16)
        session['created_at'] = datetime.now().isoformat()
        
        # Initialize session in memory storage
        MEMORY_STORAGE['active_sessions'][session['session_id']] = {
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'conversations': []
        }
        
        # Increment total users counter
        MEMORY_STORAGE['total_users'] += 1
    
    # Update last activity
    MEMORY_STORAGE['active_sessions'][session['session_id']]['last_activity'] = datetime.now()
    
    return session['session_id']

def cleanup_old_sessions():
    """
    Clean up sessions older than 24 hours to manage memory.
    This function should be called periodically.
    """
    current_time = datetime.now()
    sessions_to_remove = []
    
    for session_id, session_data in MEMORY_STORAGE['active_sessions'].items():
        if current_time - session_data['last_activity'] > timedelta(hours=24):
            sessions_to_remove.append(session_id)
    
    # Remove old sessions and their conversations
    for session_id in sessions_to_remove:
        # Remove conversations
        for conv_id in MEMORY_STORAGE['active_sessions'][session_id].get('conversations', []):
            if conv_id in MEMORY_STORAGE['conversations']:
                del MEMORY_STORAGE['conversations'][conv_id]
        
        # Remove session
        del MEMORY_STORAGE['active_sessions'][session_id]
    
    print(f"Cleaned up {len(sessions_to_remove)} old sessions")

def process_file_upload(file: FileStorage) -> Dict:
    """
    Process an uploaded file and prepare it for AI analysis.
    Returns file metadata and content information.
    """
    try:
        # Secure the filename
        filename = secure_filename(file.filename)
        
        # Check file extension
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        if file_ext not in ALLOWED_EXTENSIONS:
            return {'error': f'File type .{file_ext} not allowed'}
        
        # Read file content
        file_content = file.read()
        file_size = len(file_content)
        
        # Reset file pointer
        file.seek(0)
        
        # Prepare file data based on type
        file_data = {
            'name': filename,
            'type': file.content_type or mimetypes.guess_type(filename)[0],
            'size': file_size,
            'extension': file_ext,
            'uploaded_at': datetime.now().isoformat()
        }
        
        # For text files, store content directly
        if file_ext in ['txt', 'csv', 'json', 'xml', 'html', 'css', 'js', 'py', 'java', 'cpp', 'c', 'h', 'md', 'yaml', 'yml']:
            try:
                file_data['content'] = file_content.decode('utf-8')
                file_data['description'] = f"Text file with {len(file_data['content'])} characters"
            except:
                file_data['content'] = base64.b64encode(file_content).decode('utf-8')
                file_data['description'] = f"Binary file ({file_size} bytes)"
        
        # For images, store as base64
        elif file_ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
            file_data['content'] = base64.b64encode(file_content).decode('utf-8')
            file_data['description'] = f"Image file ({file_ext.upper()}, {file_size} bytes)"
        
        # For other files, store metadata only
        else:
            file_data['content'] = base64.b64encode(file_content).decode('utf-8')
            file_data['description'] = f"Document file ({file_ext.upper()}, {file_size} bytes)"
        
        return file_data
        
    except Exception as e:
        print(f"Error processing file upload: {str(e)}")
        return {'error': str(e)}

# HTML Template with embedded CSS and JavaScript
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jack's AI - Advanced AI Assistant</title>
    
    <!-- Modern CSS Styles for Beautiful UI -->
    <style>
        /* CSS Variables for easy theme customization */
        :root {
            --primary-color: #6366f1;
            --primary-dark: #4f46e5;
            --primary-light: #818cf8;
            --secondary-color: #10b981;
            --danger-color: #ef4444;
            --warning-color: #f59e0b;
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --text-muted: #94a3b8;
            --border-color: #475569;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
            --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --gradient-secondary: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --gradient-success: linear-gradient(135deg, #13f1fc 0%, #0470dc 100%);
        }
        
        /* Global Styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        /* Animated Background */
        .background-animation {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            background: linear-gradient(45deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
            overflow: hidden;
        }
        
        .background-animation::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.1) 0%, transparent 70%);
            animation: rotate 30s linear infinite;
        }
        
        @keyframes rotate {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Container */
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 1rem;
            position: relative;
            z-index: 1;
        }
        
        /* Header */
        .header {
            background: var(--bg-secondary);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-xl);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border-color);
            animation: slideDown 0.5s ease-out;
        }
        
        @keyframes slideDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1.5rem;
            font-weight: bold;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: pulse 2s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }
        
        .logo-icon {
            width: 40px;
            height: 40px;
            background: var(--gradient-primary);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            color: white;
            box-shadow: var(--shadow-lg);
        }
        
        .header-info {
            display: flex;
            gap: 2rem;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .status-badge {
            background: var(--gradient-success);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 2rem;
            font-size: 0.875rem;
            font-weight: 600;
            box-shadow: var(--shadow-md);
            animation: shimmer 2s ease-in-out infinite;
        }
        
        @keyframes shimmer {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.9; }
        }
        
        .beta-notice {
            background: var(--warning-color);
            color: var(--bg-primary);
            padding: 0.5rem 1rem;
            border-radius: 2rem;
            font-size: 0.875rem;
            font-weight: 600;
            animation: bounce 2s ease-in-out infinite;
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-2px); }
        }
        
        /* Main Chat Container */
        .chat-container {
            background: var(--bg-secondary);
            border-radius: 1rem;
            box-shadow: var(--shadow-xl);
            border: 1px solid var(--border-color);
            overflow: hidden;
            animation: fadeIn 0.5s ease-out;
            min-height: 600px;
            display: flex;
            flex-direction: column;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: scale(0.95);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        /* Token Usage Bar */
        .token-bar-container {
            padding: 1rem;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-color);
        }
        
        .token-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
            font-size: 0.875rem;
        }
        
        .token-label {
            color: var(--text-secondary);
        }
        
        .token-count {
            font-weight: 600;
            color: var(--primary-light);
        }
        
        .token-bar {
            width: 100%;
            height: 8px;
            background: var(--bg-primary);
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }
        
        .token-progress {
            height: 100%;
            background: var(--gradient-primary);
            border-radius: 4px;
            transition: width 0.3s ease;
            box-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
            position: relative;
            overflow: hidden;
        }
        
        .token-progress::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            bottom: 0;
            right: 0;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            animation: shimmerBar 2s infinite;
        }
        
        @keyframes shimmerBar {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        /* Chat Actions Bar */
        .chat-actions {
            padding: 1rem;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        
        .action-button {
            padding: 0.5rem 1rem;
            background: var(--bg-secondary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            cursor: pointer;
            font-size: 0.875rem;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .action-button:hover {
            background: var(--primary-color);
            color: white;
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }
        
        .action-button.danger {
            border-color: var(--danger-color);
            color: var(--danger-color);
        }
        
        .action-button.danger:hover {
            background: var(--danger-color);
            color: white;
        }
        
        .action-button.success {
            border-color: var(--secondary-color);
            color: var(--secondary-color);
        }
        
        .action-button.success:hover {
            background: var(--secondary-color);
            color: white;
        }
        
        /* Messages Area */
        .messages-container {
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            min-height: 400px;
            max-height: 600px;
        }
        
        .message {
            display: flex;
            gap: 1rem;
            animation: messageSlide 0.3s ease-out;
            max-width: 100%;
        }
        
        @keyframes messageSlide {
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
            justify-content: flex-end;
        }
        
        .message-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.875rem;
            flex-shrink: 0;
        }
        
        .user .message-avatar {
            background: var(--gradient-primary);
            color: white;
            order: 2;
        }
        
        .assistant .message-avatar {
            background: var(--gradient-secondary);
            color: white;
        }
        
        .message-content {
            background: var(--bg-tertiary);
            padding: 1rem;
            border-radius: 1rem;
            max-width: 70%;
            word-wrap: break-word;
            box-shadow: var(--shadow-md);
            position: relative;
        }
        
        .user .message-content {
            background: var(--primary-color);
            color: white;
            border-bottom-right-radius: 0.25rem;
        }
        
        .assistant .message-content {
            background: var(--bg-tertiary);
            border-bottom-left-radius: 0.25rem;
        }
        
        .message-content pre {
            background: var(--bg-primary);
            padding: 1rem;
            border-radius: 0.5rem;
            overflow-x: auto;
            margin: 0.5rem 0;
        }
        
        .message-content code {
            background: var(--bg-primary);
            padding: 0.2rem 0.4rem;
            border-radius: 0.25rem;
            font-size: 0.875rem;
        }
        
        .message-timestamp {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
        }
        
        .message-files {
            display: flex;
            gap: 0.5rem;
            margin-top: 0.5rem;
            flex-wrap: wrap;
        }
        
        .file-tag {
            background: var(--bg-primary);
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }
        
        /* Typing Indicator */
        .typing-indicator {
            display: none;
            align-items: center;
            gap: 0.5rem;
            padding: 1rem;
        }
        
        .typing-indicator.active {
            display: flex;
        }
        
        .typing-dots {
            display: flex;
            gap: 0.25rem;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            background: var(--primary-light);
            border-radius: 50%;
            animation: typingAnimation 1.4s infinite ease-in-out;
        }
        
        .typing-dot:nth-child(1) {
            animation-delay: -0.32s;
        }
        
        .typing-dot:nth-child(2) {
            animation-delay: -0.16s;
        }
        
        @keyframes typingAnimation {
            0%, 80%, 100% {
                transform: scale(1);
                opacity: 0.5;
            }
            40% {
                transform: scale(1.3);
                opacity: 1;
            }
        }
        
        /* Input Area */
        .input-container {
            padding: 1.5rem;
            background: var(--bg-tertiary);
            border-top: 1px solid var(--border-color);
        }
        
        .input-wrapper {
            display: flex;
            gap: 1rem;
            align-items: flex-end;
        }
        
        .input-group {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .file-upload-area {
            display: none;
            padding: 1rem;
            background: var(--bg-secondary);
            border: 2px dashed var(--border-color);
            border-radius: 0.5rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .file-upload-area.active {
            display: block;
        }
        
        .file-upload-area:hover {
            border-color: var(--primary-color);
            background: rgba(99, 102, 241, 0.1);
        }
        
        .file-upload-area.dragover {
            border-color: var(--primary-light);
            background: rgba(99, 102, 241, 0.2);
        }
        
        .file-list {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-top: 0.5rem;
        }
        
        .file-item {
            background: var(--bg-secondary);
            padding: 0.5rem;
            border-radius: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.875rem;
            border: 1px solid var(--border-color);
        }
        
        .file-remove {
            cursor: pointer;
            color: var(--danger-color);
            font-weight: bold;
        }
        
        .file-remove:hover {
            color: var(--text-primary);
        }
        
        .message-input {
            width: 100%;
            padding: 1rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            color: var(--text-primary);
            font-size: 1rem;
            resize: vertical;
            min-height: 50px;
            max-height: 200px;
            transition: all 0.3s ease;
        }
        
        .message-input:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }
        
        .input-actions {
            display: flex;
            gap: 0.5rem;
        }
        
        .icon-button {
            padding: 0.75rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .icon-button:hover {
            background: var(--primary-color);
            color: white;
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }
        
        .send-button {
            padding: 0.75rem 1.5rem;
            background: var(--gradient-primary);
            color: white;
            border: none;
            border-radius: 0.5rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .send-button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }
        
        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        /* Prompt Improvement Modal */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(5px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            animation: fadeIn 0.3s ease-out;
        }
        
        .modal.active {
            display: flex;
        }
        
        .modal-content {
            background: var(--bg-secondary);
            border-radius: 1rem;
            padding: 2rem;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: var(--shadow-xl);
            border: 1px solid var(--border-color);
            animation: slideUp 0.3s ease-out;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .modal-header {
            margin-bottom: 1.5rem;
        }
        
        .modal-title {
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }
        
        .modal-subtitle {
            color: var(--text-secondary);
            font-size: 0.875rem;
        }
        
        .prompt-comparison {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .prompt-option {
            padding: 1rem;
            background: var(--bg-tertiary);
            border: 2px solid var(--border-color);
            border-radius: 0.5rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .prompt-option:hover {
            border-color: var(--primary-color);
            transform: translateX(5px);
        }
        
        .prompt-option.selected {
            border-color: var(--primary-light);
            background: rgba(99, 102, 241, 0.1);
        }
        
        .prompt-label {
            font-weight: 600;
            color: var(--primary-light);
            margin-bottom: 0.5rem;
            font-size: 0.875rem;
        }
        
        .prompt-text {
            color: var(--text-primary);
            line-height: 1.5;
        }
        
        .modal-actions {
            display: flex;
            gap: 1rem;
            margin-top: 1.5rem;
            justify-content: flex-end;
        }
        
        .modal-button {
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            border: none;
        }
        
        .modal-button.primary {
            background: var(--gradient-primary);
            color: white;
        }
        
        .modal-button.primary:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }
        
        .modal-button.secondary {
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }
        
        .modal-button.secondary:hover {
            background: var(--bg-primary);
        }
        
        /* Welcome Screen */
        .welcome-screen {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem;
            text-align: center;
            animation: fadeIn 0.5s ease-out;
        }
        
        .welcome-title {
            font-size: 2.5rem;
            font-weight: bold;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }
        
        .welcome-subtitle {
            font-size: 1.25rem;
            color: var(--text-secondary);
            margin-bottom: 2rem;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
            width: 100%;
            max-width: 800px;
        }
        
        .feature-card {
            background: var(--bg-tertiary);
            padding: 1.5rem;
            border-radius: 0.75rem;
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: var(--shadow-xl);
            border-color: var(--primary-color);
        }
        
        .feature-icon {
            font-size: 2rem;
            margin-bottom: 0.75rem;
        }
        
        .feature-title {
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }
        
        .feature-description {
            font-size: 0.875rem;
            color: var(--text-secondary);
        }
        
        /* Loading Spinner */
        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid var(--border-color);
            border-top: 3px solid var(--primary-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .container {
                padding: 0.5rem;
            }
            
            .header {
                padding: 1rem;
            }
            
            .header-content {
                flex-direction: column;
                text-align: center;
            }
            
            .header-info {
                flex-direction: column;
                gap: 0.5rem;
            }
            
            .message-content {
                max-width: 85%;
            }
            
            .input-wrapper {
                flex-direction: column;
            }
            
            .features-grid {
                grid-template-columns: 1fr;
            }
            
            .welcome-title {
                font-size: 2rem;
            }
            
            .welcome-subtitle {
                font-size: 1rem;
            }
            
            .modal-content {
                padding: 1.5rem;
            }
        }
        
        @media (max-width: 480px) {
            .message-content {
                max-width: 90%;
            }
            
            .logo {
                font-size: 1.25rem;
            }
            
            .logo-icon {
                width: 32px;
                height: 32px;
            }
            
            .action-button {
                padding: 0.4rem 0.8rem;
                font-size: 0.75rem;
            }
            
            .welcome-title {
                font-size: 1.5rem;
            }
        }
        
        /* Print Styles */
        @media print {
            body {
                background: white;
                color: black;
            }
            
            .header, .input-container, .chat-actions, .token-bar-container {
                display: none;
            }
            
            .messages-container {
                max-height: none;
            }
            
            .message-content {
                box-shadow: none;
                border: 1px solid #ddd;
            }
        }
        
        /* Accessibility */
        .sr-only {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border-width: 0;
        }
        
        /* Focus Styles */
        button:focus-visible,
        input:focus-visible,
        textarea:focus-visible {
            outline: 2px solid var(--primary-light);
            outline-offset: 2px;
        }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--bg-tertiary);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--border-color);
        }
        
        /* Toast Notifications */
        .toast-container {
            position: fixed;
            top: 1rem;
            right: 1rem;
            z-index: 2000;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .toast {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: var(--shadow-xl);
            display: flex;
            align-items: center;
            gap: 0.75rem;
            min-width: 300px;
            animation: slideInRight 0.3s ease-out;
        }
        
        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(100%);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        .toast.success {
            border-color: var(--secondary-color);
        }
        
        .toast.error {
            border-color: var(--danger-color);
        }
        
        .toast.warning {
            border-color: var(--warning-color);
        }
        
        .toast-icon {
            font-size: 1.25rem;
        }
        
        .toast.success .toast-icon {
            color: var(--secondary-color);
        }
        
        .toast.error .toast-icon {
            color: var(--danger-color);
        }
        
        .toast.warning .toast-icon {
            color: var(--warning-color);
        }
        
        .toast-message {
            flex: 1;
            color: var(--text-primary);
        }
        
        .toast-close {
            cursor: pointer;
            color: var(--text-muted);
            font-size: 1.25rem;
        }
        
        .toast-close:hover {
            color: var(--text-primary);
        }
    </style>
</head>
<body>
    <!-- Animated Background -->
    <div class="background-animation"></div>
    
    <!-- Main Container -->
    <div class="container">
        <!-- Header -->
        <header class="header">
            <div class="header-content">
                <div class="logo">
                    <div class="logo-icon">🤖</div>
                    <span>Jack's AI</span>
                </div>
                <div class="header-info">
                    <span class="status-badge">🟢 Online</span>
                    <span class="beta-notice">🎉 Temporarily Free!</span>
                </div>
            </div>
        </header>
        
        <!-- Main Chat Container -->
        <main class="chat-container">
            <!-- Token Usage Bar -->
            <div class="token-bar-container">
                <div class="token-info">
                    <span class="token-label">Context Window Usage</span>
                    <span class="token-count">
                        <span id="currentTokens">0</span> / <span id="maxTokens">125,000</span> tokens
                    </span>
                </div>
                <div class="token-bar">
                    <div class="token-progress" id="tokenProgress" style="width: 0%"></div>
                </div>
            </div>
            
            <!-- Chat Actions Bar -->
            <div class="chat-actions">
                <button class="action-button" onclick="startNewChat()">
                    <span>🆕</span>
                    <span>New Chat</span>
                </button>
                <button class="action-button success" onclick="compactConversation()">
                    <span>📦</span>
                    <span>Compact</span>
                </button>
                <button class="action-button" onclick="exportChat()">
                    <span>💾</span>
                    <span>Export</span>
                </button>
                <button class="action-button danger" onclick="clearChat()">
                    <span>🗑️</span>
                    <span>Clear</span>
                </button>
            </div>
            
            <!-- Messages Container -->
            <div class="messages-container" id="messagesContainer">
                <!-- Welcome Screen -->
                <div class="welcome-screen" id="welcomeScreen">
                    <h1 class="welcome-title">Welcome to Jack's AI</h1>
                    <p class="welcome-subtitle">Your Advanced AI Assistant with 125K Context Window</p>
                    
                    <div class="features-grid">
                        <div class="feature-card">
                            <div class="feature-icon">🧠</div>
                            <div class="feature-title">Smart Prompts</div>
                            <div class="feature-description">AI-powered prompt optimization for better results</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-icon">📎</div>
                            <div class="feature-title">File Support</div>
                            <div class="feature-description">Upload images, documents, and more for analysis</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-icon">💬</div>
                            <div class="feature-title">Long Context</div>
                            <div class="feature-description">125,000 token context window for extended conversations</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-icon">⚡</div>
                            <div class="feature-title">Fast & Reliable</div>
                            <div class="feature-description">Multiple API keys with automatic failover</div>
                        </div>
                    </div>
                    
                    <p style="margin-top: 2rem; color: var(--text-muted); font-size: 0.875rem;">
                        Start by typing a message or uploading a file below!
                    </p>
                </div>
            </div>
            
            <!-- Typing Indicator -->
            <div class="typing-indicator" id="typingIndicator">
                <div class="message-avatar">🤖</div>
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
                <span style="color: var(--text-muted); margin-left: 0.5rem;">Jack's AI is thinking...</span>
            </div>
            
            <!-- Input Container -->
            <div class="input-container">
                <div class="input-wrapper">
                    <div class="input-group">
                        <!-- File Upload Area -->
                        <div class="file-upload-area" id="fileUploadArea" onclick="document.getElementById('fileInput').click()">
                            <span>📁 Drop files here or click to upload</span>
                            <div class="file-list" id="fileList"></div>
                        </div>
                        
                        <!-- Message Input -->
                        <textarea 
                            class="message-input" 
                            id="messageInput" 
                            placeholder="Type your message here... (Shift+Enter for new line, Enter to send)"
                            rows="1"></textarea>
                    </div>
                    
                    <!-- Input Actions -->
                    <div class="input-actions">
                        <button class="icon-button" onclick="toggleFileUpload()" title="Attach files">
                            📎
                        </button>
                        <button class="send-button" id="sendButton" onclick="sendMessage()">
                            <span>Send</span>
                            <span>➤</span>
                        </button>
                    </div>
                </div>
                
                <!-- Hidden File Input -->
                <input type="file" id="fileInput" multiple style="display: none;" onchange="handleFileSelect(event)">
            </div>
        </main>
    </div>
    
    <!-- Prompt Improvement Modal -->
    <div class="modal" id="promptModal">
        <div class="modal-content">
            <div class="modal-header">
                <h2 class="modal-title">Prompt Optimization</h2>
                <p class="modal-subtitle">We've improved your prompt for better results. Choose which version to use:</p>
            </div>
            
            <div class="prompt-comparison">
                <div class="prompt-option" id="originalPromptOption" onclick="selectPromptOption('original')">
                    <div class="prompt-label">Original Prompt</div>
                    <div class="prompt-text" id="originalPromptText"></div>
                </div>
                
                <div class="prompt-option selected" id="improvedPromptOption" onclick="selectPromptOption('improved')">
                    <div class="prompt-label">✨ Improved Prompt (Recommended)</div>
                    <div class="prompt-text" id="improvedPromptText"></div>
                </div>
            </div>
            
            <div class="modal-actions">
                <button class="modal-button secondary" onclick="closePromptModal()">Cancel</button>
                <button class="modal-button primary" onclick="confirmPromptSelection()">Use Selected</button>
            </div>
        </div>
    </div>
    
    <!-- Toast Container -->
    <div class="toast-container" id="toastContainer"></div>
    
    <!-- JavaScript Code -->
    <script>
        // Global variables for managing application state
        let conversationId = null;  // Current conversation ID
        let currentTokens = 0;  // Current token usage
        let selectedPromptType = 'improved';  // Selected prompt type in modal
        let pendingMessage = null;  // Message waiting for prompt selection
        let uploadedFiles = [];  // Currently uploaded files
        
        // Initialize the application when page loads
        document.addEventListener('DOMContentLoaded', function() {
            // Initialize conversation
            initializeConversation();
            
            // Set up event listeners
            setupEventListeners();
            
            // Auto-resize textarea
            setupAutoResize();
            
            // Setup drag and drop for file upload
            setupDragAndDrop();
            
            // Load any existing conversation
            loadConversation();
        });
        
        // Initialize a new conversation
        function initializeConversation() {
            // Generate a unique conversation ID
            conversationId = generateId();
            currentTokens = 0;
            updateTokenBar(0);
        }
        
        // Generate a unique ID for conversations
        function generateId() {
            return 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        }
        
        // Set up all event listeners
        function setupEventListeners() {
            // Enter key to send message (Shift+Enter for new line)
            document.getElementById('messageInput').addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
            
            // File input change
            document.getElementById('fileInput').addEventListener('change', handleFileSelect);
            
            // Click outside modal to close
            document.getElementById('promptModal').addEventListener('click', function(e) {
                if (e.target === this) {
                    closePromptModal();
                }
            });
        }
        
        // Set up auto-resize for textarea
        function setupAutoResize() {
            const textarea = document.getElementById('messageInput');
            textarea.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 200) + 'px';
            });
        }
        
        // Set up drag and drop functionality
        function setupDragAndDrop() {
            const uploadArea = document.getElementById('fileUploadArea');
            
            // Prevent default drag behaviors
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, preventDefaults, false);
                document.body.addEventListener(eventName, preventDefaults, false);
            });
            
            // Highlight drop area when item is dragged over it
            ['dragenter', 'dragover'].forEach(eventName => {
                uploadArea.addEventListener(eventName, highlight, false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, unhighlight, false);
            });
            
            // Handle dropped files
            uploadArea.addEventListener('drop', handleDrop, false);
        }
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        function highlight(e) {
            document.getElementById('fileUploadArea').classList.add('dragover');
        }
        
        function unhighlight(e) {
            document.getElementById('fileUploadArea').classList.remove('dragover');
        }
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFiles(files);
        }
        
        // Toggle file upload area visibility
        function toggleFileUpload() {
            const uploadArea = document.getElementById('fileUploadArea');
            uploadArea.classList.toggle('active');
        }
        
        // Handle file selection
        function handleFileSelect(e) {
            const files = e.target.files;
            handleFiles(files);
        }
        
        // Process selected files
        function handleFiles(files) {
            for (let file of files) {
                // Check file size (max 10MB per file)
                if (file.size > 10 * 1024 * 1024) {
                    showToast('error', `File ${file.name} is too large (max 10MB)`);
                    continue;
                }
                
                // Add file to uploaded files
                uploadedFiles.push(file);
                
                // Display file in list
                displayFile(file);
            }
            
            // Show upload area if files are added
            if (uploadedFiles.length > 0) {
                document.getElementById('fileUploadArea').classList.add('active');
            }
        }
        
        // Display uploaded file in the list
        function displayFile(file) {
            const fileList = document.getElementById('fileList');
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <span>📄 ${file.name}</span>
                <span class="file-remove" onclick="removeFile('${file.name}')">×</span>
            `;
            fileList.appendChild(fileItem);
        }
        
        // Remove file from upload list
        function removeFile(fileName) {
            uploadedFiles = uploadedFiles.filter(f => f.name !== fileName);
            
            // Update display
            const fileList = document.getElementById('fileList');
            fileList.innerHTML = '';
            uploadedFiles.forEach(file => displayFile(file));
            
            // Hide upload area if no files
            if (uploadedFiles.length === 0) {
                document.getElementById('fileUploadArea').classList.remove('active');
            }
        }
        
        // Send message to the server
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message && uploadedFiles.length === 0) {
                showToast('warning', 'Please enter a message or upload a file');
                return;
            }
            
            // Disable send button
            document.getElementById('sendButton').disabled = true;
            
            // Hide welcome screen
            const welcomeScreen = document.getElementById('welcomeScreen');
            if (welcomeScreen) {
                welcomeScreen.style.display = 'none';
            }
            
            // Display user message
            displayMessage('user', message, uploadedFiles);
            
            // Clear input and files
            input.value = '';
            input.style.height = 'auto';
            const filesToSend = [...uploadedFiles];
            uploadedFiles = [];
            document.getElementById('fileList').innerHTML = '';
            document.getElementById('fileUploadArea').classList.remove('active');
            
            // Show typing indicator
            document.getElementById('typingIndicator').classList.add('active');
            
            try {
                // Prepare form data
                const formData = new FormData();
                formData.append('message', message);
                formData.append('conversation_id', conversationId);
                
                // Add files to form data
                filesToSend.forEach(file => {
                    formData.append('files', file);
                });
                
                // Send request to server
                const response = await fetch('/chat', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Failed to send message');
                }
                
                const data = await response.json();
                
                // Check if prompt was improved
                if (data.prompt_improved) {
                    // Show prompt improvement modal
                    pendingMessage = data;
                    showPromptModal(message, data.improved_prompt);
                } else {
                    // Display assistant response
                    displayMessage('assistant', data.response);
                    
                    // Update token usage
                    updateTokenBar(data.total_tokens);
                    
                    // Check if approaching token limit
                    if (data.total_tokens > 100000) {
                        showToast('warning', 'Approaching token limit. Consider starting a new chat or compacting the conversation.');
                    }
                }
                
            } catch (error) {
                console.error('Error:', error);
                showToast('error', 'Failed to send message. Please try again.');
                
                // Remove the user message if sending failed
                const messages = document.querySelectorAll('.message');
                if (messages.length > 0) {
                    messages[messages.length - 1].remove();
                }
            } finally {
                // Hide typing indicator
                document.getElementById('typingIndicator').classList.remove('active');
                
                // Re-enable send button
                document.getElementById('sendButton').disabled = false;
            }
        }
        
        // Display a message in the chat
        function displayMessage(role, content, files = []) {
            const container = document.getElementById('messagesContainer');
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = role === 'user' ? 'U' : 'AI';
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            
            // Process content for formatting
            const formattedContent = formatMessageContent(content);
            contentDiv.innerHTML = formattedContent;
            
            // Add timestamp
            const timestamp = document.createElement('div');
            timestamp.className = 'message-timestamp';
            timestamp.textContent = new Date().toLocaleTimeString();
            contentDiv.appendChild(timestamp);
            
            // Add files if present
            if (files && files.length > 0) {
                const filesDiv = document.createElement('div');
                filesDiv.className = 'message-files';
                files.forEach(file => {
                    const fileTag = document.createElement('span');
                    fileTag.className = 'file-tag';
                    fileTag.innerHTML = `📎 ${file.name || file}`;
                    filesDiv.appendChild(fileTag);
                });
                contentDiv.appendChild(filesDiv);
            }
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(contentDiv);
            container.appendChild(messageDiv);
            
            // Scroll to bottom
            container.scrollTop = container.scrollHeight;
        }
        
        // Format message content (handle code blocks, etc.)
        function formatMessageContent(content) {
            // Escape HTML
            content = content.replace(/&/g, '&amp;')
                           .replace(/</g, '&lt;')
                           .replace(/>/g, '&gt;');
            
            // Format code blocks
            content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, function(match, lang, code) {
                return `<pre><code class="language-${lang || 'plaintext'}">${code}</code></pre>`;
            });
            
            // Format inline code
            content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
            
            // Format bold text
            content = content.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
            
            // Format italic text
            content = content.replace(/\*([^*]+)\*/g, '<em>$1</em>');
            
            // Format line breaks
            content = content.replace(/\n/g, '<br>');
            
            return content;
        }
        
        // Show prompt improvement modal
        function showPromptModal(original, improved) {
            document.getElementById('originalPromptText').textContent = original;
            document.getElementById('improvedPromptText').textContent = improved;
            document.getElementById('promptModal').classList.add('active');
            selectedPromptType = 'improved';
            selectPromptOption('improved');
        }
        
        // Close prompt improvement modal
        function closePromptModal() {
            document.getElementById('promptModal').classList.remove('active');
            pendingMessage = null;
        }
        
        // Select prompt option in modal
        function selectPromptOption(type) {
            selectedPromptType = type;
            
            // Update UI
            document.getElementById('originalPromptOption').classList.toggle('selected', type === 'original');
            document.getElementById('improvedPromptOption').classList.toggle('selected', type === 'improved');
        }
        
        // Confirm prompt selection and continue
        async function confirmPromptSelection() {
            if (!pendingMessage) return;
            
            closePromptModal();
            
            // Show typing indicator
            document.getElementById('typingIndicator').classList.add('active');
            
            try {
                // Send confirmation to server
                const response = await fetch('/confirm-prompt', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        conversation_id: conversationId,
                        use_improved: selectedPromptType === 'improved'
                    })
                });
                
                if (!response.ok) {
                    throw new Error('Failed to confirm prompt');
                }
                
                const data = await response.json();
                
                // Display assistant response
                displayMessage('assistant', data.response);
                
                // Update token usage
                updateTokenBar(data.total_tokens);
                
            } catch (error) {
                console.error('Error:', error);
                showToast('error', 'Failed to process response. Please try again.');
            } finally {
                // Hide typing indicator
                document.getElementById('typingIndicator').classList.remove('active');
                pendingMessage = null;
            }
        }
        
        // Update token usage bar
        function updateTokenBar(tokens) {
            currentTokens = tokens;
            const percentage = (tokens / 125000) * 100;
            
            document.getElementById('currentTokens').textContent = tokens.toLocaleString();
            document.getElementById('tokenProgress').style.width = percentage + '%';
            
            // Change color based on usage
            const progressBar = document.getElementById('tokenProgress');
            if (percentage > 80) {
                progressBar.style.background = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)';
            } else if (percentage > 60) {
                progressBar.style.background = 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)';
            } else {
                progressBar.style.background = 'var(--gradient-primary)';
            }
        }
        
        // Start a new chat
        function startNewChat() {
            if (confirm('Start a new conversation? Current conversation will be saved.')) {
                // Clear messages
                const container = document.getElementById('messagesContainer');
                container.innerHTML = '';
                
                // Show welcome screen
                const welcomeScreen = document.createElement('div');
                welcomeScreen.className = 'welcome-screen';
                welcomeScreen.id = 'welcomeScreen';
                welcomeScreen.innerHTML = `
                    <h1 class="welcome-title">Welcome to Jack's AI</h1>
                    <p class="welcome-subtitle">Your Advanced AI Assistant with 125K Context Window</p>
                    
                    <div class="features-grid">
                        <div class="feature-card">
                            <div class="feature-icon">🧠</div>
                            <div class="feature-title">Smart Prompts</div>
                            <div class="feature-description">AI-powered prompt optimization for better results</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-icon">📎</div>
                            <div class="feature-title">File Support</div>
                            <div class="feature-description">Upload images, documents, and more for analysis</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-icon">💬</div>
                            <div class="feature-title">Long Context</div>
                            <div class="feature-description">125,000 token context window for extended conversations</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-icon">⚡</div>
                            <div class="feature-title">Fast & Reliable</div>
                            <div class="feature-description">Multiple API keys with automatic failover</div>
                        </div>
                    </div>
                    
                    <p style="margin-top: 2rem; color: var(--text-muted); font-size: 0.875rem;">
                        Start by typing a message or uploading a file below!
                    </p>
                `;
                container.appendChild(welcomeScreen);
                
                // Initialize new conversation
                initializeConversation();
                
                showToast('success', 'New conversation started');
            }
        }
        
        // Compact the current conversation
        async function compactConversation() {
            if (!confirm('Compact this conversation? This will summarize the chat history to save tokens.')) {
                return;
            }
            
            // Show loading
            showToast('info', 'Compacting conversation...');
            
            try {
                const response = await fetch('/compact', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        conversation_id: conversationId
                    })
                });
                
                if (!response.ok) {
                    throw new Error('Failed to compact conversation');
                }
                
                const data = await response.json();
                
                // Clear current messages
                const container = document.getElementById('messagesContainer');
                container.innerHTML = '';
                
                // Display compacted summary
                displayMessage('assistant', `📦 Conversation Compacted\n\n${data.summary}`);
                
                // Update token usage
                updateTokenBar(data.total_tokens);
                
                showToast('success', 'Conversation compacted successfully');
                
            } catch (error) {
                console.error('Error:', error);
                showToast('error', 'Failed to compact conversation');
            }
        }
        
        // Export chat history
        function exportChat() {
            const messages = document.querySelectorAll('.message');
            let chatText = 'Jack\'s AI - Chat Export\n';
            chatText += '========================\n\n';
            
            messages.forEach(msg => {
                const role = msg.classList.contains('user') ? 'User' : 'Assistant';
                const content = msg.querySelector('.message-content').textContent;
                const timestamp = msg.querySelector('.message-timestamp').textContent;
                
                chatText += `[${timestamp}] ${role}:\n${content}\n\n`;
            });
            
            // Create download link
            const blob = new Blob([chatText], { type: 'text/plain' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `jacks-ai-chat-${new Date().toISOString().split('T')[0]}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            showToast('success', 'Chat exported successfully');
        }
        
        // Clear chat history
        function clearChat() {
            if (confirm('Clear all messages? This cannot be undone.')) {
                const container = document.getElementById('messagesContainer');
                container.innerHTML = '';
                
                // Show welcome screen
                const welcomeScreen = document.createElement('div');
                welcomeScreen.className = 'welcome-screen';
                welcomeScreen.id = 'welcomeScreen';
                welcomeScreen.innerHTML = `
                    <h1 class="welcome-title">Welcome to Jack's AI</h1>
                    <p class="welcome-subtitle">Your Advanced AI Assistant with 125K Context Window</p>
                    
                    <div class="features-grid">
                        <div class="feature-card">
                            <div class="feature-icon">🧠</div>
                            <div class="feature-title">Smart Prompts</div>
                            <div class="feature-description">AI-powered prompt optimization for better results</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-icon">📎</div>
                            <div class="feature-title">File Support</div>
                            <div class="feature-description">Upload images, documents, and more for analysis</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-icon">💬</div>
                            <div class="feature-title">Long Context</div>
                            <div class="feature-description">125,000 token context window for extended conversations</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-icon">⚡</div>
                            <div class="feature-title">Fast & Reliable</div>
                            <div class="feature-description">Multiple API keys with automatic failover</div>
                        </div>
                    </div>
                    
                    <p style="margin-top: 2rem; color: var(--text-muted); font-size: 0.875rem;">
                        Start by typing a message or uploading a file below!
                    </p>
                `;
                container.appendChild(welcomeScreen);
                
                updateTokenBar(0);
                showToast('success', 'Chat cleared');
            }
        }
        
        // Show toast notification
        function showToast(type, message) {
            const container = document.getElementById('toastContainer');
            
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            
            const icon = type === 'success' ? '✓' : type === 'error' ? '✗' : type === 'warning' ? '⚠' : 'ℹ';
            
            toast.innerHTML = `
                <span class="toast-icon">${icon}</span>
                <span class="toast-message">${message}</span>
                <span class="toast-close" onclick="this.parentElement.remove()">×</span>
            `;
            
            container.appendChild(toast);
            
            // Auto remove after 5 seconds
            setTimeout(() => {
                toast.remove();
            }, 5000);
        }
        
        // Load existing conversation (if any)
        async function loadConversation() {
            try {
                const response = await fetch('/load-conversation', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        conversation_id: conversationId
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    
                    if (data.messages && data.messages.length > 0) {
                        // Hide welcome screen
                        const welcomeScreen = document.getElementById('welcomeScreen');
                        if (welcomeScreen) {
                            welcomeScreen.style.display = 'none';
                        }
                        
                        // Display messages
                        data.messages.forEach(msg => {
                            displayMessage(msg.role, msg.content, msg.files);
                        });
                        
                        // Update token usage
                        updateTokenBar(data.total_tokens);
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

@app.route('/')
def index():
    """Serve the main application page."""
    session_id = get_or_create_session()
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages and file uploads."""
    try:
        session_id = get_or_create_session()
        
        # Get message and conversation ID
        message = request.form.get('message', '')
        conversation_id = request.form.get('conversation_id')
        
        # Process uploaded files
        files_data = []
        if 'files' in request.files:
            files = request.files.getlist('files')
            for file in files:
                file_data = process_file_upload(file)
                if 'error' not in file_data:
                    files_data.append(file_data)
        
        # Get or create conversation
        if conversation_id not in MEMORY_STORAGE['conversations']:
            MEMORY_STORAGE['conversations'][conversation_id] = Conversation(id=conversation_id)
        
        conversation = MEMORY_STORAGE['conversations'][conversation_id]
        
        # Create user message
        user_message = Message(
            role='user',
            content=message,
            files=files_data
        )
        conversation.messages.append(user_message)
        
        # Improve prompt
        improved_prompt, prompt_improved = gemini_manager.improve_prompt(message)
        
        if prompt_improved and improved_prompt != message:
            # Store improved prompt for later use
            user_message.prompt_improved = True
            user_message.original_prompt = message
            
            return jsonify({
                'prompt_improved': True,
                'original_prompt': message,
                'improved_prompt': improved_prompt,
                'conversation_id': conversation_id
            })
        
        # Generate response
        response_text, tokens_used = gemini_manager.generate_response(message, files_data)
        
        # Create assistant message
        assistant_message = Message(
            role='assistant',
            content=response_text,
            tokens_used=tokens_used
        )
        conversation.messages.append(assistant_message)
        
        # Update token count
        conversation.total_tokens += tokens_used
        
        return jsonify({
            'response': response_text,
            'total_tokens': conversation.total_tokens,
            'prompt_improved': False
        })
        
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/confirm-prompt', methods=['POST'])
def confirm_prompt():
    """Handle prompt selection confirmation."""
    try:
        data = request.json
        conversation_id = data.get('conversation_id')
        use_improved = data.get('use_improved', True)
        
        # Get conversation
        conversation = MEMORY_STORAGE['conversations'].get(conversation_id)
        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404
        
        # Get last user message
        last_message = None
        for msg in reversed(conversation.messages):
            if msg.role == 'user':
                last_message = msg
                break
        
        if not last_message:
            return jsonify({'error': 'No user message found'}), 404
        
        # Use selected prompt
        prompt = last_message.content if not use_improved else last_message.original_prompt
        if last_message.prompt_improved and use_improved:
            # The content is already the improved version
            prompt = last_message.content
        
        # Generate response
        response_text, tokens_used = gemini_manager.generate_response(prompt, last_message.files)
        
        # Create assistant message
        assistant_message = Message(
            role='assistant',
            content=response_text,
            tokens_used=tokens_used
        )
        conversation.messages.append(assistant_message)
        
        # Update token count
        conversation.total_tokens += tokens_used
        
        return jsonify({
            'response': response_text,
            'total_tokens': conversation.total_tokens
        })
        
    except Exception as e:
        print(f"Error confirming prompt: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/compact', methods=['POST'])
def compact():
    """Compact a conversation to reduce token usage."""
    try:
        data = request.json
        conversation_id = data.get('conversation_id')
        
        # Get conversation
        conversation = MEMORY_STORAGE['conversations'].get(conversation_id)
        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404
        
        # Compact the conversation
        summary = gemini_manager.compact_conversation(conversation.messages)
        
        # Create new compacted conversation
        new_conversation = Conversation(id=conversation_id + '_compacted')
        compacted_message = Message(
            role='system',
            content=summary,
            tokens_used=len(summary.split()) * 1.3  # Rough estimate
        )
        new_conversation.messages = [compacted_message]
        new_conversation.total_tokens = compacted_message.tokens_used
        new_conversation.compacted = True
        
        # Replace old conversation
        MEMORY_STORAGE['conversations'][conversation_id] = new_conversation
        
        return jsonify({
            'summary': summary,
            'total_tokens': new_conversation.total_tokens
        })
        
    except Exception as e:
        print(f"Error compacting conversation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/load-conversation', methods=['POST'])
def load_conversation():
    """Load an existing conversation."""
    try:
        data = request.json
        conversation_id = data.get('conversation_id')
        
        # Get conversation
        conversation = MEMORY_STORAGE['conversations'].get(conversation_id)
        if not conversation:
            return jsonify({'messages': [], 'total_tokens': 0})
        
        # Convert messages to dict format
        messages = []
        for msg in conversation.messages:
            messages.append({
                'role': msg.role,
                'content': msg.content,
                'files': [f['name'] for f in msg.files] if msg.files else [],
                'timestamp': msg.timestamp.isoformat()
            })
        
        return jsonify({
            'messages': messages,
            'total_tokens': conversation.total_tokens
        })
        
    except Exception as e:
        print(f"Error loading conversation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(MEMORY_STORAGE['active_sessions']),
        'total_conversations': len(MEMORY_STORAGE['conversations']),
        'api_keys_configured': len(API_KEYS)
    })

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Main entry point
if __name__ == '__main__':
    # Clean up old sessions periodically (you might want to run this in a background thread)
    cleanup_old_sessions()
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the application
    app.run(host='0.0.0.0', port=port, debug=False)