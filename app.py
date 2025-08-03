"""
Jack's AI - Ultra Modern Web Application with Stunning UI
This is a full-stack application with advanced UI/UX design
Author: Jack's AI System
Version: 2.0.0
"""

import os
import json
import base64
import hashlib
import secrets
import traceback
from datetime import datetime, timedelta
from functools import wraps
from io import BytesIO
import mimetypes
import random
import string

# Import Flask and related libraries for web framework
from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for, make_response
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# Import OpenAI library to interact with Gemini API
from openai import OpenAI
import tiktoken  # For token counting
from PIL import Image  # For image processing
import PyPDF2  # For PDF processing
import docx  # For Word document processing
import openpyxl  # For Excel processing

# Initialize Flask application
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Max file size: 100MB

# In-memory storage
USERS_DB = {}
CHAT_SESSIONS = {}
USER_TOKENS = {}
API_KEYS_STATUS = {}
USER_PREFERENCES = {}  # Store user preferences like theme

# HTML Template - Ultra Modern UI with stunning design
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jack's AI - Next Generation Assistant</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        /* CSS Variables for theming */
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --success-gradient: linear-gradient(135deg, #13B497 0%, #59D4A7 100%);
            --warning-gradient: linear-gradient(135deg, #FA8231 0%, #FFD14C 100%);
            --dark-gradient: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            --glass-bg: rgba(255, 255, 255, 0.95);
            --glass-border: rgba(255, 255, 255, 0.18);
            --shadow-color: rgba(0, 0, 0, 0.1);
            --text-primary: #1a202c;
            --text-secondary: #4a5568;
            --border-radius: 20px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        /* Dark theme variables */
        [data-theme="dark"] {
            --glass-bg: rgba(30, 30, 30, 0.95);
            --glass-border: rgba(255, 255, 255, 0.1);
            --shadow-color: rgba(0, 0, 0, 0.5);
            --text-primary: #f7fafc;
            --text-secondary: #a0aec0;
            --bg-primary: #0f0f0f;
            --bg-secondary: #1a1a1a;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #0f0f0f;
            min-height: 100vh;
            overflow-x: hidden;
            position: relative;
        }
        
        /* Animated gradient background */
        .animated-bg {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(270deg, #667eea, #764ba2, #f093fb, #f5576c);
            background-size: 800% 800%;
            animation: gradientShift 20s ease infinite;
            z-index: -2;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Floating particles effect */
        .particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
        }
        
        .particle {
            position: absolute;
            width: 4px;
            height: 4px;
            background: rgba(255, 255, 255, 0.5);
            border-radius: 50%;
            animation: float 15s infinite;
        }
        
        @keyframes float {
            0%, 100% {
                transform: translateY(0) translateX(0);
                opacity: 0;
            }
            10% {
                opacity: 1;
            }
            90% {
                opacity: 1;
            }
            100% {
                transform: translateY(-100vh) translateX(100px);
                opacity: 0;
            }
        }
        
        /* Main container with glass morphism */
        .main-container {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        
        /* Chat container */
        .chat-container {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 30px;
            box-shadow: 
                0 20px 40px var(--shadow-color),
                0 0 80px rgba(102, 126, 234, 0.1),
                inset 0 0 20px rgba(255, 255, 255, 0.05);
            width: 100%;
            max-width: 1400px;
            height: 90vh;
            display: flex;
            overflow: hidden;
            animation: slideUp 0.5s ease-out;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Sidebar */
        .sidebar {
            width: 280px;
            background: rgba(255, 255, 255, 0.05);
            border-right: 1px solid var(--glass-border);
            display: flex;
            flex-direction: column;
            transition: var(--transition);
        }
        
        .sidebar-header {
            padding: 30px 20px;
            background: var(--primary-gradient);
            position: relative;
            overflow: hidden;
        }
        
        .sidebar-header::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: pulse 3s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.5; }
            50% { transform: scale(1.1); opacity: 0.8; }
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
            color: white;
            position: relative;
            z-index: 1;
        }
        
        .logo-icon {
            width: 50px;
            height: 50px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .logo-text {
            flex: 1;
        }
        
        .logo-text h1 {
            font-size: 20px;
            font-weight: 800;
            margin-bottom: 4px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .logo-text p {
            font-size: 12px;
            opacity: 0.9;
            font-weight: 500;
        }
        
        /* Chat history */
        .chat-history {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        .chat-history::-webkit-scrollbar {
            width: 8px;
        }
        
        .chat-history::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
        }
        
        .chat-history::-webkit-scrollbar-thumb {
            background: var(--primary-gradient);
            border-radius: 10px;
        }
        
        .chat-item {
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 15px;
            cursor: pointer;
            transition: var(--transition);
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
            background: var(--primary-gradient);
            opacity: 0.1;
            transition: left 0.3s ease;
        }
        
        .chat-item:hover::before {
            left: 0;
        }
        
        .chat-item:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateX(5px);
        }
        
        .chat-item-title {
            color: var(--text-primary);
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .chat-item-preview {
            color: var(--text-secondary);
            font-size: 13px;
            opacity: 0.8;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .chat-item-time {
            color: var(--text-secondary);
            font-size: 11px;
            margin-top: 5px;
            opacity: 0.6;
        }
        
        /* Sidebar actions */
        .sidebar-actions {
            padding: 20px;
            border-top: 1px solid var(--glass-border);
        }
        
        .new-chat-btn {
            width: 100%;
            padding: 15px;
            background: var(--primary-gradient);
            color: white;
            border: none;
            border-radius: 15px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .new-chat-btn::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: translate(-50%, -50%);
            transition: width 0.4s, height 0.4s;
        }
        
        .new-chat-btn:hover::before {
            width: 300px;
            height: 300px;
        }
        
        .new-chat-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
        }
        
        /* Main chat area */
        .chat-main {
            flex: 1;
            display: flex;
            flex-direction: column;
            position: relative;
        }
        
        /* Chat header */
        .chat-header {
            padding: 25px 30px;
            background: rgba(255, 255, 255, 0.05);
            border-bottom: 1px solid var(--glass-border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            backdrop-filter: blur(10px);
        }
        
        .chat-header-info {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .model-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: var(--primary-gradient);
            color: white;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .model-badge i {
            font-size: 14px;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: var(--success-gradient);
            color: white;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            animation: pulse 2s infinite;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            background: white;
            border-radius: 50%;
            animation: blink 1s infinite;
        }
        
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        
        /* Chat header actions */
        .chat-header-actions {
            display: flex;
            gap: 10px;
        }
        
        .header-btn {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            border: 1px solid var(--glass-border);
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: var(--transition);
            position: relative;
        }
        
        .header-btn:hover {
            background: var(--primary-gradient);
            color: white;
            transform: scale(1.1);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        
        .tooltip {
            position: absolute;
            bottom: -35px;
            left: 50%;
            transform: translateX(-50%);
            background: #333;
            color: white;
            padding: 5px 10px;
            border-radius: 8px;
            font-size: 12px;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s;
        }
        
        .header-btn:hover .tooltip {
            opacity: 1;
        }
        
        /* Messages area */
        .messages-container {
            flex: 1;
            overflow-y: auto;
            padding: 30px;
            scroll-behavior: smooth;
            position: relative;
        }
        
        .messages-container::-webkit-scrollbar {
            width: 10px;
        }
        
        .messages-container::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
        }
        
        .messages-container::-webkit-scrollbar-thumb {
            background: var(--primary-gradient);
            border-radius: 10px;
        }
        
        /* Date divider */
        .date-divider {
            text-align: center;
            margin: 30px 0;
            position: relative;
        }
        
        .date-divider::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            height: 1px;
            background: var(--glass-border);
        }
        
        .date-divider span {
            background: var(--glass-bg);
            padding: 0 20px;
            color: var(--text-secondary);
            font-size: 12px;
            font-weight: 600;
            position: relative;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Message bubbles */
        .message {
            margin-bottom: 20px;
            animation: messageSlide 0.3s ease-out;
            display: flex;
            align-items: flex-start;
            gap: 15px;
        }
        
        @keyframes messageSlide {
            from {
                opacity: 0;
                transform: translateY(20px);
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
            width: 40px;
            height: 40px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .message.user .message-avatar {
            background: var(--primary-gradient);
            color: white;
        }
        
        .message.assistant .message-avatar {
            background: var(--secondary-gradient);
            color: white;
        }
        
        .message-content-wrapper {
            max-width: 70%;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .message-header {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0 10px;
        }
        
        .message-author {
            font-weight: 600;
            font-size: 13px;
            color: var(--text-primary);
        }
        
        .message-time {
            font-size: 11px;
            color: var(--text-secondary);
            opacity: 0.6;
        }
        
        .message-bubble {
            padding: 16px 20px;
            border-radius: 20px;
            position: relative;
            word-wrap: break-word;
            line-height: 1.6;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        }
        
        .message.user .message-bubble {
            background: var(--primary-gradient);
            color: white;
            border-bottom-right-radius: 5px;
        }
        
        .message.assistant .message-bubble {
            background: rgba(255, 255, 255, 0.9);
            color: var(--text-primary);
            border: 1px solid var(--glass-border);
            border-bottom-left-radius: 5px;
        }
        
        [data-theme="dark"] .message.assistant .message-bubble {
            background: rgba(40, 40, 40, 0.9);
        }
        
        /* Message actions */
        .message-actions {
            display: flex;
            gap: 8px;
            padding: 0 10px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .message:hover .message-actions {
            opacity: 1;
        }
        
        .message-action {
            padding: 6px 10px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid var(--glass-border);
            border-radius: 8px;
            font-size: 12px;
            cursor: pointer;
            transition: var(--transition);
            color: var(--text-secondary);
        }
        
        .message-action:hover {
            background: var(--primary-gradient);
            color: white;
            transform: scale(1.05);
        }
        
        /* Typing indicator */
        .typing-indicator {
            display: none;
            align-items: center;
            gap: 15px;
            padding: 20px 30px;
        }
        
        .typing-indicator.active {
            display: flex;
            animation: fadeIn 0.3s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .typing-dots {
            display: flex;
            gap: 4px;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            background: var(--primary-gradient);
            border-radius: 50%;
            animation: typingAnimation 1.4s infinite;
        }
        
        .typing-dot:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-dot:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typingAnimation {
            0%, 60%, 100% {
                transform: translateY(0);
                opacity: 0.7;
            }
            30% {
                transform: translateY(-10px);
                opacity: 1;
            }
        }
        
        /* Input area */
        .input-section {
            padding: 20px 30px 30px;
            background: rgba(255, 255, 255, 0.05);
            border-top: 1px solid var(--glass-border);
            backdrop-filter: blur(10px);
        }
        
        /* Token usage bar */
        .token-usage-bar {
            margin-bottom: 20px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            border: 1px solid var(--glass-border);
        }
        
        .token-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .token-label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: var(--text-primary);
            font-weight: 600;
        }
        
        .token-count {
            font-size: 13px;
            color: var(--text-secondary);
            font-weight: 500;
        }
        
        .token-progress {
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }
        
        .token-fill {
            height: 100%;
            background: var(--primary-gradient);
            border-radius: 10px;
            transition: width 0.5s ease;
            position: relative;
            overflow: hidden;
        }
        
        .token-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            bottom: 0;
            right: 0;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(255, 255, 255, 0.3),
                transparent
            );
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        /* Input container */
        .input-container {
            display: flex;
            gap: 15px;
            align-items: flex-end;
        }
        
        .input-wrapper {
            flex: 1;
            position: relative;
        }
        
        .input-box {
            width: 100%;
            min-height: 50px;
            max-height: 150px;
            padding: 15px 50px 15px 20px;
            background: rgba(255, 255, 255, 0.1);
            border: 2px solid var(--glass-border);
            border-radius: 20px;
            color: var(--text-primary);
            font-size: 15px;
            resize: none;
            transition: var(--transition);
            font-family: 'Inter', sans-serif;
            line-height: 1.5;
        }
        
        .input-box:focus {
            outline: none;
            border-color: #667eea;
            background: rgba(255, 255, 255, 0.15);
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
        }
        
        .input-box::placeholder {
            color: var(--text-secondary);
            opacity: 0.6;
        }
        
        /* Input actions */
        .input-actions {
            position: absolute;
            right: 10px;
            bottom: 10px;
            display: flex;
            gap: 5px;
        }
        
        .input-action-btn {
            width: 32px;
            height: 32px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid var(--glass-border);
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: var(--transition);
        }
        
        .input-action-btn:hover {
            background: var(--primary-gradient);
            color: white;
            transform: scale(1.1);
        }
        
        /* File upload area */
        .file-upload-section {
            margin-bottom: 15px;
            display: none;
        }
        
        .file-upload-section.active {
            display: block;
            animation: slideDown 0.3s ease;
        }
        
        @keyframes slideDown {
            from {
                opacity: 0;
                max-height: 0;
            }
            to {
                opacity: 1;
                max-height: 200px;
            }
        }
        
        .files-preview {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            border: 2px dashed var(--glass-border);
            min-height: 80px;
            position: relative;
        }
        
        .file-preview-item {
            padding: 10px 15px;
            background: var(--primary-gradient);
            color: white;
            border-radius: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            animation: fadeIn 0.3s;
            position: relative;
            overflow: hidden;
        }
        
        .file-preview-item::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.1);
            transform: translateX(-100%);
            transition: transform 0.3s;
        }
        
        .file-preview-item:hover::before {
            transform: translateX(0);
        }
        
        .file-remove {
            cursor: pointer;
            opacity: 0.8;
            transition: opacity 0.3s;
        }
        
        .file-remove:hover {
            opacity: 1;
            transform: scale(1.2);
        }
        
        .drop-zone-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: var(--text-secondary);
            font-size: 14px;
            pointer-events: none;
            opacity: 0.6;
        }
        
        /* Send button */
        .send-button {
            padding: 15px 30px;
            background: var(--primary-gradient);
            color: white;
            border: none;
            border-radius: 20px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            display: flex;
            align-items: center;
            gap: 10px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        }
        
        .send-button::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: translate(-50%, -50%);
            transition: width 0.4s, height 0.4s;
        }
        
        .send-button:hover::before {
            width: 300px;
            height: 300px;
        }
        
        .send-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        }
        
        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        /* Quick actions bar */
        .quick-actions {
            display: flex;
            gap: 10px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        
        .quick-action {
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            color: var(--text-secondary);
            font-size: 13px;
            cursor: pointer;
            transition: var(--transition);
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .quick-action:hover {
            background: var(--primary-gradient);
            color: white;
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        
        /* Auth container */
        .auth-container {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 30px;
            padding: 50px;
            width: 100%;
            max-width: 480px;
            box-shadow: 
                0 20px 40px var(--shadow-color),
                0 0 80px rgba(102, 126, 234, 0.1),
                inset 0 0 20px rgba(255, 255, 255, 0.05);
            animation: slideUp 0.5s ease-out;
            position: relative;
            overflow: hidden;
        }
        
        .auth-container::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
            animation: rotate 20s linear infinite;
        }
        
        @keyframes rotate {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .auth-header {
            text-align: center;
            margin-bottom: 40px;
            position: relative;
            z-index: 1;
        }
        
        .auth-logo {
            width: 80px;
            height: 80px;
            margin: 0 auto 20px;
            background: var(--primary-gradient);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 36px;
            color: white;
            box-shadow: 0 15px 30px rgba(102, 126, 234, 0.3);
            animation: bounce 2s infinite;
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        .auth-title {
            font-size: 28px;
            font-weight: 800;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .auth-subtitle {
            color: var(--text-secondary);
            font-size: 14px;
            font-weight: 500;
        }
        
        /* Form styling */
        .form-group {
            margin-bottom: 25px;
            position: relative;
            z-index: 1;
        }
        
        .form-label {
            display: block;
            margin-bottom: 8px;
            color: var(--text-primary);
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .form-input {
            width: 100%;
            padding: 15px 20px;
            background: rgba(255, 255, 255, 0.05);
            border: 2px solid var(--glass-border);
            border-radius: 15px;
            color: var(--text-primary);
            font-size: 15px;
            transition: var(--transition);
        }
        
        .form-input:focus {
            outline: none;
            border-color: #667eea;
            background: rgba(255, 255, 255, 0.1);
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
        }
        
        .form-input::placeholder {
            color: var(--text-secondary);
            opacity: 0.5;
        }
        
        /* Password strength indicator */
        .password-strength {
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            margin-top: 8px;
            overflow: hidden;
        }
        
        .password-strength-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.3s, background 0.3s;
        }
        
        .password-strength-fill.weak {
            width: 33%;
            background: #f5576c;
        }
        
        .password-strength-fill.medium {
            width: 66%;
            background: #ffa726;
        }
        
        .password-strength-fill.strong {
            width: 100%;
            background: #66bb6a;
        }
        
        /* Submit button */
        .submit-btn {
            width: 100%;
            padding: 18px;
            background: var(--primary-gradient);
            color: white;
            border: none;
            border-radius: 15px;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            transition: var(--transition);
            position: relative;
            overflow: hidden;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 30px;
            box-shadow: 0 15px 30px rgba(102, 126, 234, 0.3);
        }
        
        .submit-btn::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: translate(-50%, -50%);
            transition: width 0.5s, height 0.5s;
        }
        
        .submit-btn:hover::before {
            width: 400px;
            height: 400px;
        }
        
        .submit-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 20px 40px rgba(102, 126, 234, 0.4);
        }
        
        /* Toggle form link */
        .form-toggle {
            text-align: center;
            margin-top: 30px;
            color: var(--text-secondary);
            font-size: 14px;
            position: relative;
            z-index: 1;
        }
        
        .form-toggle a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
            transition: var(--transition);
        }
        
        .form-toggle a:hover {
            color: #764ba2;
            text-decoration: underline;
        }
        
        /* Notifications */
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 20px 25px;
            border-radius: 15px;
            color: white;
            font-size: 14px;
            font-weight: 500;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            z-index: 10000;
            display: flex;
            align-items: center;
            gap: 15px;
            animation: slideInRight 0.3s ease-out;
            backdrop-filter: blur(10px);
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
        
        .notification.success {
            background: var(--success-gradient);
        }
        
        .notification.error {
            background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
        }
        
        .notification.warning {
            background: var(--warning-gradient);
        }
        
        .notification-icon {
            font-size: 20px;
        }
        
        .notification-close {
            margin-left: auto;
            cursor: pointer;
            opacity: 0.8;
            transition: opacity 0.3s;
        }
        
        .notification-close:hover {
            opacity: 1;
        }
        
        /* Loading overlay */
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(5px);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        }
        
        .loading-overlay.active {
            display: flex;
        }
        
        .loading-spinner {
            width: 60px;
            height: 60px;
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-top: 4px solid white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .sidebar {
                display: none;
            }
            
            .chat-container {
                height: 100vh;
                border-radius: 0;
            }
            
            .message-content-wrapper {
                max-width: 85%;
            }
            
            .auth-container {
                padding: 30px;
            }
        }
        
        /* Prompt enhancement modal */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(10px);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 9998;
            padding: 20px;
        }
        
        .modal-overlay.active {
            display: flex;
            animation: fadeIn 0.3s;
        }
        
        .modal-content {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 25px;
            padding: 40px;
            max-width: 600px;
            width: 100%;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            animation: slideUp 0.3s ease-out;
        }
        
        .modal-header {
            margin-bottom: 30px;
        }
        
        .modal-title {
            font-size: 24px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 10px;
        }
        
        .modal-subtitle {
            color: var(--text-secondary);
            font-size: 14px;
        }
        
        .prompt-option {
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            border: 2px solid var(--glass-border);
            border-radius: 15px;
            margin-bottom: 20px;
            cursor: pointer;
            transition: var(--transition);
        }
        
        .prompt-option:hover {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.1);
        }
        
        .prompt-option.selected {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.2);
        }
        
        .prompt-option-title {
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 10px;
        }
        
        .prompt-option-text {
            color: var(--text-secondary);
            font-size: 14px;
            line-height: 1.6;
        }
        
        .modal-actions {
            display: flex;
            gap: 15px;
            margin-top: 30px;
        }
        
        .modal-btn {
            flex: 1;
            padding: 15px;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            text-align: center;
        }
        
        .modal-btn-primary {
            background: var(--primary-gradient);
            color: white;
            border: none;
        }
        
        .modal-btn-secondary {
            background: transparent;
            color: var(--text-primary);
            border: 2px solid var(--glass-border);
        }
        
        .modal-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.2);
        }
    </style>
</head>
<body>
    <!-- Animated background -->
    <div class="animated-bg"></div>
    
    <!-- Floating particles -->
    <div class="particles" id="particles"></div>
    
    <!-- Loading overlay -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-spinner"></div>
    </div>
    
    <!-- Main container -->
    <div class="main-container">
        <!-- Chat Interface -->
        <div class="chat-container" id="chatContainer" style="display: none;">
            <!-- Sidebar -->
            <aside class="sidebar">
                <div class="sidebar-header">
                    <div class="logo">
                        <div class="logo-icon">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div class="logo-text">
                            <h1>Jack's AI</h1>
                            <p>Next Generation Assistant</p>
                        </div>
                    </div>
                </div>
                
                <div class="chat-history" id="chatHistory">
                    <!-- Chat history items will be added here -->
                </div>
                
                <div class="sidebar-actions">
                    <button class="new-chat-btn" onclick="newChat()">
                        <i class="fas fa-plus"></i> New Chat
                    </button>
                </div>
            </aside>
            
            <!-- Main chat area -->
            <main class="chat-main">
                <!-- Chat header -->
                <header class="chat-header">
                    <div class="chat-header-info">
                        <div class="model-badge">
                            <i class="fas fa-microchip"></i>
                            <span>Gemini 2.5 Pro</span>
                        </div>
                        <div class="status-indicator">
                            <span class="status-dot"></span>
                            <span>Online</span>
                        </div>
                    </div>
                    
                    <div class="chat-header-actions">
                        <button class="header-btn" onclick="toggleTheme()">
                            <i class="fas fa-moon" id="themeIcon"></i>
                            <span class="tooltip">Toggle Theme</span>
                        </button>
                        <button class="header-btn" onclick="compactChat()">
                            <i class="fas fa-compress"></i>
                            <span class="tooltip">Compact Chat</span>
                        </button>
                        <button class="header-btn" onclick="exportChat()">
                            <i class="fas fa-download"></i>
                            <span class="tooltip">Export Chat</span>
                        </button>
                        <button class="header-btn" onclick="logout()">
                            <i class="fas fa-sign-out-alt"></i>
                            <span class="tooltip">Logout</span>
                        </button>
                    </div>
                </header>
                
                <!-- Messages container -->
                <div class="messages-container" id="messagesContainer">
                    <div class="date-divider">
                        <span>Today</span>
                    </div>
                    
                    <!-- Welcome message -->
                    <div class="message assistant">
                        <div class="message-avatar">
                            <i class="fas fa-robot"></i>
                        </div>
                        <div class="message-content-wrapper">
                            <div class="message-header">
                                <span class="message-author">Jack's AI</span>
                                <span class="message-time">Now</span>
                            </div>
                            <div class="message-bubble">
                                👋 Welcome! I'm your advanced AI assistant powered by Gemini 2.5 Pro. I can help you with complex tasks, analyze documents, generate code, and much more. How can I assist you today?
                            </div>
                            <div class="message-actions">
                                <button class="message-action" onclick="copyMessage(this)">
                                    <i class="fas fa-copy"></i> Copy
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Typing indicator -->
                <div class="typing-indicator" id="typingIndicator">
                    <div class="message-avatar">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="typing-dots">
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                    </div>
                </div>
                
                <!-- Input section -->
                <div class="input-section">
                    <!-- Token usage bar -->
                    <div class="token-usage-bar">
                        <div class="token-header">
                            <div class="token-label">
                                <i class="fas fa-chart-line"></i>
                                <span>Context Window</span>
                            </div>
                            <div class="token-count">
                                <span id="tokenCount">0</span> / 125,000 tokens
                            </div>
                        </div>
                        <div class="token-progress">
                            <div class="token-fill" id="tokenFill" style="width: 0%"></div>
                        </div>
                    </div>
                    
                    <!-- File upload section -->
                    <div class="file-upload-section" id="fileUploadSection">
                        <div class="files-preview" id="filesPreview">
                            <span class="drop-zone-text">Drop files here or click to browse</span>
                        </div>
                    </div>
                    
                    <!-- Input container -->
                    <div class="input-container">
                        <div class="input-wrapper">
                            <textarea 
                                class="input-box" 
                                id="messageInput" 
                                placeholder="Type your message here... (Shift+Enter for new line)"
                                rows="1"
                            ></textarea>
                            <div class="input-actions">
                                <button class="input-action-btn" onclick="toggleFileUpload()">
                                    <i class="fas fa-paperclip"></i>
                                </button>
                                <button class="input-action-btn" onclick="toggleVoiceInput()">
                                    <i class="fas fa-microphone"></i>
                                </button>
                            </div>
                        </div>
                        <button class="send-button" id="sendButton" onclick="sendMessage()">
                            <i class="fas fa-paper-plane"></i>
                            <span>Send</span>
                        </button>
                    </div>
                    
                    <!-- Quick actions -->
                    <div class="quick-actions">
                        <button class="quick-action" onclick="insertPrompt('Explain in detail')">
                            <i class="fas fa-info-circle"></i> Explain
                        </button>
                        <button class="quick-action" onclick="insertPrompt('Analyze and provide insights')">
                            <i class="fas fa-chart-bar"></i> Analyze
                        </button>
                        <button class="quick-action" onclick="insertPrompt('Generate code for')">
                            <i class="fas fa-code"></i> Code
                        </button>
                        <button class="quick-action" onclick="insertPrompt('Create a comprehensive')">
                            <i class="fas fa-file-alt"></i> Create
                        </button>
                        <button class="quick-action" onclick="insertPrompt('Solve step by step')">
                            <i class="fas fa-calculator"></i> Solve
                        </button>
                    </div>
                </div>
            </main>
        </div>
        
        <!-- Auth Container -->
        <div class="auth-container" id="authContainer">
            <div class="auth-header">
                <div class="auth-logo">
                    <i class="fas fa-robot"></i>
                </div>
                <h1 class="auth-title">Jack's AI</h1>
                <p class="auth-subtitle">Next Generation AI Assistant</p>
            </div>
            
            <!-- Login Form -->
            <div id="loginForm">
                <form onsubmit="return login(event)">
                    <div class="form-group">
                        <label class="form-label" for="loginUsername">Username</label>
                        <input 
                            type="text" 
                            id="loginUsername" 
                            class="form-input" 
                            placeholder="Enter your username"
                            required
                        >
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label" for="loginPassword">Password</label>
                        <input 
                            type="password" 
                            id="loginPassword" 
                            class="form-input" 
                            placeholder="Enter your password"
                            required
                        >
                    </div>
                    
                    <button type="submit" class="submit-btn">
                        Sign In
                    </button>
                </form>
                
                <div class="form-toggle">
                    Don't have an account? 
                    <a href="#" onclick="toggleAuthForm('register')">Create one</a>
                </div>
            </div>
            
            <!-- Register Form -->
            <div id="registerForm" style="display: none;">
                <form onsubmit="return register(event)">
                    <div class="form-group">
                        <label class="form-label" for="registerUsername">Username</label>
                        <input 
                            type="text" 
                            id="registerUsername" 
                            class="form-input" 
                            placeholder="Choose a username"
                            required
                            minlength="3"
                        >
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label" for="registerPassword">Password</label>
                        <input 
                            type="password" 
                            id="registerPassword" 
                            class="form-input" 
                            placeholder="Create a strong password"
                            required
                            minlength="6"
                            onkeyup="checkPasswordStrength(this.value)"
                        >
                        <div class="password-strength">
                            <div class="password-strength-fill" id="passwordStrength"></div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label" for="confirmPassword">Confirm Password</label>
                        <input 
                            type="password" 
                            id="confirmPassword" 
                            class="form-input" 
                            placeholder="Confirm your password"
                            required
                            minlength="6"
                        >
                    </div>
                    
                    <button type="submit" class="submit-btn">
                        Create Account
                    </button>
                </form>
                
                <div class="form-toggle">
                    Already have an account? 
                    <a href="#" onclick="toggleAuthForm('login')">Sign in</a>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Prompt Enhancement Modal -->
    <div class="modal-overlay" id="promptModal">
        <div class="modal-content">
            <div class="modal-header">
                <h2 class="modal-title">✨ Enhance Your Prompt</h2>
                <p class="modal-subtitle">Choose how you'd like to ask your question</p>
            </div>
            
            <div class="prompt-option" id="originalOption" onclick="selectPromptOption('original')">
                <div class="prompt-option-title">Original Prompt</div>
                <div class="prompt-option-text" id="originalPromptText"></div>
            </div>
            
            <div class="prompt-option selected" id="enhancedOption" onclick="selectPromptOption('enhanced')">
                <div class="prompt-option-title">Enhanced Prompt (Recommended)</div>
                <div class="prompt-option-text" id="enhancedPromptText"></div>
            </div>
            
            <div class="modal-actions">
                <button class="modal-btn modal-btn-secondary" onclick="closePromptModal()">
                    Cancel
                </button>
                <button class="modal-btn modal-btn-primary" onclick="confirmPromptSelection()">
                    Use Selected Prompt
                </button>
            </div>
        </div>
    </div>
    
    <!-- File input (hidden) -->
    <input type="file" id="fileInput" multiple style="display: none;">
    
    <script>
        // Application state
        let currentUser = null;
        let chatHistory = [];
        let tokenUsage = 0;
        let selectedPromptType = 'enhanced';
        let currentPrompt = '';
        let enhancedPrompt = '';
        let attachedFiles = [];
        let isDarkTheme = true;
        
        // Initialize particles
        function createParticles() {
            const particlesContainer = document.getElementById('particles');
            for (let i = 0; i < 50; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 15 + 's';
                particle.style.animationDuration = (15 + Math.random() * 10) + 's';
                particlesContainer.appendChild(particle);
            }
        }
        
        // Theme toggle
        function toggleTheme() {
            isDarkTheme = !isDarkTheme;
            document.documentElement.setAttribute('data-theme', isDarkTheme ? 'dark' : 'light');
            const icon = document.getElementById('themeIcon');
            icon.className = isDarkTheme ? 'fas fa-moon' : 'fas fa-sun';
            
            // Save preference
            localStorage.setItem('theme', isDarkTheme ? 'dark' : 'light');
        }
        
        // Load theme preference
        function loadThemePreference() {
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme) {
                isDarkTheme = savedTheme === 'dark';
                document.documentElement.setAttribute('data-theme', savedTheme);
                const icon = document.getElementById('themeIcon');
                if (icon) {
                    icon.className = isDarkTheme ? 'fas fa-moon' : 'fas fa-sun';
                }
            }
        }
        
        // Show notification
        function showNotification(message, type = 'success') {
            const notification = document.createElement('div');
            notification.className = `notification ${type}`;
            notification.innerHTML = `
                <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'} notification-icon"></i>
                <span>${message}</span>
                <i class="fas fa-times notification-close" onclick="this.parentElement.remove()"></i>
            `;
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.remove();
            }, 5000);
        }
        
        // Toggle auth form
        function toggleAuthForm(form) {
            if (form === 'register') {
                document.getElementById('loginForm').style.display = 'none';
                document.getElementById('registerForm').style.display = 'block';
            } else {
                document.getElementById('loginForm').style.display = 'block';
                document.getElementById('registerForm').style.display = 'none';
            }
        }
        
        // Check password strength
        function checkPasswordStrength(password) {
            const strengthBar = document.getElementById('passwordStrength');
            let strength = 0;
            
            if (password.length >= 8) strength++;
            if (password.match(/[a-z]/) && password.match(/[A-Z]/)) strength++;
            if (password.match(/[0-9]/)) strength++;
            if (password.match(/[^a-zA-Z0-9]/)) strength++;
            
            strengthBar.className = 'password-strength-fill';
            if (strength <= 1) {
                strengthBar.classList.add('weak');
            } else if (strength === 2) {
                strengthBar.classList.add('medium');
            } else {
                strengthBar.classList.add('strong');
            }
        }
        
        // Register function
        async function register(event) {
            event.preventDefault();
            
            const username = document.getElementById('registerUsername').value;
            const password = document.getElementById('registerPassword').value;
            const confirmPassword = document.getElementById('confirmPassword').value;
            
            if (password !== confirmPassword) {
                showNotification('Passwords do not match!', 'error');
                return false;
            }
            
            showLoading();
            
            try {
                const response = await fetch('/register', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        username: username,
                        password: password
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    showNotification('Account created successfully! Please login.', 'success');
                    toggleAuthForm('login');
                    document.getElementById('registerForm').reset();
                } else {
                    showNotification(data.error || 'Registration failed', 'error');
                }
            } catch (error) {
                showNotification('Network error. Please try again.', 'error');
            } finally {
                hideLoading();
            }
            
            return false;
        }
        
        // Login function
        async function login(event) {
            event.preventDefault();
            
            const username = document.getElementById('loginUsername').value;
            const password = document.getElementById('loginPassword').value;
            
            showLoading();
            
            try {
                const response = await fetch('/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        username: username,
                        password: password
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    currentUser = username;
                    showNotification('Welcome back, ' + username + '!', 'success');
                    document.getElementById('authContainer').style.display = 'none';
                    document.getElementById('chatContainer').style.display = 'flex';
                    loadChatHistory();
                } else {
                    showNotification(data.error || 'Login failed', 'error');
                }
            } catch (error) {
                showNotification('Network error. Please try again.', 'error');
            } finally {
                hideLoading();
            }
            
            return false;
        }
        
        // Logout function
        async function logout() {
            if (confirm('Are you sure you want to logout?')) {
                try {
                    await fetch('/logout', {
                        method: 'POST'
                    });
                    
                    currentUser = null;
                    document.getElementById('authContainer').style.display = 'block';
                    document.getElementById('chatContainer').style.display = 'none';
                    showNotification('Logged out successfully', 'success');
                } catch (error) {
                    console.error('Logout error:', error);
                }
            }
        }
        
        // Load chat history
        async function loadChatHistory() {
            try {
                const response = await fetch('/get_chat_history');
                const data = await response.json();
                
                if (data.success) {
                    chatHistory = data.history || [];
                    tokenUsage = data.token_usage || 0;
                    updateTokenBar();
                    
                    // Display messages
                    const container = document.getElementById('messagesContainer');
                    container.innerHTML = `
                        <div class="date-divider">
                            <span>Today</span>
                        </div>
                    `;
                    
                    chatHistory.forEach(msg => {
                        addMessageToUI(msg.role, msg.content);
                    });
                }
            } catch (error) {
                console.error('Error loading chat history:', error);
            }
        }
        
        // Auto-resize textarea
        function autoResizeTextarea() {
            const textarea = document.getElementById('messageInput');
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
        }
        
        // Handle input
        document.addEventListener('DOMContentLoaded', function() {
            const messageInput = document.getElementById('messageInput');
            if (messageInput) {
                messageInput.addEventListener('input', autoResizeTextarea);
                messageInput.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                    }
                });
            }
            
            // File input handler
            document.getElementById('fileInput').addEventListener('change', handleFileSelect);
            
            // Initialize
            createParticles();
            loadThemePreference();
            checkSession();
        });
        
        // Check session
        async function checkSession() {
            try {
                const response = await fetch('/check_session');
                const data = await response.json();
                
                if (data.logged_in) {
                    currentUser = data.username;
                    document.getElementById('authContainer').style.display = 'none';
                    document.getElementById('chatContainer').style.display = 'flex';
                    loadChatHistory();
                }
            } catch (error) {
                console.error('Session check error:', error);
            }
        }
        
        // Send message
        async function sendMessage() {
            const messageInput = document.getElementById('messageInput');
            const message = messageInput.value.trim();
            
            if (!message && attachedFiles.length === 0) {
                return;
            }
            
            currentPrompt = message;
            messageInput.value = '';
            autoResizeTextarea();
            document.getElementById('sendButton').disabled = true;
            
            // Add user message to UI
            if (message) {
                addMessageToUI('user', message);
            }
            
            // Show typing indicator
            document.getElementById('typingIndicator').classList.add('active');
            
            // Get enhanced prompt
            try {
                const enhanceResponse = await fetch('/enhance_prompt', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt: message
                    })
                });
                
                const enhanceData = await enhanceResponse.json();
                
                if (enhanceData.success) {
                    enhancedPrompt = enhanceData.enhanced_prompt;
                    showPromptModal(message, enhancedPrompt);
                } else {
                    await processMessage(message);
                }
            } catch (error) {
                console.error('Error:', error);
                await processMessage(message);
            }
        }
        
        // Show prompt modal
        function showPromptModal(original, enhanced) {
            document.getElementById('originalPromptText').textContent = original;
            document.getElementById('enhancedPromptText').textContent = enhanced;
            document.getElementById('promptModal').classList.add('active');
            document.getElementById('typingIndicator').classList.remove('active');
            document.getElementById('sendButton').disabled = false;
        }
        
        // Select prompt option
        function selectPromptOption(type) {
            selectedPromptType = type;
            document.getElementById('originalOption').classList.toggle('selected', type === 'original');
            document.getElementById('enhancedOption').classList.toggle('selected', type === 'enhanced');
        }
        
        // Close prompt modal
        function closePromptModal() {
            document.getElementById('promptModal').classList.remove('active');
            document.getElementById('sendButton').disabled = false;
        }
        
        // Confirm prompt selection
        async function confirmPromptSelection() {
            closePromptModal();
            const promptToUse = selectedPromptType === 'original' ? currentPrompt : enhancedPrompt;
            await processMessage(promptToUse);
        }
        
        // Process message
        async function processMessage(prompt) {
            document.getElementById('typingIndicator').classList.add('active');
            
            const formData = new FormData();
            formData.append('message', prompt);
            
            // Add files
            for (let file of attachedFiles) {
                formData.append('files', file);
            }
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    addMessageToUI('assistant', data.response);
                    tokenUsage = data.token_usage || tokenUsage;
                    updateTokenBar();
                    
                    if (tokenUsage > 100000) {
                        showNotification('Approaching context limit. Consider starting a new chat.', 'warning');
                    }
                } else {
                    showNotification(data.error || 'Failed to get response', 'error');
                }
            } catch (error) {
                console.error('Error:', error);
                showNotification('Network error. Please try again.', 'error');
            } finally {
                document.getElementById('typingIndicator').classList.remove('active');
                document.getElementById('sendButton').disabled = false;
                clearFiles();
            }
        }
        
        // Add message to UI
        function addMessageToUI(role, content) {
            const container = document.getElementById('messagesContainer');
            const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            messageDiv.innerHTML = `
                <div class="message-avatar">
                    <i class="fas fa-${role === 'user' ? 'user' : 'robot'}"></i>
                </div>
                <div class="message-content-wrapper">
                    <div class="message-header">
                        <span class="message-author">${role === 'user' ? 'You' : "Jack's AI"}</span>
                        <span class="message-time">${time}</span>
                    </div>
                    <div class="message-bubble">${content}</div>
                    <div class="message-actions">
                        <button class="message-action" onclick="copyMessage(this)">
                            <i class="fas fa-copy"></i> Copy
                        </button>
                        ${role === 'assistant' ? `
                            <button class="message-action" onclick="regenerateMessage(this)">
                                <i class="fas fa-redo"></i> Regenerate
                            </button>
                        ` : ''}
                    </div>
                </div>
            `;
            
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }
        
        // Copy message
        function copyMessage(button) {
            const content = button.closest('.message-content-wrapper').querySelector('.message-bubble').textContent;
            navigator.clipboard.writeText(content);
            showNotification('Message copied to clipboard', 'success');
        }
        
        // Update token bar
        function updateTokenBar() {
            const percentage = Math.min((tokenUsage / 125000) * 100, 100);
            document.getElementById('tokenFill').style.width = percentage + '%';
            document.getElementById('tokenCount').textContent = tokenUsage.toLocaleString();
            
            // Change color based on usage
            const fill = document.getElementById('tokenFill');
            if (percentage > 80) {
                fill.style.background = 'linear-gradient(135deg, #f5576c 0%, #f093fb 100%)';
            } else if (percentage > 60) {
                fill.style.background = 'linear-gradient(135deg, #FA8231 0%, #FFD14C 100%)';
            }
        }
        
        // Toggle file upload
        function toggleFileUpload() {
            const fileSection = document.getElementById('fileUploadSection');
            if (fileSection.classList.contains('active')) {
                fileSection.classList.remove('active');
            } else {
                fileSection.classList.add('active');
                document.getElementById('fileInput').click();
            }
        }
        
        // Handle file select
        function handleFileSelect(event) {
            const files = event.target.files;
            const preview = document.getElementById('filesPreview');
            
            for (let file of files) {
                if (file.size > 100 * 1024 * 1024) {
                    showNotification(`File ${file.name} is too large. Max size is 100MB.`, 'error');
                    continue;
                }
                
                attachedFiles.push(file);
                
                const fileItem = document.createElement('div');
                fileItem.className = 'file-preview-item';
                fileItem.innerHTML = `
                    <i class="fas fa-file"></i>
                    <span>${file.name}</span>
                    <i class="fas fa-times file-remove" onclick="removeFile('${file.name}')"></i>
                `;
                
                preview.appendChild(fileItem);
            }
            
            // Remove placeholder text
            const placeholder = preview.querySelector('.drop-zone-text');
            if (placeholder && attachedFiles.length > 0) {
                placeholder.style.display = 'none';
            }
        }
        
        // Remove file
        function removeFile(fileName) {
            attachedFiles = attachedFiles.filter(f => f.name !== fileName);
            
            const preview = document.getElementById('filesPreview');
            const items = preview.querySelectorAll('.file-preview-item');
            items.forEach(item => {
                if (item.textContent.includes(fileName)) {
                    item.remove();
                }
            });
            
            // Show placeholder if no files
            if (attachedFiles.length === 0) {
                const placeholder = preview.querySelector('.drop-zone-text');
                if (placeholder) {
                    placeholder.style.display = 'block';
                }
            }
        }
        
        // Clear files
        function clearFiles() {
            attachedFiles = [];
            const preview = document.getElementById('filesPreview');
            preview.innerHTML = '<span class="drop-zone-text">Drop files here or click to browse</span>';
            document.getElementById('fileUploadSection').classList.remove('active');
        }
        
        // New chat
        async function newChat() {
            if (confirm('Start a new chat? Current conversation will be saved.')) {
                try {
                    const response = await fetch('/new_chat', {
                        method: 'POST'
                    });
                    
                    if (response.ok) {
                        chatHistory = [];
                        tokenUsage = 0;
                        updateTokenBar();
                        
                        const container = document.getElementById('messagesContainer');
                        container.innerHTML = `
                            <div class="date-divider">
                                <span>Today</span>
                            </div>
                        `;
                        
                        addMessageToUI('assistant', 'New chat started! How can I help you today?');
                        showNotification('New chat created', 'success');
                    }
                } catch (error) {
                    console.error('Error:', error);
                    showNotification('Failed to create new chat', 'error');
                }
            }
        }
        
        // Compact chat
        async function compactChat() {
            if (chatHistory.length < 10) {
                showNotification('Chat is too short to compact', 'warning');
                return;
            }
            
            if (confirm('Compact this conversation to reduce token usage?')) {
                showLoading();
                
                try {
                    const response = await fetch('/compact_chat', {
                        method: 'POST'
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        const container = document.getElementById('messagesContainer');
                        container.innerHTML = `
                            <div class="date-divider">
                                <span>Compacted</span>
                            </div>
                        `;
                        
                        addMessageToUI('assistant', 'Chat compacted. Summary:\n\n' + data.summary);
                        tokenUsage = data.token_usage || 0;
                        updateTokenBar();
                        showNotification('Chat successfully compacted', 'success');
                    } else {
                        showNotification(data.error || 'Failed to compact chat', 'error');
                    }
                } catch (error) {
                    console.error('Error:', error);
                    showNotification('Failed to compact chat', 'error');
                } finally {
                    hideLoading();
                }
            }
        }
        
        // Export chat
        function exportChat() {
            const messages = document.querySelectorAll('.message');
            let exportText = 'Jack\'s AI Chat Export\n';
            exportText += '========================\n\n';
            
            messages.forEach(msg => {
                const author = msg.querySelector('.message-author').textContent;
                const time = msg.querySelector('.message-time').textContent;
                const content = msg.querySelector('.message-bubble').textContent;
                exportText += `[${time}] ${author}:\n${content}\n\n`;
            });
            
            const blob = new Blob([exportText], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `chat-export-${new Date().toISOString()}.txt`;
            a.click();
            
            showNotification('Chat exported successfully', 'success');
        }
        
        // Insert prompt
        function insertPrompt(text) {
            const input = document.getElementById('messageInput');
            input.value = text + ' ';
            input.focus();
            autoResizeTextarea();
        }
        
        // Toggle voice input (placeholder)
        function toggleVoiceInput() {
            showNotification('Voice input coming soon!', 'warning');
        }
        
        // Show/hide loading
        function showLoading() {
            document.getElementById('loadingOverlay').classList.add('active');
        }
        
        function hideLoading() {
            document.getElementById('loadingOverlay').classList.remove('active');
        }
        
        // Regenerate message (placeholder)
        function regenerateMessage(button) {
            showNotification('Regenerate feature coming soon!', 'warning');
        }
    </script>
</body>
</html>
"""

# API key rotation system
API_KEYS = []
current_key_index = 0

def get_api_keys():
    """Get API keys from environment variables"""
    global API_KEYS
    for i in range(1, 11):
        key = os.environ.get(f'GEMINI_API_KEY_{i}')
        if key:
            API_KEYS.append(key)
            API_KEYS_STATUS[key] = {'failures': 0, 'last_used': None}
    
    if not API_KEYS:
        default_key = os.environ.get('GEMINI_API_KEY')
        if default_key:
            API_KEYS.append(default_key)
            API_KEYS_STATUS[default_key] = {'failures': 0, 'last_used': None}
    
    if not API_KEYS:
        print("WARNING: No API keys found! Please set GEMINI_API_KEY environment variables.")
        API_KEYS.append("YOUR_API_KEY_HERE")

def get_next_api_key():
    """Rotate through available API keys"""
    global current_key_index
    
    if not API_KEYS:
        get_api_keys()
    
    attempts = 0
    while attempts < len(API_KEYS):
        key = API_KEYS[current_key_index]
        
        if API_KEYS_STATUS.get(key, {}).get('failures', 0) < 3:
            API_KEYS_STATUS[key]['last_used'] = datetime.now()
            current_key_index = (current_key_index + 1) % len(API_KEYS)
            return key
        
        current_key_index = (current_key_index + 1) % len(API_KEYS)
        attempts += 1
    
    for key in API_KEYS:
        API_KEYS_STATUS[key]['failures'] = 0
    
    return API_KEYS[0] if API_KEYS else "YOUR_API_KEY_HERE"

def mark_api_key_failure(api_key):
    """Mark an API key as having failed"""
    if api_key in API_KEYS_STATUS:
        API_KEYS_STATUS[api_key]['failures'] += 1

# System prompts for the AI models
PROMPT_ENHANCER_SYSTEM = """You are a prompt enhancement specialist. Your job is to take user prompts and make them clearer, more detailed, and more effective for an AI assistant.

Rules:
1. Preserve the user's original intent completely
2. Add clarity and context where helpful
3. Structure the prompt for better AI understanding
4. Include specific details that will help get a better response
5. Make the prompt comprehensive but not overly long
6. If the prompt involves analysis of files or images, specify what kind of analysis would be most helpful

Take the user's prompt and rewrite it to be more effective. Return ONLY the enhanced prompt, nothing else."""

MAIN_AI_SYSTEM = """You are Jack's AI, an ultra-advanced artificial intelligence assistant powered by cutting-edge Gemini 2.5 Pro technology. You are incredibly capable, intelligent, and helpful.

CORE PRINCIPLES:

1. COMPREHENSIVE RESPONSES
   - Provide extremely detailed, thorough answers
   - Never use placeholders or shortcuts
   - Include all necessary code, explanations, and examples
   - Every response should be production-ready

2. CLARITY AND EDUCATION
   - Explain complex concepts in simple terms
   - Use analogies and examples liberally
   - Break down steps clearly
   - Write as if teaching someone new to the topic

3. MAXIMUM VALUE
   - Use the full context window when beneficial
   - Provide multiple solutions when applicable
   - Include best practices and recommendations
   - Anticipate follow-up questions

4. CAPABILITIES
   - Advanced code generation in any language
   - Complex document and image analysis
   - Creative problem solving
   - Data analysis and visualization
   - Research and synthesis
   - Mathematical computations

5. PERSONALITY
   - Professional yet friendly
   - Enthusiastic and engaging
   - Proactive and helpful
   - Use emojis appropriately 🚀

Remember: The user has unlimited access to your capabilities. Give them exceptional, comprehensive responses that exceed expectations."""

CHAT_COMPACTOR_SYSTEM = """You are a conversation summarizer. Create a comprehensive summary that preserves all important information while reducing token usage.

Requirements:
1. Keep all key facts, decisions, and outcomes
2. Maintain chronological flow
3. Preserve technical details and code
4. Summarize repetitive discussions efficiently
5. Include all solutions and answers provided
6. Make the summary detailed enough for seamless continuation

Create a thorough yet efficient summary."""

def create_ai_client(api_key):
    """Create an OpenAI client configured for Gemini"""
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

def count_tokens(text):
    """Estimate token count for text"""
    return len(text) // 4

def process_file_for_ai(file):
    """Process uploaded file and convert to AI-readable format"""
    try:
        file_content = ""
        file_type = file.content_type
        
        if file_type.startswith('image/'):
            img = Image.open(file)
            buffered = BytesIO()
            img.save(buffered, format=img.format if img.format else 'PNG')
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            file_content = f"[Image file: {file.filename}]\n[Image data available for analysis]"
            return file_content, img_base64
            
        elif file_type == 'application/pdf':
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            file_content = f"[PDF file: {file.filename}]\nContent:\n{text}"
            
        elif file_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']:
            doc = docx.Document(file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            file_content = f"[Word document: {file.filename}]\nContent:\n{text}"
            
        elif file_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']:
            workbook = openpyxl.load_workbook(file)
            text = ""
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                text += f"\nSheet: {sheet_name}\n"
                for row in sheet.iter_rows(values_only=True):
                    text += "\t".join([str(cell) if cell else "" for cell in row]) + "\n"
            file_content = f"[Excel file: {file.filename}]\nContent:\n{text}"
            
        elif file_type.startswith('text/'):
            text = file.read().decode('utf-8', errors='ignore')
            file_content = f"[Text file: {file.filename}]\nContent:\n{text}"
            
        else:
            file_content = f"[File: {file.filename}]\n[Type: {file_type}]\n[Unable to process]"
        
        return file_content, None
        
    except Exception as e:
        return f"[Error processing {file.filename}: {str(e)}]", None

# Flask routes
def login_required(f):
    """Decorator to check if user is logged in"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return jsonify({'error': 'Not logged in'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/register', methods=['POST'])
def register():
    """Handle user registration"""
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'error': 'All fields are required'}), 400
        
        if len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        if username in USERS_DB:
            return jsonify({'error': 'Username already exists'}), 400
        
        USERS_DB[username] = {
            'password_hash': generate_password_hash(password),
            'created_at': datetime.now().isoformat(),
            'last_login': None
        }
        
        CHAT_SESSIONS[username] = []
        USER_TOKENS[username] = 0
        USER_PREFERENCES[username] = {'theme': 'dark'}
        
        return jsonify({'success': True, 'message': 'Registration successful'}), 200
        
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/login', methods=['POST'])
def login():
    """Handle user login"""
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if username not in USERS_DB:
            return jsonify({'error': 'Invalid username or password'}), 401
        
        if not check_password_hash(USERS_DB[username]['password_hash'], password):
            return jsonify({'error': 'Invalid username or password'}), 401
        
        USERS_DB[username]['last_login'] = datetime.now().isoformat()
        session['username'] = username
        
        if username not in CHAT_SESSIONS:
            CHAT_SESSIONS[username] = []
        if username not in USER_TOKENS:
            USER_TOKENS[username] = 0
        if username not in USER_PREFERENCES:
            USER_PREFERENCES[username] = {'theme': 'dark'}
        
        return jsonify({'success': True, 'message': 'Login successful'}), 200
        
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    """Handle user logout"""
    session.clear()
    return jsonify({'success': True}), 200

@app.route('/check_session', methods=['GET'])
def check_session():
    """Check if user is logged in"""
    if 'username' in session:
        return jsonify({'logged_in': True, 'username': session['username']}), 200
    return jsonify({'logged_in': False}), 200

@app.route('/get_chat_history', methods=['GET'])
@login_required
def get_chat_history():
    """Get user's chat history"""
    username = session['username']
    history = CHAT_SESSIONS.get(username, [])
    token_usage = USER_TOKENS.get(username, 0)
    
    return jsonify({
        'success': True,
        'history': history,
        'token_usage': token_usage
    }), 200

@app.route('/enhance_prompt', methods=['POST'])
@login_required
def enhance_prompt():
    """Enhance user's prompt using AI"""
    try:
        data = request.json
        original_prompt = data.get('prompt', '')
        
        if not original_prompt:
            return jsonify({'success': False, 'error': 'No prompt provided'}), 400
        
        api_key = get_next_api_key()
        client = create_ai_client(api_key)
        
        try:
            response = client.chat.completions.create(
                model="gemini-2.5-flash",  # Changed from 2.0 to 2.5
                messages=[
                    {"role": "system", "content": PROMPT_ENHANCER_SYSTEM},
                    {"role": "user", "content": original_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            enhanced_prompt = response.choices[0].message.content
            
            return jsonify({
                'success': True,
                'enhanced_prompt': enhanced_prompt
            }), 200
            
        except Exception as api_error:
            print(f"API error in enhance_prompt: {api_error}")
            mark_api_key_failure(api_key)
            return jsonify({
                'success': True,
                'enhanced_prompt': original_prompt
            }), 200
            
    except Exception as e:
        print(f"Enhance prompt error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    """Handle chat messages with AI"""
    try:
        username = session['username']
        message = request.form.get('message', '')
        files = request.files.getlist('files')
        
        file_contents = []
        image_data = None
        
        for file in files:
            if file:
                content, img_data = process_file_for_ai(file)
                file_contents.append(content)
                if img_data:
                    image_data = img_data
        
        full_message = message
        if file_contents:
            full_message += "\n\n" + "\n".join(file_contents)
        
        chat_history = CHAT_SESSIONS.get(username, [])
        
        messages = [
            {"role": "system", "content": MAIN_AI_SYSTEM}
        ]
        
        for msg in chat_history[-10:]:
            messages.append({"role": msg['role'], "content": msg['content']})
        
        messages.append({"role": "user", "content": full_message})
        
        api_key = get_next_api_key()
        client = create_ai_client(api_key)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gemini-2.5-pro",
                    messages=messages,
                    max_tokens=60000,
                    temperature=0.9
                )
                
                ai_response = response.choices[0].message.content
                
                chat_history.append({"role": "user", "content": message})
                chat_history.append({"role": "assistant", "content": ai_response})
                CHAT_SESSIONS[username] = chat_history
                
                token_usage = USER_TOKENS.get(username, 0)
                token_usage += count_tokens(full_message) + count_tokens(ai_response)
                USER_TOKENS[username] = token_usage
                
                return jsonify({
                    'success': True,
                    'response': ai_response,
                    'token_usage': token_usage
                }), 200
                
            except Exception as api_error:
                print(f"API attempt {attempt + 1} failed: {api_error}")
                mark_api_key_failure(api_key)
                
                if attempt < max_retries - 1:
                    api_key = get_next_api_key()
                    client = create_ai_client(api_key)
                else:
                    return jsonify({
                        'success': False,
                        'error': 'AI service temporarily unavailable. Please try again.'
                    }), 500
        
    except Exception as e:
        print(f"Chat error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/new_chat', methods=['POST'])
@login_required
def new_chat():
    """Start a new chat session"""
    username = session['username']
    CHAT_SESSIONS[username] = []
    USER_TOKENS[username] = 0
    
    return jsonify({'success': True}), 200

@app.route('/compact_chat', methods=['POST'])
@login_required
def compact_chat():
    """Compact the chat history to reduce tokens"""
    try:
        username = session['username']
        chat_history = CHAT_SESSIONS.get(username, [])
        
        if len(chat_history) < 10:
            return jsonify({
                'success': False,
                'error': 'Chat history too short to compact'
            }), 400
        
        conversation_text = ""
        for msg in chat_history:
            role = "User" if msg['role'] == 'user' else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n\n"
        
        api_key = get_next_api_key()
        client = create_ai_client(api_key)
        
        try:
            response = client.chat.completions.create(
                model="gemini-2.5-flash",  # Changed from 2.0 to 2.5
                messages=[
                    {"role": "system", "content": CHAT_COMPACTOR_SYSTEM},
                    {"role": "user", "content": f"Please summarize this conversation:\n\n{conversation_text}"}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            summary = response.choices[0].message.content
            
            CHAT_SESSIONS[username] = [
                {"role": "assistant", "content": summary}
            ]
            
            token_usage = count_tokens(summary)
            USER_TOKENS[username] = token_usage
            
            return jsonify({
                'success': True,
                'summary': summary,
                'token_usage': token_usage
            }), 200
            
        except Exception as api_error:
            print(f"Compact chat API error: {api_error}")
            mark_api_key_failure(api_key)
            return jsonify({
                'success': False,
                'error': 'Failed to compact chat. Please try again.'
            }), 500
            
    except Exception as e:
        print(f"Compact chat error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Initialize API keys when the app starts
get_api_keys()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)