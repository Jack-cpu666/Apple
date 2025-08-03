"""
Jack's AI - Ultra Modern Web Application (No Authentication)
Open access version - No login required
Author: Jack's AI System
Version: 3.0.0
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

from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for, make_response
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from openai import OpenAI
from PIL import Image
import PyPDF2
import docx
import openpyxl

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

CHAT_SESSIONS = {}
API_KEYS_STATUS = {}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Jack AI Beta - Next Generation Intelligence</title>
    <link href="https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@200;300;400;500;600;700;800;900&family=Inter:wght@100;200;300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --primary-light: #818cf8;
            --secondary: #ec4899;
            --accent: #14b8a6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --dark: #0f172a;
            --dark-lighter: #1e293b;
            --dark-card: #1a2332;
            --light: #f8fafc;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border: rgba(148, 163, 184, 0.1);
            --glow: rgba(99, 102, 241, 0.5);
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.25);
            --shadow-xl: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --gradient-secondary: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --gradient-accent: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --gradient-dark: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-base: 250ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-slow: 350ms cubic-bezier(0.4, 0, 0.2, 1);
            --blur-sm: 4px;
            --blur-base: 8px;
            --blur-lg: 16px;
            --blur-xl: 24px;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
            background: var(--dark);
            color: var(--text-primary);
            overflow: hidden;
            height: 100vh;
            position: relative;
        }

        .background-effects {
            position: fixed;
            inset: 0;
            z-index: 0;
            overflow: hidden;
            pointer-events: none;
        }

        .gradient-orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(100px);
            opacity: 0.5;
            animation: float 20s infinite ease-in-out;
        }

        .gradient-orb:nth-child(1) {
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, var(--primary) 0%, transparent 70%);
            top: -200px;
            left: -200px;
            animation-duration: 25s;
        }

        .gradient-orb:nth-child(2) {
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, var(--secondary) 0%, transparent 70%);
            bottom: -150px;
            right: -150px;
            animation-duration: 30s;
            animation-delay: 5s;
        }

        .gradient-orb:nth-child(3) {
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, var(--accent) 0%, transparent 70%);
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            animation-duration: 35s;
            animation-delay: 10s;
        }

        @keyframes float {
            0%, 100% { transform: translate(0, 0) scale(1); }
            25% { transform: translate(50px, -50px) scale(1.1); }
            50% { transform: translate(-30px, 30px) scale(0.9); }
            75% { transform: translate(-50px, -30px) scale(1.05); }
        }

        .grid-overlay {
            position: absolute;
            inset: 0;
            background-image: 
                linear-gradient(rgba(99, 102, 241, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(99, 102, 241, 0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            animation: grid-move 10s linear infinite;
        }

        @keyframes grid-move {
            0% { transform: translate(0, 0); }
            100% { transform: translate(50px, 50px); }
        }

        .app-container {
            position: relative;
            width: 100%;
            height: 100vh;
            display: flex;
            z-index: 1;
        }

        .sidebar {
            width: 280px;
            background: rgba(26, 35, 50, 0.6);
            backdrop-filter: blur(var(--blur-xl));
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            transition: transform var(--transition-base);
            z-index: 20;
        }

        .sidebar-header {
            padding: 1.5rem;
            border-bottom: 1px solid var(--border);
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%);
        }

        .logo-container {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .logo-icon {
            width: 48px;
            height: 48px;
            background: var(--gradient-primary);
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 30px rgba(99, 102, 241, 0.5);
            animation: pulse-glow 2s infinite;
        }

        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 30px rgba(99, 102, 241, 0.5); }
            50% { box-shadow: 0 0 50px rgba(99, 102, 241, 0.8); }
        }

        .logo-icon svg {
            width: 28px;
            height: 28px;
            fill: white;
        }

        .logo-text h1 {
            font-size: 1.25rem;
            font-weight: 700;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.25rem;
        }

        .logo-text p {
            font-size: 0.75rem;
            color: var(--text-secondary);
            font-weight: 500;
        }

        .new-chat-btn {
            margin: 1rem;
            padding: 0.875rem;
            background: var(--gradient-primary);
            border: none;
            border-radius: 12px;
            color: white;
            font-weight: 600;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all var(--transition-base);
            position: relative;
            overflow: hidden;
        }

        .new-chat-btn::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            transition: width var(--transition-slow), height var(--transition-slow);
        }

        .new-chat-btn:hover::before {
            width: 300px;
            height: 300px;
        }

        .new-chat-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
        }

        .chat-history {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
        }

        .chat-history::-webkit-scrollbar {
            width: 4px;
        }

        .chat-history::-webkit-scrollbar-track {
            background: transparent;
        }

        .chat-history::-webkit-scrollbar-thumb {
            background: var(--text-muted);
            border-radius: 2px;
        }

        .history-item {
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            border-radius: 8px;
            cursor: pointer;
            transition: all var(--transition-fast);
            border: 1px solid transparent;
        }

        .history-item:hover {
            background: rgba(99, 102, 241, 0.1);
            border-color: var(--primary);
        }

        .history-item-title {
            font-size: 0.875rem;
            font-weight: 500;
            margin-bottom: 0.25rem;
            color: var(--text-primary);
        }

        .history-item-preview {
            font-size: 0.75rem;
            color: var(--text-secondary);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .sidebar-footer {
            padding: 1rem;
            border-top: 1px solid var(--border);
            background: rgba(15, 23, 42, 0.5);
        }

        .settings-btn {
            width: 100%;
            padding: 0.75rem;
            background: transparent;
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-secondary);
            font-size: 0.875rem;
            cursor: pointer;
            transition: all var(--transition-fast);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .settings-btn:hover {
            background: rgba(99, 102, 241, 0.1);
            border-color: var(--primary);
            color: var(--primary);
        }

        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            position: relative;
            overflow: hidden;
        }

        .chat-header {
            padding: 1rem 2rem;
            background: rgba(26, 35, 50, 0.4);
            backdrop-filter: blur(var(--blur-lg));
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .chat-info {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 100px;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--success);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--success);
            border-radius: 50%;
            animation: blink 2s infinite;
        }

        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }

        .header-actions {
            display: flex;
            gap: 0.5rem;
        }

        .header-btn {
            width: 36px;
            height: 36px;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all var(--transition-fast);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }

        .header-btn:hover {
            background: var(--primary);
            color: white;
            transform: scale(1.05);
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
        }

        .tooltip {
            position: absolute;
            bottom: -35px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--dark-card);
            color: var(--text-primary);
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            font-size: 0.75rem;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity var(--transition-fast);
            z-index: 100;
        }

        .header-btn:hover .tooltip {
            opacity: 1;
        }

        .messages-container {
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
            scroll-behavior: smooth;
        }

        .messages-container::-webkit-scrollbar {
            width: 6px;
        }

        .messages-container::-webkit-scrollbar-track {
            background: transparent;
        }

        .messages-container::-webkit-scrollbar-thumb {
            background: var(--text-muted);
            border-radius: 3px;
        }

        .message {
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
            animation: message-appear 0.3s ease-out;
        }

        @keyframes message-appear {
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
            width: 40px;
            height: 40px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            position: relative;
        }

        .message.user .message-avatar {
            background: var(--gradient-primary);
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
        }

        .message.assistant .message-avatar {
            background: var(--gradient-secondary);
            box-shadow: 0 0 20px rgba(236, 72, 153, 0.3);
        }

        .message-avatar::after {
            content: '';
            position: absolute;
            inset: -2px;
            border-radius: 12px;
            padding: 2px;
            background: var(--gradient-primary);
            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            opacity: 0;
            transition: opacity var(--transition-base);
        }

        .message:hover .message-avatar::after {
            opacity: 1;
        }

        .message-content {
            max-width: 70%;
        }

        .message-meta {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
            padding: 0 0.5rem;
        }

        .message-author {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .message-time {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .message-bubble {
            padding: 1rem 1.25rem;
            border-radius: 18px;
            position: relative;
            line-height: 1.6;
            word-wrap: break-word;
        }

        .message.user .message-bubble {
            background: var(--gradient-primary);
            color: white;
            border-bottom-right-radius: 4px;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.2);
        }

        .message.assistant .message-bubble {
            background: rgba(26, 35, 50, 0.8);
            backdrop-filter: blur(var(--blur-base));
            border: 1px solid var(--border);
            color: var(--text-primary);
            border-bottom-left-radius: 4px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }

        .message-actions {
            display: flex;
            gap: 0.25rem;
            margin-top: 0.5rem;
            padding: 0 0.5rem;
            opacity: 0;
            transition: opacity var(--transition-fast);
        }

        .message:hover .message-actions {
            opacity: 1;
        }

        .message-action {
            padding: 0.25rem 0.5rem;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text-secondary);
            font-size: 0.75rem;
            cursor: pointer;
            transition: all var(--transition-fast);
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }

        .message-action:hover {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
            transform: scale(1.05);
        }

        .typing-indicator {
            display: none;
            align-items: center;
            gap: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .typing-indicator.active {
            display: flex;
            animation: fade-in 0.3s ease-out;
        }

        @keyframes fade-in {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .typing-dots {
            display: flex;
            gap: 4px;
            padding: 0.75rem 1rem;
            background: rgba(26, 35, 50, 0.8);
            backdrop-filter: blur(var(--blur-base));
            border: 1px solid var(--border);
            border-radius: 18px;
        }

        .typing-dot {
            width: 8px;
            height: 8px;
            background: var(--primary);
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

        .input-container {
            padding: 1.5rem 2rem;
            background: rgba(26, 35, 50, 0.6);
            backdrop-filter: blur(var(--blur-xl));
            border-top: 1px solid var(--border);
        }

        .input-wrapper {
            display: flex;
            gap: 1rem;
            align-items: flex-end;
        }

        .input-field {
            flex: 1;
            position: relative;
        }

        .input-box {
            width: 100%;
            min-height: 48px;
            max-height: 120px;
            padding: 0.75rem 3rem 0.75rem 1rem;
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--border);
            border-radius: 12px;
            color: var(--text-primary);
            font-size: 0.875rem;
            font-family: inherit;
            resize: none;
            transition: all var(--transition-fast);
            line-height: 1.5;
        }

        .input-box:focus {
            outline: none;
            border-color: var(--primary);
            background: rgba(15, 23, 42, 0.8);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }

        .input-box::placeholder {
            color: var(--text-muted);
        }

        .input-tools {
            position: absolute;
            right: 0.5rem;
            top: 50%;
            transform: translateY(-50%);
            display: flex;
            gap: 0.25rem;
        }

        .input-tool {
            width: 32px;
            height: 32px;
            background: transparent;
            border: none;
            border-radius: 6px;
            color: var(--text-muted);
            cursor: pointer;
            transition: all var(--transition-fast);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .input-tool:hover {
            background: rgba(99, 102, 241, 0.1);
            color: var(--primary);
        }

        .send-btn {
            padding: 0.75rem 1.5rem;
            background: var(--gradient-primary);
            border: none;
            border-radius: 12px;
            color: white;
            font-weight: 600;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all var(--transition-base);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            position: relative;
            overflow: hidden;
        }

        .send-btn::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            transition: width var(--transition-slow), height var(--transition-slow);
        }

        .send-btn:hover::before {
            width: 200px;
            height: 200px;
        }

        .send-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
        }

        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .quick-actions {
            display: flex;
            gap: 0.5rem;
            margin-top: 1rem;
            flex-wrap: wrap;
        }

        .quick-action {
            padding: 0.5rem 1rem;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid var(--border);
            border-radius: 100px;
            color: var(--text-secondary);
            font-size: 0.75rem;
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-fast);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .quick-action:hover {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
            transform: scale(1.05);
        }

        .mobile-menu-btn {
            display: none;
            position: fixed;
            top: 1rem;
            left: 1rem;
            width: 48px;
            height: 48px;
            background: rgba(26, 35, 50, 0.8);
            backdrop-filter: blur(var(--blur-base));
            border: 1px solid var(--border);
            border-radius: 12px;
            color: var(--text-primary);
            cursor: pointer;
            z-index: 30;
            align-items: center;
            justify-content: center;
            transition: all var(--transition-fast);
        }

        .mobile-menu-btn:hover {
            background: var(--primary);
            color: white;
        }

        .file-upload-area {
            display: none;
            padding: 1rem;
            margin-bottom: 1rem;
            background: rgba(99, 102, 241, 0.05);
            border: 2px dashed var(--primary);
            border-radius: 12px;
            text-align: center;
            transition: all var(--transition-fast);
        }

        .file-upload-area.active {
            display: block;
            animation: slide-down 0.3s ease-out;
        }

        @keyframes slide-down {
            from {
                opacity: 0;
                max-height: 0;
            }
            to {
                opacity: 1;
                max-height: 200px;
            }
        }

        .file-upload-area.drag-over {
            background: rgba(99, 102, 241, 0.1);
            border-color: var(--primary);
        }

        .file-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .file-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid var(--primary);
            border-radius: 8px;
            font-size: 0.75rem;
            color: var(--primary);
        }

        .file-remove {
            cursor: pointer;
            opacity: 0.7;
            transition: opacity var(--transition-fast);
        }

        .file-remove:hover {
            opacity: 1;
        }

        .notification {
            position: fixed;
            top: 2rem;
            right: 2rem;
            padding: 1rem 1.5rem;
            background: var(--dark-card);
            backdrop-filter: blur(var(--blur-lg));
            border: 1px solid var(--border);
            border-radius: 12px;
            color: var(--text-primary);
            box-shadow: var(--shadow-xl);
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 1rem;
            animation: slide-in 0.3s ease-out;
        }

        @keyframes slide-in {
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
            border-color: var(--success);
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
        }

        .notification.error {
            border-color: var(--danger);
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
        }

        .notification-close {
            margin-left: auto;
            cursor: pointer;
            opacity: 0.7;
            transition: opacity var(--transition-fast);
        }

        .notification-close:hover {
            opacity: 1;
        }

        .modal-overlay {
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(var(--blur-base));
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            padding: 2rem;
        }

        .modal-overlay.active {
            display: flex;
            animation: fade-in 0.3s ease-out;
        }

        .modal {
            background: var(--dark-card);
            backdrop-filter: blur(var(--blur-xl));
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 2rem;
            max-width: 600px;
            width: 100%;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: var(--shadow-xl);
            animation: modal-appear 0.3s ease-out;
        }

        @keyframes modal-appear {
            from {
                opacity: 0;
                transform: scale(0.9);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }

        .modal-header {
            margin-bottom: 1.5rem;
        }

        .modal-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }

        .modal-subtitle {
            color: var(--text-secondary);
            font-size: 0.875rem;
        }

        .modal-options {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .modal-option {
            padding: 1.25rem;
            background: rgba(99, 102, 241, 0.05);
            border: 2px solid var(--border);
            border-radius: 12px;
            cursor: pointer;
            transition: all var(--transition-fast);
        }

        .modal-option:hover {
            border-color: var(--primary);
            background: rgba(99, 102, 241, 0.1);
        }

        .modal-option.selected {
            border-color: var(--primary);
            background: rgba(99, 102, 241, 0.15);
        }

        .modal-option-title {
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }

        .modal-option-text {
            color: var(--text-secondary);
            font-size: 0.875rem;
            line-height: 1.5;
        }

        .modal-actions {
            display: flex;
            gap: 1rem;
        }

        .modal-btn {
            flex: 1;
            padding: 0.875rem;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all var(--transition-fast);
            border: none;
        }

        .modal-btn-primary {
            background: var(--gradient-primary);
            color: white;
        }

        .modal-btn-primary:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
        }

        .modal-btn-secondary {
            background: transparent;
            color: var(--text-primary);
            border: 1px solid var(--border);
        }

        .modal-btn-secondary:hover {
            background: rgba(99, 102, 241, 0.1);
            border-color: var(--primary);
        }

        @media (max-width: 768px) {
            .sidebar {
                position: fixed;
                left: 0;
                top: 0;
                bottom: 0;
                transform: translateX(-100%);
                width: 280px;
                z-index: 40;
            }

            .sidebar.open {
                transform: translateX(0);
            }

            .mobile-menu-btn {
                display: flex;
            }

            .main-content {
                margin-left: 0;
            }

            .chat-header {
                padding-left: 4rem;
            }

            .messages-container {
                padding: 1rem;
            }

            .message-content {
                max-width: 85%;
            }

            .modal {
                padding: 1.5rem;
            }

            .quick-actions {
                overflow-x: auto;
                flex-wrap: nowrap;
                -webkit-overflow-scrolling: touch;
            }

            .quick-actions::-webkit-scrollbar {
                display: none;
            }
        }

        @media (max-width: 480px) {
            .logo-text h1 {
                font-size: 1rem;
            }

            .chat-info {
                flex-direction: column;
                align-items: flex-start;
                gap: 0.5rem;
            }

            .header-actions {
                position: absolute;
                top: 1rem;
                right: 1rem;
            }

            .message-bubble {
                font-size: 0.875rem;
                padding: 0.875rem 1rem;
            }

            .input-container {
                padding: 1rem;
            }

            .send-btn span {
                display: none;
            }

            .send-btn {
                padding: 0.75rem;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            * {
                animation: none !important;
                transition: none !important;
            }
        }

        .loading-spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="background-effects">
        <div class="gradient-orb"></div>
        <div class="gradient-orb"></div>
        <div class="gradient-orb"></div>
        <div class="grid-overlay"></div>
    </div>

    <div class="app-container">
        <button class="mobile-menu-btn" onclick="toggleSidebar()">
            <i class="fas fa-bars"></i>
        </button>

        <aside class="sidebar" id="sidebar">
            <div class="sidebar-header">
                <div class="logo-container">
                    <div class="logo-icon">
                        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M2 17L12 22L22 17" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M2 12L12 17L22 12" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </div>
                    <div class="logo-text">
                        <h1>Jack AI Beta</h1>
                        <p>Next-Gen Intelligence</p>
                    </div>
                </div>
            </div>

            <button class="new-chat-btn" onclick="newChat()">
                <i class="fas fa-plus"></i> New Chat
            </button>

            <div class="chat-history" id="chatHistory">
                <div class="history-item">
                    <div class="history-item-title">Welcome to Jack AI</div>
                    <div class="history-item-preview">Start your conversation...</div>
                </div>
            </div>

            <div class="sidebar-footer">
                <button class="settings-btn" onclick="openSettings()">
                    <i class="fas fa-cog"></i> Settings
                </button>
            </div>
        </aside>

        <main class="main-content">
            <header class="chat-header">
                <div class="chat-info">
                    <div class="status-badge">
                        <span class="status-dot"></span>
                        <span>Online</span>
                    </div>
                </div>

                <div class="header-actions">
                    <button class="header-btn" onclick="clearChat()">
                        <i class="fas fa-trash"></i>
                        <span class="tooltip">Clear Chat</span>
                    </button>
                    <button class="header-btn" onclick="exportChat()">
                        <i class="fas fa-download"></i>
                        <span class="tooltip">Export</span>
                    </button>
                    <button class="header-btn" onclick="toggleTheme()">
                        <i class="fas fa-moon"></i>
                        <span class="tooltip">Theme</span>
                    </button>
                </div>
            </header>

            <div class="messages-container" id="messagesContainer">
                <div class="message assistant">
                    <div class="message-avatar">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="message-content">
                        <div class="message-meta">
                            <span class="message-author">Jack AI</span>
                            <span class="message-time">Now</span>
                        </div>
                        <div class="message-bubble">
                            Welcome! I'm Jack AI Beta, your advanced AI assistant. I can help you with complex tasks, analyze documents, generate code, solve problems, and much more. How can I assist you today?
                        </div>
                        <div class="message-actions">
                            <button class="message-action" onclick="copyMessage(this)">
                                <i class="fas fa-copy"></i> Copy
                            </button>
                        </div>
                    </div>
                </div>
            </div>

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

            <div class="input-container">
                <div class="file-upload-area" id="fileUploadArea">
                    <i class="fas fa-cloud-upload-alt" style="font-size: 2rem; color: var(--primary); margin-bottom: 0.5rem;"></i>
                    <p style="color: var(--text-secondary); margin-bottom: 0.5rem;">Drop files here or click to browse</p>
                    <p style="color: var(--text-muted); font-size: 0.75rem;">Supports documents, images, PDFs, and more (max 100MB)</p>
                    <div class="file-list" id="fileList"></div>
                </div>

                <div class="input-wrapper">
                    <div class="input-field">
                        <textarea 
                            class="input-box" 
                            id="messageInput" 
                            placeholder="Ask me anything..."
                            rows="1"
                        ></textarea>
                        <div class="input-tools">
                            <button class="input-tool" onclick="toggleFileUpload()">
                                <i class="fas fa-paperclip"></i>
                            </button>
                        </div>
                    </div>
                    <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                        <i class="fas fa-paper-plane"></i>
                        <span>Send</span>
                    </button>
                </div>

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

    <div class="modal-overlay" id="promptModal">
        <div class="modal">
            <div class="modal-header">
                <h2 class="modal-title">✨ Enhance Your Prompt</h2>
                <p class="modal-subtitle">Choose how you'd like to ask your question</p>
            </div>
            
            <div class="modal-options">
                <div class="modal-option" id="originalOption" onclick="selectPromptOption('original')">
                    <div class="modal-option-title">Original Prompt</div>
                    <div class="modal-option-text" id="originalPromptText"></div>
                </div>
                
                <div class="modal-option selected" id="enhancedOption" onclick="selectPromptOption('enhanced')">
                    <div class="modal-option-title">Enhanced Prompt (Recommended)</div>
                    <div class="modal-option-text" id="enhancedPromptText"></div>
                </div>
            </div>
            
            <div class="modal-actions">
                <button class="modal-btn modal-btn-secondary" onclick="closePromptModal()">
                    Cancel
                </button>
                <button class="modal-btn modal-btn-primary" onclick="confirmPromptSelection()">
                    Use Selected
                </button>
            </div>
        </div>
    </div>

    <input type="file" id="fileInput" multiple style="display: none;">

    <script>
        let chatHistory = [];
        let selectedPromptType = 'enhanced';
        let currentPrompt = '';
        let enhancedPrompt = '';
        let attachedFiles = [];
        let sessionId = null;

        function generateSessionId() {
            return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        }

        function initializeSession() {
            sessionId = localStorage.getItem('sessionId');
            if (!sessionId) {
                sessionId = generateSessionId();
                localStorage.setItem('sessionId', sessionId);
            }
            
            const savedHistory = localStorage.getItem('chatHistory');
            if (savedHistory) {
                try {
                    chatHistory = JSON.parse(savedHistory);
                    displayChatHistory();
                } catch (e) {
                    console.error('Error loading chat history:', e);
                }
            }
        }

        function displayChatHistory() {
            const container = document.getElementById('messagesContainer');
            container.innerHTML = '';
            
            chatHistory.forEach(msg => {
                addMessageToUI(msg.role, msg.content, false);
            });
        }

        function toggleSidebar() {
            const sidebar = document.getElementById('sidebar');
            sidebar.classList.toggle('open');
        }

        function autoResizeTextarea() {
            const textarea = document.getElementById('messageInput');
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
        }

        document.addEventListener('DOMContentLoaded', function() {
            initializeSession();
            
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
            
            const fileInput = document.getElementById('fileInput');
            if (fileInput) {
                fileInput.addEventListener('change', handleFileSelect);
            }

            const fileUploadArea = document.getElementById('fileUploadArea');
            if (fileUploadArea) {
                fileUploadArea.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    fileUploadArea.classList.add('drag-over');
                });
                
                fileUploadArea.addEventListener('dragleave', () => {
                    fileUploadArea.classList.remove('drag-over');
                });
                
                fileUploadArea.addEventListener('drop', (e) => {
                    e.preventDefault();
                    fileUploadArea.classList.remove('drag-over');
                    handleFiles(e.dataTransfer.files);
                });
            }
        });

        async function sendMessage() {
            const messageInput = document.getElementById('messageInput');
            const message = messageInput.value.trim();
            
            if (!message && attachedFiles.length === 0) {
                return;
            }
            
            currentPrompt = message;
            messageInput.value = '';
            autoResizeTextarea();
            document.getElementById('sendBtn').disabled = true;
            
            if (message) {
                addMessageToUI('user', message, true);
            }
            
            document.getElementById('typingIndicator').classList.add('active');
            
            try {
                const enhanceResponse = await fetch('/enhance_prompt', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt: message,
                        session_id: sessionId
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

        function showPromptModal(original, enhanced) {
            document.getElementById('originalPromptText').textContent = original;
            document.getElementById('enhancedPromptText').textContent = enhanced;
            document.getElementById('promptModal').classList.add('active');
            document.getElementById('typingIndicator').classList.remove('active');
            document.getElementById('sendBtn').disabled = false;
        }

        function selectPromptOption(type) {
            selectedPromptType = type;
            document.getElementById('originalOption').classList.toggle('selected', type === 'original');
            document.getElementById('enhancedOption').classList.toggle('selected', type === 'enhanced');
        }

        function closePromptModal() {
            document.getElementById('promptModal').classList.remove('active');
            document.getElementById('sendBtn').disabled = false;
        }

        async function confirmPromptSelection() {
            closePromptModal();
            const promptToUse = selectedPromptType === 'original' ? currentPrompt : enhancedPrompt;
            await processMessage(promptToUse);
        }

        async function processMessage(prompt) {
            document.getElementById('typingIndicator').classList.add('active');
            
            const formData = new FormData();
            formData.append('message', prompt);
            formData.append('session_id', sessionId);
            
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
                    addMessageToUI('assistant', data.response, true);
                } else {
                    showNotification(data.error || 'Failed to get response', 'error');
                }
            } catch (error) {
                console.error('Error:', error);
                showNotification('Network error. Please try again.', 'error');
            } finally {
                document.getElementById('typingIndicator').classList.remove('active');
                document.getElementById('sendBtn').disabled = false;
                clearFiles();
            }
        }

        function addMessageToUI(role, content, save = false) {
            const container = document.getElementById('messagesContainer');
            const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            messageDiv.innerHTML = `
                <div class="message-avatar">
                    <i class="fas fa-${role === 'user' ? 'user' : 'robot'}"></i>
                </div>
                <div class="message-content">
                    <div class="message-meta">
                        <span class="message-author">${role === 'user' ? 'You' : 'Jack AI'}</span>
                        <span class="message-time">${time}</span>
                    </div>
                    <div class="message-bubble">${content.replace(/\n/g, '<br>')}</div>
                    <div class="message-actions">
                        <button class="message-action" onclick="copyMessage(this)">
                            <i class="fas fa-copy"></i> Copy
                        </button>
                    </div>
                </div>
            `;
            
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
            
            if (save) {
                chatHistory.push({ role, content });
                localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
            }
        }

        function copyMessage(button) {
            const content = button.closest('.message-content').querySelector('.message-bubble').textContent;
            navigator.clipboard.writeText(content);
            showNotification('Message copied to clipboard', 'success');
        }

        function showNotification(message, type = 'success') {
            const notification = document.createElement('div');
            notification.className = `notification ${type}`;
            notification.innerHTML = `
                <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
                <span>${message}</span>
                <i class="fas fa-times notification-close" onclick="this.parentElement.remove()"></i>
            `;
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.remove();
            }, 5000);
        }

        function toggleFileUpload() {
            const fileArea = document.getElementById('fileUploadArea');
            if (fileArea.classList.contains('active')) {
                fileArea.classList.remove('active');
            } else {
                fileArea.classList.add('active');
                document.getElementById('fileInput').click();
            }
        }

        function handleFileSelect(event) {
            handleFiles(event.target.files);
        }

        function handleFiles(files) {
            const fileList = document.getElementById('fileList');
            
            for (let file of files) {
                if (file.size > 100 * 1024 * 1024) {
                    showNotification(`File ${file.name} is too large. Max size is 100MB.`, 'error');
                    continue;
                }
                
                attachedFiles.push(file);
                
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.innerHTML = `
                    <i class="fas fa-file"></i>
                    <span>${file.name}</span>
                    <i class="fas fa-times file-remove" onclick="removeFile('${file.name}')"></i>
                `;
                
                fileList.appendChild(fileItem);
            }
        }

        function removeFile(fileName) {
            attachedFiles = attachedFiles.filter(f => f.name !== fileName);
            
            const fileList = document.getElementById('fileList');
            const items = fileList.querySelectorAll('.file-item');
            items.forEach(item => {
                if (item.textContent.includes(fileName)) {
                    item.remove();
                }
            });
            
            if (attachedFiles.length === 0) {
                document.getElementById('fileUploadArea').classList.remove('active');
            }
        }

        function clearFiles() {
            attachedFiles = [];
            document.getElementById('fileList').innerHTML = '';
            document.getElementById('fileUploadArea').classList.remove('active');
        }

        function newChat() {
            if (confirm('Start a new chat? Current conversation will be saved.')) {
                chatHistory = [];
                localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
                
                const container = document.getElementById('messagesContainer');
                container.innerHTML = '';
                
                addMessageToUI('assistant', 'New chat started! How can I help you today?', true);
                showNotification('New chat created', 'success');
            }
        }

        function clearChat() {
            if (confirm('Clear all messages? This cannot be undone.')) {
                chatHistory = [];
                localStorage.removeItem('chatHistory');
                
                const container = document.getElementById('messagesContainer');
                container.innerHTML = '';
                
                addMessageToUI('assistant', 'Chat cleared! How can I help you today?', false);
                showNotification('Chat cleared', 'success');
            }
        }

        function exportChat() {
            const messages = document.querySelectorAll('.message');
            let exportText = 'Jack AI Beta Chat Export\n';
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
            a.download = `jack-ai-chat-${new Date().toISOString()}.txt`;
            a.click();
            
            showNotification('Chat exported successfully', 'success');
        }

        function insertPrompt(text) {
            const input = document.getElementById('messageInput');
            input.value = text + ' ';
            input.focus();
            autoResizeTextarea();
        }

        function toggleTheme() {
            showNotification('Theme customization coming soon!', 'success');
        }

        function openSettings() {
            showNotification('Settings panel coming soon!', 'success');
        }
    </script>
</body>
</html>
"""

API_KEYS = []
current_key_index = 0

def get_api_keys():
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
    if api_key in API_KEYS_STATUS:
        API_KEYS_STATUS[api_key]['failures'] += 1

PROMPT_ENHANCER_SYSTEM = """You are a prompt enhancement specialist. Your job is to take user prompts and make them clearer, more detailed, and more effective for an AI assistant.

Rules:
1. Preserve the user's original intent completely
2. Add clarity and context where helpful
3. Structure the prompt for better AI understanding
4. Include specific details that will help get a better response
5. Make the prompt comprehensive but not overly long
6. If the prompt involves analysis of files or images, specify what kind of analysis would be most helpful

Take the user's prompt and rewrite it to be more effective. Return ONLY the enhanced prompt, nothing else."""

MAIN_AI_SYSTEM = """You are Jack AI Beta, an ultra-advanced artificial intelligence assistant. You are incredibly capable, intelligent, and helpful.

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
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

def count_tokens(text):
    return len(text) // 4

def process_file_for_ai(file):
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

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/enhance_prompt', methods=['POST'])
def enhance_prompt():
    try:
        data = request.json
        original_prompt = data.get('prompt', '')
        session_id = data.get('session_id', 'default')
        
        if not original_prompt:
            return jsonify({'success': False, 'error': 'No prompt provided'}), 400
        
        api_key = get_next_api_key()
        client = create_ai_client(api_key)
        
        try:
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
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
def chat():
    try:
        message = request.form.get('message', '')
        session_id = request.form.get('session_id', 'default')
        files = request.files.getlist('files')
        
        if session_id not in CHAT_SESSIONS:
            CHAT_SESSIONS[session_id] = {
                'history': [],
                'token_usage': 0
            }
        
        session = CHAT_SESSIONS[session_id]
        
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
        
        messages = [
            {"role": "system", "content": MAIN_AI_SYSTEM}
        ]
        
        for msg in session['history'][-10:]:
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
                    max_tokens=8000,
                    temperature=0.8
                )
                
                ai_response = response.choices[0].message.content
                
                session['history'].append({"role": "user", "content": message})
                session['history'].append({"role": "assistant", "content": ai_response})
                
                token_usage = session['token_usage']
                token_usage += count_tokens(full_message) + count_tokens(ai_response)
                session['token_usage'] = token_usage
                
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

@app.route('/compact_chat', methods=['POST'])
def compact_chat():
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        if session_id not in CHAT_SESSIONS:
            return jsonify({
                'success': False,
                'error': 'No chat history found'
            }), 400
        
        session = CHAT_SESSIONS[session_id]
        chat_history = session['history']
        
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
                model="gemini-2.5-flash",
                messages=[
                    {"role": "system", "content": CHAT_COMPACTOR_SYSTEM},
                    {"role": "user", "content": f"Please summarize this conversation:\n\n{conversation_text}"}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            summary = response.choices[0].message.content
            
            session['history'] = [
                {"role": "assistant", "content": summary}
            ]
            
            token_usage = count_tokens(summary)
            session['token_usage'] = token_usage
            
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

get_api_keys()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)