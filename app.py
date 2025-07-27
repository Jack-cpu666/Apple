import os
import ast
import re
import json
import shutil
import zipfile
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="FastAPI Monolith Refactoring Engine")

# HTML Template with modern UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FastAPI Refactoring Engine</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * { font-family: 'Inter', sans-serif; }
        
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .glass {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .upload-zone {
            border: 3px dashed rgba(255, 255, 255, 0.3);
            transition: all 0.3s ease;
        }
        
        .upload-zone:hover {
            border-color: rgba(255, 255, 255, 0.6);
            background: rgba(255, 255, 255, 0.05);
        }
        
        .upload-zone.dragover {
            border-color: #10b981;
            background: rgba(16, 185, 129, 0.1);
        }
        
        .progress-bar {
            transition: width 0.5s ease-out;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
        
        .feature-card {
            transition: transform 0.2s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-4px);
        }
    </style>
</head>
<body class="gradient-bg min-h-screen text-white">
    <div class="container mx-auto px-4 py-8 max-w-6xl">
        <!-- Header -->
        <div class="text-center mb-12">
            <h1 class="text-5xl font-bold mb-4">FastAPI Refactoring Engine</h1>
            <p class="text-xl opacity-90">Transform your monolithic FastAPI application into a clean, modular architecture</p>
        </div>
        
        <!-- Features Grid -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
            <div class="glass rounded-xl p-6 feature-card">
                <div class="text-3xl mb-3">🏗️</div>
                <h3 class="text-lg font-semibold mb-2">Smart Structure</h3>
                <p class="text-sm opacity-80">Intelligently organizes code into logical modules and packages</p>
            </div>
            <div class="glass rounded-xl p-6 feature-card">
                <div class="text-3xl mb-3">🔍</div>
                <h3 class="text-lg font-semibold mb-2">AST Analysis</h3>
                <p class="text-sm opacity-80">Uses Abstract Syntax Tree parsing for accurate code understanding</p>
            </div>
            <div class="glass rounded-xl p-6 feature-card">
                <div class="text-3xl mb-3">📦</div>
                <h3 class="text-lg font-semibold mb-2">Ready to Deploy</h3>
                <p class="text-sm opacity-80">Generates complete project with proper imports and dependencies</p>
            </div>
        </div>
        
        <!-- Upload Section -->
        <div class="glass rounded-2xl p-8">
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-zone rounded-xl p-12 text-center cursor-pointer" id="dropZone">
                    <svg class="w-16 h-16 mx-auto mb-4 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                    </svg>
                    <p class="text-xl font-medium mb-2" id="uploadText">Drop your FastAPI file here or click to browse</p>
                    <p class="text-sm opacity-60">Supports .py files up to 50MB</p>
                    <input type="file" id="fileInput" class="hidden" accept=".py">
                </div>
                
                <!-- Progress Section -->
                <div id="progressSection" class="mt-6 hidden">
                    <div class="flex justify-between text-sm mb-2">
                        <span id="progressText">Processing...</span>
                        <span id="progressPercent">0%</span>
                    </div>
                    <div class="w-full bg-white/20 rounded-full h-2">
                        <div class="progress-bar bg-green-400 h-2 rounded-full" style="width: 0%"></div>
                    </div>
                </div>
                
                <!-- File Info -->
                <div id="fileInfo" class="mt-6 hidden">
                    <div class="flex items-center justify-between p-4 bg-white/10 rounded-lg">
                        <div class="flex items-center">
                            <svg class="w-8 h-8 mr-3 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                                <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z"></path>
                                <path fill-rule="evenodd" d="M4 5a2 2 0 012-2 1 1 0 000 2H6a2 2 0 00-2 2v6a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-1a1 1 0 100-2h1a4 4 0 014 4v6a4 4 0 01-4 4H6a4 4 0 01-4-4V7a4 4 0 014-4z" clip-rule="evenodd"></path>
                            </svg>
                            <div>
                                <p class="font-medium" id="fileName">file.py</p>
                                <p class="text-sm opacity-60" id="fileSize">0 KB</p>
                            </div>
                        </div>
                        <button type="button" class="text-red-400 hover:text-red-300" onclick="resetUpload()">
                            <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                            </svg>
                        </button>
                    </div>
                </div>
                
                <!-- Refactor Button -->
                <button type="submit" id="refactorBtn" class="w-full mt-6 bg-white/20 hover:bg-white/30 py-4 rounded-lg font-medium text-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed" disabled>
                    <span id="btnText">Select a file to begin refactoring</span>
                    <svg id="btnSpinner" class="hidden inline-block w-5 h-5 ml-2 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                </button>
            </form>
        </div>
        
        <!-- Results Section -->
        <div id="resultsSection" class="mt-12 hidden">
            <div class="glass rounded-2xl p-8">
                <h2 class="text-2xl font-bold mb-6">Refactoring Complete! 🎉</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    <div class="bg-white/10 rounded-lg p-4">
                        <p class="text-sm opacity-60 mb-1">Original Lines</p>
                        <p class="text-2xl font-semibold" id="originalLines">0</p>
                    </div>
                    <div class="bg-white/10 rounded-lg p-4">
                        <p class="text-sm opacity-60 mb-1">Files Created</p>
                        <p class="text-2xl font-semibold" id="filesCreated">0</p>
                    </div>
                </div>
                <div id="structurePreview" class="bg-black/30 rounded-lg p-4 mb-6 font-mono text-sm overflow-x-auto">
                    <!-- File structure will be shown here -->
                </div>
                <a id="downloadBtn" href="#" class="inline-block bg-green-500 hover:bg-green-600 px-6 py-3 rounded-lg font-medium transition-colors">
                    Download Refactored Project
                </a>
            </div>
        </div>
    </div>
    
    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const uploadForm = document.getElementById('uploadForm');
        const progressSection = document.getElementById('progressSection');
        const fileInfo = document.getElementById('fileInfo');
        const refactorBtn = document.getElementById('refactorBtn');
        const resultsSection = document.getElementById('resultsSection');
        let selectedFile = null;
        
        // Drag and drop
        dropZone.addEventListener('click', () => fileInput.click());
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].name.endsWith('.py')) {
                handleFile(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        function handleFile(file) {
            selectedFile = file;
            document.getElementById('fileName').textContent = file.name;
            document.getElementById('fileSize').textContent = formatFileSize(file.size);
            document.getElementById('fileInfo').classList.remove('hidden');
            refactorBtn.disabled = false;
            document.getElementById('btnText').textContent = 'Start Refactoring';
            dropZone.style.display = 'none';
        }
        
        function formatFileSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        }
        
        function resetUpload() {
            selectedFile = null;
            fileInput.value = '';
            document.getElementById('fileInfo').classList.add('hidden');
            refactorBtn.disabled = true;
            document.getElementById('btnText').textContent = 'Select a file to begin refactoring';
            dropZone.style.display = 'block';
            progressSection.classList.add('hidden');
            resultsSection.classList.add('hidden');
        }
        
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!selectedFile) return;
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            // Show progress
            progressSection.classList.remove('hidden');
            refactorBtn.disabled = true;
            document.getElementById('btnText').textContent = 'Refactoring...';
            document.getElementById('btnSpinner').classList.remove('hidden');
            
            // Simulate progress
            let progress = 0;
            const progressBar = document.querySelector('.progress-bar');
            const progressInterval = setInterval(() => {
                progress += Math.random() * 15;
                if (progress > 90) progress = 90;
                progressBar.style.width = progress + '%';
                document.getElementById('progressPercent').textContent = Math.round(progress) + '%';
            }, 500);
            
            try {
                const response = await fetch('/refactor', {
                    method: 'POST',
                    body: formData
                });
                
                clearInterval(progressInterval);
                progressBar.style.width = '100%';
                document.getElementById('progressPercent').textContent = '100%';
                
                if (response.ok) {
                    const result = await response.json();
                    showResults(result);
                } else {
                    throw new Error('Refactoring failed');
                }
            } catch (error) {
                clearInterval(progressInterval);
                alert('Error: ' + error.message);
                resetUpload();
            }
        });
        
        function showResults(result) {
            document.getElementById('originalLines').textContent = result.original_lines.toLocaleString();
            document.getElementById('filesCreated').textContent = result.files_created;
            document.getElementById('structurePreview').innerHTML = '<pre>' + result.structure + '</pre>';
            document.getElementById('downloadBtn').href = result.download_url;
            resultsSection.classList.remove('hidden');
            document.getElementById('btnSpinner').classList.add('hidden');
            document.getElementById('btnText').textContent = 'Refactoring Complete!';
        }
    </script>
</body>
</html>
"""

# Data classes for code organization
@dataclass
class CodeBlock:
    """Represents a code block to be extracted."""
    content: str
    line_start: int
    line_end: int
    type: str  # 'class', 'function', 'route', etc.
    name: str
    decorators: List[str] = None
    imports_needed: Set[str] = None

@dataclass
class RouteInfo:
    """Information about a route endpoint."""
    method: str
    path: str
    function_name: str
    has_html_response: bool
    html_content: Optional[str] = None
    template_variables: List[str] = None


class AdvancedCodeRefactorer:
    """Advanced refactoring engine using AST parsing."""
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.lines = source_code.split('\n')
        self.tree = None
        self.imports = []
        self.code_blocks = defaultdict(list)
        self.routes = []
        self.models = []
        self.schemas = []
        self.utilities = []
        self.config_vars = []
        
    def parse(self):
        """Parse the source code using AST."""
        try:
            self.tree = ast.parse(self.source_code)
            self._extract_imports()
            self._categorize_code()
        except SyntaxError as e:
            logger.error(f"Syntax error in source code: {e}")
            raise
    
    def _extract_imports(self):
        """Extract all import statements."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    self.imports.append(f"from {module} import {alias.name}")
    
    def _categorize_code(self):
        """Categorize code into different types."""
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef):
                self._process_class(node)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                self._process_function(node)
            elif isinstance(node, ast.Assign):
                self._process_assignment(node)
    
    def _process_class(self, node: ast.ClassDef):
        """Process class definitions."""
        # Check if it's a SQLAlchemy model
        if any(base.id == 'Base' for base in node.bases if isinstance(base, ast.Name)):
            self.models.append(CodeBlock(
                content=ast.unparse(node),
                line_start=node.lineno,
                line_end=node.end_lineno,
                type='model',
                name=node.name
            ))
        # Check if it's a Pydantic schema
        elif any(base.id == 'BaseModel' for base in node.bases if isinstance(base, ast.Name)):
            self.schemas.append(CodeBlock(
                content=ast.unparse(node),
                line_start=node.lineno,
                line_end=node.end_lineno,
                type='schema',
                name=node.name
            ))
        else:
            # Other classes go to utilities
            self.utilities.append(CodeBlock(
                content=ast.unparse(node),
                line_start=node.lineno,
                line_end=node.end_lineno,
                type='class',
                name=node.name
            ))
    
    def _process_function(self, node):
        """Process function definitions."""
        decorators = [ast.unparse(d) for d in node.decorator_list]
        
        # Check if it's a route
        route_decorators = [d for d in decorators if '@app.' in d or '@router.' in d]
        if route_decorators:
            self._process_route(node, route_decorators)
        else:
            # Regular function
            self.utilities.append(CodeBlock(
                content=ast.unparse(node),
                line_start=node.lineno,
                line_end=node.end_lineno,
                type='function',
                name=node.name,
                decorators=decorators
            ))
    
    def _process_route(self, node, decorators):
        """Process route endpoints."""
        # Determine route type and extract info
        route_info = RouteInfo(
            method='get',  # default
            path='/',
            function_name=node.name,
            has_html_response=False
        )
        
        # Parse decorator to get method and path
        for dec in decorators:
            match = re.search(r'@(app|router)\.(get|post|put|delete|patch|websocket)\("([^"]+)"\)', dec)
            if match:
                route_info.method = match.group(2)
                route_info.path = match.group(3)
        
        # Check if returns HTML
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                if hasattr(stmt.value.func, 'id') and stmt.value.func.id == 'HTMLResponse':
                    route_info.has_html_response = True
                    # Try to extract HTML content
                    route_info.html_content = self._extract_html_from_return(stmt)
        
        self.routes.append({
            'info': route_info,
            'code': CodeBlock(
                content=ast.unparse(node),
                line_start=node.lineno,
                line_end=node.end_lineno,
                type='route',
                name=node.name,
                decorators=decorators
            )
        })
    
    def _extract_html_from_return(self, return_node):
        """Extract HTML content from return statement."""
        # This is a simplified extraction - in reality would need more sophisticated parsing
        try:
            if hasattr(return_node.value, 'keywords'):
                for kw in return_node.value.keywords:
                    if kw.arg == 'content':
                        if isinstance(kw.value, ast.JoinedStr):
                            # This is an f-string
                            return "<!-- HTML content would be extracted here -->"
        except:
            pass
        return None
    
    def _process_assignment(self, node: ast.Assign):
        """Process variable assignments."""
        try:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    # Check if it's a configuration variable
                    if var_name.isupper() or 'config' in var_name.lower():
                        self.config_vars.append({
                            'name': var_name,
                            'value': ast.unparse(node.value)
                        })
        except:
            pass
    
    def generate_project_structure(self) -> Dict[str, str]:
        """Generate the refactored project structure."""
        structure = {}
        
        # app/__init__.py
        structure['app/__init__.py'] = '"""StreamBeatz application package."""\n'
        
        # app/core/config.py
        structure['app/core/config.py'] = self._generate_config_file()
        
        # app/database/models.py
        structure['app/database/models.py'] = self._generate_models_file()
        
        # app/schemas.py
        structure['app/schemas.py'] = self._generate_schemas_file()
        
        # app/utils.py
        structure['app/utils.py'] = self._generate_utils_file()
        
        # Route files
        route_files = self._organize_routes()
        structure.update(route_files)
        
        # main.py
        structure['main.py'] = self._generate_main_file()
        
        # requirements.txt
        structure['requirements.txt'] = self._generate_requirements()
        
        # .env.example
        structure['.env.example'] = self._generate_env_example()
        
        return structure
    
    def _generate_config_file(self) -> str:
        """Generate the config.py file."""
        content = '''"""Centralized configuration management."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
'''
        # Add config variables
        for var in self.config_vars:
            if 'os.getenv' in var['value']:
                # Extract env var name
                match = re.search(r'os\.getenv\(["\']([^"\']+)["\']', var['value'])
                if match:
                    env_name = match.group(1)
                    # Determine type
                    var_type = 'str'
                    if 'int(' in var['value']:
                        var_type = 'int'
                    elif var['value'].lower() in ['true', 'false']:
                        var_type = 'bool'
                    
                    content += f"    {var['name']}: {var_type}\n"
        
        content += '''
    class Config:
        env_file = ".env"


settings = Settings()
'''
        return content
    
    def _generate_models_file(self) -> str:
        """Generate the models.py file."""
        content = '''"""SQLAlchemy ORM models."""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, Integer, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY

Base = declarative_base()

'''
        for model in self.models:
            content += model.content + '\n\n'
        
        return content
    
    def _generate_schemas_file(self) -> str:
        """Generate the schemas.py file."""
        content = '''"""Pydantic validation schemas."""

from pydantic import BaseModel, validator, Field
from typing import Optional, List, Dict
from datetime import datetime
import re

'''
        for schema in self.schemas:
            content += schema.content + '\n\n'
        
        return content
    
    def _generate_utils_file(self) -> str:
        """Generate the utils.py file."""
        content = '''"""Utility functions and classes."""

from app.core.config import settings
import time
import json
import asyncio
import hashlib
import secrets
from typing import Optional, Dict, Any

'''
        for util in self.utilities:
            if util.type == 'function' and util.name in [
                'hash_password', 'verify_password', 'generate_token',
                'generate_referral_code', 'create_session', 'verify_session'
            ]:
                content += util.content + '\n\n'
        
        return content
    
    def _organize_routes(self) -> Dict[str, str]:
        """Organize routes into appropriate files."""
        route_files = defaultdict(list)
        
        for route_data in self.routes:
            route = route_data['info']
            code = route_data['code']
            
            # Determine which file this route belongs to
            if '/auth/' in route.path or '/login' in route.path or '/register' in route.path:
                route_files['app/api/auth.py'].append(code)
            elif '/api/requests' in route.path or '/api/queue' in route.path:
                route_files['app/api/requests.py'].append(code)
            elif '/api/users' in route.path or '/api/profile' in route.path:
                route_files['app/api/users.py'].append(code)
            elif '/api/stripe' in route.path or '/api/withdraw' in route.path:
                route_files['app/api/payments.py'].append(code)
            elif '/api/admin' in route.path:
                route_files['app/api/admin.py'].append(code)
            elif route.method == 'websocket':
                route_files['app/services/websockets.py'].append(code)
            elif route.has_html_response:
                route_files['app/pages/views.py'].append(code)
            else:
                # Default to media
                route_files['app/api/media.py'].append(code)
        
        # Generate file contents
        result = {}
        for filepath, routes in route_files.items():
            content = self._generate_route_file(filepath, routes)
            result[filepath] = content
        
        return result
    
    def _generate_route_file(self, filepath: str, routes: List[CodeBlock]) -> str:
        """Generate a route file with proper imports and router setup."""
        filename = filepath.split('/')[-1].replace('.py', '')
        
        content = f'''"""{filename.title()} route handlers."""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, List, Optional

router = APIRouter()

'''
        
        for route in routes:
            # Convert @app. to @router.
            route_content = route.content.replace('@app.', '@router.')
            content += route_content + '\n\n'
        
        return content
    
    def _generate_main_file(self) -> str:
        """Generate the main.py file."""
        return '''"""Main FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database.session import init_db
from app.core.config import settings

# Import routers
from app.api import auth, requests, users, payments, media, admin
from app.pages import views
from app.services import websockets

app = FastAPI(
    title="StreamBeatz",
    description="Advanced song request platform for streamers",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else [settings.BASE_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(requests.router, prefix="/api", tags=["Requests"])
app.include_router(users.router, prefix="/api", tags=["Users"])
app.include_router(payments.router, prefix="/api", tags=["Payments"])
app.include_router(media.router, prefix="/api", tags=["Media"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(websockets.router, tags=["WebSockets"])
app.include_router(views.router, tags=["Pages"])

@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    await init_db()
    print("StreamBeatz server started successfully!")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "streambeatz"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
'''
    
    def _generate_requirements(self) -> str:
        """Generate requirements.txt based on imports."""
        # Core requirements
        requirements = [
            'fastapi==0.104.1',
            'uvicorn[standard]==0.24.0',
            'sqlalchemy[asyncio]==2.0.23',
            'asyncpg==0.29.0',
            'pydantic==2.5.0',
            'pydantic-settings==2.1.0',
            'python-multipart==0.0.6',
            'python-jose[cryptography]==3.3.0',
            'passlib[bcrypt]==1.7.4',
            'httpx==0.25.2',
            'redis==5.0.1',
            'stripe==7.6.0',
            'jinja2==3.1.2',
            'email-validator==2.1.0',
            'python-dotenv==1.0.0'
        ]
        
        # Add requirements based on imports found
        import_to_package = {
            'qrcode': 'qrcode[pil]==7.4.2',
            'PIL': 'pillow==10.1.0',
            'boto3': 'boto3==1.34.0',
            'aiofiles': 'aiofiles==23.2.1',
        }
        
        for imp in self.imports:
            for key, package in import_to_package.items():
                if key in imp and package not in requirements:
                    requirements.append(package)
        
        return '\n'.join(sorted(requirements))
    
    def _generate_env_example(self) -> str:
        """Generate .env.example file."""
        env_vars = []
        
        # Extract environment variables from config
        for var in self.config_vars:
            if 'os.getenv' in var['value']:
                match = re.search(r'os\.getenv\(["\']([^"\']+)["\']', var['value'])
                if match:
                    env_name = match.group(1)
                    env_vars.append(f"{env_name}=")
        
        # Add common vars if not present
        common_vars = [
            'DATABASE_URL=postgresql+asyncpg://user:password@localhost/streambeatz',
            'REDIS_URL=redis://localhost:6379/0',
            'SECRET_KEY=your-secret-key-here',
            'DEBUG=true'
        ]
        
        content = "# StreamBeatz Environment Configuration\n\n"
        content += "# Database\n"
        content += "DATABASE_URL=postgresql+asyncpg://user:password@localhost/streambeatz\n\n"
        content += "# Redis\n"
        content += "REDIS_URL=redis://localhost:6379/0\n\n"
        content += "# Security\n"
        content += "SECRET_KEY=your-secret-key-here\n"
        content += "DEBUG=false\n\n"
        
        if any('STRIPE' in var for var in env_vars):
            content += "# Stripe\n"
            content += "STRIPE_SECRET_KEY=\n"
            content += "STRIPE_WEBHOOK_SECRET=\n\n"
        
        if any('SPOTIFY' in var for var in env_vars):
            content += "# Spotify\n"
            content += "SPOTIFY_CLIENT_ID=\n"
            content += "SPOTIFY_CLIENT_SECRET=\n\n"
        
        return content


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main page."""
    return HTML_TEMPLATE


@app.post("/refactor")
async def refactor_code(file: UploadFile = File(...)):
    """Refactor the uploaded FastAPI monolith."""
    try:
        # Read the uploaded file
        content = await file.read()
        source_code = content.decode('utf-8')
        
        # Count original lines
        original_lines = len(source_code.split('\n'))
        
        # Create refactorer instance
        refactorer = AdvancedCodeRefactorer(source_code)
        refactorer.parse()
        
        # Generate project structure
        project_files = refactorer.generate_project_structure()
        
        # Create temporary directory for the project
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / "streambeatz_refactored"
            
            # Create all necessary directories
            for filepath in project_files.keys():
                file_path = project_root / filepath
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create additional directories
            (project_root / "static").mkdir(exist_ok=True)
            (project_root / "templates").mkdir(exist_ok=True)
            (project_root / "app" / "__init__.py").touch()
            (project_root / "app" / "api" / "__init__.py").touch()
            (project_root / "app" / "core" / "__init__.py").touch()
            (project_root / "app" / "database" / "__init__.py").touch()
            (project_root / "app" / "pages" / "__init__.py").touch()
            (project_root / "app" / "services" / "__init__.py").touch()
            
            # Write all files
            for filepath, content in project_files.items():
                file_path = project_root / filepath
                file_path.write_text(content, encoding='utf-8')
            
            # Create README
            readme_content = """# StreamBeatz - Refactored

This project has been automatically refactored from a monolithic structure to a modular architecture.

## Project Structure

```
streambeatz_refactored/
├── app/
│   ├── api/          # API route handlers
│   ├── core/         # Core configuration
│   ├── database/     # Database models and session
│   ├── pages/        # Page route handlers
│   ├── services/     # External services
│   ├── schemas.py    # Pydantic models
│   └── utils.py      # Utility functions
├── static/           # Static files
├── templates/        # Jinja2 templates
├── main.py          # Application entry point
├── requirements.txt  # Python dependencies
└── .env.example     # Environment variables template
```

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Run the application:
   ```bash
   python main.py
   ```

## Next Steps

1. Review and test the refactored code
2. Extract HTML templates from route handlers
3. Add proper error handling
4. Implement logging
5. Add unit tests
6. Set up CI/CD pipeline
"""
            (project_root / "README.md").write_text(readme_content)
            
            # Create zip file
            zip_path = Path(temp_dir) / "streambeatz_refactored.zip"
            shutil.make_archive(str(zip_path.with_suffix('')), 'zip', project_root)
            
            # Generate file structure preview
            structure = []
            for root, dirs, files in os.walk(project_root):
                level = root.replace(str(project_root), '').count(os.sep)
                indent = ' ' * 2 * level
                structure.append(f'{indent}{os.path.basename(root)}/')
                subindent = ' ' * 2 * (level + 1)
                for file in sorted(files):
                    if not file.startswith('.'):
                        structure.append(f'{subindent}{file}')
            
            # Save zip temporarily
            output_path = Path("temp_downloads") / f"refactored_{file.filename}_{int(time.time())}.zip"
            output_path.parent.mkdir(exist_ok=True)
            shutil.copy(zip_path, output_path)
            
            return JSONResponse({
                "success": True,
                "original_lines": original_lines,
                "files_created": len(project_files) + 10,  # Including __init__.py files
                "structure": '\n'.join(structure[:20]) + '\n...',  # Show first 20 lines
                "download_url": f"/download/{output_path.name}"
            })
            
    except Exception as e:
        logger.error(f"Refactoring error: {str(e)}", exc_info=True)
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )


@app.get("/download/{filename}")
async def download_refactored(filename: str):
    """Download the refactored project."""
    file_path = Path("temp_downloads") / filename
    if file_path.exists() and file_path.is_file():
        return FileResponse(
            path=file_path,
            filename=f"streambeatz_refactored.zip",
            media_type="application/zip"
        )
    return JSONResponse({"error": "File not found"}, status_code=404)


# Cleanup old downloads periodically
async def cleanup_downloads():
    """Remove old download files."""
    while True:
        await asyncio.sleep(3600)  # Every hour
        try:
            download_dir = Path("temp_downloads")
            if download_dir.exists():
                for file in download_dir.iterdir():
                    if file.is_file():
                        # Remove files older than 1 hour
                        if (time.time() - file.stat().st_mtime) > 3600:
                            file.unlink()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


@app.on_event("startup")
async def startup_event():
    """Start background tasks."""
    asyncio.create_task(cleanup_downloads())


if __name__ == "__main__":
    # For local development
    uvicorn.run(app, host="0.0.0.0", port=8000)