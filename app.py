import os
import ast
import re
import shutil
import zipfile
import tempfile
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Helper to unparse AST nodes back to code, which is standard in Python 3.9+
unparse = ast.unparse

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
        .gradient-bg { background: linear-gradient(135deg, #1e3a8a 0%, #4c1d95 100%); }
        .glass { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); }
        .upload-zone { border: 3px dashed rgba(255, 255, 255, 0.3); transition: all 0.3s ease; }
        .upload-zone:hover { border-color: rgba(255, 255, 255, 0.6); background: rgba(255, 255, 255, 0.05); }
        .upload-zone.dragover { border-color: #34d399; background: rgba(16, 185, 129, 0.1); }
        .progress-bar { transition: width 0.5s ease-out; }
        .feature-card { transition: transform 0.2s ease; }
        .feature-card:hover { transform: translateY(-4px); }
        pre { white-space: pre-wrap; word-wrap: break-word; }
    </style>
</head>
<body class="gradient-bg min-h-screen text-gray-200">
    <div class="container mx-auto px-4 py-8 max-w-6xl">
        <div class="text-center mb-12">
            <h1 class="text-5xl font-bold text-white mb-4">FastAPI Refactoring Engine</h1>
            <p class="text-xl text-gray-300">Transform your monolithic FastAPI application into a clean, modular architecture</p>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
            <div class="glass rounded-xl p-6 feature-card text-center"><div class="text-4xl mb-3">🏗️</div><h3 class="text-lg font-semibold text-white">Smart Structure</h3><p class="text-sm text-gray-400">Organizes code into logical modules: api, core, models, schemas.</p></div>
            <div class="glass rounded-xl p-6 feature-card text-center"><div class="text-4xl mb-3">🔍</div><h3 class="text-lg font-semibold text-white">AST Analysis</h3><p class="text-sm text-gray-400">Uses Abstract Syntax Tree parsing for accurate code understanding.</p></div>
            <div class="glass rounded-xl p-6 feature-card text-center"><div class="text-4xl mb-3">📦</div><h3 class="text-lg font-semibold text-white">Ready to Deploy</h3><p class="text-sm text-gray-400">Generates a runnable project with proper imports and dependencies.</p></div>
        </div>
        <div class="glass rounded-2xl p-8">
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-zone rounded-xl p-12 text-center cursor-pointer" id="dropZone">
                    <svg class="w-16 h-16 mx-auto mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                    <p class="text-xl font-medium text-white mb-2" id="uploadText">Drop your FastAPI file here or click to browse</p>
                    <p class="text-sm text-gray-400">Supports .py files up to 50MB</p>
                    <input type="file" id="fileInput" class="hidden" accept=".py">
                </div>
                <div id="fileInfo" class="mt-6 hidden"><div class="flex items-center justify-between p-4 bg-gray-900/50 rounded-lg"><div class="flex items-center"><svg class="w-8 h-8 mr-3 text-green-400" fill="currentColor" viewBox="0 0 20 20"><path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z"></path><path fill-rule="evenodd" d="M4 5a2 2 0 012-2 1 1 0 000 2H6a2 2 0 00-2 2v6a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-1a1 1 0 100-2h1a4 4 0 014 4v6a4 4 0 01-4 4H6a4 4 0 01-4-4V7a4 4 0 014-4z" clip-rule="evenodd"></path></svg><div><p class="font-medium text-white" id="fileName">file.py</p><p class="text-sm text-gray-400" id="fileSize">0 KB</p></div></div><button type="button" class="text-red-400 hover:text-red-300" onclick="resetUpload()">×</button></div></div>
                <button type="submit" id="refactorBtn" class="w-full mt-6 bg-indigo-600 hover:bg-indigo-700 py-4 rounded-lg font-medium text-lg text-white transition-all disabled:opacity-50 disabled:cursor-not-allowed" disabled><span id="btnText">Select a file to begin</span><svg id="btnSpinner" class="hidden inline-block w-5 h-5 ml-2 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg></button>
            </form>
        </div>
        <div id="resultsSection" class="mt-12 hidden"><div class="glass rounded-2xl p-8"><h2 class="text-3xl font-bold mb-6 text-white">Refactoring Complete! 🎉</h2><div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6"><div class="bg-gray-900/50 rounded-lg p-4"><p class="text-sm text-gray-400 mb-1">Original Lines</p><p class="text-3xl font-semibold text-white" id="originalLines">0</p></div><div class="bg-gray-900/50 rounded-lg p-4"><p class="text-sm text-gray-400 mb-1">Files Created</p><p class="text-3xl font-semibold text-white" id="filesCreated">0</p></div></div><h3 class="text-xl font-semibold mb-4 text-white">Generated Project Structure:</h3><div id="structurePreview" class="bg-black/50 rounded-lg p-4 mb-6 font-mono text-sm overflow-x-auto"></div><a id="downloadBtn" href="#" class="inline-block w-full text-center bg-green-500 hover:bg-green-600 px-6 py-3 rounded-lg font-medium text-white transition-colors">Download Refactored Project (.zip)</a></div></div>
    </div>
    <script>
        const dropZone = document.getElementById('dropZone'), fileInput = document.getElementById('fileInput'), uploadForm = document.getElementById('uploadForm'), fileInfo = document.getElementById('fileInfo'), refactorBtn = document.getElementById('refactorBtn'), resultsSection = document.getElementById('resultsSection');
        let selectedFile = null;
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); if (e.dataTransfer.files.length > 0 && e.dataTransfer.files[0].name.endsWith('.py')) handleFile(e.dataTransfer.files[0]); });
        fileInput.addEventListener('change', (e) => { if (e.target.files.length > 0) handleFile(e.target.files[0]); });
        function handleFile(file) { selectedFile = file; document.getElementById('fileName').textContent = file.name; document.getElementById('fileSize').textContent = (file.size / 1024).toFixed(1) + ' KB'; fileInfo.classList.remove('hidden'); refactorBtn.disabled = false; document.getElementById('btnText').textContent = 'Start Refactoring'; dropZone.style.display = 'none'; }
        function resetUpload() { selectedFile = null; fileInput.value = ''; fileInfo.classList.add('hidden'); refactorBtn.disabled = true; document.getElementById('btnText').textContent = 'Select a file to begin'; dropZone.style.display = 'block'; resultsSection.classList.add('hidden'); }
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault(); if (!selectedFile) return;
            const formData = new FormData(); formData.append('file', selectedFile);
            refactorBtn.disabled = true; document.getElementById('btnText').textContent = 'Analyzing Code...'; document.getElementById('btnSpinner').classList.remove('hidden');
            try {
                const response = await fetch('/refactor', { method: 'POST', body: formData });
                document.getElementById('btnText').textContent = 'Generating Files...';
                if (response.ok) { const result = await response.json(); showResults(result); } 
                else { const err = await response.json(); throw new Error(err.detail || 'Refactoring failed on the server.'); }
            } catch (error) { alert('Error: ' + error.message); resetUpload(); }
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

class CodeRefactorer:
    def __init__(self, source_code: str, project_name: str):
        self.source_code = source_code
        self.project_name = project_name
        self.tree = ast.parse(source_code)
        self.app_name = self._find_app_name()
        self.local_modules = self._detect_local_modules()
        
        self.nodes = defaultdict(list)
        self.routes = defaultdict(list)
        self.html_templates: Dict[str, str] = {}

    def _find_app_name(self) -> str:
        for node in ast.walk(self.tree):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and
                    getattr(node.value.func, 'id', '') == 'FastAPI'):
                return node.targets[0].id
        return "app"

    def _detect_local_modules(self) -> Set[str]:
        known_packages = set(sys.stdlib_module_names) | {
            "fastapi", "uvicorn", "sqlalchemy", "pydantic", "httpx", "stripe",
            "qrcode", "itsdangerous", "passlib", "email_validator", "redis", "boto3",
            "starlette", "jinja2"
        }
        local_mods = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if '.' not in alias.name and alias.name not in known_packages:
                        local_mods.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                if '.' not in node.module and node.module not in known_packages:
                    local_mods.add(node.module)
        return local_mods

    def refactor(self) -> str:
        self._categorize_nodes()
        return self._create_zip_file()

    def _categorize_nodes(self):
        for node in self.tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)): continue
            
            if isinstance(node, ast.Assign):
                name = node.targets[0].id if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) else None
                if name and name.isupper(): self.nodes['config'].append(node)
                elif name == self.app_name: self.nodes['main_app'].append(node)
                elif name in ["engine", "AsyncSessionLocal", "Base", "redis_client"]: self.nodes['db_setup'].append(node)
                else: self.nodes['globals'].append(node)
            elif isinstance(node, ast.ClassDef):
                bases = {getattr(b, 'id', None) for b in node.bases}
                if 'Base' in bases: self.nodes['models'].append(node)
                elif 'BaseModel' in bases: self.nodes['schemas'].append(node)
                else: self.nodes['globals'].append(node)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                is_route, is_startup = False, False
                for d in node.decorator_list:
                    if isinstance(d, ast.Call) and getattr(d.func, 'value', {}).id == self.app_name:
                        attr = getattr(d.func, 'attr', '')
                        if attr in {'get', 'post', 'put', 'delete', 'websocket'}: is_route = True
                        elif attr == 'on_event' and d.args[0].value == 'startup': is_startup = True
                
                if is_startup: self.nodes['startup'].append(node)
                elif is_route: self._process_route(node)
                elif node.name.startswith(('get_', 'require_', 'check_')): self.nodes['dependencies'].append(node)
                else: self.nodes['services'].append(node)
            else:
                self.nodes['globals'].append(node)

    def _process_route(self, node):
        path_arg = node.decorator_list[0].args[0]
        path = path_arg.value if isinstance(path_arg, ast.Constant) else "/unknown"
        prefix = path.split('/')[1] if path.count('/') > 0 and path != "/" else "root"
        
        # Rewrite decorator from @app.get to @router.get
        node.decorator_list[0].func.value.id = 'router'

        # Check for and extract HTML
        if 'HTMLResponse' in unparse(node):
            html_fstring = next((n for n in ast.walk(node) if isinstance(n, ast.JoinedStr)), None)
            if html_fstring:
                html_content = "".join([s.value for s in html_fstring.values if isinstance(s, ast.Constant)])
                template_name = f"{prefix}_{node.name}.html"
                self.html_templates[template_name] = dedent(html_content)
                # Replace return statement
                new_return = ast.Return(value=ast.Call(
                    func=ast.Name(id="templates.TemplateResponse", ctx=ast.Load()),
                    args=[ast.Constant(value=template_name), ast.Dict(keys=[ast.Constant(value="request")], values=[ast.Name(id="request", ctx=ast.Load())])],
                    keywords=[]
                ))
                node.body[-1] = new_return
        
        self.routes[prefix].append(node)

    def _create_zip_file(self):
        temp_dir = tempfile.mkdtemp()
        project_root = Path(temp_dir) / self.project_name
        
        # Create directories
        app_dir = project_root / "app"
        api_dir = app_dir / "api"
        dirs = [project_root, app_dir, api_dir, app_dir / "core", app_dir / "db", 
                app_dir / "models", app_dir / "schemas", app_dir / "services", 
                app_dir / "dependencies", project_root / "templates"]
        for d in dirs: d.mkdir(parents=True, exist_ok=True)
        for d in dirs: (d / "__init__.py").touch() if d != project_root else None

        # Write files
        (app_dir / "core" / "config.py").write_text("import os\n\n" + "\n".join(unparse(n) for n in self.nodes['config']))
        (app_dir / "db" / "base.py").write_text("from sqlalchemy.orm import declarative_base\n\nBase = declarative_base()\n")
        (app_dir / "db" / "session.py").write_text(dedent("""
            from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
            from app.core.config import DATABASE_URL
            engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
            AsyncSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine)
        """))
        for model in self.nodes['models']: (app_dir / "models" / f"{model.name.lower()}.py").write_text(dedent("""
            import uuid; from datetime import datetime
            from sqlalchemy import Column, String, Boolean, Integer, Float, DateTime, ForeignKey, Text, func
            from sqlalchemy.orm import relationship
            from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
            from app.db.base import Base
            """) + f"\n\n{unparse(model)}")
        for schema in self.nodes['schemas']: (app_dir / "schemas" / f"{schema.name.lower()}.py").write_text(dedent("""
            from pydantic import BaseModel, validator, Field
            from typing import Optional, List, Dict, Any
            from datetime import datetime
            import re
            try: from email_validator import validate_email, EmailNotValidError
            except ImportError: EmailNotValidError = ValueError
            """) + f"\n\n{unparse(schema)}")
        (app_dir / "services" / "utils.py").write_text(dedent("""
            import os, time, secrets, hmac, hashlib
            from passlib.context import CryptContext
            from itsdangerous import URLSafeTimedSerializer
            from app.core.config import SESSION_SECRET
            """) + "\n\n" + "\n\n".join(unparse(n) for n in self.nodes['globals'] + self.nodes['services']))
        (app_dir / "dependencies" / "deps.py").write_text(dedent("""
            from fastapi import Depends, HTTPException, Request
            from sqlalchemy.ext.asyncio import AsyncSession
            from app.db.session import AsyncSessionLocal
            async def get_db():
                async with AsyncSessionLocal() as session: yield session
            """) + "\n\n" + "\n\n".join(unparse(n) for n in self.nodes['dependencies']))
        
        router_names = []
        for prefix, routes in self.routes.items():
            router_name = prefix.replace('/', '')
            router_names.append(router_name)
            imports = dedent("""
                import os, uuid, time, json, asyncio, hashlib
                from datetime import datetime, timedelta
                from typing import Optional, List, Dict, Any
                from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, HTTPException, Request, Response, Form, Query, BackgroundTasks, File, UploadFile
                from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse, JSONResponse
                from sqlalchemy.ext.asyncio import AsyncSession
                from app.dependencies.deps import get_db, get_current_user, require_user, require_streamer
                # TODO: Add specific model, schema, and service imports
            """)
            router_code = f"{imports}\nrouter = APIRouter(tags=['{router_name}'])" + "\n\n" + "\n\n".join(unparse(r) for r in routes)
            (api_dir / f"{router_name}.py").write_text(router_code)
            
        for name, html in self.html_templates.items(): (project_root / "templates" / name).write_text(html)
        
        # Main file
        main_imports = "\n".join(f"from app.api import {name} as {name}_router" for name in router_names)
        router_includes = "\n".join(f"app.include_router({name}_router.router)" for name in router_names)
        (app_dir / "main.py").write_text(dedent(f"""
            from fastapi import FastAPI
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.templating import Jinja2Templates
            {main_imports}
            
            app = FastAPI(title="{self.project_name.replace('_', ' ').title()}")
            templates = Jinja2Templates(directory="../templates")
            app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
            
            {router_includes}
            
            @app.on_event("startup")
            async def startup():
                # Add your startup logic here
                pass
        """))

        # Local module placeholders
        for module in self.local_modules:
            (app_dir / f"{module}.py").write_text(f"# Paste the content of {module}.py here\n")

        # requirements.txt and README
        (project_root / "requirements.txt").write_text("\n".join(sorted(["fastapi", "uvicorn[standard]", "SQLAlchemy[asyncio]", "asyncpg", "pydantic", "httpx", "stripe", "qrcode", "itsdangerous", "passlib[bcrypt]", "email-validator", "redis", "python-multipart", "Jinja2"])))
        (project_root / "README.md").write_text(dedent(f"""
            # {self.project_name}
            **IMPORTANT**: This project requires local modules: `{', '.join(self.local_modules)}`. Paste their contents into the corresponding empty files in the `app/` directory.
            
            ## Run
            1. `pip install -r requirements.txt`
            2. `uvicorn app.main:app --reload`
        """))

        zip_path_str = shutil.make_archive(project_root.name, 'zip', temp_dir, self.project_name)
        return zip_path_str

# --- FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE

@app.post("/refactor")
async def refactor_endpoint(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    try:
        source_code = (await file.read()).decode('utf-8')
        project_name = file.filename.replace('.py', '') + "_refactored"
        
        refactorer = CodeRefactorer(source_code, project_name)
        zip_path_str = refactorer.refactor()
        
        # Move zip to a downloadable location
        final_zip_name = f"{project_name}_{int(time.time())}.zip"
        downloadable_path = Path("temp_downloads") / final_zip_name
        downloadable_path.parent.mkdir(exist_ok=True)
        shutil.move(zip_path_str, downloadable_path)

        structure = []
        with zipfile.ZipFile(downloadable_path, 'r') as zip_ref:
            for name in sorted(zip_ref.namelist()):
                level = name.strip('/').count('/')
                indent = '  ' * level
                structure.append(f"{indent}{Path(name).name}")

        return JSONResponse({
            "success": True, "original_lines": len(source_code.splitlines()),
            "files_created": len(structure), "structure": "\n".join(structure[:25]),
            "download_url": f"/download/{final_zip_name}"
        })
    except Exception as e:
        logger.error("Refactoring error", exc_info=True)
        return JSONResponse({"success": False, "detail": str(e)}, status_code=500)
    finally:
        shutil.rmtree(temp_dir)

@app.get("/download/{filename}")
async def download_zip(filename: str):
    path = Path("temp_downloads") / filename
    if path.exists():
        return FileResponse(path, media_type='application/zip', filename=f"{filename.split('_')[0]}.zip")
    raise HTTPException(status_code=404, detail="File not found or expired.")

if __name__ == "__main__":
    if not os.path.exists("temp_downloads"): os.makedirs("temp_downloads")
    uvicorn.run(app, host="0.0.0.0", port=8000)