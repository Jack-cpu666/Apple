import os
import ast
import zipfile
import sys
from io import BytesIO
from textwrap import dedent
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse

# Helper to unparse AST nodes back to code, available in Python 3.9+
# For older versions, an external library like astor would be needed.
unparse = ast.unparse

# --- Main Application ---
app = FastAPI(
    title="Monolith to Structured Project Converter",
    description="Upload your monolithic FastAPI script to convert it into a structured project with placeholders for local dependencies."
)

# --- Static Content (HTML Template) ---

HOME_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StreamBeatz Code Restructurer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        body {
            font-family: 'Inter', sans-serif;
            background-color: #0a0a0a;
            color: #e5e7eb;
        }
        .container { max-width: 800px; margin: auto; padding: 2rem; }
        .card { background-color: #111827; border: 1px solid #374151; border-radius: 0.75rem; }
        .btn {
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            color: white; font-weight: 600; padding: 0.75rem 1.5rem;
            border-radius: 0.5rem; transition: all 0.3s;
            border: none; cursor: pointer;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
        .file-label {
            border: 2px dashed #4b5563; padding: 2rem; text-align: center;
            border-radius: 0.5rem; cursor: pointer; transition: all 0.3s;
        }
        .file-label:hover { border-color: #6366f1; background-color: #1f2937; }
        #spinner { display: none; }
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="text-center mb-8">
            <h1 class="text-4xl font-bold mb-2">StreamBeatz Code Restructurer</h1>
            <p class="text-gray-400">Transform your single-file application into a structured, maintainable project.</p>
        </div>

        <div class="card p-8">
            <form id="uploadForm" action="/restructure" method="post" enctype="multipart/form-data">
                <div class="mb-6">
                    <label for="project_name" class="block mb-2 text-sm font-medium text-gray-300">New Project Name</label>
                    <input type="text" id="project_name" name="project_name" value="streambeatz_project" required
                           class="w-full bg-gray-900 border border-gray-600 text-white rounded-lg p-2.5 focus:ring-indigo-500 focus:border-indigo-500">
                </div>
                <div class="mb-6">
                    <label class="file-label" for="file_upload">
                        <span id="file-text">Click to upload your monolithic Python file</span>
                        <input type="file" id="file_upload" name="file" class="hidden" accept=".py" required>
                    </label>
                </div>
                <button type="submit" class="btn w-full">
                    <span id="button-text">Restructure Code</span>
                    <div id="spinner" class="inline-block h-5 w-5 animate-spin rounded-full border-2 border-solid border-current border-r-transparent align-[-0.125em] motion-reduce:animate-[spin_1.5s_linear_infinite]"></div>
                </button>
            </form>
        </div>
        <footer class="text-center mt-8 text-gray-500 text-sm">
            <p>Your code is processed in memory and never stored on the server.</p>
        </footer>
    </div>
    <script>
        const form = document.getElementById('uploadForm');
        const fileInput = document.getElementById('file_upload');
        const fileText = document.getElementById('file-text');
        const spinner = document.getElementById('spinner');
        const buttonText = document.getElementById('button-text');

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                fileText.textContent = `File selected: \${fileInput.files[0].name}`;
            } else {
                fileText.textContent = 'Click to upload your monolithic Python file';
            }
        });

        form.addEventListener('submit', (event) => {
            if (fileInput.files.length === 0) {
                alert('Please select a file to upload.');
                event.preventDefault();
                return;
            }
            buttonText.textContent = 'Processing...';
            spinner.style.display = 'inline-block';
            form.querySelector('button').disabled = true;
        });
    </script>
</body>
</html>
"""

# --- Code Analysis and Restructuring Logic ---

class LocalModuleDetector(ast.NodeVisitor):
    """
    AST Visitor to detect imports that are likely local modules.
    """
    def __init__(self):
        # A set of known third-party libraries from the monolith's requirements.
        self.known_packages = {
            "fastapi", "uvicorn", "sqlalchemy", "pydantic", "httpx", "stripe",
            "qrcode", "itsdangerous", "passlib", "email_validator", "redis",
            "os", "uuid", "time", "json", "asyncio", "hashlib", "hmac", "logging",
            "re", "secrets", "sys", "textwrap", "typing", "io"
        }
        # Add Python's standard library modules for more accuracy
        self.known_packages.update(sys.stdlib_module_names)
        self.local_modules = set()

    def visit_Import(self, node):
        for alias in node.names:
            # A local module likely won't have dots in its top-level import
            if '.' not in alias.name and alias.name not in self.known_packages:
                self.local_modules.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and '.' not in node.module and node.module not in self.known_packages:
            self.local_modules.add(node.module)
        self.generic_visit(node)


class CodeCategorizer(ast.NodeVisitor):
    """
    AST Visitor to categorize code elements into logical groups.
    """
    def __init__(self):
        self.categories = {
            "imports": [], "config": [], "models": [], "schemas": [],
            "db_setup": [], "services": [], "dependencies": [],
            "routes": [], "main_app": [], "other_globals": []
        }
        self.fastapi_app_name = "app"

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.isupper(): self.categories["config"].append(node)
            elif name in ["engine", "AsyncSessionLocal", "Base", "redis_client"]: self.categories["db_setup"].append(node)
            elif name == "app": self.fastapi_app_name = name; self.categories["main_app"].append(node)
            else: self.categories["other_globals"].append(node)
        else: self.categories["other_globals"].append(node)

    def visit_ClassDef(self, node):
        bases = {getattr(b, 'id', None) for b in node.bases}
        if 'Base' in bases: self.categories["models"].append(node)
        elif 'BaseModel' in bases: self.categories["schemas"].append(node)
        else: self.categories["other_globals"].append(node)

    def visit_FunctionDef(self, node):
        is_route = any(
            isinstance(d, ast.Call) and
            getattr(d.func, 'value', None) and
            getattr(d.func.value, 'id', None) == self.fastapi_app_name and
            getattr(d.func, 'attr', None) in {'get', 'post', 'put', 'delete', 'websocket', 'on_event', 'exception_handler'}
            for d in node.decorator_list
        )
        if is_route: self.categories["routes"].append(node)
        elif node.name in ['get_db', 'get_current_user', 'require_user', 'require_streamer']: self.categories["dependencies"].append(node)
        else: self.categories["services"].append(node)


def restructure_code(code: str, project_name: str):
    """Main function to parse and restructure the code."""
    tree = ast.parse(code)

    # 1. Detect local modules first
    detector = LocalModuleDetector()
    detector.visit(tree)
    local_modules = detector.local_modules
    print(f"Detected local modules: {local_modules}")

    # 2. Categorize all code elements
    categorizer = CodeCategorizer()
    categorizer.visit(tree)
    
    # In-memory zip file
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Create directory structure
        base_path = project_name
        app_path = f"{base_path}/app"
        dirs = [
            app_path, f"{app_path}/api", f"{app_path}/core", f"{app_path}/db",
            f"{app_path}/models", f"{app_path}/schemas", f"{app_path}/services",
            f"{app_path}/dependencies", f"{base_path}/templates"
        ]
        for d in dirs:
            zf.writestr(f"{d}/__init__.py", "")

        # 3. Create placeholder files for detected local modules in `app/`
        for module_name in local_modules:
            placeholder_content = dedent(f"""
            # This is an auto-generated placeholder for your local module '{module_name}.py'.
            # The restructuring algorithm detected that your main script imports this file.
            # 
            # PLEASE PASTE THE ORIGINAL CONTENTS OF '{module_name}.py' HERE.
            """)
            zf.writestr(f"{app_path}/{module_name}.py", placeholder_content)

        # 4. Create structured files from categorized code
        
        # Config
        config_code = "import os\n\n" + "\n".join([unparse(node) for node in categorizer.categories["config"]])
        zf.writestr(f"{app_path}/core/config.py", config_code)

        # DB Session
        db_code = "from sqlalchemy.orm import declarative_base\n\n" + unparse(categorizer.categories["db_setup"][0])
        zf.writestr(f"{app_path}/db/base.py", db_code)
        
        db_session_code = dedent("""
            from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
            from app.core.config import DATABASE_URL
            
            engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
            AsyncSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine)
        """)
        zf.writestr(f"{app_path}/db/session.py", db_session_code)

        # Models
        model_imports = dedent("""
            import uuid
            from datetime import datetime
            from sqlalchemy import (Column, String, Boolean, Integer, Float, DateTime, 
                                    ForeignKey, Text, Index, UniqueConstraint, func)
            from sqlalchemy.orm import relationship
            from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
            from app.db.base import Base
        """)
        for model in categorizer.categories["models"]:
            zf.writestr(f"{app_path}/models/{model.name.lower()}.py", model_imports + "\n\n" + unparse(model))

        # Schemas
        schema_imports = dedent("""
            from pydantic import BaseModel, validator, Field
            from typing import Optional, List
            import re
            from email_validator import validate_email, EmailNotValidError
        """)
        for schema in categorizer.categories["schemas"]:
             zf.writestr(f"{app_path}/schemas/{schema.name.lower()}.py", schema_imports + "\n\n" + unparse(schema))
        
        # Services & Other Globals
        services_code = dedent("""
            # This file contains business logic, helper functions, and other global objects.
            import os
            import time
            import secrets
            from passlib.context import CryptContext
            from itsdangerous import URLSafeTimedSerializer
            from app.core.config import SESSION_SECRET
        """) + "\n\n" + "\n\n".join(unparse(n) for n in categorizer.categories['other_globals'] + categorizer.categories['services'])
        zf.writestr(f"{app_path}/services/utils.py", services_code)

        # Dependencies
        deps_code = dedent("""
            from fastapi import Depends, HTTPException, Request
            from sqlalchemy.ext.asyncio import AsyncSession
            from app.db.session import AsyncSessionLocal
            # Add other necessary imports here
        """) + "\n\n" + "\n\n".join(unparse(n) for n in categorizer.categories['dependencies'])
        zf.writestr(f"{app_path}/dependencies.py", deps_code)

        # API Routers and HTML Templates
        routes_by_prefix = {}
        # ... (same logic as before to group routes by prefix) ...
        for route in categorizer.categories["routes"]:
            path_arg = route.decorator_list[0].args[0]
            if isinstance(path_arg, ast.Constant):
                path = path_arg.value
                prefix = "/" + path.split('/')[1] if path.count('/') > 0 else "/root"
                if prefix not in routes_by_prefix: routes_by_prefix[prefix] = []
                routes_by_prefix[prefix].append(route)
        
        all_router_names = []
        for prefix, routes in routes_by_prefix.items():
            router_name = prefix.strip('/').replace('-', '_') or 'root'
            all_router_names.append(router_name)
            router_code = "from fastapi import APIRouter, Depends, Request, Response, Form, Query, UploadFile, File\n"
            router_code += "from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse\n"
            router_code += "# Add all necessary imports from your project here, e.g.:\n"
            router_code += "# from app.schemas.user import UserCreate\n"
            router_code += "# from app.services.utils import hash_password\n\n"
            router_code += "router = APIRouter()\n\n"
            
            for route in routes:
                decorator_func = route.decorator_list[0].func
                decorator_func.value.id = "router"
                
                # Extract and save HTML
                if "HTMLResponse" in unparse(route):
                    html_fstring_node = next((n for n in ast.walk(route.body[-1]) if isinstance(n, ast.JoinedStr)), None)
                    if html_fstring_node:
                        html_content = "".join([s.value for s in html_fstring_node.values if isinstance(s, ast.Constant)])
                        template_name = f"{router_name}_{route.name}.html"
                        zf.writestr(f"{base_path}/templates/{template_name}", dedent(html_content))
                        
                        # Replace return with a placeholder for template rendering
                        route.body[-1] = ast.Expr(value=ast.Constant(value=f"... # Return TemplateResponse('{template_name}') here"))
                
                router_code += unparse(route) + "\n\n"

            zf.writestr(f"{app_path}/api/{router_name}_routes.py", router_code)

        # Main app file
        main_py_code = dedent(f"""
            from fastapi import FastAPI
            from fastapi.middleware.cors import CORSMiddleware
        """)
        for name in all_router_names:
            main_py_code += f"from app.api import {name}_routes\n"
        
        main_py_code += dedent(f"""

            app = FastAPI(title="{project_name}")

            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"], allow_credentials=True,
                allow_methods=["*"], allow_headers=["*"],
            )
        """)
        for name in all_router_names:
            main_py_code += f"app.include_router({name}_routes.router, prefix='/{name if name != 'root' else ''}', tags=['{name}'])"

        zf.writestr(f"{base_path}/app/main.py", main_py_code)

        # Requirements.txt
        requirements = [
            "fastapi", "uvicorn[standard]", "SQLAlchemy==2.0.23", "asyncpg",
            "pydantic", "httpx", "stripe", "qrcode", "itsdangerous",
            "passlib[bcrypt]", "email-validator", "redis", "python-multipart", "Jinja2"
        ]
        zf.writestr(f"{base_path}/requirements.txt", "\n".join(requirements))
        
        # README
        readme = dedent(f"""
        # {project_name.replace('_', ' ').title()}

        This project was automatically restructured.

        **IMPORTANT:** The script detected that you import the following local modules: `{', '.join(local_modules)}`. 
        Empty placeholder files have been created for them inside the `app/` directory. 
        **You must paste your original code into these files for the application to work.**

        ## How to Run
        1. `python -m venv venv`
        2. `source venv/bin/activate`
        3. `pip install -r requirements.txt`
        4. `uvicorn app.main:app --reload`
        """)
        zf.writestr(f"{base_path}/README.md", readme)


    zip_buffer.seek(0)
    return zip_buffer

# --- FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serves the main HTML upload page."""
    return HOME_PAGE_HTML

@app.post("/restructure")
async def restructure_endpoint(project_name: str = Form(...), file: UploadFile = File(...)):
    """Handles file upload, restructuring, and returns a zip file."""
    if not file.filename.endswith(".py"):
        return HTMLResponse(content="<h1>Error: Please upload a Python (.py) file.</h1>", status_code=400)

    contents = await file.read()
    code_string = contents.decode("utf-8")

    try:
        zip_buffer = restructure_code(code_string, project_name)
    except Exception as e:
        import traceback
        error_html = f"""
        <h1>Restructuring Failed</h1>
        <p>An error occurred: {e}</p>
        <pre style='background:#222;padding:1rem;border-radius:5px;color:#ff5555;'>{traceback.format_exc()}</pre>
        """
        return HTMLResponse(content=error_html, status_code=500)

    return StreamingResponse(
        iter([zip_buffer.read()]),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={project_name}.zip"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))