import os
import ast
import zipfile
from io import BytesIO
from textwrap import dedent

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse

# Helper to unparse AST nodes back to code, available in Python 3.9+
# For older versions, an external library like astor would be needed.
unparse = ast.unparse

# --- Main Application ---
app = FastAPI(
    title="Monolith to Microservice Converter",
    description="Upload your monolithic FastAPI script to convert it into a structured project."
)

# --- Static Content (HTML Templates, etc.) ---

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
                    <label for="project_name" class="block mb-2 text-sm font-medium text-gray-300">Project Name</label>
                    <input type="text" id="project_name" name="project_name" value="streambeatz_project" required
                           class="w-full bg-gray-900 border border-gray-600 text-white rounded-lg p-2.5 focus:ring-indigo-500 focus:border-indigo-500">
                </div>
                <div class="mb-6">
                    <label class="file-label" for="file_upload">
                        <span id="file-text">Click to upload your monolithic Python file</span>
                        <input type="file" id="file_upload" name="file" class="hidden" accept=".py">
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

        form.addEventListener('submit', () => {
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

class CodeCategorizer(ast.NodeVisitor):
    def __init__(self):
        self.categories = {
            "imports": [], "config": [], "models": [], "schemas": [],
            "dbsession": [], "services": [], "dependencies": [],
            "routes": [], "main_app": [], "startup_events": [], "other": []
        }
        self.fastapi_app_name = "app"

    def visit_Import(self, node):
        self.categories["imports"].append(node)

    def visit_ImportFrom(self, node):
        self.categories["imports"].append(node)

    def visit_Assign(self, node):
        # This is a simplified check. A real-world scenario might be more complex.
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.isupper():
                self.categories["config"].append(node)
            elif name in ["engine", "AsyncSessionLocal", "Base", "redis_client", "pwd_context", "signer"]:
                self.categories["dbsession"].append(node)
            elif name == "app":
                self.fastapi_app_name = name
                self.categories["main_app"].append(node)
            else:
                self.categories["other"].append(node)
        else:
            self.categories["other"].append(node)

    def visit_ClassDef(self, node):
        # Check for base classes to categorize
        if any(getattr(b, 'id', None) == 'Base' for b in node.bases):
            self.categories["models"].append(node)
        elif any(getattr(b, 'id', None) == 'BaseModel' for b in node.bases):
            self.categories["schemas"].append(node)
        else:
            self.categories["other"].append(node)

    def visit_FunctionDef(self, node):
        is_route = False
        is_startup = False
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                if getattr(decorator.func.value, 'id', None) == self.fastapi_app_name:
                    if decorator.func.attr in ['get', 'post', 'put', 'delete', 'websocket', 'exception_handler']:
                        is_route = True
                    elif decorator.func.attr == 'on_event' and decorator.args and isinstance(decorator.args[0], ast.Constant) and decorator.args[0].value == 'startup':
                        is_startup = True

        if is_route:
            self.categories["routes"].append(node)
        elif is_startup:
            self.categories["startup_events"].append(node)
        elif node.name in ['get_db', 'get_current_user', 'require_user', 'require_streamer', 'get_admin_user']:
            self.categories["dependencies"].append(node)
        elif node.name.startswith(('__', 'test_')):
            self.categories["other"].append(node) # Ignore private/test functions for now
        else:
            self.categories["services"].append(node)

def restructure_code(code: str, project_name: str):
    """Main function to parse and restructure the code."""
    tree = ast.parse(code)
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

        # --- Create individual files ---

        # 1. Config
        config_code = "# --- Configuration variables ---\n\nimport os\n\n"
        config_code += "\n".join([unparse(node) for node in categorizer.categories["config"]])
        zf.writestr(f"{app_path}/core/config.py", config_code)

        # 2. DB Session
        db_session_code = dedent("""
            import os
            from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
            from sqlalchemy.orm import declarative_base
            from .config import DATABASE_URL, REDIS_URL

            if DATABASE_URL and DATABASE_URL.startswith('postgresql'):
                engine = create_async_engine(DATABASE_URL, echo=False)
                AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)
            else:
                engine = None
                AsyncSessionLocal = None
            
            Base = declarative_base()

            # Redis (optional)
            redis_client = None
            try:
                import redis.asyncio as redis
                if REDIS_URL:
                    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            except ImportError:
                pass
        """)
        zf.writestr(f"{app_path}/db/session.py", db_session_code)
        
        # 3. Models
        model_imports = dedent("""
            import uuid
            from datetime import datetime
            from sqlalchemy import (Column, String, Boolean, Integer, Float, DateTime, 
                                    ForeignKey, Text, Index, UniqueConstraint, func)
            from sqlalchemy.orm import relationship
            from sqlalchemy.dialects.postgresql import UUID, JSONB
            from app.db.session import Base
        """)
        for model_class in categorizer.categories["models"]:
            filename = f"{model_class.name.lower()}.py"
            file_content = model_imports + "\n\n" + unparse(model_class)
            zf.writestr(f"{app_path}/models/{filename}", file_content)

        # 4. Schemas
        schema_imports = dedent("""
            from typing import Optional, List, Dict, Any
            from pydantic import BaseModel, validator, Field
            from datetime import datetime
            import re
            from email_validator import validate_email, EmailNotValidError
        """)
        for schema_class in categorizer.categories["schemas"]:
            filename = f"{schema_class.name.lower()}.py"
            file_content = schema_imports + "\n\n" + unparse(schema_class)
            zf.writestr(f"{app_path}/schemas/{filename}", file_content)

        # 5. Services & Dependencies
        zf.writestr(f"{app_path}/services.py", "# --- Business Logic and Helper Functions ---\n\n" +
                    "import time\nimport secrets\nfrom datetime import datetime\nfrom passlib.context import CryptContext\n\n" +
                    "\n\n".join([unparse(s) for s in categorizer.categories["services"]]))
        
        zf.writestr(f"{app_path}/dependencies.py", "# --- FastAPI Dependencies ---\n\n" +
                    "from typing import Optional, Dict\nfrom fastapi import Depends, HTTPException, Request\nfrom sqlalchemy.ext.asyncio import AsyncSession\n\n"
                    "from .db.session import AsyncSessionLocal\n\n" +
                    "\n\n".join([unparse(d) for d in categorizer.categories["dependencies"]]))
        
        # 6. API Routers and HTML Templates
        routes_by_prefix = {}
        for route in categorizer.categories["routes"]:
            # Extract URL path to group routes
            path = route.decorator_list[0].args[0].value
            prefix = "/" + path.split('/')[1]
            if prefix not in routes_by_prefix:
                routes_by_prefix[prefix] = []
            
            # Check for and extract HTML
            if "HTMLResponse" in unparse(route):
                # This is a complex task. We'll simplify by finding the f-string.
                html_fstring_node = None
                for node in ast.walk(route.body[-1]): # Assume return is the last statement
                    if isinstance(node, ast.JoinedStr):
                        html_fstring_node = node
                        break
                
                if html_fstring_node:
                    html_content = "".join([s.value for s in html_fstring_node.values if isinstance(s, ast.Constant)])
                    # Create a sensible filename
                    template_name = path.replace('/', '_').strip('_') + ".html"
                    zf.writestr(f"{base_path}/templates/{template_name}", dedent(html_content))
                    
                    # Replace the return statement with a template render
                    new_return = ast.Return(value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id="templates", ctx=ast.Load()), attr="TemplateResponse", ctx=ast.Load()),
                        args=[ast.Constant(value=template_name), ast.Constant(value={"request": ast.Name(id="request", ctx=ast.Load())})], # Simplified context
                        keywords=[]
                    ))
                    route.body[-1] = new_return

            routes_by_prefix[prefix].append(route)
        
        # Write router files
        router_files = []
        for prefix, routes in routes_by_prefix.items():
            router_name = prefix.replace('/', '') or 'root'
            router_filename = f"{router_name}.py"
            router_files.append(router_name)
            
            router_code = "from fastapi import APIRouter, Depends, Request, Response\n"
            router_code += "from fastapi.responses import HTMLResponse, RedirectResponse\n"
            router_code += "from sqlalchemy.ext.asyncio import AsyncSession\n"
            router_code += "from app.dependencies import get_db, require_user, require_streamer\n\n"
            router_code += "router = APIRouter()\n\n"
            
            for route in routes:
                # Change decorator from app.get to router.get
                decorator_func = route.decorator_list[0].func
                decorator_func.value.id = "router"
                router_code += unparse(route) + "\n\n"
            zf.writestr(f"{app_path}/api/{router_filename}", router_code)

        # 7. Main app file
        main_py_code = dedent(f"""
            from fastapi import FastAPI
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.templating import Jinja2Templates
            from app.core.config import BASE_URL
        """)
        for router_name in router_files:
            main_py_code += f"from app.api import {router_name}\n"
        
        main_py_code += dedent(f"""

            app = FastAPI(
                title="{project_name}",
                version="1.0.0"
            )

            templates = Jinja2Templates(directory="templates")

            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            # Include API routers
        """)
        for router_name in router_files:
            main_py_code += f"app.include_router({router_name}.router, tags=[\"{router_name}\"])\n"

        zf.writestr(f"{app_path}/main.py", main_py_code)

        # 8. Requirements.txt
        # Simplified based on the provided script
        requirements = [
            "fastapi", "uvicorn[standard]", "SQLAlchemy", "asyncpg",
            "pydantic", "httpx", "stripe", "qrcode", "itsdangerous",
            "passlib[bcrypt]", "email_validator", "redis", "python-multipart", "Jinja2"
        ]
        zf.writestr(f"{base_path}/requirements.txt", "\n".join(requirements))
        
        # 9. .gitignore
        gitignore = dedent("""
            # Byte-compiled / optimized / DLL files
            __pycache__/
            *.py[cod]
            *$py.class

            # Environment
            .env
            venv/
            
            # IDEs
            .vscode/
            .idea/
        """)
        zf.writestr(f"{base_path}/.gitignore", gitignore)

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
        # Provide a more helpful error message
        error_html = f"""
        <h1>Restructuring Failed</h1>
        <p>An error occurred while processing your file:</p>
        <pre style='background: #222; padding: 1rem; border-radius: 5px; color: #ff5555;'>{e}</pre>
        <p>This can happen if the script has complex syntax that the parser cannot automatically handle. Please check the file and try again.</p>
        """
        return HTMLResponse(content=error_html, status_code=500)

    return StreamingResponse(
        iter([zip_buffer.read()]),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={project_name}.zip"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)