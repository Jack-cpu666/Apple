import os
import ast
import re
import json
import shutil
import zipfile
import tempfile
import logging
import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HAS_AST_UNPARSE = hasattr(ast, 'unparse')
if not HAS_AST_UNPARSE:
    logger.warning("Python 3.8 or older detected. AST unparsing will be less reliable. Python 3.9+ is recommended.")

@dataclass
class CodeBlock:
    node: ast.AST
    name: str
    type: str
    content: str
    dependencies: Set[str] = field(default_factory=set)
    decorators: List[str] = field(default_factory=list)

@dataclass
class RouteInfo:
    method: str
    path: str
    is_html_response: bool = False
    template_name: Optional[str] = None

class DependencyVisitor(ast.NodeVisitor):
    def __init__(self, global_scope_names: Set[str]):
        self.dependencies = set()
        self.global_scope = global_scope_names
        self.local_scope = set()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.local_scope.update(arg.arg for arg in node.args.args)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.local_scope.update(arg.arg for arg in node.args.args)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id not in self.local_scope and node.id in self.global_scope:
            self.dependencies.add(node.id)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.local_scope.add(target.id)
        self.generic_visit(node)

class HTMLResponseVisitor(ast.NodeVisitor):
    def __init__(self, lines: List[str]):
        self.html_content: Optional[str] = None
        self.lines = lines

    def visit_Return(self, node: ast.Return):
        if (isinstance(node.value, ast.Call) and
            isinstance(node.value.func, ast.Name) and
            node.value.func.id == 'HTMLResponse'):
            if node.value.args and isinstance(node.value.args[0], ast.JoinedStr):
                start_line = node.value.args[0].lineno - 1
                end_line = getattr(node.value.args[0], 'end_lineno', start_line + 1)
                
                raw_lines = self.lines[start_line:end_line]
                full_fstring = "".join(raw_lines)
                
                match = re.search(r'f("""|\'\'\'|"|\')(.*)(\1)', full_fstring, re.DOTALL)
                if match:
                    self.html_content = match.group(2)


class MonolithRefactorer:
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.lines = source_code.split('\n')
        self.tree = ast.parse(self.source_code)
        self.imports: Dict[str, str] = {}
        self.models: Dict[str, CodeBlock] = {}
        self.schemas: Dict[str, CodeBlock] = {}
        self.routes: Dict[str, Tuple[RouteInfo, CodeBlock]] = {}
        self.utilities: Dict[str, CodeBlock] = {}
        self.infra: Dict[str, CodeBlock] = {}
        self.config: Dict[str, str] = {}
        self.app_setup: Dict[str, Any] = {
            "middleware": [], "startup": [], "shutdown": []
        }
        self.defined_names: Set[str] = set()

    def refactor(self) -> Dict[str, Any]:
        logger.info("Step 1: Parsing and categorizing code blocks...")
        self._categorize_all_nodes()
        logger.info("Step 2: Analyzing dependencies for each code block...")
        self._analyze_all_dependencies()
        logger.info("Step 3: Generating new project file structure and content...")
        project_files = self._generate_project_files()
        logger.info(f"Refactoring complete. Identified {len(self.routes)} routes, {len(self.models)} models, {len(self.schemas)} schemas.")
        return project_files

    def _categorize_all_nodes(self):
        for node in self.tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._categorize_import(node)
            elif isinstance(node, ast.ClassDef):
                self._categorize_class(node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._categorize_function(node)
            elif isinstance(node, ast.Assign):
                self._categorize_assignment(node)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                self._categorize_expression(node)

    def _get_node_content(self, node: ast.AST) -> str:
        if HAS_AST_UNPARSE:
            return ast.unparse(node)
        start_line = node.lineno - 1
        end_line = getattr(node, 'end_lineno', start_line + 1)
        return '\n'.join(self.lines[start_line:end_line])

    def _categorize_import(self, node: Union[ast.Import, ast.ImportFrom]):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                self.imports[name] = self._get_node_content(node)
                self.defined_names.add(name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                name = alias.asname or alias.name
                level = getattr(node, 'level', 0)
                full_import = f"from {'.' * level}{module} import {alias.name}"
                if alias.asname:
                    full_import += f" as {alias.asname}"
                self.imports[name] = full_import
                self.defined_names.add(name)

    def _categorize_class(self, node: ast.ClassDef):
        self.defined_names.add(node.name)
        content = self._get_node_content(node)
        base_names = {b.id for b in node.bases if isinstance(b, ast.Name)}
        if 'Base' in base_names:
            self.models[node.name] = CodeBlock(node=node, name=node.name, type='model', content=content)
        elif 'BaseModel' in base_names:
            self.schemas[node.name] = CodeBlock(node=node, name=node.name, type='schema', content=content)
        else:
            self.utilities[node.name] = CodeBlock(node=node, name=node.name, type='utility_class', content=content)

    def _categorize_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        self.defined_names.add(node.name)
        decorators = [self._get_node_content(d) for d in node.decorator_list]
        content = self._get_node_content(node)
        code_block = CodeBlock(node=node, name=node.name, type='utility', content=content, decorators=decorators)
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                if isinstance(dec.func.value, ast.Name) and dec.func.value.id == 'app':
                    if dec.func.attr in ['get', 'post', 'put', 'delete', 'patch', 'websocket']:
                        self._process_route(node, dec)
                        return
                    elif dec.func.attr == 'on_event':
                        event_type = dec.args[0].value
                        self.app_setup[event_type].append(content)
                        return
        self.utilities[node.name] = code_block

    def _process_route(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], dec: ast.Call):
        method = dec.func.attr
        path = dec.args[0].value if dec.args and isinstance(dec.args[0], ast.Constant) else "/"
        is_html = any(isinstance(n, ast.Return) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'HTMLResponse' for n in ast.walk(node))
        route_info = RouteInfo(method=method, path=path, is_html_response=is_html)
        content = self._get_node_content(node)
        code_block = CodeBlock(node=node, name=node.name, type='route', content=content, decorators=[self._get_node_content(d) for d in node.decorator_list])
        self.routes[node.name] = (route_info, code_block)

    def _categorize_assignment(self, node: ast.Assign):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            self.defined_names.add(name)
            content = self._get_node_content(node)
            if name.isupper():
                self.config[name] = content
            elif name in ['pwd_context', 'signer', 'queue_manager', 'rate_limiter', 'redis_client', 'engine', 'AsyncSessionLocal']:
                self.infra[name] = CodeBlock(node=node, name=name, type='infra', content=content)
            elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'FastAPI':
                self.app_setup['app_instance'] = content

    def _categorize_expression(self, node: ast.Expr):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            if isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'app':
                if node.value.func.attr == 'add_middleware':
                    self.app_setup['middleware'].append(self._get_node_content(node))

    def _analyze_all_dependencies(self):
        all_blocks = list(self.models.values()) + list(self.schemas.values()) + [r[1] for r in self.routes.values()] + list(self.utilities.values()) + list(self.infra.values())
        for block in all_blocks:
            visitor = DependencyVisitor(self.defined_names)
            visitor.visit(block.node)
            block.dependencies = visitor.dependencies

    def _generate_project_files(self) -> Dict[str, Any]:
        files: Dict[str, str] = {}
        templates: Dict[str, str] = {}
        route_modules = self._group_routes()
        files['app/core/config.py'] = self._build_config_file()
        files['app/database/session.py'] = self._build_session_file()
        files['app/database/models.py'] = self._build_models_file()
        files['app/schemas.py'] = self._build_schemas_file()
        files['app/core/dependencies.py'] = self._build_dependencies_file()
        files['app/utils/security.py'] = self._build_security_utils_file()
        for module_path, routes_in_module in route_modules.items():
            if "pages" in module_path:
                content, extracted_html = self._build_page_routes_file(routes_in_module)
                files[module_path] = content
                templates.update(extracted_html)
            else:
                files[module_path] = self._build_api_routes_file(module_path, routes_in_module)
        files['main.py'] = self._build_main_file(route_modules)
        files['requirements.txt'] = self._build_requirements_file()
        files['.env.example'] = self._build_env_example_file()
        return {"files": files, "templates": templates}

    def _collect_imports_for_blocks(self, blocks: List[CodeBlock]) -> str:
        required_imports = set()
        for block in blocks:
            for dep in block.dependencies:
                if dep in self.imports:
                    required_imports.add(self.imports[dep])
        return '\n'.join(sorted(list(required_imports))) + '\n\n'

    def _build_config_file(self) -> str:
        content = "from pydantic_settings import BaseSettings\n\n\nclass Settings(BaseSettings):\n"
        for name, assign_str in self.config.items():
            var_type = "str"
            if "int(" in assign_str: var_type = "int"
            if ".lower() == 'true'" in assign_str: var_type = "bool"
            content += f"    {name}: {var_type}\n"
        content += "\n    class Config:\n        env_file = '.env'\n\n\nsettings = Settings()\n"
        return content

    def _build_session_file(self) -> str:
        db_infra = [b for n, b in self.infra.items() if n in ['engine', 'AsyncSessionLocal']]
        db_utils = [b for n, b in self.utilities.items() if n in ['get_db', 'init_db']]
        all_blocks = db_infra + db_utils
        imports = self._collect_imports_for_blocks(all_blocks)
        imports += "from app.core.config import settings\n"
        content = imports + "\n".join(b.content for b in all_blocks)
        return content

    def _build_models_file(self) -> str:
        imports = self._collect_imports_for_blocks(list(self.models.values()))
        base_def = "from sqlalchemy.orm import declarative_base\n\nBase = declarative_base()\n\n"
        content = imports + base_def + "\n\n".join(b.content for b in self.models.values())
        return content

    def _build_schemas_file(self) -> str:
        imports = self._collect_imports_for_blocks(list(self.schemas.values()))
        content = imports + "\n\n".join(b.content for b in self.schemas.values())
        return content

    def _build_dependencies_file(self) -> str:
        dep_blocks = [b for n, b in self.infra.items() if n in ['pwd_context', 'signer']]
        imports = self._collect_imports_for_blocks(dep_blocks)
        content = imports + "\n\n".join(b.content for b in dep_blocks)
        return content

    def _build_security_utils_file(self) -> str:
        sec_utils = [b for n, b in self.utilities.items() if 'password' in n or 'token' in n or 'session' in n]
        imports = self._collect_imports_for_blocks(sec_utils)
        imports += "from app.core.dependencies import pwd_context, signer\n"
        content = imports + "\n\n".join(b.content for b in sec_utils)
        return content

    def _group_routes(self) -> Dict[str, list]:
        modules = defaultdict(list)
        for name, (route_info, code_block) in self.routes.items():
            path = route_info.path
            if route_info.is_html_response or path in ['/', '/login', '/register', '/dashboard']:
                modules['app/pages/views.py'].append(name)
            elif path.startswith('/api/auth') or path.startswith('/auth/'):
                modules['app/api/auth.py'].append(name)
            elif path.startswith('/api/requests') or path.startswith('/api/queue'):
                modules['app/api/requests.py'].append(name)
            elif path.startswith('/api/admin'):
                modules['app/api/admin.py'].append(name)
            elif path.startswith('/api/stripe') or path.startswith('/api/withdraw'):
                modules['app/api/payments.py'].append(name)
            elif path.startswith('/ws/'):
                modules['app/services/websockets.py'].append(name)
            else:
                modules['app/api/users.py'].append(name)
        return modules

    def _build_api_routes_file(self, module_path: str, route_names: List[str]) -> str:
        route_blocks = [self.routes[name][1] for name in route_names]
        imports = self._collect_imports_for_blocks(route_blocks)
        content = "from fastapi import APIRouter\n\n"
        content += imports
        content += "router = APIRouter()\n\n"
        content += "\n\n".join(b.content.replace('@app.', '@router.') for b in route_blocks)
        return content

    def _build_page_routes_file(self, route_names: List[str]) -> Tuple[str, Dict[str, str]]:
        route_blocks = [self.routes[name][1] for name in route_names]
        imports = self._collect_imports_for_blocks(route_blocks)
        templates = {}
        content = "from fastapi import APIRouter, Request\n"
        content += "from fastapi.responses import HTMLResponse\n"
        content += "from fastapi.templating import Jinja2Templates\n\n"
        content += imports
        content += "router = APIRouter()\n"
        content += "templates = Jinja2Templates(directory='templates')\n\n"
        for name in route_names:
            route_info, code_block = self.routes[name]
            if not route_info.is_html_response:
                content += code_block.content.replace('@app.', '@router.') + '\n\n'
                continue
            visitor = HTMLResponseVisitor(self.lines)
            visitor.visit(code_block.node)
            if visitor.html_content:
                template_content, context_vars = self._convert_fstring_to_jinja(visitor.html_content)
                template_name = f"{name}.html"
                templates[template_name] = template_content
                new_func_def = f"async def {name}(request: Request):\n"
                
                # Corrected f-string escaping
                context_str = f"context = {{'request': request, {', '.join(f'\"{v}\": {v}' for v in context_vars)}}}\n"

                return_str = f"return templates.TemplateResponse('{template_name}', context)\n"
                new_content = "\n".join(code_block.decorators) + '\n'
                new_content += new_func_def
                new_content += f"    # TODO: Implement logic to define context variables: {list(context_vars)}\n"
                for var in context_vars:
                    new_content += f"    {var} = {{}}\n"
                new_content += '    ' + context_str
                new_content += '    ' + return_str
                content += new_content.replace('@app.', '@router.') + '\n\n'
            else:
                content += code_block.content.replace('@app.', '@router.') + '\n\n'
        return content, templates

    def _convert_fstring_to_jinja(self, fstring_content: str) -> Tuple[str, Set[str]]:
        context_vars = set()
        def replacer(match):
            expression = match.group(1).strip()
            context_vars.add(expression.split('[')[0].split('.')[0])
            return f"{{{{ {expression} }}}}"
        template_content = re.sub(r'\{([^}]+)\}', replacer, fstring_content)
        return template_content, context_vars

    def _build_main_file(self, route_modules: Dict[str, list]) -> str:
        router_imports = ""
        router_includes = ""
        for module_path in sorted(route_modules.keys()):
            module_name = module_path.replace('app/', '').replace('/', '.').replace('.py', '')
            router_name = module_name.split('.')[-1]
            router_imports += f"from app.{module_name} import router as {router_name}_router\n"
            prefix = "/api" if "api" in module_path else ""
            tags = [router_name.capitalize()]
            router_includes += f"app.include_router({router_name}_router, prefix='{prefix}', tags={tags})\n"
        
        main_content = f"""from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.database.session import init_db

{router_imports}

app = FastAPI(title="StreamBeatz Refactored", version="1.0.0")

{"\n".join(self.app_setup['middleware'])}

@app.on_event("startup")
async def startup_event():
    await init_db()
    {"\n    ".join(self.app_setup.get('startup', []))}

@app.on_event("shutdown")
async def shutdown_event():
    {"\n    ".join(self.app_setup.get('shutdown', []))}

{router_includes}

@app.get("/health", tags=["Health"])
def health_check():
    return {{"status": "ok"}}
"""
        return main_content
    
    def _build_requirements_file(self) -> str:
        return "\n".join([
            "fastapi", "uvicorn[standard]", "sqlalchemy[asyncpg]", "pydantic",
            "pydantic-settings", "python-multipart", "passlib[bcrypt]", "httpx",
            "redis", "stripe", "jinja2", "email-validator", "python-dotenv", "qrcode[pil]"
        ])

    def _build_env_example_file(self) -> str:
        content = ""
        for name, assign_str in self.config.items():
            match = re.search(r"os\.getenv\(['\"]([^'\"]+)['\"]", assign_str)
            if match:
                default_val = re.search(r",\s*['\"]?([^'\"]+)['\"]?\)", assign_str)
                default = f'"{default_val.group(1)}"' if default_val else ""
                content += f"{match.group(1)}={default}\n"
        return content

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
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .glass { background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2); }
        .upload-zone { border: 3px dashed rgba(255, 255, 255, 0.3); transition: all 0.3s ease; }
        .upload-zone:hover { border-color: rgba(255, 255, 255, 0.6); background: rgba(255, 255, 255, 0.05); }
        .upload-zone.dragover { border-color: #10b981; background: rgba(16, 185, 129, 0.1); }
        .progress-bar { transition: width 0.5s ease-out; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
        .feature-card { transition: transform 0.2s ease; }
        .feature-card:hover { transform: translateY(-4px); }
    </style>
</head>
<body class="gradient-bg min-h-screen text-white">
    <div class="container mx-auto px-4 py-8 max-w-6xl">
        <div class="text-center mb-12">
            <h1 class="text-5xl font-bold mb-4">FastAPI Refactoring Engine</h1>
            <p class="text-xl opacity-90">Transform your monolithic FastAPI application into a clean, modular architecture</p>
        </div>
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
                <div id="progressSection" class="mt-6 hidden">
                    <div class="flex justify-between text-sm mb-2">
                        <span id="progressText">Processing...</span>
                        <span id="progressPercent">0%</span>
                    </div>
                    <div class="w-full bg-white/20 rounded-full h-2">
                        <div class="progress-bar bg-green-400 h-2 rounded-full" style="width: 0%"></div>
                    </div>
                </div>
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
                <button type="submit" id="refactorBtn" class="w-full mt-6 bg-white/20 hover:bg-white/30 py-4 rounded-lg font-medium text-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed" disabled>
                    <span id="btnText">Select a file to begin refactoring</span>
                    <svg id="btnSpinner" class="hidden inline-block w-5 h-5 ml-2 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                </button>
            </form>
        </div>
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
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('dragover'); });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].name.endsWith('.py')) { handleFile(files[0]); }
        });
        fileInput.addEventListener('change', (e) => { if (e.target.files.length > 0) { handleFile(e.target.files[0]); } });
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
            progressSection.classList.remove('hidden');
            refactorBtn.disabled = true;
            document.getElementById('btnText').textContent = 'Refactoring...';
            document.getElementById('btnSpinner').classList.remove('hidden');
            let progress = 0;
            const progressBar = document.querySelector('.progress-bar');
            const progressInterval = setInterval(() => {
                progress += Math.random() * 15;
                if (progress > 90) progress = 90;
                progressBar.style.width = progress + '%';
                document.getElementById('progressPercent').textContent = Math.round(progress) + '%';
            }, 500);
            try {
                const response = await fetch('/refactor', { method: 'POST', body: formData });
                clearInterval(progressInterval);
                progressBar.style.width = '100%';
                document.getElementById('progressPercent').textContent = '100%';
                if (response.ok) {
                    const result = await response.json();
                    showResults(result);
                } else { throw new Error('Refactoring failed'); }
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

app = FastAPI(title="FastAPI Monolith Refactoring Engine")

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE

@app.post("/refactor")
async def refactor_endpoint(file: UploadFile = File(...)):
    if not file.filename.endswith('.py'):
        return JSONResponse({"error": "Only .py files are supported"}, status_code=400)
    
    source_code_bytes = await file.read()
    try:
        source_code = source_code_bytes.decode('utf-8')
    except UnicodeDecodeError:
        return JSONResponse({"error": "File must be UTF-8 encoded."}, status_code=400)
        
    original_lines = len(source_code.split('\n'))

    try:
        refactorer = MonolithRefactorer(source_code)
        project = refactorer.refactor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / "streambeatz_refactored"
            
            for rel_path, content in project.get("files", {}).items():
                path = project_root / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding='utf-8')
            
            templates_dir = project_root / "templates"
            if project.get("templates"):
                templates_dir.mkdir(exist_ok=True)
                for name, content in project.get("templates", {}).items():
                    (templates_dir / name).write_text(content, encoding='utf-8')
            
            for d in project_root.rglob(''):
                if d.is_dir() and not list(d.glob('__init__.py')):
                    (d / '__init__.py').touch()
            
            (project_root / "README.md").write_text("# Refactored Project\n\nThis project was automatically refactored.")

            structure_preview = []
            for path in sorted(project_root.rglob('*')):
                if len(path.relative_to(project_root).parts) > 4: continue
                depth = len(path.relative_to(project_root).parts) - 1
                indent = '  ' * depth
                structure_preview.append(f"{indent}{'|-- ' if depth > 0 else ''}{path.name}{'/' if path.is_dir() else ''}")

            zip_filename = f"refactored_{int(time.time())}"
            zip_path = Path(tempfile.gettempdir()) / zip_filename
            shutil.make_archive(str(zip_path), 'zip', project_root)

            return JSONResponse({
                "success": True,
                "original_lines": original_lines,
                "files_created": len(project['files']) + len(project['templates']),
                "structure": "\n".join(structure_preview),
                "download_url": f"/download/{zip_filename}.zip"
            })

    except Exception as e:
        logger.error(f"Refactoring failed: {e}", exc_info=True)
        return JSONResponse({"error": f"An error occurred: {e}"}, status_code=500)

@app.get("/download/{filename}")
async def download_zip(filename: str):
    path = Path(tempfile.gettempdir()) / filename
    if path.exists():
        return FileResponse(path, media_type='application/zip', filename='streambeatz_refactored.zip')
    return JSONResponse({"error": "File not found or expired"}, status_code=404)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)