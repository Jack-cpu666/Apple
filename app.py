# FastAPI Monolith Refactoring Tool - Full Algorithm
# This script uses AST parsing to refactor a large FastAPI file into a modular project.
# It handles dependency analysis, infrastructure separation, route grouping, and HTML extraction.

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
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# --- Configuration and Setup ---

# Configure logging to provide insight into the refactoring process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check Python version for `ast.unparse` availability (Python 3.9+)
# This provides a much cleaner way to convert AST nodes back to source code.
HAS_AST_UNPARSE = hasattr(ast, 'unparse')
if not HAS_AST_UNPARSE:
    logger.warning("Python 3.8 or older detected. AST unparsing will be less reliable. Python 3.9+ is recommended.")

# --- Data Structures for Code Analysis ---

@dataclass
class CodeBlock:
    """Represents a categorized block of code (class, function, etc.) from the monolith."""
    node: ast.AST                    # The raw AST node
    name: str                        # Name of the class or function
    type: str                        # e.g., 'model', 'schema', 'route', 'utility'
    content: str                     # The string content of the code block
    dependencies: Set[str] = field(default_factory=set) # Names of other objects this block depends on
    decorators: List[str] = field(default_factory=list) # Decorator strings, if any

@dataclass
class RouteInfo:
    """Stores specific metadata for an API route."""
    method: str
    path: str
    is_html_response: bool = False
    template_name: Optional[str] = None # Filename for extracted HTML template

# --- Core Refactoring Engine ---

class MonolithRefactorer:
    """
    The main class that orchestrates the parsing, analysis, and generation
    of the refactored FastAPI project.
    """

    def __init__(self, source_code: str):
        self.source_code = source_code
        self.lines = source_code.split('\n')
        self.tree = ast.parse(self.source_code)

        # These dictionaries will store all the categorized code blocks.
        self.imports: Dict[str, str] = {}              # Maps imported name to the full import statement
        self.models: Dict[str, CodeBlock] = {}         # SQLAlchemy models
        self.schemas: Dict[str, CodeBlock] = {}        # Pydantic schemas
        self.routes: Dict[str, Tuple[RouteInfo, CodeBlock]] = {} # API routes
        self.utilities: Dict[str, CodeBlock] = {}      # Helper functions and classes
        self.infra: Dict[str, CodeBlock] = {}          # Core infrastructure instances (pwd_context, etc.)
        self.config: Dict[str, str] = {}               # Configuration variables
        self.app_setup: Dict[str, Any] = {             # Middleware, event handlers
            "middleware": [],
            "startup": [],
            "shutdown": []
        }

        # This set keeps track of all defined names to help with dependency resolution.
        self.defined_names: Set[str] = set()

    def refactor(self) -> Dict[str, Any]:
        """
        Executes the full refactoring process and returns the generated project data.
        """
        logger.info("Step 1: Parsing and categorizing code blocks...")
        self._categorize_all_nodes()

        logger.info("Step 2: Analyzing dependencies for each code block...")
        self._analyze_all_dependencies()

        logger.info("Step 3: Generating new project file structure and content...")
        project_files = self._generate_project_files()

        logger.info(f"Refactoring complete. Identified {len(self.routes)} routes, {len(self.models)} models, {len(self.schemas)} schemas.")
        return project_files

    # --- Step 1: Code Categorization ---

    def _categorize_all_nodes(self):
        """
        Iterates through the top-level nodes of the AST and sends them to the correct
        categorization method.
        """
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
                # This handles things like `app.add_middleware`
                self._categorize_expression(node)

    def _get_node_content(self, node: ast.AST) -> str:
        """
        Safely converts an AST node back into its original source code string.
        Falls back to line-based extraction for older Python versions.
        """
        if HAS_AST_UNPARSE:
            return ast.unparse(node)
        # Fallback for Python < 3.9
        start_line = node.lineno - 1
        end_line = getattr(node, 'end_lineno', start_line)
        return '\n'.join(self.lines[start_line:end_line])

    def _categorize_import(self, node: Union[ast.Import, ast.ImportFrom]):
        """Parses import statements and stores them for later use."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                self.imports[name] = self._get_node_content(node)
                self.defined_names.add(name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                name = alias.asname or alias.name
                full_import = f"from {'.' * node.level}{module} import {alias.name}"
                if alias.asname:
                    full_import += f" as {alias.asname}"
                self.imports[name] = full_import
                self.defined_names.add(name)

    def _categorize_class(self, node: ast.ClassDef):
        """Determines if a class is a Model, Schema, or a Utility class."""
        self.defined_names.add(node.name)
        content = self._get_node_content(node)
        base_names = {b.id for b in node.bases if isinstance(b, ast.Name)}

        if 'Base' in base_names:
            self.models[node.name] = CodeBlock(node=node, name=node.name, type='model', content=content)
        elif 'BaseModel' in base_names:
            self.schemas[node.name] = CodeBlock(node=node, name=node.name, type='schema', content=content)
        else:
            # Any other class is considered a utility/service class
            self.utilities[node.name] = CodeBlock(node=node, name=node.name, type='utility_class', content=content)

    def _categorize_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        """Determines if a function is a Route, an Event Handler, or a Utility."""
        self.defined_names.add(node.name)
        decorators = [self._get_node_content(d) for d in node.decorator_list]
        content = self._get_node_content(node)
        code_block = CodeBlock(node=node, name=node.name, type='utility', content=content, decorators=decorators)

        # Check for FastAPI decorators
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                if isinstance(dec.func.value, ast.Name) and dec.func.value.id == 'app':
                    # This is a route or event handler
                    if dec.func.attr in ['get', 'post', 'put', 'delete', 'patch', 'websocket']:
                        self._process_route(node, dec)
                        return
                    elif dec.func.attr == 'on_event':
                        event_type = dec.args[0].value
                        self.app_setup[event_type].append(content)
                        return

        # If it's not a route or event, it's a utility
        self.utilities[node.name] = code_block

    def _process_route(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], dec: ast.Call):
        """Extracts metadata from a route decorator and categorizes the route."""
        method = dec.func.attr
        path = dec.args[0].value if dec.args else "/"
        
        # Check if the route returns an HTMLResponse to identify page routes
        is_html = any(
            isinstance(n, ast.Return) and
            isinstance(n.value, ast.Call) and
            isinstance(n.value.func, ast.Name) and
            n.value.func.id == 'HTMLResponse'
            for n in ast.walk(node)
        )

        route_info = RouteInfo(method=method, path=path, is_html_response=is_html)
        content = self._get_node_content(node)
        code_block = CodeBlock(node=node, name=node.name, type='route', content=content)
        self.routes[node.name] = (route_info, code_block)

    def _categorize_assignment(self, node: ast.Assign):
        """Identifies configuration variables and infrastructure instances."""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            self.defined_names.add(name)
            content = self._get_node_content(node)
            
            # Identify config variables (typically uppercase)
            if name.isupper():
                self.config[name] = content
            # Identify key infrastructure instances
            elif name in ['pwd_context', 'signer', 'queue_manager', 'rate_limiter', 'redis_client', 'engine', 'AsyncSessionLocal']:
                self.infra[name] = CodeBlock(node=node, name=name, type='infra', content=content)
            # Identify the FastAPI app instance
            elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'FastAPI':
                self.app_setup['app_instance'] = content

    def _categorize_expression(self, node: ast.Expr):
        """Identifies calls like `app.add_middleware`."""
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            if isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'app':
                if node.value.func.attr == 'add_middleware':
                    self.app_setup['middleware'].append(self._get_node_content(node))

    # --- Step 2: Dependency Analysis ---

    def _analyze_all_dependencies(self):
        """Run dependency analysis on every categorized code block."""
        all_blocks = list(self.models.values()) + list(self.schemas.values()) + \
                     [r[1] for r in self.routes.values()] + list(self.utilities.values()) + \
                     list(self.infra.values())
        
        for block in all_blocks:
            visitor = DependencyVisitor(self.defined_names)
            visitor.visit(block.node)
            block.dependencies = visitor.dependencies

    # --- Step 3: Project File Generation ---

    def _generate_project_files(self) -> Dict[str, Any]:
        """Assembles the content for each file in the new project structure."""
        files: Dict[str, str] = {}
        templates: Dict[str, str] = {}

        # Organize routes into different API modules
        route_modules = self._group_routes()

        # Generate content for each file
        files['app/core/config.py'] = self._build_config_file()
        files['app/database/session.py'] = self._build_session_file()
        files['app/database/models.py'] = self._build_models_file()
        files['app/schemas.py'] = self._build_schemas_file()
        files['app/core/dependencies.py'] = self._build_dependencies_file()
        files['app/utils/security.py'] = self._build_security_utils_file()
        
        for module_path, routes_in_module in route_modules.items():
            if "pages" in module_path:
                # Page routes might have HTML to extract
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
        """Gathers all necessary import statements for a list of code blocks."""
        required_imports = set()
        for block in blocks:
            for dep in block.dependencies:
                if dep in self.imports:
                    required_imports.add(self.imports[dep])
        return '\n'.join(sorted(list(required_imports))) + '\n\n'

    def _build_config_file(self) -> str:
        # Build the core/config.py file using Pydantic settings
        content = "from pydantic_settings import BaseSettings\n\n\nclass Settings(BaseSettings):\n"
        for name, assign_str in self.config.items():
            var_type = "str" # Default type
            if "int(" in assign_str: var_type = "int"
            if ".lower() == 'true'" in assign_str: var_type = "bool"
            content += f"    {name}: {var_type}\n"
        content += "\n    class Config:\n        env_file = '.env'\n\n\nsettings = Settings()\n"
        return content

    def _build_session_file(self) -> str:
        # Build the database/session.py file
        db_infra = [b for n, b in self.infra.items() if n in ['engine', 'AsyncSessionLocal']]
        db_utils = [b for n, b in self.utilities.items() if n in ['get_db', 'init_db']]
        all_blocks = db_infra + db_utils
        
        imports = self._collect_imports_for_blocks(all_blocks)
        imports += "from app.core.config import settings\n" # Add settings import
        
        content = "# Database session management\n\n" + imports
        content += "\n".join(b.content for b in all_blocks)
        return content

    def _build_models_file(self) -> str:
        # Build the database/models.py file
        imports = self._collect_imports_for_blocks(list(self.models.values()))
        # Ensure Base is defined
        base_def = "from sqlalchemy.orm import declarative_base\n\nBase = declarative_base()\n\n"
        content = "# SQLAlchemy ORM Models\n\n" + imports + base_def
        content += "\n\n".join(b.content for b in self.models.values())
        return content

    def _build_schemas_file(self) -> str:
        # Build the schemas.py file
        imports = self._collect_imports_for_blocks(list(self.schemas.values()))
        content = "# Pydantic Schemas\n\n" + imports
        content += "\n\n".join(b.content for b in self.schemas.values())
        return content

    def _build_dependencies_file(self) -> str:
        # Build a file for shared dependencies like pwd_context, signer, etc.
        dep_blocks = [b for n, b in self.infra.items() if n in ['pwd_context', 'signer']]
        imports = self._collect_imports_for_blocks(dep_blocks)
        content = "# Core application dependencies\n\n" + imports
        content += "\n\n".join(b.content for b in dep_blocks)
        return content

    def _build_security_utils_file(self) -> str:
        # Build a file for security-related utility functions
        sec_utils = [b for n, b in self.utilities.items() if 'password' in n or 'token' in n or 'session' in n]
        imports = self._collect_imports_for_blocks(sec_utils)
        imports += "from app.core.dependencies import pwd_context, signer\n" # These are often needed
        content = "# Security-related utility functions\n\n" + imports
        content += "\n\n".join(b.content for b in sec_utils)
        return content

    def _group_routes(self) -> Dict[str, list]:
        """Group routes into logical modules based on their URL path."""
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
                modules['app/api/users.py'].append(name) # Default for other user-related endpoints
        return modules

    def _build_api_routes_file(self, module_path: str, route_names: List[str]) -> str:
        """Builds a standard API route file with a router."""
        route_blocks = [self.routes[name][1] for name in route_names]
        imports = self._collect_imports_for_blocks(route_blocks)
        
        content = f"# API routes for {module_path.split('/')[-1].replace('.py','')}\n\n"
        content += "from fastapi import APIRouter\n\n"
        content += imports
        content += "router = APIRouter()\n\n"
        # Replace @app decorators with @router
        content += "\n\n".join(b.content.replace('@app.', '@router.') for b in route_blocks)
        return content

    def _build_page_routes_file(self, route_names: List[str]) -> Tuple[str, Dict[str, str]]:
        """Builds the file for HTML-serving routes and extracts templates."""
        route_blocks = [self.routes[name][1] for name in route_names]
        imports = self._collect_imports_for_blocks(route_blocks)
        templates = {}

        content = "# Page-serving routes\n\n"
        content += "from fastapi import APIRouter, Request\n"
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
            
            # --- HTML Extraction Logic ---
            visitor = HTMLResponseVisitor()
            visitor.visit(code_block.node)
            if visitor.html_content:
                # Convert f-string expressions to Jinja2 variables
                template_content, context_vars = self._convert_fstring_to_jinja(visitor.html_content)
                template_name = f"{name}.html"
                templates[template_name] = template_content
                
                # Rewrite the function to use TemplateResponse
                new_func_def = f"async def {name}(request: Request):\n"
                context_str = f"context = {{'request': request, {', '.join(f'\"{v}\": {v}' for v in context_vars)}}}}\n"
                return_str = f"return templates.TemplateResponse('{template_name}', context)\n"
                
                # Reconstruct the function with decorators
                new_content = "\n".join(code_block.decorators) + '\n'
                new_content += new_func_def
                new_content += "    # This function was auto-refactored to use a Jinja2 template.\n"
                new_content += f"    # Original logic needs to be moved here to define context variables: {list(context_vars)}\n"
                new_content += "    # Example context variables (replace with actual logic):\n"
                for var in context_vars:
                    new_content += f"    {var} = '{{'Sample Data'}}' # TODO: Replace with actual data\n"
                new_content += '    ' + context_str
                new_content += '    ' + return_str
                content += new_content.replace('@app.', '@router.') + '\n\n'
            else:
                # Couldn't extract, so just add the original function
                content += code_block.content.replace('@app.', '@router.') + '\n\n'

        return content, templates

    def _convert_fstring_to_jinja(self, fstring_content: str) -> Tuple[str, Set[str]]:
        """A simple converter from Python f-string syntax to Jinja2 syntax."""
        context_vars = set()
        
        def replacer(match):
            expression = match.group(1).strip()
            # Simple expressions can be kept as is
            # Complex expressions would need more logic
            context_vars.add(expression.split('[')[0].split('.')[0]) # Get the base variable name
            return f"{{{{ {expression} }}}}"

        # Remove the outer f"", f'', etc.
        if fstring_content.startswith('f"') and fstring_content.endswith('"'):
            fstring_content = fstring_content[2:-1]
        elif fstring_content.startswith("f'") and fstring_content.endswith("'"):
            fstring_content = fstring_content[2:-1]
            
        template_content = re.sub(r'\{([^}]+)\}', replacer, fstring_content)
        return template_content, context_vars

    def _build_main_file(self, route_modules: Dict[str, list]) -> str:
        # Generates the main.py entrypoint file
        # Imports routers from their new locations
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

# --- Auto-generated router imports ---
{router_imports}

app = FastAPI(title="StreamBeatz Refactored", version="1.0.0")

# --- Middleware setup ---
{"\n".join(self.app_setup['middleware'])}

# --- Event handlers ---
@app.on_event("startup")
async def startup_event():
    await init_db()
    {"\n    ".join(self.app_setup.get('startup', []))}
    print("Application startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    {"\n    ".join(self.app_setup.get('shutdown', []))}
    print("Application shutdown.")

# --- Router includes ---
{router_includes}

@app.get("/health", tags=["Health"])
def health_check():
    return {{"status": "ok"}}
"""
        return main_content
    
    def _build_requirements_file(self) -> str:
        # A simplified requirements generator based on common packages.
        # A more robust solution would inspect all import statements.
        return "\n".join([
            "fastapi", "uvicorn[standard]", "sqlalchemy[asyncpg]", "pydantic",
            "pydantic-settings", "python-multipart", "passlib[bcrypt]", "httpx",
            "redis", "stripe", "jinja2", "email-validator", "python-dotenv", "qrcode[pil]"
        ])

    def _build_env_example_file(self) -> str:
        # Creates a .env.example from the detected config variables
        content = "# Environment variables for the application\n\n"
        for name, assign_str in self.config.items():
            match = re.search(r"os\.getenv\(['\"]([^'\"]+)['\"]", assign_str)
            if match:
                default_val = re.search(r",\s*['\"]?([^'\"]+)['\"]?\)", assign_str)
                default = f'"{default_val.group(1)}"' if default_val else ""
                content += f"{match.group(1)}={default}\n"
        return content

# --- AST Visitor for Dependency Analysis ---

class DependencyVisitor(ast.NodeVisitor):
    """
    An AST visitor that walks a code block (like a function) and records all
    the names it uses, which represent its dependencies.
    """
    def __init__(self, global_scope_names: Set[str]):
        self.dependencies = set()
        self.global_scope = global_scope_names
        self.local_scope = set()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Add function arguments to the local scope so we don't count them as dependencies.
        self.local_scope.update(arg.arg for arg in node.args.args)
        self.generic_visit(node) # Continue traversal into the function body

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.local_scope.update(arg.arg for arg in node.args.args)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        # This is the core logic. If a name is used that isn't defined locally,
        # and it exists in the global scope of the original file, it's a dependency.
        if node.id not in self.local_scope and node.id in self.global_scope:
            self.dependencies.add(node.id)

    def visit_Assign(self, node: ast.Assign):
        # Add newly assigned variables to the local scope.
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.local_scope.add(target.id)
        self.generic_visit(node)

# --- AST Visitor for HTML Extraction ---

class HTMLResponseVisitor(ast.NodeVisitor):
    """Finds and extracts the content from an HTMLResponse call inside a function."""
    def __init__(self):
        self.html_content = None

    def visit_Return(self, node: ast.Return):
        if (isinstance(node.value, ast.Call) and
            isinstance(node.value.func, ast.Name) and
            node.value.func.id == 'HTMLResponse'):
            # Found it. Now get the content. It's likely an f-string (JoinedStr).
            if node.value.args and isinstance(node.value.args[0], ast.JoinedStr):
                # We need to reconstruct the f-string source
                start = node.value.args[0].lineno - 1
                end = node.value.args[0].end_lineno
                self.html_content = "f'''" + "\n".join(self.lines[start:end]) + "'''"


# --- FastAPI Application ---

app = FastAPI(title="FastAPI Monolith Refactoring Engine")
# (HTML_TEMPLATE and other FastAPI setup code would go here, identical to the user's provided code)
# ... The HTML_TEMPLATE and FastAPI routes from the user's request go here ...
# For brevity, I'll omit the web UI code which is already correct.

@app.post("/refactor")
async def refactor_endpoint(file: UploadFile = File(...)):
    """The main API endpoint that receives the file and triggers the refactoring."""
    if not file.filename.endswith('.py'):
        return JSONResponse({"error": "Only .py files are supported"}, status_code=400)
    
    source_code = (await file.read()).decode('utf-8')
    original_lines = len(source_code.split('\n'))

    try:
        refactorer = MonolithRefactorer(source_code)
        project = refactorer.refactor()
        
        # Create a temporary directory to build the zip file
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / "streambeatz_refactored"
            
            # Write python files
            for rel_path, content in project.get("files", {}).items():
                path = project_root / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding='utf-8')
            
            # Write template files
            templates_dir = project_root / "templates"
            templates_dir.mkdir(exist_ok=True)
            for name, content in project.get("templates", {}).items():
                (templates_dir / name).write_text(content, encoding='utf-8')
            
            # Add __init__.py files
            for d in project_root.rglob(''):
                if d.is_dir() and not list(d.glob('__init__.py')):
                    (d / '__init__.py').touch()
            
            # Create a simple README
            (project_root / "README.md").write_text("# Refactored Project\n\nThis project was automatically refactored.")

            # Generate file structure for preview
            structure_preview = []
            for path in sorted(project_root.rglob('*')):
                depth = len(path.relative_to(project_root).parts) - 1
                indent = '  ' * depth
                structure_preview.append(f"{indent}{'|-- ' if depth > 0 else ''}{path.name}{'/' if path.is_dir() else ''}")

            # Create the zip file
            zip_filename = f"refactored_{int(time.time())}"
            zip_path = Path(tempfile.gettempdir()) / zip_filename
            shutil.make_archive(str(zip_path), 'zip', project_root)

            # Return the result
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
    """Serves the generated zip file for download."""
    path = Path(tempfile.gettempdir()) / filename
    if path.exists():
        return FileResponse(path, media_type='application/zip', filename='streambeatz_refactored.zip')
    return JSONResponse({"error": "File not found or expired"}, status_code=404)

# Add a simple root endpoint with the UI
@app.get("/", response_class=HTMLResponse)
async def root():
    # The HTML_TEMPLATE from the user's prompt would be returned here
    return """
    <!-- The full 300+ line HTML_TEMPLATE from the prompt goes here -->
    <!DOCTYPE html>...
    """

# Main entry point for running the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)