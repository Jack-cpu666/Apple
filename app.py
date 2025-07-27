import os
import re
import shutil
import zipfile
from flask import Flask, request, render_template_string, send_from_directory, after_this_request
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- HTML Template for the Web UI ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Monolith Code Refactor Engine</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #111827; color: #d1d5db; font-family: 'Inter', sans-serif; }
        .container { max-width: 800px; margin: auto; padding: 2rem; }
        .card { background-color: #1f2937; border: 1px solid #374151; border-radius: 0.75rem; }
        .btn {
            display: inline-block;
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            transition: background-color 0.2s;
            cursor: pointer;
        }
        .btn-primary { background-color: #4f46e5; color: white; }
        .btn-primary:hover { background-color: #4338ca; }
        .file-input-label {
            display: block;
            padding: 2rem;
            border: 2px dashed #4b5567;
            border-radius: 0.5rem;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.2s, background-color 0.2s;
        }
        .file-input-label:hover { border-color: #6366f1; background-color: #374151; }
        #spinner { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card p-8 text-center">
            <h1 class="text-3xl font-bold text-white mb-2">Monolith Refactor Engine</h1>
            <p class="mb-6">Upload your single-file FastAPI script to automatically restructure it into a scalable project.</p>
            
            <form id="uploadForm" action="/upload" method="post" enctype="multipart/form-data">
                <label for="file-upload" class="file-input-label" id="drop-zone">
                    <svg class="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                        <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
                    </svg>
                    <span class="mt-2 block font-medium" id="file-name-display">Click to upload or drag & drop</span>
                    <span class="block text-xs text-gray-500">Python file (.py)</span>
                </label>
                <input id="file-upload" name="file" type="file" class="sr-only" accept=".py">
                <button type="submit" class="btn btn-primary w-full mt-6">
                    <span id="button-text">Refactor Code</span>
                    <div id="spinner" role="status" class="inline-block h-4 w-4 animate-spin rounded-full border-2 border-solid border-current border-r-transparent align-[-0.125em] motion-reduce:animate-[spin_1.5s_linear_infinite]"></div>
                </button>
            </form>
        </div>
    </div>

<script>
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('file-upload');
    const dropZone = document.getElementById('drop-zone');
    const fileNameDisplay = document.getElementById('file-name-display');
    const buttonText = document.getElementById('button-text');
    const spinner = document.getElementById('spinner');

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            fileNameDisplay.textContent = fileInput.files[0].name;
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-indigo-500', 'bg-gray-700');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-indigo-500', 'bg-gray-700');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-indigo-500', 'bg-gray-700');
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            fileNameDisplay.textContent = fileInput.files[0].name;
        }
    });

    form.addEventListener('submit', () => {
        buttonText.style.display = 'none';
        spinner.style.display = 'inline-block';
    });
</script>
</body>
</html>
"""

# --- Core Refactoring Logic ---

# Define the project structure and content mapping
# Keys are file paths, values are lists of identifiers (class/function names or regex patterns)
STRUCTURE_MAP = {
    # Core Application
    'app/core/config.py': [r'^BASE_URL =', r'^STRIPE_SECRET_KEY =', r'^STRIPE_WEBHOOK_SECRET =', r'^SPOTIFY_CLIENT_ID =', r'^SPOTIFY_CLIENT_SECRET =', r'^TWITCH_CLIENT_ID =', r'^TWITCH_CLIENT_SECRET =', r'^DISCORD_CLIENT_ID =', r'^DISCORD_CLIENT_SECRET =', r'^DISCORD_REDIRECT_URI =', r'^SESSION_SECRET =', r'^DATABASE_URL =', r'^REDIS_URL =', r'^DEBUG =', r'^FREE_REQUEST_LIMIT =', r'^FREE_REQUEST_WINDOW_SECONDS =', r'^ADMIN_PASSWORD =', r'^YOUTUBE_API_KEY ='],
    # Database
    'app/database/models.py': [r'class \w+\(Base\):'],
    'app/database/session.py': [r'^engine =', r'^AsyncSessionLocal =', r'^DB_AVAILABLE =', r'def get_db\('],
    # Schemas
    'app/schemas.py': [r'class \w+\(BaseModel\):'],
    # Utils
    'app/utils.py': ['def hash_password', 'def verify_password', 'def generate_token', 'def generate_referral_code', 'def create_session', 'def verify_session', 'def create_admin_session', 'def verify_admin_session', 'class QueueManager', 'class RateLimiter', 'def calculate_priority', 'def calculate_queue_eta', 'def create_notification', 'def check_achievements', 'def update_analytics'],
    # Services
    'app/services/external_apis.py': ['def get_twitch_token', 'def check_twitch_live_status', 'def get_spotify_token', 'def get_youtube_video_info', 'def extract_youtube_id'],
    'app/services/websockets.py': [r'@app\.websocket\("/ws/streamer/\{streamer_id\}"\)', r'@app\.websocket\("/ws/viewer/\{username\}"\)', 'def broadcast_to_streamer', 'def broadcast_to_viewers'],
    # API Routers
    'app/api/auth.py': [r'@app\.post\("/api/auth/register"\)', r'@app\.post\("/api/auth/login"\)', r'@app\.post\("/api/auth/logout"\)', r'@app\.get\("/auth/discord/login"\)', r'@app\.get\("/auth/discord/callback"\)', r'@app\.get\("/auth/spotify/login"\)', r'@app\.get\("/auth/spotify/callback"\)', r'@app\.post\("/api/spotify/disconnect"\)', 'def get_current_user', 'def require_user', 'def require_streamer'],
    'app/api/requests.py': [r'@app\.post\("/api/requests"\)', r'@app\.post\("/api/requests/\{request_id\}/vote"\)', r'@app\.post\("/api/requests/\{request_id\}/play"\)', r'@app\.post\("/api/requests/\{request_id\}/complete"\)', r'@app\.post\("/api/requests/\{request_id\}/skip"\)', r'@app\.delete\("/api/requests/\{request_id\}"\)', r'@app\.post\("/api/queue/clear"\)', r'@app\.get\("/api/queue/\{username\}"\)', r'@app\.get\("/api/viewer/\{username\}/history"\)'],
    'app/api/users.py': [r'@app\.post\("/api/settings"\)', r'@app\.post\("/api/profile"\)', r'@app\.post\("/api/users/\{user_id\}/follow"\)', r'@app\.delete\("/api/users/\{user_id\}/follow"\)', r'@app\.post\("/api/playlists"\)', r'@app\.post\("/api/playlists/\{playlist_id\}/tracks"\)', r'@app\.get\("/api/notifications"\)', r'@app\.post\("/api/notifications/\{notification_id\}/read"\)', r'@app\.post\("/api/user/sync-balance"\)'],
    'app/api/payments.py': [r'@app\.post\("/api/withdraw"\)', r'@app\.get\("/api/withdrawals"\)', r'@app\.post\("/api/stripe/webhook"'],
    'app/api/media.py': [r'@app\.get\("/api/spotify/search"\)', r'@app\.post\("/api/youtube/info"\)', r'@app\.get\("/api/qr/\{username\}"\)', r'@app\.post\("/api/audio/upload"\)', r'@app\.get\("/api/audio/stream/\{audio_id\}"\)', r'@app\.get\("/api/audio/list"\)', r'@app\.delete\("/api/audio/\{audio_id\}"\)', r'@app\.post\("/api/user/audio/upload"\)', r'@app\.get\("/api/user/audio/list"\)', r'@app\.get\("/api/played-songs"\)', r'@app\.post\("/api/played-songs"\)', r'@app\.delete\("/api/played-songs/\{song_id\}"\)', r'@app\.post\("/api/played-songs/clear"\)'],
    'app/api/admin.py': [r'@app\.post\("/api/admin/login"\)', r'@app\.post\("/api/admin/logout"\)', r'@app\.post\("/api/admin/withdrawals/\{withdrawal_id\}/approve"\)', r'@app\.post\("/api/admin/withdrawals/\{withdrawal_id\}/reject"\)', r'def get_admin_user', 'def check_admin_attempts', 'def log_admin_attempt', r'@app\.post\("/api/admin/fix-analytics/\{user_id\}"\)', r'@app\.post\("/api/admin/sync-balance/\{user_id\}"\)', r'@app\.post\("/api/admin/sync-all-balances"\)', r'@app\.get\("/api/admin/streamers-data"\)', r'@app\.get\("/api/admin/users-data"\)', r'@app\.get\("/api/admin/financial-data"\)', r'@app\.post\("/api/admin/test-payment/\{request_id\}"\)'],
    # Page Routers
    'app/pages/views.py': [r'@app\.get\("/"\)', r'@app\.get\("/login"\)', r'@app\.get\("/register"\)', r'@app\.get\("/onboarding"\)', r'@app\.get\("/dashboard"\)', r'@app\.get\("/streamer/dashboard"\)', r'@app\.get\("/viewer/dashboard"\)', r'@app\.get\("/admin/login"\)', r'@app\.get\("/admin"\)', r'@app\.get\("/\{username\}"\)', r'@app\.get\("/overlay/\{username\}"\)', r'@app\.get\("/features"\)', r'@app\.get\("/pricing"\)', r'@app\.get\("/api-docs"\)', r'@app\.get\("/changelog"\)', r'@app\.get\("/about"\)', r'@app\.get\("/blog"\)', r'@app\.get\("/careers"\)', r'@app\.get\("/contact"\)', r'@app\.get\("/privacy"\)', r'@app\.get\("/terms"\)', r'@app\.get\("/dmca"\)', r'@app\.get\("/cookies"\)', r'@app\.get\("/streamer/dashboard/Queue"\)', r'@app\.get\("/streamer/dashboard/Played-Songs"\)', r'@app\.get\("/streamer/dashboard/Analytics"\)', r'@app\.get\("/streamer/dashboard/Settings"\)', r'@app\.get\("/streamer/dashboard/Overlay"\)', r'@app\.get\("/streamer/dashboard/Payouts"\)', r'@app\.get\("/streamer/dashboard/Profile"\)'],
}

def get_indentation(line):
    return len(line) - len(line.lstrip())

def extract_code_block(start_line_index, lines):
    block = [lines[start_line_index]]
    initial_indent = get_indentation(lines[start_line_index])
    
    # Handle decorators
    current_index = start_line_index - 1
    while current_index >= 0 and lines[current_index].strip().startswith('@'):
        block.insert(0, lines[current_index])
        current_index -= 1

    # Find the end of the block
    for line in lines[start_line_index + 1:]:
        if line.strip() == "":
            block.append(line)
            continue
        indent = get_indentation(line)
        if indent <= initial_indent:
            break
        block.append(line)
    
    return "\n".join(block)

def extract_html_from_fstring(code_block):
    match = re.search(r'f"""\s*(<!DOCTYPE html>.*)"""', code_block, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_and_refactor():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    temp_dir = tempfile.mkdtemp()
    project_root = os.path.join(temp_dir, 'streambeatz_project')
    os.makedirs(project_root)

    try:
        monolith_content = file.read().decode('utf-8')
        lines = monolith_content.split('\n')
        
        # --- 1. Create directory structure ---
        logging.info("Creating project directory structure...")
        for path in STRUCTURE_MAP.keys():
            dir_path = os.path.dirname(os.path.join(project_root, path))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        os.makedirs(os.path.join(project_root, 'templates'))
        os.makedirs(os.path.join(project_root, 'static'))
        
        # --- 2. Extract and distribute code blocks ---
        logging.info("Extracting and distributing code blocks...")
        extracted_indices = set()
        code_map = {path: [] for path in STRUCTURE_MAP}
        
        for path, identifiers in STRUCTURE_MAP.items():
            for i, line in enumerate(lines):
                if i in extracted_indices:
                    continue
                for identifier in identifiers:
                    if re.search(identifier, line.strip()):
                        block = extract_code_block(i, lines)
                        code_map[path].append(block)
                        start_line = i - (len(block.split('\n')) - len(extract_code_block(i, lines).split('\n'))) # Account for decorators
                        end_line = start_line + len(block.split('\n'))
                        for j in range(start_line, end_line):
                            extracted_indices.add(j)
                        break
        
        # --- 3. Gather all imports ---
        logging.info("Gathering all import statements...")
        all_imports = []
        for line in lines:
            if line.strip().startswith(('import ', 'from ')):
                all_imports.append(line)
        all_imports = sorted(list(set(all_imports)))
        
        # --- 4. Write files ---
        logging.info("Writing refactored files...")
        
        # Handle special case for app/pages/views.py for Jinja2 setup
        page_view_imports = [
            "from fastapi import APIRouter, Request, Depends",
            "from fastapi.responses import HTMLResponse",
            "from fastapi.templating import Jinja2Templates",
            "from sqlalchemy.ext.asyncio import AsyncSession",
            "from ..database.session import get_db",
            "from ..api.auth import get_current_user, require_user, require_streamer",
            "import os" # For BASE_URL in template context
        ]

        # Write each file with necessary imports
        for path, blocks in code_map.items():
            file_path = os.path.join(project_root, path)
            with open(file_path, 'w', encoding='utf-8') as f:
                # Add base imports
                f.write("\n".join(all_imports) + "\n\n")

                if path == 'app/pages/views.py':
                    f.write("\n".join(page_view_imports) + "\n\n")
                    f.write("router = APIRouter()\n")
                    f.write("templates = Jinja2Templates(directory=\"templates\")\n\n")
                elif path.startswith('app/api/') or path.startswith('app/services/websockets'):
                    f.write("from fastapi import APIRouter\nrouter = APIRouter()\n\n")

                f.write("\n\n".join(blocks))
        
        # --- 5. Handle HTML extraction and route modification ---
        logging.info("Extracting HTML and updating view routes...")
        page_views_path = os.path.join(project_root, 'app/pages/views.py')
        with open(page_views_path, 'r+', encoding='utf-8') as f:
            content = f.read()
            # Replace decorators
            content = content.replace('@app.get(', '@router.get(')
            content = content.replace('@app.post(', '@router.post(')
            
            # Find routes returning HTML and modify them
            html_routes = re.findall(r'(@router\.get\(.*?\).*?)\n(async def .*?\(.*?\):\n.*?return f"""\s*(<!DOCTYPE html>.*?)"""\s*)', content, re.DOTALL)
            
            template_map = {
                r'"/"': "index.html",
                r'"/login"': "login.html",
                r'"/register"': "register.html",
                r'"/streamer/dashboard"': "streamer_dashboard.html",
                r'"/viewer/dashboard"': "viewer_dashboard.html",
                r'"/admin"': "admin_dashboard.html",
                r'"/\{username\}"': "streamer_public_page.html",
                r'"/overlay/\{username\}"': "overlay.html"
            }

            for full_decorator, full_function_body in html_routes:
                html_content = extract_html_from_fstring(full_function_body)
                if not html_content: continue
                
                template_name = "default.html"
                for pattern, name in template_map.items():
                    if re.search(pattern, full_decorator):
                        template_name = name
                        break
                
                # Write HTML to template file
                with open(os.path.join(project_root, 'templates', template_name), 'w', encoding='utf-8') as tf:
                    # Basic conversion from f-string to Jinja2
                    html_content = re.sub(r'\{([^}]+)\}', r'{{ \1 }}', html_content)
                    tf.write(html_content)

                # Modify python route to use TemplateResponse
                # This part is complex to automate perfectly, so we do a best-effort replacement
                context_vars = re.findall(r'{{ (.*?) }}', html_content)
                context_dict = ", ".join(f'"{var.split(".")[0]}": {var.split(".")[0]}' for var in context_vars if var != 'request' and not "'" in var and not '"' in var)
                
                new_return_statement = f'    return templates.TemplateResponse("{template_name}", {{"request": request, {context_dict}}})'
                modified_function = re.sub(r'return f""".*"""', new_return_statement, full_function_body, flags=re.DOTALL)
                
                content = content.replace(full_function_body, modified_function)
            
            f.seek(0)
            f.write(content)
            f.truncate()
            
        # --- 6. Create main.py, requirements.txt, etc. ---
        logging.info("Creating main entrypoint and supplementary files...")
        
        # main.py
        with open(os.path.join(project_root, 'main.py'), 'w', encoding='utf-8') as f:
            f.write("""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import auth, requests, users, payments, media, admin
from app.pages import views
from app.services import websockets
from app.database.session import init_db

app = FastAPI(title="StreamBeatz Refactored")

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(views.router, tags=["Pages"])
app.include_router(auth.router, prefix="/api", tags=["Authentication"])
app.include_router(requests.router, prefix="/api", tags=["Requests"])
app.include_router(users.router, prefix="/api", tags=["Users"])
app.include_router(payments.router, prefix="/api", tags=["Payments"])
app.include_router(media.router, prefix="/api", tags=["Media"])
app.include_router(admin.router, prefix="/api", tags=["Admin"])
app.include_router(websockets.router, tags=["WebSockets"])

@app.on_event("startup")
async def on_startup():
    await init_db()
    print("StreamBeatz server started...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
""")
        
        # requirements.txt
        with open(os.path.join(project_root, 'requirements.txt'), 'w', encoding='utf-8') as f:
            f.write("""fastapi
uvicorn[standard]
sqlalchemy[asyncio]
asyncpg
pydantic
httpx
passlib[bcrypt]
itsdangerous
stripe
qrcode[pil]
python-multipart
redis
email_validator
""")

        # run.py (simple runner)
        with open(os.path.join(project_root, 'run.py'), 'w', encoding='utf-8') as f:
            f.write("import uvicorn\n\nif __name__ == '__main__':\n    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)\n")
        
        # .env.example
        with open(os.path.join(project_root, '.env.example'), 'w', encoding='utf-8') as f:
            f.write("""
DATABASE_URL=postgresql+asyncpg://user:password@host:port/dbname
REDIS_URL=redis://localhost:6379/0
BASE_URL=http://localhost:8000
SESSION_SECRET=a_very_secret_key
ADMIN_PASSWORD=change_this_password
# Add other secrets like Stripe, Spotify, Twitch keys here
STRIPE_SECRET_KEY=
STRIPE_WEBHOOK_SECRET=
SPOTIFY_CLIENT_ID=
SPOTIFY_CLIENT_SECRET=
# ... etc
""")

        # --- 7. Zip the project ---
        logging.info("Zipping the final project...")
        zip_path = os.path.join(temp_dir, 'streambeatz_refactored_project.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(project_root):
                for file in files:
                    file_path = os.path.join(root, file)
                    archive_name = os.path.relpath(file_path, project_root)
                    zipf.write(file_path, archive_name)
                    
        @after_this_request
        def cleanup(response):
            try:
                shutil.rmtree(temp_dir)
                logging.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logging.error(f"Error cleaning up temp directory {temp_dir}: {e}")
            return response

        logging.info("Sending zip file for download.")
        return send_from_directory(temp_dir, 'streambeatz_refactored_project.zip', as_attachment=True)
        
    except Exception as e:
        logging.error(f"An error occurred during refactoring: {e}", exc_info=True)
        shutil.rmtree(temp_dir)
        return f"An error occurred: {e}", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))