# app.py
"""
AI Model Root-Cause Analyzer & Diagnostic Platform
Enterprise Web Application Backend Server
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from email.parser import BytesParser
from email.policy import default
import html
import io
import json
import os
import urllib.parse
from typing import Dict, Any, Optional

import pandas as pd

from analysis.model_analyzer import analyze_model
from analysis.data_quality import analyze_data_quality
from analysis.ai_explainer import AIExplainer
from analysis.visualization import generate_all_visualizations
from analysis.evidence import safe_primitive

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))

# Server session state
CURRENT_DATAFRAME: Optional[pd.DataFrame] = None
CURRENT_RESULT: Optional[Dict[str, Any]] = None
CURRENT_DATA_QUALITY: Optional[Dict[str, Any]] = None
CURRENT_FILENAME: Optional[str] = None
CURRENT_TARGET: Optional[str] = None
CURRENT_MODE: str = "auto"
CURRENT_ERROR: Optional[str] = None
CURRENT_AI_EXPLAINER: Optional[AIExplainer] = None
CURRENT_AI_REPORT: Optional[str] = None
CURRENT_FIX_SCRIPT: Optional[str] = None
CURRENT_VISUALIZATIONS: Optional[Dict[str, Any]] = None

# User Session & Cloud Vault Storage Directory
SESSION_DIR = os.path.join(BASE_DIR, "data", "sessions")
os.makedirs(SESSION_DIR, exist_ok=True)


# ============================================================
# SESSION & USER VAULT STORAGE HELPERS
# ============================================================

def sanitize_user_key(user_id: str) -> str:
    import re
    cleaned = re.sub(r'[^a-zA-Z0-9_\-@.]', '_', str(user_id or "anonymous").lower())
    return cleaned if cleaned else "anonymous"


def get_user_sessions_list(user_id: str) -> list:
    import json
    uid = sanitize_user_key(user_id)
    user_folder = os.path.join(SESSION_DIR, uid)
    if not os.path.exists(user_folder):
        return []
    sessions = []
    for fname in os.listdir(user_folder):
        if fname.endswith(".json"):
            fpath = os.path.join(user_folder, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append({
                        "session_id": fname[:-5],
                        "session_name": data.get("session_name") or data.get("filename") or fname[:-5],
                        "filename": data.get("filename", "dataset.csv"),
                        "task_type": data.get("task_type", "classification"),
                        "risk_score": data.get("risk_score", 0),
                        "risk_level": data.get("risk_level", "LOW"),
                        "saved_at": data.get("saved_at", ""),
                        "rows": data.get("rows", 0),
                        "columns_count": len(data.get("columns", [])),
                        "target": data.get("target", "")
                    })
            except Exception:
                continue
    sessions.sort(key=lambda s: s.get("saved_at", ""), reverse=True)
    return sessions


def save_user_session_payload(user_id: str, payload: dict) -> str:
    import datetime
    uid = sanitize_user_key(user_id)
    user_folder = os.path.join(SESSION_DIR, uid)
    os.makedirs(user_folder, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = payload.get("session_id") or f"diag_{timestamp}"
    payload["session_id"] = session_id
    payload["saved_at"] = datetime.datetime.now().isoformat()

    fpath = os.path.join(user_folder, f"{session_id}.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(safe_primitive(payload), f, indent=2, default=str)
    return session_id


def load_user_session_payload(user_id: str, session_id: str) -> Optional[dict]:
    uid = sanitize_user_key(user_id)
    sid = sanitize_user_key(session_id)
    fpath = os.path.join(SESSION_DIR, uid, f"{sid}.json")
    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def delete_user_session_payload(user_id: str, session_id: str) -> bool:
    uid = sanitize_user_key(user_id)
    sid = sanitize_user_key(session_id)
    fpath = os.path.join(SESSION_DIR, uid, f"{sid}.json")
    if os.path.exists(fpath):
        try:
            os.remove(fpath)
            return True
        except Exception:
            return False
    return False


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def reset_state():
    """Resets server session state."""
    global CURRENT_DATAFRAME, CURRENT_RESULT, CURRENT_DATA_QUALITY
    global CURRENT_FILENAME, CURRENT_TARGET, CURRENT_MODE, CURRENT_ERROR
    global CURRENT_AI_EXPLAINER, CURRENT_AI_REPORT, CURRENT_FIX_SCRIPT, CURRENT_VISUALIZATIONS

    CURRENT_DATAFRAME = None
    CURRENT_RESULT = None
    CURRENT_DATA_QUALITY = None
    CURRENT_FILENAME = None
    CURRENT_TARGET = None
    CURRENT_MODE = "auto"
    CURRENT_ERROR = None
    CURRENT_AI_EXPLAINER = None
    CURRENT_AI_REPORT = None
    CURRENT_FIX_SCRIPT = None
    CURRENT_VISUALIZATIONS = None


def load_uploaded_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    """Safely loads an uploaded CSV or Excel file into a pandas DataFrame."""
    lower = filename.lower()
    if lower.endswith(".csv") or lower.endswith(".txt"):
        for encoding in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
            try:
                return pd.read_csv(io.BytesIO(content), encoding=encoding)
            except Exception:
                continue
        return pd.read_csv(io.BytesIO(content))
    elif lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    else:
        # Default try CSV
        return pd.read_csv(io.BytesIO(content))


def smart_suggest_target(df: pd.DataFrame) -> str:
    """Infers the most likely target column from DataFrame column names."""
    candidates = [
        "default_status", "default", "churn", "target", "label", "class",
        "price", "salary", "outcome", "status", "fraud", "is_fraud",
        "survived", "diagnosis", "y", "output"
    ]
    col_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in col_map:
            return col_map[cand]
    # Default to the last column
    return df.columns[-1] if len(df.columns) > 0 else ""


def smart_suggest_mode(df: pd.DataFrame, target_col: str) -> str:
    """Infers whether task is classification or regression."""
    if not target_col or target_col not in df.columns:
        return "auto"
    s = df[target_col].dropna()
    if s.empty:
        return "auto"
    
    unique_count = s.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(s)

    if not is_numeric:
        return "classification"
    if unique_count <= 10:
        return "classification"
    return "regression"


def get_dataset_summary(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
    """Generates JSON-serializable dataset metadata and preview rows."""
    target = smart_suggest_target(df)
    mode = smart_suggest_mode(df, target)

    dtypes_dict = {}
    for col in df.columns:
        dt = str(df[col].dtype)
        if "int" in dt or "float" in dt:
            dtypes_dict[col] = "numeric"
        elif "datetime" in dt:
            dtypes_dict[col] = "datetime"
        elif "bool" in dt:
            dtypes_dict[col] = "boolean"
        else:
            dtypes_dict[col] = "categorical"

    preview_rows = df.head(10).fillna("").to_dict(orient="records")

    return {
        "status": "success",
        "filename": filename,
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": dtypes_dict,
        "suggested_target": target,
        "suggested_mode": mode,
        "preview": preview_rows,
    }


def parse_multipart(body: bytes, content_type: str):
    """Parses multipart/form-data request body."""
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    message = BytesParser(policy=default).parsebytes(header + body)

    fields = {}
    files = {}

    if not message.is_multipart():
        return fields, files

    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        if not disposition:
            continue

        params = dict(part.get_params(header="Content-Disposition"))
        name = params.get("name")
        filename = params.get("filename")
        payload = part.get_payload(decode=True) or b""

        if filename:
            files[name] = {
                "filename": filename,
                "content": payload,
                "content_type": part.get_content_type(),
            }
        elif name:
            text = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
            fields[name] = text

    return fields, files


# ============================================================
# HTTP REQUEST HANDLER
# ============================================================

class AppHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        """Clean terminal logging."""
        print(f"[{self.log_date_time_string()}] {format % args}")

    def send_json(self, data: Any, status: int = 200):
        """Helper to send JSON response with CORS headers."""
        payload = json.dumps(safe_primitive(data), indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_file_response(self, filepath: str, content_type: str):
        """Helper to serve static files."""
        resolved_path = filepath if os.path.isabs(filepath) else os.path.join(BASE_DIR, filepath)
        if not os.path.exists(resolved_path) and os.path.exists(filepath):
            resolved_path = filepath
        if os.path.exists(resolved_path):
            with open(resolved_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_json({"error": f"File '{filepath}' not found"}, 404)

    def send_download(self, content_type: str, filename: str, payload: bytes):
        """Helper to send downloadable attachments."""
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        """Handle CORS pre-flight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ========================================================
    # HEAD ENDPOINTS
    # ========================================================

    def do_HEAD(self):
        """Handle HEAD requests cleanly for search engine crawlers and health checkers."""
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        path_lower = path.lower().rstrip("/")

        # 1. Sitemap HEAD
        if path_lower in {"/sitemap.xml", "/sitemap", "/sitemaps.xml", "/sitemap_index.xml"}:
            sitemap_path = os.path.join(BASE_DIR, "sitemap.xml")
            if os.path.exists(sitemap_path):
                try:
                    with open(sitemap_path, "rb") as f:
                        content_len = len(f.read())
                except Exception:
                    content_len = 291
            else:
                content_len = 291
            self.send_response(200)
            self.send_header("Content-Type", "application/xml; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "public, max-age=3600")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(content_len))
            self.end_headers()
            return

        # 2. Robots.txt HEAD
        if path_lower == "/robots.txt":
            robots_path = os.path.join(BASE_DIR, "robots.txt")
            if os.path.exists(robots_path):
                try:
                    with open(robots_path, "rb") as f:
                        content_len = len(f.read())
                except Exception:
                    content_len = 95
            else:
                content_len = 95
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "public, max-age=3600")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(content_len))
            self.end_headers()
            return

        # 3. Static Web App Files HEAD
        if path in {"/", "/index.html"}:
            index_path = os.path.join(BASE_DIR, "index.html")
            content_len = os.path.getsize(index_path) if os.path.exists(index_path) else 0
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(content_len))
            self.end_headers()
            return

        if path in {"/healthz", "/api/health", "/health"}:
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            return

        self.send_response(200)
        self.end_headers()

    # ========================================================
    # GET ENDPOINTS
    # ========================================================

    def do_GET(self):
        global CURRENT_DATAFRAME, CURRENT_FILENAME, CURRENT_TARGET, CURRENT_MODE
        global CURRENT_RESULT, CURRENT_DATA_QUALITY, CURRENT_AI_REPORT, CURRENT_FIX_SCRIPT, CURRENT_AI_EXPLAINER

        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)
        path_lower = path.lower().rstrip("/")

        # 1. Static Web App Files & SEO routes
        if path in {"/", "/index.html"}:
            self.send_file_response("index.html", "text/html; charset=utf-8")
            return

        if path == "/styles.css":
            self.send_file_response("styles.css", "text/css; charset=utf-8")
            return

        if path == "/app.js":
            self.send_file_response("app.js", "application/javascript; charset=utf-8")
            return

        if path_lower == "/robots.txt":
            robots_path = os.path.join(BASE_DIR, "robots.txt")
            if os.path.exists(robots_path):
                try:
                    with open(robots_path, "rb") as f:
                        robots_content = f.read()
                except Exception:
                    robots_content = b"User-agent: *\nAllow: /\n\nSitemap: https://ai-model-root-cause-analyzer.onrender.com/sitemap.xml\n"
            else:
                robots_content = b"User-agent: *\nAllow: /\n\nSitemap: https://ai-model-root-cause-analyzer.onrender.com/sitemap.xml\n"

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "public, max-age=3600")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(robots_content)))
            self.end_headers()
            self.wfile.write(robots_content)
            return

        if path_lower in {"/sitemap.xml", "/sitemap", "/sitemaps.xml", "/sitemap_index.xml"}:
            sitemap_path = os.path.join(BASE_DIR, "sitemap.xml")
            if os.path.exists(sitemap_path):
                try:
                    with open(sitemap_path, "rb") as f:
                        sitemap_content = f.read()
                except Exception:
                    sitemap_content = (
                        '<?xml version="1.0" encoding="UTF-8"?>\n'
                        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
                        '  <url>\n'
                        '    <loc>https://ai-model-root-cause-analyzer.onrender.com/</loc>\n'
                        '    <lastmod>2026-09-19</lastmod>\n'
                        '    <changefreq>daily</changefreq>\n'
                        '    <priority>1.0</priority>\n'
                        '  </url>\n'
                        '</urlset>\n'
                    ).encode("utf-8")
            else:
                sitemap_content = (
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
                    '  <url>\n'
                    '    <loc>https://ai-model-root-cause-analyzer.onrender.com/</loc>\n'
                    '    <lastmod>2026-09-19</lastmod>\n'
                    '    <changefreq>daily</changefreq>\n'
                    '    <priority>1.0</priority>\n'
                    '  </url>\n'
                    '</urlset>\n'
                ).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/xml; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "public, max-age=3600")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(sitemap_content)))
            self.end_headers()
            self.wfile.write(sitemap_content)
            return

        if path in {"/manifest.json", "/site.webmanifest"}:
            self.send_file_response("manifest.json", "application/manifest+json; charset=utf-8")
            return

        # Health check endpoint for cloud load balancers & Render
        if path in {"/healthz", "/api/health", "/health"}:
            self.send_json({"status": "ok", "service": "AI-Model-Root-Cause-Analyzer"})
            return

        # 2. REST API: Current State
        if path == "/api/status":
            self.send_json({
                "has_dataset": CURRENT_DATAFRAME is not None,
                "has_analysis": CURRENT_RESULT is not None,
                "filename": CURRENT_FILENAME,
                "rows": len(CURRENT_DATAFRAME) if CURRENT_DATAFRAME is not None else 0,
                "columns": list(CURRENT_DATAFRAME.columns) if CURRENT_DATAFRAME is not None else [],
                "target": CURRENT_TARGET,
                "mode": CURRENT_MODE,
                "overall_risk": CURRENT_RESULT.get("overall_risk") if CURRENT_RESULT else None,
                "risk_score": CURRENT_RESULT.get("risk_score") if CURRENT_RESULT else None,
            })
            return

        # 3. REST API: Load Demo Dataset
        if path == "/api/sample":
            scenario_type = query.get("type", ["classification"])[0].lower()
            if scenario_type == "regression":
                sample_file = "continuous_regression_test_dataset.csv"
            else:
                sample_file = "random_test_dataset.csv"

            if not os.path.exists(sample_file):
                self.send_json({"error": f"Sample dataset '{sample_file}' not found on server."}, 404)
                return

            try:
                df = pd.read_csv(sample_file)
                CURRENT_DATAFRAME = df
                CURRENT_FILENAME = sample_file
                CURRENT_RESULT = None
                CURRENT_DATA_QUALITY = None
                CURRENT_TARGET = None
                CURRENT_MODE = "regression" if scenario_type == "regression" else "classification"
                CURRENT_AI_REPORT = None
                CURRENT_FIX_SCRIPT = None

                summary = get_dataset_summary(df, sample_file)
                self.send_json(summary)
                return
            except Exception as exc:
                self.send_json({"error": f"Failed to load sample dataset: {str(exc)}"}, 500)
                return

        # 4. REST API: Reset
        if path == "/api/reset":
            reset_state()
            self.send_json({"status": "success", "message": "State reset successfully."})
            return

        # 5. File Downloads
        if path == "/download/dataset.csv":
            if CURRENT_DATAFRAME is None:
                self.send_json({"error": "No dataset currently loaded."}, 404)
                return
            buffer = io.StringIO()
            CURRENT_DATAFRAME.to_csv(buffer, index=False)
            self.send_download("text/csv; charset=utf-8", CURRENT_FILENAME or "dataset.csv", buffer.getvalue().encode("utf-8-sig"))
            return

        if path in {"/download/summary.csv", "/download/analysis_summary.csv"}:
            if CURRENT_RESULT is None:
                self.send_json({"error": "No analysis available to export."}, 404)
                return
            
            perf = CURRENT_RESULT.get("model_performance", {})
            cv = CURRENT_RESULT.get("cross_validation", {})
            rows = [
                {"metric": "filename", "value": CURRENT_FILENAME},
                {"metric": "selected_model", "value": CURRENT_RESULT.get("selected_model", "")},
                {"metric": "task_type", "value": CURRENT_RESULT.get("task_type", "")},
                {"metric": "risk_score", "value": CURRENT_RESULT.get("risk_score", 0)},
                {"metric": "overall_risk", "value": CURRENT_RESULT.get("overall_risk", "")},
                {"metric": "cv_average", "value": cv.get("average_score", cv.get("mean", 0))},
            ]
            for k, v in perf.items():
                rows.append({"metric": f"metric_{k}", "value": v})
            
            buffer = io.StringIO()
            pd.DataFrame(rows).to_csv(buffer, index=False)
            self.send_download("text/csv; charset=utf-8", "analysis_summary.csv", buffer.getvalue().encode("utf-8-sig"))
            return

        if path in {"/download/analysis.json", "/download/full_analysis.json"}:
            if CURRENT_RESULT is None:
                self.send_json({"error": "No analysis available to export."}, 404)
                return
            payload = {
                "dataset_info": {
                    "filename": CURRENT_FILENAME,
                    "rows": len(CURRENT_DATAFRAME) if CURRENT_DATAFRAME is not None else 0,
                    "columns": list(CURRENT_DATAFRAME.columns) if CURRENT_DATAFRAME is not None else [],
                    "target": CURRENT_TARGET,
                    "mode": CURRENT_MODE,
                },
                "analysis": CURRENT_RESULT,
                "data_quality": CURRENT_DATA_QUALITY,
            }
            content = json.dumps(safe_primitive(payload), indent=2, default=str).encode("utf-8")
            self.send_download("application/json; charset=utf-8", "full_analysis.json", content)
            return

        if path in {"/download/fix_pipeline.py", "/download/remediation_script.py"}:
            if CURRENT_FIX_SCRIPT is None:
                if CURRENT_RESULT is not None:
                    if CURRENT_AI_EXPLAINER is None:
                        CURRENT_AI_EXPLAINER = AIExplainer()
                    CURRENT_FIX_SCRIPT = CURRENT_AI_EXPLAINER.generate_fix_script(CURRENT_RESULT, CURRENT_DATA_QUALITY)
                else:
                    self.send_json({"error": "No analysis available to export."}, 404)
                    return
            self.send_download("text/x-python; charset=utf-8", "fix_pipeline.py", CURRENT_FIX_SCRIPT.encode("utf-8"))
            return

        # User Saved Sessions List
        if path == "/api/user/sessions":
            query_params = urllib.parse.parse_qs(parsed_url.query)
            user_id = query_params.get("user_id", [""])[0] or query_params.get("email", [""])[0]
            if not user_id:
                self.send_json({"status": "error", "error": "Missing user_id parameter."}, 400)
                return
            sessions = get_user_sessions_list(user_id)
            self.send_json({"status": "success", "user_id": user_id, "sessions": sessions})
            return

        self.send_json({"error": f"Route '{path}' not found."}, 404)

    # ========================================================
    # POST ENDPOINTS
    # ========================================================

    def do_POST(self):
        global CURRENT_DATAFRAME, CURRENT_FILENAME, CURRENT_TARGET, CURRENT_MODE
        global CURRENT_RESULT, CURRENT_DATA_QUALITY, CURRENT_ERROR, CURRENT_AI_EXPLAINER
        global CURRENT_AI_REPORT, CURRENT_FIX_SCRIPT, CURRENT_VISUALIZATIONS

        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # 1. POST /api/upload: Upload Dataset File
        if path == "/api/upload":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("No file content received.")
                if length > 100 * 1024 * 1024:
                    raise ValueError("File exceeds maximum allowable size (100 MB).")

                body = self.rfile.read(length)
                content_type = self.headers.get("Content-Type", "")

                fields, files = parse_multipart(body, content_type)
                uploaded = files.get("dataset") or files.get("file")

                if not uploaded:
                    try:
                        data = json.loads(body.decode("utf-8", errors="replace"))
                        if "csv_text" in data:
                            filename = data.get("filename", "uploaded_dataset.csv")
                            df = pd.read_csv(io.StringIO(data["csv_text"]))
                        elif "rows" in data:
                            filename = data.get("filename", "uploaded_dataset.csv")
                            df = pd.DataFrame(data["rows"])
                        else:
                            raise ValueError("Please provide a CSV or Excel dataset file.")
                    except Exception:
                        raise ValueError("Please provide a valid CSV or Excel dataset file.")
                else:
                    filename = uploaded["filename"]
                    content = uploaded["content"]
                    df = load_uploaded_dataframe(filename, content)

                if df.empty:
                    raise ValueError("The uploaded dataset contains 0 observations (empty).")

                if len(df.columns) < 2:
                    raise ValueError("The dataset must contain at least 2 columns (features and target).")

                CURRENT_DATAFRAME = df
                CURRENT_FILENAME = filename
                CURRENT_RESULT = None
                CURRENT_DATA_QUALITY = None
                CURRENT_TARGET = None
                CURRENT_MODE = "auto"
                CURRENT_AI_REPORT = None
                CURRENT_FIX_SCRIPT = None
                CURRENT_VISUALIZATIONS = None

                summary = get_dataset_summary(df, filename)
                self.send_json(summary)
                return

            except Exception as exc:
                self.send_json({"status": "error", "error": str(exc)}, 400)
                return

        # 2. POST /api/analyze: Run Root-Cause Diagnostics
        if path == "/api/analyze":
            try:
                if CURRENT_DATAFRAME is None:
                    raise ValueError("No dataset loaded. Please upload a dataset first.")

                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length > 0 else b""
                content_type = self.headers.get("Content-Type", "")

                target = ""
                mode = "auto"

                if "application/json" in content_type:
                    try:
                        data = json.loads(body.decode("utf-8", errors="replace"))
                        target = data.get("target", "").strip()
                        mode = data.get("mode", "auto").strip().lower()
                    except Exception:
                        pass
                else:
                    fields = urllib.parse.parse_qs(body.decode("utf-8", errors="replace"))
                    target = fields.get("target", [""])[0].strip()
                    mode = fields.get("mode", ["auto"])[0].strip().lower()

                if not target:
                    target = smart_suggest_target(CURRENT_DATAFRAME)

                if target not in CURRENT_DATAFRAME.columns:
                    raise ValueError(f"Target column '{target}' not found in dataset columns.")

                if mode not in {"classification", "regression", "auto"}:
                    mode = "auto"

                CURRENT_TARGET = target
                CURRENT_MODE = mode

                print(f"[*] Starting diagnostic analysis for '{CURRENT_FILENAME}' (Target: {target}, Mode: {mode})...")

                # Run Deterministic ML Diagnostics
                result = analyze_model(CURRENT_DATAFRAME, target, analysis_type=mode)
                data_quality = result.get("data_quality") or analyze_data_quality(CURRENT_DATAFRAME)

                # Generate Plotly Visualizations
                visualizations = generate_all_visualizations(result)

                # Generate AI Explanations & Auto-Fix Script
                explainer = AIExplainer()
                ai_explanation = explainer.explain(result, data_quality)
                ai_summary = ai_explanation.get("executive_summary", "")
                fix_script = explainer.generate_fix_script(result, data_quality)

                CURRENT_RESULT = result
                CURRENT_DATA_QUALITY = data_quality
                CURRENT_VISUALIZATIONS = visualizations
                CURRENT_AI_REPORT = ai_summary
                CURRENT_FIX_SCRIPT = fix_script
                CURRENT_AI_EXPLAINER = explainer

                payload = {
                    "status": "success",
                    "filename": CURRENT_FILENAME,
                    "target": CURRENT_TARGET,
                    "mode": CURRENT_MODE,
                    "task_type": result.get("task_type", mode),
                    "rows": len(CURRENT_DATAFRAME),
                    "columns": list(CURRENT_DATAFRAME.columns),
                    "result": result,
                    "data_quality": data_quality,
                    "visualizations": visualizations,
                    "ai_summary": ai_summary,
                    "fix_script": fix_script,
                }

                self.send_json(payload)
                return

            except Exception as exc:
                print(f"[!] Analysis error: {exc}")
                self.send_json({"status": "error", "error": str(exc)}, 400)
                return

        # 3. POST /api/copilot/chat: AI Explainer Copilot
        if path == "/api/copilot/chat":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length > 0 else b""
                question = ""
                content_type = self.headers.get("Content-Type", "")

                if "application/json" in content_type:
                    try:
                        data = json.loads(body.decode("utf-8", errors="replace"))
                        question = data.get("question", "").strip()
                    except Exception:
                        pass
                else:
                    fields = urllib.parse.parse_qs(body.decode("utf-8", errors="replace"))
                    question = fields.get("question", [""])[0].strip()

                if not question:
                    raise ValueError("Please provide a question.")

                if CURRENT_RESULT is None:
                    raise ValueError("No active dataset analysis available. Please analyze a dataset first.")

                if CURRENT_AI_EXPLAINER is None:
                    CURRENT_AI_EXPLAINER = AIExplainer()

                reply = CURRENT_AI_EXPLAINER.ask_copilot(question, CURRENT_RESULT, CURRENT_DATA_QUALITY)
                self.send_json({
                    "status": "success",
                    "question": question,
                    "reply": reply,
                    "provider": CURRENT_AI_EXPLAINER.get_active_provider_name()
                })
                return

            except Exception as exc:
                self.send_json({"status": "error", "reply": f"Copilot Error: {str(exc)}"}, 400)
                return

        # 4. POST /api/auth/google: Google Sign-In & Verification
        if path == "/api/auth/google":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(body.decode("utf-8", errors="replace"))

                user_email = data.get("email", "").strip()
                user_name = data.get("name", "").strip() or (user_email.split("@")[0] if user_email else "Google User")
                user_picture = data.get("picture", "").strip()
                user_id = data.get("sub", "") or user_email

                # If credential JWT token is passed, parse claims safely
                credential = data.get("credential") or data.get("token")
                if credential and isinstance(credential, str) and "." in credential:
                    try:
                        import base64
                        parts = credential.split(".")
                        if len(parts) >= 2:
                            padded = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
                            jwt_payload = json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8", errors="replace"))
                            user_email = jwt_payload.get("email", user_email)
                            user_name = jwt_payload.get("name", user_name)
                            user_picture = jwt_payload.get("picture", user_picture)
                            user_id = jwt_payload.get("sub", user_id) or user_email
                    except Exception as jwt_err:
                        print(f"[!] JWT parse notice: {jwt_err}")

                if not user_email and not user_id:
                    raise ValueError("Google authentication requires an email or Google ID.")

                sessions = get_user_sessions_list(user_email or user_id)

                self.send_json({
                    "status": "success",
                    "user": {
                        "user_id": user_id,
                        "email": user_email,
                        "name": user_name,
                        "picture": user_picture,
                        "provider": "google"
                    },
                    "sessions": sessions,
                    "message": f"Authenticated successfully with Google Account ({user_email})."
                })
                return

            except Exception as exc:
                self.send_json({"status": "error", "error": f"Authentication failed: {str(exc)}"}, 400)
                return

        # 5. POST /api/user/save-session: Save Active Diagnostics to Google Cloud Vault
        if path == "/api/user/save-session":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(body.decode("utf-8", errors="replace"))

                user_id = data.get("user_id") or data.get("email", "").strip()
                if not user_id:
                    raise ValueError("You must be signed in with Google to save dataset diagnostics.")

                session_name = data.get("session_name", "").strip() or data.get("filename", "Dataset Analysis")

                save_payload = {
                    "session_name": session_name,
                    "filename": data.get("filename", CURRENT_FILENAME or "dataset.csv"),
                    "target": data.get("target", CURRENT_TARGET or "target"),
                    "mode": data.get("mode", CURRENT_MODE or "auto"),
                    "task_type": data.get("task_type") or (CURRENT_RESULT.get("task_type") if CURRENT_RESULT else "classification"),
                    "rows": data.get("rows") or (len(CURRENT_DATAFRAME) if CURRENT_DATAFRAME is not None else 0),
                    "columns": data.get("columns") or (list(CURRENT_DATAFRAME.columns) if CURRENT_DATAFRAME is not None else []),
                    "risk_score": data.get("risk_score") or (CURRENT_RESULT.get("risk_score") if CURRENT_RESULT else 0),
                    "risk_level": data.get("risk_level") or (CURRENT_RESULT.get("overall_risk") if CURRENT_RESULT else "LOW"),
                    "dataset": data.get("dataset"),
                    "result": data.get("result") or CURRENT_RESULT,
                    "data_quality": data.get("data_quality") or CURRENT_DATA_QUALITY,
                    "visualizations": data.get("visualizations") or CURRENT_VISUALIZATIONS,
                    "ai_summary": data.get("ai_summary") or CURRENT_AI_REPORT,
                    "fix_script": data.get("fix_script") or CURRENT_FIX_SCRIPT,
                }

                sid = save_user_session_payload(user_id, save_payload)
                updated_sessions = get_user_sessions_list(user_id)

                self.send_json({
                    "status": "success",
                    "session_id": sid,
                    "message": f"Analysis '{session_name}' securely saved to your Google cloud vault.",
                    "sessions": updated_sessions
                })
                return

            except Exception as exc:
                self.send_json({"status": "error", "error": f"Failed to save session: {str(exc)}"}, 400)
                return

        # 6. POST /api/user/load-session: Reload Saved Diagnostic Output
        if path == "/api/user/load-session":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(body.decode("utf-8", errors="replace"))

                user_id = data.get("user_id") or data.get("email", "").strip()
                session_id = data.get("session_id", "").strip()

                if not user_id or not session_id:
                    raise ValueError("Missing user_id or session_id.")

                session_data = load_user_session_payload(user_id, session_id)
                if not session_data:
                    raise ValueError(f"Saved session '{session_id}' not found.")

                # Populate server active memory with loaded session
                if session_data.get("result"):
                    CURRENT_RESULT = session_data.get("result")
                if session_data.get("data_quality"):
                    CURRENT_DATA_QUALITY = session_data.get("data_quality")
                if session_data.get("visualizations"):
                    CURRENT_VISUALIZATIONS = session_data.get("visualizations")
                if session_data.get("ai_summary"):
                    CURRENT_AI_REPORT = session_data.get("ai_summary")
                if session_data.get("fix_script"):
                    CURRENT_FIX_SCRIPT = session_data.get("fix_script")
                if session_data.get("filename"):
                    CURRENT_FILENAME = session_data.get("filename")
                if session_data.get("target"):
                    CURRENT_TARGET = session_data.get("target")

                self.send_json({
                    "status": "success",
                    "message": f"Successfully loaded session '{session_data.get('session_name')}'.",
                    "session": session_data
                })
                return

            except Exception as exc:
                self.send_json({"status": "error", "error": f"Failed to load session: {str(exc)}"}, 400)
                return

        # 7. POST /api/user/delete-session: Delete Saved Session
        if path == "/api/user/delete-session":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(body.decode("utf-8", errors="replace"))

                user_id = data.get("user_id") or data.get("email", "").strip()
                session_id = data.get("session_id", "").strip()

                if not user_id or not session_id:
                    raise ValueError("Missing user_id or session_id.")

                delete_user_session_payload(user_id, session_id)
                updated_sessions = get_user_sessions_list(user_id)

                self.send_json({
                    "status": "success",
                    "message": "Session deleted.",
                    "sessions": updated_sessions
                })
                return

            except Exception as exc:
                self.send_json({"status": "error", "error": f"Failed to delete session: {str(exc)}"}, 400)
                return

        self.send_json({"error": f"POST route '{path}' not found."}, 404)


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    display_host = "localhost" if HOST == "0.0.0.0" else HOST
    print("=" * 65)
    print("       AI MODEL ROOT-CAUSE ANALYZER & DIAGNOSTICS")
    print("=" * 65)
    print()
    print(f"Server URL: http://{display_host}:{PORT}")
    print(f"Local URL:  http://127.0.0.1:{PORT}")
    print()
    print("Endpoints:")
    print(f"  • Web App Dashboard: http://{display_host}:{PORT}/")
    print(f"  • Ingest / Upload:   POST http://{display_host}:{PORT}/api/upload")
    print(f"  • Run Diagnostics:   POST http://{display_host}:{PORT}/api/analyze")
    print(f"  • Demo Datasets:     GET  http://{display_host}:{PORT}/api/sample?type=classification|regression")
    print()
    print("Press CTRL+C to stop the server.")
    print()

    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Server stopped by user.")
    finally:
        server.server_close()