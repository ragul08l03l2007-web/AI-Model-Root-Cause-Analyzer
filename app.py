# app.py

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from email.parser import BytesParser
from email.policy import default
import html
import io
import json
import os
import urllib.parse

import pandas as pd

from analysis.model_analyzer import analyze_model
from analysis.data_quality import analyze_data_quality
from analysis.ai_explainer import AIExplainer
from analysis.visualization import generate_all_visualizations


HOST = "localhost"
PORT = 8000

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


# ============================================================
# HELPERS & TYPE NORMALIZATION
# ============================================================

def safe(value):
    if value is None:
        return ""
    return html.escape(str(value))


def number(value, default=0.0):
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def percent(value):
    val = number(value)
    if 0 <= val <= 1:
        return val * 100
    return val


def total_missing(values):
    if isinstance(values, dict):
        total = 0
        for value in values.values():
            try:
                total += int(value)
            except (TypeError, ValueError):
                pass
        return total
    try:
        return int(values)
    except (TypeError, ValueError):
        return 0


def total_outliers(values):
    if isinstance(values, dict):
        total = 0
        for value in values.values():
            try:
                total += int(value)
            except (TypeError, ValueError):
                pass
        return total
    try:
        return int(values)
    except (TypeError, ValueError):
        return 0


def record_row(record):
    if "row" in record:
        return record.get("row")
    if "dataset_row" in record:
        return record.get("dataset_row")
    return "N/A"


def confidence(record):
    if "confidence" in record:
        return percent(record.get("confidence", 0))
    if "confidence_percentage" in record:
        return number(record.get("confidence_percentage", 0))
    return 0.0


def error_type(record):
    if "error_type" in record:
        return str(record.get("error_type"))
    act = str(record.get("actual", ""))
    pred = str(record.get("predicted", ""))
    return f"Misclassification ({act} → {pred})"


def get_task_type(result):
    if not isinstance(result, dict):
        return CURRENT_MODE if CURRENT_MODE in {"classification", "regression"} else "classification"
    task_type = str(result.get("task_type", CURRENT_MODE)).strip().lower()
    if task_type in {"classification", "regression"}:
        return task_type
    return "classification"


def mode_label(mode):
    labels = {
        "classification": "Classification",
        "regression": "Regression",
        "auto": "Auto Detect",
    }
    return labels.get(str(mode).strip().lower(), "Auto Detect")


def parse_multipart(body, content_type):
    header = (
        f"Content-Type: {content_type}\r\n"
        f"MIME-Version: 1.0\r\n\r\n"
    ).encode("utf-8")

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
            charset = part.get_content_charset() or "utf-8"
            fields[name] = payload.decode(charset, errors="replace")

    return fields, files


def load_uploaded_dataframe(filename, content):
    extension = os.path.splitext(filename)[1].lower()

    if extension == ".csv":
        return pd.read_csv(io.BytesIO(content))

    if extension in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(io.BytesIO(content))
        except ImportError as exc:
            raise ValueError(
                "Excel support requires openpyxl. Install it with "
                "'.\\.venv\\Scripts\\python.exe -m pip install openpyxl'."
            ) from exc

    raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


def reset_state():
    global CURRENT_DATAFRAME
    global CURRENT_RESULT
    global CURRENT_DATA_QUALITY
    global CURRENT_FILENAME
    global CURRENT_TARGET
    global CURRENT_MODE
    global CURRENT_ERROR
    global CURRENT_AI_EXPLAINER
    global CURRENT_AI_REPORT
    global CURRENT_FIX_SCRIPT
    global CURRENT_VISUALIZATIONS

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


# ============================================================
# STYLING & BASE TEMPLATE
# ============================================================

BASE_CSS = """
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #f8fafc;
    color: #0f172a;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.5;
}

.header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%);
    color: white;
    padding: 30px 20px;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.15);
}

.header-inner {
    width: min(1280px, 95%);
    margin: auto;
}

.header h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.header p {
    margin: 8px 0 0;
    color: #bfdbfe;
    font-size: 15px;
}

.container {
    width: min(1280px, 95%);
    margin: 28px auto 60px;
}

.card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    margin-bottom: 20px;
}

.section-title {
    margin: 34px 0 14px;
    font-size: 19px;
    font-weight: 800;
    color: #1e293b;
    display: flex;
    align-items: center;
    gap: 8px;
}

.upload-card {
    padding: 32px;
}

.dropzone {
    display: block;
    border: 2px dashed #93c5fd;
    border-radius: 14px;
    padding: 40px 20px;
    text-align: center;
    background: #eff6ff;
    cursor: pointer;
    transition: all 0.2s ease;
}

.dropzone:hover {
    background: #dbeafe;
    border-color: #3b82f6;
}

.dropzone input {
    display: none;
}

.drop-title {
    font-size: 20px;
    font-weight: 800;
    color: #1e3a8a;
}

.drop-sub {
    margin-top: 8px;
    color: #64748b;
    font-size: 14px;
}

.file-name {
    margin-top: 16px;
    font-weight: 700;
    color: #2563eb;
    background: white;
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    border: 1px solid #bfdbfe;
}

.form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-top: 24px;
}

label {
    display: block;
    font-size: 14px;
    font-weight: 700;
    color: #334155;
    margin-bottom: 8px;
}

input[type=text],
select {
    width: 100%;
    padding: 12px 14px;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    background: white;
    font-size: 15px;
    color: #0f172a;
    transition: border-color 0.2s ease;
}

input[type=text]:focus,
select:focus {
    outline: none;
    border-color: #2563eb;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
}

.btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 11px 20px;
    border: none;
    border-radius: 10px;
    text-decoration: none;
    font-weight: 700;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.15s ease;
    gap: 6px;
}

.btn-primary {
    background: #2563eb;
    color: white;
}

.btn-primary:hover {
    background: #1d4ed8;
}

.btn-secondary {
    background: #e2e8f0;
    color: #1e293b;
}

.btn-secondary:hover {
    background: #cbd5e1;
}

.btn-row {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 20px;
    align-items: center;
}

.note {
    margin-top: 20px;
    padding: 16px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    color: #64748b;
    font-size: 13px;
    line-height: 1.6;
}

.info-box {
    margin-top: 20px;
    padding: 16px 20px;
    background: #f0fdf4;
    border-left: 4px solid #16a34a;
    border-radius: 8px;
    color: #14532d;
    font-size: 14px;
    line-height: 1.6;
}

.mode-info-box {
    margin-top: 20px;
    padding: 16px 20px;
    background: #eff6ff;
    border-left: 4px solid #2563eb;
    border-radius: 8px;
    color: #1e3a8a;
    font-size: 14px;
    line-height: 1.6;
}

.error {
    margin-bottom: 20px;
    padding: 16px 20px;
    background: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
    border-radius: 10px;
    font-weight: 700;
}

.success {
    margin-bottom: 20px;
    padding: 16px 20px;
    background: #f0fdf4;
    color: #166534;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    font-weight: 700;
}

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
}

.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.label {
    font-size: 13px;
    color: #64748b;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric {
    margin-top: 6px;
    font-size: 26px;
    font-weight: 800;
    color: #0f172a;
}

.sub {
    margin-top: 4px;
    font-size: 13px;
    color: #64748b;
}

.badge {
    display: inline-flex;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 800;
}

.badge-good {
    background: #dcfce7;
    color: #166534;
}

.badge-high {
    background: #fee2e2;
    color: #991b1b;
}

.badge-critical {
    background: #fdf2f8;
    color: #9d174d;
    border: 1px solid #fbcfe8;
}

.badge-medium {
    background: #fef3c7;
    color: #92400e;
}

.badge-info {
    background: #e0f2fe;
    color: #075985;
}

.table-wrap {
    overflow-x: auto;
    border-radius: 10px;
}

table {
    width: 100%;
    border-collapse: collapse;
    min-width: 650px;
}

th, td {
    text-align: left;
    padding: 12px 14px;
    border-bottom: 1px solid #e2e8f0;
}

th {
    background: #f8fafc;
    color: #475569;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

tr:hover td {
    background: #f8fafc;
}

.numbered-row {
    display: flex;
    gap: 14px;
    padding: 14px 0;
    border-bottom: 1px solid #e2e8f0;
    line-height: 1.5;
    align-items: flex-start;
}

.numbered-row:last-child {
    border-bottom: none;
}

.row-number {
    min-width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #eff6ff;
    color: #2563eb;
    border-radius: 50%;
    font-weight: 800;
    font-size: 12px;
}

.root-cause-card {
    padding: 18px 20px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 5px solid #2563eb;
    border-radius: 10px;
    margin-bottom: 14px;
}

.root-cause-card.critical {
    border-left-color: #9d174d;
    background: #fdf2f8;
}

.root-cause-card.high {
    border-left-color: #dc2626;
    background: #fff5f5;
}

.root-cause-card.medium {
    border-left-color: #d97706;
    background: #fffbeb;
}

.root-cause-card.low {
    border-left-color: #2563eb;
    background: #f8fafc;
}

/* Verification Engine Styling */
.verif-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}
.verif-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 12px;
    padding-bottom: 10px;
    border-bottom: 1px solid #f1f5f9;
}
.verif-metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin-bottom: 14px;
}
.verif-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 14px;
}
.verif-box-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 4px;
}
.verif-box-val {
    font-size: 18px;
    font-weight: 800;
    color: #0f172a;
}
.verif-box-sub {
    font-size: 12px;
    color: #64748b;
    margin-top: 2px;
}
.verif-evidence-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    align-items: center;
    margin-bottom: 10px;
    font-size: 13px;
    color: #334155;
}
.remed-card {
    background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
    border: 1px solid #bbf7d0;
    border-left: 6px solid #16a34a;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 24px;
    box-shadow: 0 4px 16px rgba(22, 163, 74, 0.06);
}

/* AI Explainer & Copilot Styling */
.ai-card {
    background: linear-gradient(135deg, #f8faff 0%, #ffffff 100%);

    border: 1px solid #c7d2fe;
    border-left: 6px solid #4f46e5;
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 4px 20px rgba(79, 70, 229, 0.08);
    margin-bottom: 24px;
}

.ai-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid #e0e7ff;
}

.ai-title {
    font-size: 20px;
    font-weight: 800;
    color: #312e81;
    display: flex;
    align-items: center;
    gap: 8px;
}

.ai-summary-text {
    font-size: 15px;
    line-height: 1.6;
    color: #1e1b4b;
    margin-bottom: 16px;
}

.roadmap-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-top: 14px;
}

.roadmap-item {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 14px;
    color: #334155;
    display: flex;
    align-items: flex-start;
    gap: 10px;
}

.roadmap-num {
    background: #4f46e5;
    color: white;
    font-weight: 700;
    font-size: 12px;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.code-drawer {
    margin-top: 18px;
    border: 1px solid #334155;
    border-radius: 10px;
    background: #0f172a;
    overflow: hidden;
}

.code-drawer-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #1e293b;
    padding: 10px 16px;
    color: #94a3b8;
    font-size: 13px;
    font-weight: 600;
}

.code-pre {
    margin: 0;
    padding: 16px;
    color: #38bdf8;
    background: #0f172a;
    font-family: Consolas, Monaco, "Courier New", monospace;
    font-size: 13px;
    line-height: 1.5;
    overflow-x: auto;
    max-height: 400px;
}

/* Copilot Chat UI */
.copilot-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    overflow: hidden;
    margin-bottom: 24px;
}

.copilot-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    color: white;
    padding: 18px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.chat-history {
    padding: 20px;
    min-height: 240px;
    max-height: 450px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 14px;
    background: #f8fafc;
}

.chat-bubble {
    max-width: 85%;
    padding: 12px 16px;
    border-radius: 12px;
    font-size: 14px;
    line-height: 1.5;
}

.chat-bubble.ai {
    background: white;
    border: 1px solid #e2e8f0;
    color: #0f172a;
    align-self: flex-start;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
}

.chat-bubble.user {
    background: #4f46e5;
    color: white;
    align-self: flex-end;
}

.chips-row {
    padding: 12px 20px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    background: #f1f5f9;
    border-top: 1px solid #e2e8f0;
}

.prompt-chip {
    background: white;
    border: 1px solid #cbd5e1;
    border-radius: 16px;
    padding: 5px 12px;
    font-size: 12px;
    font-weight: 600;
    color: #334155;
    cursor: pointer;
    transition: all 0.15s ease;
}

.prompt-chip:hover {
    background: #e0e7ff;
    border-color: #818cf8;
    color: #312e81;
}

.chat-input-row {
    display: flex;
    gap: 10px;
    padding: 16px 20px;
    background: white;
    border-top: 1px solid #e2e8f0;
}

.chat-input {
    flex: 1;
    padding: 10px 14px;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    font-size: 14px;
    outline: none;
}

.chat-input:focus {
    border-color: #4f46e5;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 32px 0;
    font-size: 13px;
}

@media(max-width:768px) {
    .form-row {
        grid-template-columns: 1fr;
    }
    .header h1 {
        font-size: 24px;
    }
    .chart-grid {
        grid-template-columns: 1fr !important;
    }
}

/* Interactive Visual Charts & Tabs */
.chart-tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 20px;
    flex-wrap: wrap;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 12px;
}

.chart-tab-btn {
    background: #f1f5f9;
    color: #475569;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s ease;
}

.chart-tab-btn:hover {
    background: #e2e8f0;
    color: #0f172a;
}

.chart-tab-btn.active {
    background: #2563eb;
    color: white;
    border-color: #1d4ed8;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
}

.chart-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
    gap: 20px;
}

.chart-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
    transition: all 0.2s ease;
}

.chart-box:hover {
    box-shadow: 0 4px 18px rgba(15, 23, 42, 0.08);
}

.chart-container {
    width: 100%;
    min-height: 280px;
}
"""

UPLOAD_JS = """
<script>
document.addEventListener("DOMContentLoaded", function() {
    var fileInput = document.getElementById("dataset");
    var fileName = document.getElementById("file-name");
    if (fileInput && fileName) {
        fileInput.addEventListener("change", function() {
            if (this.files && this.files.length > 0) {
                fileName.textContent = this.files[0].name;
            } else {
                fileName.textContent = "No file selected";
            }
        });
    }
});
</script>
"""

COPILOT_JS = """
<script>
function askCopilot(questionText) {
    var input = document.getElementById("copilot-input");
    var q = questionText || (input ? input.value.trim() : "");
    if (!q) return;

    var history = document.getElementById("chat-history");
    if (!history) return;

    // Append user bubble
    var userBubble = document.createElement("div");
    userBubble.className = "chat-bubble user";
    userBubble.textContent = q;
    history.appendChild(userBubble);

    if (input) input.value = "";

    // Append loading bubble
    var loadingBubble = document.createElement("div");
    loadingBubble.className = "chat-bubble ai";
    loadingBubble.innerHTML = "<em>Analyzing diagnostic evidence...</em>";
    history.appendChild(loadingBubble);
    history.scrollTop = history.scrollHeight;

    fetch("/api/copilot/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q })
    })
    .then(function(res) { return res.json(); })
    .then(function(data) {
        var reply = data.reply || "No response received.";
        // Format markdown bolding and newlines
        reply = reply.replace(/\\*\\*(.*?)\\*\\*/g, "<strong>$1</strong>");
        reply = reply.replace(/\\n/g, "<br>");
        loadingBubble.innerHTML = reply;
        history.scrollTop = history.scrollHeight;
    })
    .catch(function(err) {
        loadingBubble.innerHTML = "<span style='color:#dc2626;'>Error contacting Copilot. Please try again.</span>";
    });
}

function copyFixCode() {
    var codeElem = document.getElementById("fix-script-code");
    if (codeElem) {
        navigator.clipboard.writeText(codeElem.textContent).then(function() {
            var btn = document.getElementById("copy-code-btn");
            if (btn) {
                var oldText = btn.textContent;
                btn.textContent = "Copied!";
                setTimeout(function() { btn.textContent = oldText; }, 2000);
            }
        });
    }
}

function switchChartTab(cat, btnElem) {
    var buttons = document.querySelectorAll(".chart-tab-btn");
    buttons.forEach(function(b) { b.classList.remove("active"); });
    if (btnElem) btnElem.classList.add("active");

    var boxes = document.querySelectorAll(".chart-box");
    boxes.forEach(function(box) {
        if (cat === "all" || box.getAttribute("data-category") === cat) {
            box.style.display = "block";
            var childChart = box.querySelector(".chart-container");
            if (childChart && typeof Plotly !== "undefined") {
                Plotly.Plots.resize(childChart);
            }
        } else {
            box.style.display = "none";
        }
    });
}
</script>
"""


def page_start(title="AI Model Root-Cause Analyzer"):
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        '<meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate, max-age=0">\n'
        '<meta http-equiv="Pragma" content="no-cache">\n'
        '<meta http-equiv="Expires" content="0">\n'
        "<title>" + safe(title) + "</title>\n"
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>\n'
        "<style>\n" + BASE_CSS + "\n</style>\n"
        "</head>\n"
        "<body>\n"
        '<header class="header">\n'
        '<div class="header-inner">\n'
        "<h1>AI Model Root-Cause Analyzer</h1>\n"
        "<p>Evidence-Driven Machine Learning Diagnostics, Data-Quality Assessment &amp; Root-Cause Explanation</p>\n"
        "</div>\n"
        "</header>\n"
        '<div class="container">\n'
    )


def page_end():
    return (
        '<div class="footer">\n'
        "AI Model Root-Cause Analyzer &bull; Automated Diagnostic System\n"
        "</div>\n"
        "</div>\n"
        + COPILOT_JS + "\n"
        "</body>\n"
        "</html>\n"
    )


# ============================================================
# UPLOAD PAGE
# ============================================================

def upload_page(message=""):
    html_parts = [page_start("Upload Dataset - AI Model Root-Cause Analyzer")]

    if message:
        html_parts.append(
            '<div class="error">' + safe(message) + "</div>\n"
        )

    selected_mode = CURRENT_MODE if CURRENT_MODE in {"classification", "regression", "auto"} else "auto"

    upload_body = (
        '<div class="section-title">Start a New Analysis</div>\n'
        '<div class="card upload-card">\n'
        '<form method="POST" action="/upload" enctype="multipart/form-data">\n'
        '<label for="dataset">Dataset File (.csv, .xlsx, .xls)</label>\n'
        '<label class="dropzone" for="dataset">\n'
        '<input id="dataset" name="dataset" type="file" accept=".csv,.xlsx,.xls" required>\n'
        '<div class="drop-title">Choose or drag a CSV or Excel dataset</div>\n'
        '<div class="drop-sub">Your file is processed locally on this machine with zero hardcoded assumptions.</div>\n'
        '<div id="file-name" class="file-name">No file selected</div>\n'
        '</label>\n'
        '<div class="form-row">\n'
        '<div>\n'
        '<label for="mode">Analysis Mode</label>\n'
        '<select id="mode" name="mode">\n'
        f'<option value="auto"{" selected" if selected_mode == "auto" else ""}>Auto Detect (Recommended)</option>\n'
        f'<option value="classification"{" selected" if selected_mode == "classification" else ""}>Classification</option>\n'
        f'<option value="regression"{" selected" if selected_mode == "regression" else ""}>Regression</option>\n'
        '</select>\n'
        '</div>\n'
        '<div>\n'
        '<label>&nbsp;</label>\n'
        '<div style="font-size:13px; color:#64748b; padding-top:10px;">'
        'Target column will be selected on the preview screen after profiling.'
        '</div>\n'
        '</div>\n'
        '</div>\n'
        '<div class="mode-info-box">\n'
        '<strong>Auto Detect:</strong> Dynamically determines whether target is classification or regression based on distinct values and data type.<br><br>\n'
        '<strong>Classification:</strong> Model discrete categorical or class-based targets.<br><br>\n'
        '<strong>Regression:</strong> Model continuous numeric targets with residual and error concentration diagnostics.\n'
        '</div>\n'
        '<div class="btn-row">\n'
        '<button class="btn btn-primary" type="submit">Upload &amp; Continue</button>\n'
        '</div>\n'
        '<div class="note">\n'
        'The analyzer performs deep dataset profiling, target analysis, multi-model evaluation, cross-validation stability checks, '
        'feature-target relationships, subgroup analysis, confusion patterns, error segment discovery, evidence-based root causes, '
        'and prescriptive recommendations.\n'
        '</div>\n'
        '</form>\n'
        '</div>\n'
        '<div class="section-title">Diagnostic Capabilities</div>\n'
        '<div class="grid">\n'
        '<div class="metric-card">\n'
        '<div class="label">Dataset Profiling</div>\n'
        '<div class="metric">✓</div>\n'
        '<div class="sub">Types, missing %, IQR outliers &amp; correlations</div>\n'
        '</div>\n'
        '<div class="metric-card">\n'
        '<div class="label">Model Comparison</div>\n'
        '<div class="metric">✓</div>\n'
        '<div class="sub">Multi-model evaluation &amp; CV stability</div>\n'
        '</div>\n'
        '<div class="metric-card">\n'
        '<div class="label">Feature Impact</div>\n'
        '<div class="metric">✓</div>\n'
        '<div class="sub">Non-causal interpretation &amp; quantile rates</div>\n'
        '</div>\n'
        '<div class="metric-card">\n'
        '<div class="label">Root Causes</div>\n'
        '<div class="metric">✓</div>\n'
        '<div class="sub">Evidence, impact &amp; actionable recommendations</div>\n'
        '</div>\n'
        '</div>\n'
        + UPLOAD_JS
    )

    html_parts.append(upload_body)
    html_parts.append(page_end())
    return "".join(html_parts)


# ============================================================
# PREVIEW PAGE
# ============================================================

def preview_page(df, filename, mode, target=None, error=None):
    columns = list(df.columns)
    mode_text = mode_label(mode)

    rows = []
    for _, row in df.head(8).iterrows():
        cells = "".join(f"<td>{safe(val)}</td>" for val in row.tolist())
        rows.append(f"<tr>{cells}</tr>")

    # Deep Column Profiling Table
    type_rows = []
    for column in columns:
        series = df[column]
        dtype = str(series.dtype)
        missing = int(series.isna().sum())
        missing_pct = (missing / len(df) * 100) if len(df) > 0 else 0.0
        nunique = int(series.nunique(dropna=True))
        
        # Inferred role
        if pd.api.types.is_numeric_dtype(series):
            if nunique <= 2 and set(series.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
                role = '<span class="badge badge-info">Boolean / Binary</span>'
            elif nunique <= 10:
                role = '<span class="badge badge-info">Discrete Numeric</span>'
            else:
                role = '<span class="badge badge-good">Continuous Numeric</span>'
        else:
            if nunique > 20 or (len(df) >= 50 and (nunique / len(df)) > 0.40):
                role = '<span class="badge badge-medium">High Cardinality</span>'
            else:
                role = '<span class="badge badge-good">Categorical</span>'

        missing_badge = f'<span class="badge badge-high">{missing:,} ({missing_pct:.1f}%)</span>' if missing > 0 else '<span class="badge badge-good">0 (0%)</span>'

        type_rows.append(
            f"<tr><td><strong>{safe(column)}</strong></td><td><code>{safe(dtype)}</code></td><td>{missing_badge}</td><td>{nunique:,}</td><td>{role}</td></tr>"
        )

    options = []
    for col in columns:
        is_selected = " selected" if (target is not None and str(target) == str(col)) else ""
        options.append(f'<option value="{safe(col)}"{is_selected}>{safe(col)}</option>')

    error_banner = ""
    if error:
        error_banner = f'<div class="error"><strong>Analysis Error:</strong> {safe(error)}</div>'

    html = (
        page_start("Dataset Preview - AI Model Root-Cause Analyzer")
        + f'<div class="success">Dataset loaded: <strong>{safe(filename)}</strong> &mdash; {len(df):,} rows &times; {len(df.columns)} columns.</div>\n'
        + error_banner
        + '<div class="section-title">Selected Analysis Configuration</div>\n'
        + '<div class="card">\n'
        + '<div class="grid">\n'
        + '<div class="metric-card">\n'
        + '<div class="label">Analysis Mode</div>\n'
        + f'<div class="metric" style="font-size:20px;">{safe(mode_text)}</div>\n'
        + '</div>\n'
        + '<div class="metric-card">\n'
        + '<div class="label">Total Rows</div>\n'
        + f'<div class="metric">{len(df):,}</div>\n'
        + '</div>\n'
        + '<div class="metric-card">\n'
        + '<div class="label">Total Columns</div>\n'
        + f'<div class="metric">{len(df.columns):,}</div>\n'
        + '</div>\n'
        + '<div class="metric-card">\n'
        + '<div class="label">Numeric Columns</div>\n'
        + f'<div class="metric">{len(df.select_dtypes(include="number").columns):,}</div>\n'
        + '</div>\n'
        + '</div>\n'
        + '</div>\n'
        + '<div class="section-title">Dataset Preview (First 8 Rows)</div>\n'
        + '<div class="card">\n'
        + '<div class="table-wrap">\n'
        + '<table>\n'
        + '<thead><tr>' + "".join(f"<th>{safe(c)}</th>" for c in columns) + '</tr></thead>\n'
        + '<tbody>' + "".join(rows) + '</tbody>\n'
        + '</table>\n'
        + '</div>\n'
        + '</div>\n'
        + '<div class="section-title">Column Profile &amp; Structural Discovery</div>\n'
        + '<div class="card table-wrap">\n'
        + '<table>\n'
        + '<thead><tr><th>Column Name</th><th>Data Type</th><th>Missing Count (%)</th><th>Unique Values</th><th>Inferred Role</th></tr></thead>\n'
        + '<tbody>' + "".join(type_rows) + '</tbody>\n'
        + '</table>\n'
        + '</div>\n'
        + '<div class="section-title">Choose Target Column &amp; Run Diagnostic Analysis</div>\n'
        + '<div class="card">\n'
        + '<form method="POST" action="/analyze">\n'
        + '<input type="hidden" name="mode" value="' + safe(mode) + '">\n'
        + '<label for="target">Target Column</label>\n'
        + '<select id="target" name="target" required>\n'
        + '<option value="">-- Select a target column --</option>\n'
        + "".join(options) + "\n"
        + '</select>\n'
        + '<div class="btn-row">\n'
        + '<button class="btn btn-primary" type="submit">Run Diagnostic Analysis</button>\n'
        + '<a class="btn btn-secondary" href="/back-to-upload">Back to Upload</a>\n'
        + '</div>\n'
        + '</form>\n'
        + '</div>\n'
        + page_end()
    )
    return html


# ============================================================
# DASHBOARD PAGE
# ============================================================

def dashboard_page():
    df = CURRENT_DATAFRAME
    result = CURRENT_RESULT
    quality = CURRENT_DATA_QUALITY or {}

    task_type = get_task_type(result)

    performance = result.get("model_performance", {})
    cv = result.get("cross_validation", {})
    stability = result.get("model_stability", {})
    errors = result.get("error_analysis", {})
    model_comparison = result.get("model_comparison", [])
    task_reason = result.get("task_reason", result.get("auto_detection_reason", ""))
    target_profile = result.get("target_profile", result.get("target_analysis", {}))

    risk = str(result.get("overall_risk", "UNKNOWN")).upper()
    risk_score = int(number(result.get("risk_score", 0)))

    risk_class = {
        "LOW": "badge-good",
        "MEDIUM": "badge-medium",
        "HIGH": "badge-high",
    }.get(risk, "badge-info")

    missing = total_missing(quality.get("missing_values", {}))
    outliers = total_outliers(quality.get("outliers", {}))

    # --------------------------------------------------------
    # 1. Feature Impact & Importance Table
    # --------------------------------------------------------
    feature_impact = result.get("feature_impact", {})
    feature_html = ""
    if isinstance(feature_impact, dict):
        feature_items = []
        for name, data in feature_impact.items():
            if CURRENT_TARGET is not None and str(name) == str(CURRENT_TARGET):
                continue
            if isinstance(data, dict):
                feature_items.append((
                    name,
                    number(data.get("importance", 0)),
                    number(data.get("relative_share_pct", 0)),
                    data.get("influence_tier", "Moderate Model Influence"),
                    data.get("direction", "")
                ))
            else:
                feature_items.append((name, number(data, 0), 0.0, "Moderate Model Influence", ""))
        feature_items.sort(key=lambda x: x[1], reverse=True)
        for rank, (name, imp, rel_share, tier, direct) in enumerate(feature_items, 1):
            share_text = f"{rel_share:.1f}%" if rel_share > 0 else "-"
            tier_class = "badge-high" if "Very Strong" in tier else ("badge-medium" if "Strong" in tier else "badge-info")
            feature_html += (
                f"<tr><td>#{rank}</td><td><strong>{safe(name)}</strong></td>"
                f"<td>{imp:.4f}</td><td>{share_text}</td>"
                f"<td><span class=\"badge {tier_class}\">{safe(tier)}</span></td>"
                f"<td>{safe(direct)}</td></tr>\n"
            )

    # --------------------------------------------------------
    # 2. Feature -> Target Empirical Relationships
    # --------------------------------------------------------
    feature_rel = result.get("feature_relationships", [])
    rel_cards_html = ""
    if feature_rel and isinstance(feature_rel, list):
        for f_rel in feature_rel[:4]:
            feat_name = f_rel.get("feature", "")
            f_type = f_rel.get("feature_type", "numeric")
            
            if f_type == "numeric" and "bins" in f_rel:
                b_rows = []
                for b in f_rel["bins"]:
                    if task_type == "regression":
                        b_rows.append(
                            f"<tr><td>{safe(b.get('bin'))}</td><td>{b.get('sample_count')}</td>"
                            f"<td>{b.get('target_mean', 0):.4f}</td><td>{b.get('target_median', 0):.4f}</td></tr>"
                        )
                    else:
                        b_rows.append(
                            f"<tr><td>{safe(b.get('bin'))}</td><td>{b.get('sample_count')}</td>"
                            f"<td><strong>{safe(b.get('dominant_class'))}</strong> ({b.get('dominant_class_pct')}%)</td></tr>"
                        )
                
                header_th = "<th>Target Mean</th><th>Target Median</th>" if task_type == "regression" else "<th>Dominant Class (%)</th>"
                rel_cards_html += (
                    f'<div style="margin-bottom:18px;">\n'
                    f'<strong style="color:#1e3a8a;">Feature: {safe(feat_name)}</strong> (Adaptive Quantile Segments)\n'
                    f'<table style="margin-top:6px;">\n'
                    f'<thead><tr><th>Value Range</th><th>Observations</th>{header_th}</tr></thead>\n'
                    f'<tbody>' + "".join(b_rows) + '</tbody>\n'
                    f'</table>\n'
                    f'</div>\n'
                )
            elif "categories" in f_rel:
                c_rows = []
                for c in f_rel["categories"]:
                    if task_type == "regression":
                        c_rows.append(
                            f"<tr><td>'{safe(c.get('category'))}'</td><td>{c.get('sample_count')}</td>"
                            f"<td>{c.get('target_mean', 0):.4f}</td><td>{c.get('target_median', 0):.4f}</td></tr>"
                        )
                    else:
                        c_rows.append(
                            f"<tr><td>'{safe(c.get('category'))}'</td><td>{c.get('sample_count')}</td>"
                            f"<td><strong>{safe(c.get('dominant_class'))}</strong> ({c.get('dominant_class_pct')}%)</td></tr>"
                        )
                header_th = "<th>Target Mean</th><th>Target Median</th>" if task_type == "regression" else "<th>Dominant Class (%)</th>"
                rel_cards_html += (
                    f'<div style="margin-bottom:18px;">\n'
                    f'<strong style="color:#1e3a8a;">Feature: {safe(feat_name)}</strong> (Top Categories)\n'
                    f'<table style="margin-top:6px;">\n'
                    f'<thead><tr><th>Category</th><th>Observations</th>{header_th}</tr></thead>\n'
                    f'<tbody>' + "".join(c_rows) + '</tbody>\n'
                    f'</table>\n'
                    f'</div>\n'
                )

    # --------------------------------------------------------
    # 3. Subgroup & Segment Analysis
    # --------------------------------------------------------
    segments = result.get("segment_analysis", [])
    segment_rows = []
    if segments and isinstance(segments, list):
        for seg in segments:
            segment_rows.append(
                f"<tr><td><strong>{safe(seg.get('segment_name'))}</strong></td>"
                f"<td>{seg.get('population')} ({seg.get('population_pct')}%)</td>"
                f"<td>{safe(seg.get('target_metric'))}: <strong>{safe(seg.get('segment_target_value'))}</strong> (baseline {safe(seg.get('baseline_target_value'))})</td>"
                f"<td>{safe(seg.get('key_difference'))}</td></tr>"
            )

    # --------------------------------------------------------
    # 4. Evidence-Based Root-Cause Diagnostic Chain (Issues C, D, N, P)
    # --------------------------------------------------------
    root_causes_structured = result.get("root_causes_structured", [])
    root_causes_html = ""

    if root_causes_structured and isinstance(root_causes_structured, list):
        for rc in root_causes_structured:
            sev = str(rc.get("severity", "MEDIUM")).upper()
            sev_class = {
                "CRITICAL": "badge-critical",
                "HIGH": "badge-high",
                "MEDIUM": "badge-medium",
                "LOW": "badge-good"
            }.get(sev, "badge-info")
            card_class = sev.lower() if sev.lower() in {"critical", "high", "medium", "low"} else "medium"

            finding_title = safe(rc.get("finding", rc.get("root_cause", "Diagnostic Finding")))
            category_label = safe(rc.get("category", "Root Cause Candidate"))
            evidence_text = safe(rc.get("evidence", ""))
            initial_evidence = safe(rc.get("initial_evidence", evidence_text))
            hypothesis_text = safe(rc.get("hypothesis", rc.get("potential_explanation", "")))
            diagnostic_status = safe(rc.get("diagnostic_status", "SIGNAL DETECTED"))
            verif_status = safe(rc.get("verification_status", "NOT TESTED"))
            verif_score = safe(rc.get("verification_score", ""))
            verif_evidence = safe(rc.get("verification_evidence", ""))
            interpretation_text = safe(rc.get("interpretation", ""))
            explanation_text = safe(rc.get("potential_explanation", ""))
            impact_text = safe(rc.get("impact", ""))
            confidence_level = safe(rc.get("confidence", "High"))
            recommendation_text = safe(rc.get("recommended_action", ""))
            affected_area = safe(rc.get("affected_metric", "Generalization"))

            status_badge_class = "badge-high" if "VERIFIED" in diagnostic_status else ("badge-good" if "NO MEASURABLE" in diagnostic_status else "badge-info")

            root_causes_html += (
                f'<div class="root-cause-card {card_class}">\n'
                f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; flex-wrap:wrap; gap:8px;">\n'
                f'<div><strong style="font-size:16px; color:#0f172a;">{finding_title}</strong> &mdash; <span class="badge badge-info">{category_label}</span></div>\n'
                f'<div style="display:flex; gap:6px; align-items:center; flex-wrap:wrap;">'
                f'<span class="badge {status_badge_class}">Status: {diagnostic_status}</span> '
                f'<span class="badge {sev_class}">{sev} Severity</span> '
                f'<span class="badge badge-good">{confidence_level} Confidence</span></div>\n'
                f'</div>\n'
                f'<div style="font-size:14px; margin-bottom:6px; color:#0f172a;"><strong>Signal &amp; Initial Evidence:</strong> {initial_evidence}</div>\n'
                f'<div style="font-size:13px; color:#334155; margin-bottom:5px;"><strong>Candidate Hypothesis:</strong> {hypothesis_text}</div>\n'
                + (f'<div style="font-size:13px; color:#1e3a8a; margin-bottom:5px; background:#eff6ff; padding:8px 12px; border-radius:6px;"><strong>Controlled Verification:</strong> {verif_status}' + (f' &mdash; <em>{verif_score}</em>' if verif_score and verif_score != 'N/A' else '') + (f'<div style="font-size:12px; color:#475569; margin-top:2px;">{verif_evidence}</div>' if verif_evidence else '') + '</div>\n' if verif_status and verif_status != 'NOT TESTED' else '')
                + f'<div style="font-size:13px; color:#334155; margin-bottom:5px;"><strong>Diagnostic Interpretation:</strong> {interpretation_text}</div>\n'
                + f'<div style="font-size:13px; color:#991b1b; margin-bottom:5px;"><strong>Operational Impact:</strong> {impact_text}</div>\n'
                + f'<div style="font-size:13px; color:#14532d; margin-bottom:5px; background:#f0fdf4; padding:8px 12px; border-radius:6px; border:1px solid #dcfce7;"><strong>Actionable Recommendation:</strong> {recommendation_text}</div>\n'
                + f'<div style="font-size:12px; color:#64748b; margin-top:4px;"><strong>Diagnostic Domain:</strong> {affected_area}</div>\n'
                + f'</div>\n'
            )
    else:
        root_causes_strings = result.get("root_causes", [])
        for i, item in enumerate(root_causes_strings, 1):
            root_causes_html += (
                f'<div class="numbered-row"><div class="row-number">{i}</div><div>{safe(item)}</div></div>\n'
            )

    # --------------------------------------------------------
    # 5. Priority Findings Table
    # --------------------------------------------------------
    priority_findings = result.get("priority_findings", result.get("priority_issues", []))
    p_rows = []
    if priority_findings and isinstance(priority_findings, list):
        for pf in priority_findings:
            sev = str(pf.get("priority", pf.get("severity", "MEDIUM"))).upper()
            sev_class = {
                "CRITICAL": "badge-critical",
                "HIGH": "badge-high",
                "MEDIUM": "badge-medium",
                "LOW": "badge-good"
            }.get(sev, "badge-info")
            p_rows.append(
                f"<tr><td><span class=\"badge {sev_class}\">{safe(sev)}</span></td>"
                f"<td><strong>{safe(pf.get('finding', pf.get('issue', '')))}</strong></td>"
                f"<td>{safe(pf.get('evidence', pf.get('reason', '')))}</td>"
                f"<td>{safe(pf.get('confidence', 'High'))}</td></tr>"
            )

    # --------------------------------------------------------
    # 6. Prescriptive Recommendations
    # --------------------------------------------------------
    recommendations_html = "".join(
        f'<div class="numbered-row"><div class="row-number">{i}</div><div>{safe(item)}</div></div>\n'
        for i, item in enumerate(result.get("recommendations", []), 1)
    )

    # --------------------------------------------------------
    # 7. Cross-Validation Folds & Stability
    # --------------------------------------------------------
    cv_scores = [percent(x) for x in cv.get("fold_scores", [])]
    cv_average = percent(cv.get("average_score", cv.get("cross_validation_mean", 0)))
    cv_std = number(cv.get("standard_deviation", 0))
    cv_min = percent(cv.get("cv_min", 0))
    cv_max = percent(cv.get("cv_max", 0))
    cv_range = percent(cv.get("cv_range", 0))

    cv_html = ""
    for i, score in enumerate(cv_scores, 1):
        lbl = f"R² Fold {i}" if task_type == "regression" else f"Fold {i}"
        cv_html += f'<div class="card metric-card"><div class="label">{lbl}</div><div class="metric">{score:.1f}%</div></div>\n'
    cv_html += f'<div class="card metric-card"><div class="label">CV Average</div><div class="metric">{cv_average:.1f}%</div><div class="sub">Std: {cv_std:.4f}</div></div>\n'
    cv_html += f'<div class="card metric-card"><div class="label">Fold Spread</div><div class="metric" style="font-size:18px;">{cv_min:.1f}% &ndash; {cv_max:.1f}%</div><div class="sub">Range: {cv_range:.1f}%</div></div>\n'

    cv_stability_status = safe(stability.get("stability_status", "STABLE"))
    cv_stability_explanation = safe(stability.get("stability_explanation", "Cross-validation folds show consistent performance across partitions."))
    overfitting_diagnostic = safe(stability.get("overfitting_diagnostic", "No Overfitting Detected"))
    overfitting_explanation = safe(stability.get("overfitting_explanation", "Training and test evaluation metrics demonstrate aligned generalization."))

    overfit_badge = "badge-high" if "Overfitting" in overfitting_diagnostic and "No" not in overfitting_diagnostic else "badge-good"

    stability_cards_html = (
        '<div class="card" style="margin-top:16px;">\n'
        f'<div style="font-weight:700; color:#1e3a8a; margin-bottom:4px;">Cross-Validation Consistency: <span class="badge badge-info">{cv_stability_status}</span></div>\n'
        f'<div style="font-size:14px; color:#334155; margin-bottom:12px;">{cv_stability_explanation}</div>\n'
        f'<div style="font-weight:700; color:#1e3a8a; margin-bottom:4px;">Generalization &amp; Overfitting Assessment: <span class="badge {overfit_badge}">{overfitting_diagnostic}</span></div>\n'
        f'<div style="font-size:14px; color:#334155;">{overfitting_explanation}</div>\n'
        '</div>\n'
    )

    # --------------------------------------------------------
    # 8. Model Selection & Generalization Overview Card
    # --------------------------------------------------------
    model_selection = result.get("model_selection", {})
    selected_model_name = model_selection.get("selected_model", result.get("selected_model", performance.get("selected_model", "Selected Model")))
    sel_criterion = model_selection.get("criterion", "Highest Cross-Validation Weighted F1" if task_type == "classification" else "Highest Cross-Validation R² Mean")
    sel_metric = model_selection.get("metric", "weighted_f1" if task_type == "classification" else "r2")
    sel_score = model_selection.get("selected_score", cv.get("average_score", 0.0))
    sel_cv_std = model_selection.get("cv_std", cv.get("standard_deviation", 0.0))
    sel_explanation = model_selection.get("selection_explanation", "")

    if task_type == "regression":
        reg_m = result.get("regression_metrics", {})
        mae = number(reg_m.get("mae", performance.get("mae", 0)))
        mse = number(reg_m.get("mse", performance.get("mse", 0)))
        rmse = number(reg_m.get("rmse", performance.get("rmse", 0)))
        r2 = number(reg_m.get("r2", performance.get("r2", 0)))
        score_display = f"{number(sel_score):.4f} (± {number(sel_cv_std):.4f})"
        test_perf_display = f"MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}"
    else:
        accuracy = percent(performance.get("accuracy", 0))
        precision_val = percent(performance.get("precision", 0))
        recall_val = percent(performance.get("recall", 0))
        f1 = percent(performance.get("f1_score", 0))
        score_display = f"{percent(sel_score):.1f}% (± {percent(sel_cv_std):.1f}%)"
        test_perf_display = f"Accuracy: {accuracy:.1f}% | Precision: {precision_val:.1f}% | Recall: {recall_val:.1f}% | F1: {f1:.1f}%"

    model_selection_card_html = (
        '<div class="section-title">Model Selection &amp; Generalization Policy</div>\n'
        '<div class="card" style="margin-bottom:20px; border-left: 5px solid #2563eb; background: #f8fafc;">\n'
        '<div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">\n'
        '<div>\n'
        '<div style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; color:#64748b; margin-bottom:4px;">Selected Model</div>\n'
        f'<div style="font-size:24px; font-weight:800; color:#0f172a;">{safe(selected_model_name)} <span class="badge badge-good" style="font-size:12px; vertical-align:middle; margin-left:6px;">SELECTED</span></div>\n'
        '</div>\n'
        '<div style="text-align:right;">\n'
        '<div style="font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; color:#64748b; margin-bottom:4px;">Selection Score (Cross-Validation)</div>\n'
        f'<div style="font-size:22px; font-weight:800; color:#2563eb;">{score_display}</div>\n'
        f'<div style="font-size:12px; color:#64748b;">Criterion: <code>{safe(sel_metric)}</code> (CV Mean ± Std)</div>\n'
        '</div>\n'
        '</div>\n'
        '<div style="margin-top:14px; padding-top:12px; border-top:1px solid #e2e8f0; display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:14px;">\n'
        '<div>\n'
        f'<div style="font-size:13px; color:#475569;"><strong>Selection Criterion:</strong> {safe(sel_criterion)}</div>\n'
        '<div style="font-size:13px; color:#475569; margin-top:4px;"><strong>Selection Basis:</strong> Cross-validation partition generalization estimate</div>\n'
        + (f'<div style="font-size:13px; color:#1e3a8a; margin-top:6px; background:#eff6ff; padding:8px 12px; border-radius:6px;"><strong>Selection Rationale:</strong> {safe(sel_explanation)}</div>\n' if sel_explanation else '')
        + '</div>\n'
        '<div>\n'
        '<div style="font-size:13px; color:#475569;"><strong>Held-Out Test Set Performance:</strong></div>\n'
        f'<div style="font-size:13px; color:#0f172a; font-weight:600; margin-top:4px;">{test_perf_display}</div>\n'
        '<div style="font-size:12px; color:#64748b; margin-top:8px; line-height:1.4;">\n'
        '<em>Note:</em> Test-set metrics and cross-validation metrics measure different aspects of model performance. Selection is driven by cross-validation to estimate partition-averaged generalization performance rather than relying solely on a single test split.\n'
        '</div>\n'
        '</div>\n'
        '</div>\n'
        '</div>\n'
    )

    # --------------------------------------------------------
    # 9. Multi-Model Comparison Table (Clearly separating Test vs CV)
    # --------------------------------------------------------
    model_comp_html = ""
    cross_model_note = safe(result.get("cross_model_status", ""))
    if model_comparison and isinstance(model_comparison, list):
        if task_type == "regression":
            m_rows = []
            for item in model_comparison:
                m_name = item.get("model", "")
                m_mae = number(item.get("test_mae", item.get("mae", 0)))
                m_mse = number(item.get("test_mse", item.get("mse", 0)))
                m_rmse = number(item.get("test_rmse", item.get("rmse", 0)))
                m_r2 = number(item.get("test_r2", item.get("r2", 0)))
                m_cv = number(item.get("cv_mean", 0))
                m_cv_std = number(item.get("cv_std", 0))
                is_selected = bool(item.get("is_selected") or str(item.get("selection_status", "")).lower() == "selected")
                status = item.get("status", "success")
                highlight = ' style="background:#eff6ff; font-weight:700;"' if is_selected else ""
                sel_badge = '<span class="badge badge-good">SELECTED</span>' if is_selected else '<span class="badge badge-info">Unselected</span>'

                if status == "success":
                    m_rows.append(
                        f'<tr{highlight}><td><strong>{safe(m_name)}</strong></td>'
                        f'<td>{m_mae:.4f}</td><td>{m_mse:.4f}</td><td>{m_rmse:.4f}</td><td>{m_r2:.4f}</td>'
                        f'<td>{m_cv:.4f}</td><td>±{m_cv_std:.4f}</td><td>{sel_badge}</td></tr>'
                    )
                else:
                    m_rows.append(
                        f'<tr><td><strong>{safe(m_name)}</strong></td><td colspan="6" style="color:#991b1b;">Failed: {safe(item.get("error", "Error"))}</td><td><span class="badge badge-high">Failed</span></td></tr>'
                    )

            model_comp_html = (
                '<div class="section-title">Multi-Model Comparison (Test-Set vs Cross-Validation)</div>\n'
                '<div class="card table-wrap">\n'
                + (f'<div class="info-box" style="margin-bottom:16px;"><strong>Cross-Model Consensus:</strong> {cross_model_note}</div>\n' if cross_model_note else '')
                + '<table>\n'
                + '<thead><tr><th>Model</th><th>Test MAE</th><th>Test MSE</th><th>Test RMSE</th><th>Test R²</th><th>CV Mean (R²)</th><th>CV Std</th><th>Selection Status</th></tr></thead>\n'
                + '<tbody>' + "".join(m_rows) + '</tbody>\n'
                + '</table>\n'
                + '</div>\n'
            )
        else:
            m_rows = []
            for item in model_comparison:
                m_name = item.get("model", "")
                m_acc = percent(item.get("test_accuracy", item.get("accuracy", 0)))
                m_prec = percent(item.get("test_precision", item.get("precision", 0)))
                m_rec = percent(item.get("test_recall", item.get("recall", 0)))
                m_f1 = percent(item.get("test_f1_score", item.get("f1_score", 0)))
                m_cv = percent(item.get("cv_mean", 0))
                m_cv_std = percent(item.get("cv_std", 0))
                is_selected = bool(item.get("is_selected") or str(item.get("selection_status", "")).lower() == "selected")
                status = item.get("status", "success")
                highlight = ' style="background:#eff6ff; font-weight:700;"' if is_selected else ""
                sel_badge = '<span class="badge badge-good">SELECTED</span>' if is_selected else '<span class="badge badge-info">Unselected</span>'

                if status == "success":
                    m_rows.append(
                        f'<tr{highlight}><td><strong>{safe(m_name)}</strong></td>'
                        f'<td>{m_acc:.1f}%</td><td>{m_prec:.1f}%</td><td>{m_rec:.1f}%</td><td>{m_f1:.1f}%</td>'
                        f'<td>{m_cv:.1f}%</td><td>±{m_cv_std:.1f}%</td><td>{sel_badge}</td></tr>'
                    )
                else:
                    m_rows.append(
                        f'<tr><td><strong>{safe(m_name)}</strong></td><td colspan="6" style="color:#991b1b;">Failed: {safe(item.get("error", "Error"))}</td><td><span class="badge badge-high">Failed</span></td></tr>'
                    )

            model_comp_html = (
                '<div class="section-title">Multi-Model Comparison (Test-Set vs Cross-Validation)</div>\n'
                '<div class="card table-wrap">\n'
                + (f'<div class="info-box" style="margin-bottom:16px;"><strong>Cross-Model Consensus:</strong> {cross_model_note}</div>\n' if cross_model_note else '')
                + '<table>\n'
                + '<thead><tr><th>Model</th><th>Test Accuracy</th><th>Test Precision</th><th>Test Recall</th><th>Test F1</th><th>CV Mean (Weighted F1)</th><th>CV Std</th><th>Selection Status</th></tr></thead>\n'
                + '<tbody>' + "".join(m_rows) + '</tbody>\n'
                + '</table>\n'
                + '</div>\n'
            )

    # --------------------------------------------------------
    # 10. Model Performance Overview Cards
    # --------------------------------------------------------
    if task_type == "regression":
        performance_html = (
            '<div class="section-title">Model Performance — Regression (Held-Out Test Set)</div>\n'
            '<div class="grid">\n'
            f'<div class="card metric-card"><div class="label">Selected Model</div><div class="metric" style="font-size:20px;">{safe(selected_model_name)}</div></div>\n'
            f'<div class="card metric-card"><div class="label">MAE</div><div class="metric">{mae:.4f}</div><div class="sub">Mean Absolute Error</div></div>\n'
            f'<div class="card metric-card"><div class="label">MSE</div><div class="metric">{mse:.4f}</div><div class="sub">Mean Squared Error</div></div>\n'
            f'<div class="card metric-card"><div class="label">RMSE</div><div class="metric">{rmse:.4f}</div><div class="sub">Root Mean Squared Error</div></div>\n'
            f'<div class="card metric-card"><div class="label">R² Score</div><div class="metric">{r2:.4f}</div><div class="sub">Coefficient of Determination</div></div>\n'
            '</div>\n'
        )
    else:
        performance_html = (
            '<div class="section-title">Model Performance — Classification (Held-Out Test Set)</div>\n'
            '<div class="grid">\n'
            f'<div class="card metric-card"><div class="label">Selected Model</div><div class="metric" style="font-size:20px;">{safe(selected_model_name)}</div></div>\n'
            f'<div class="card metric-card"><div class="label">Test Accuracy</div><div class="metric">{accuracy:.1f}%</div><div class="sub">Overall Correct</div></div>\n'
            f'<div class="card metric-card"><div class="label">Test Precision</div><div class="metric">{precision_val:.1f}%</div><div class="sub">Weighted Precision</div></div>\n'
            f'<div class="card metric-card"><div class="label">Test Recall</div><div class="metric">{recall_val:.1f}%</div><div class="sub">Weighted Recall</div></div>\n'
            f'<div class="card metric-card"><div class="label">Test F1 Score</div><div class="metric">{f1:.1f}%</div><div class="sub">Harmonic Mean</div></div>\n'
            '</div>\n'
        )

    # --------------------------------------------------------
    # 10. Target Analysis Section
    # --------------------------------------------------------
    target_stats_html = ""
    if task_type == "classification":
        c_dist_rows = []
        c_dist = target_profile.get("class_distribution", {})
        c_pcts = target_profile.get("class_percentages", {})
        total_obs = sum(c_dist.values()) if c_dist else 0
        min_cnt = min(c_dist.values()) if c_dist else 0
        max_cnt = max(c_dist.values()) if c_dist else 1

        is_binary = target_profile.get("is_binary", len(c_dist) == 2)
        class_count = len(c_dist)

        for c_k, c_v in c_dist.items():
            pct_v = c_pcts.get(c_k, (c_v / total_obs * 100) if total_obs > 0 else 0.0)
            if c_v == min_cnt and min_cnt < max_cnt:
                role = "Least Frequent" if not is_binary else "Minority Class"
                role_badge = "badge-info"
            elif c_v == max_cnt and min_cnt < max_cnt:
                role = "Most Frequent" if not is_binary else "Majority Class"
                role_badge = "badge-good"
            else:
                role = "Intermediate Frequency"
                role_badge = "badge-info"
            c_dist_rows.append(
                f"<tr><td><strong>Class '{safe(c_k)}'</strong></td><td>{c_v:,}</td><td>{pct_v:.2f}%</td><td><span class=\"badge {role_badge}\">{role}</span></td></tr>"
            )

        imbalance_note = target_profile.get("imbalance_explanation", "")

        target_stats_html = (
            '<div class="section-title">Target Analysis &amp; Class Distribution</div>\n'
            '<div class="card">\n'
            + f'<div class="info-box" style="margin-bottom:16px;">\n'
            + f'<strong>Distinct Classes:</strong> {class_count} classes ({total_obs:,} total observations)<br>\n'
            + (f'<div style="margin-top:6px; font-size:13px; color:#14532d;">{safe(imbalance_note)}</div>' if imbalance_note else '')
            + '</div>\n'
            + '<div class="table-wrap">\n'
            + '<table>\n'
            + '<thead><tr><th>Target Class</th><th>Observation Count</th><th>Percentage of Dataset</th><th>Frequency Role</th></tr></thead>\n'
            + '<tbody>' + "".join(c_dist_rows) + '</tbody>\n'
            + '</table>\n'
            + '</div>\n'
            + '</div>\n'
        )
    else:
        compat_note = target_profile.get("binary_target_compatibility_note", result.get("binary_target_compatibility_note"))
        target_stats_html = (
            '<div class="section-title">Target Distribution &amp; Numerical Statistics</div>\n'
            '<div class="card">\n'
            + (f'<div class="info-box" style="margin-bottom:14px; background:#fffbeb; border-left:4px solid #f59e0b; color:#92400e;"><strong>Compatibility Note:</strong> {safe(compat_note)}</div>\n' if compat_note else '')
            + '<div class="grid">\n'
            f'<div class="metric-card"><div class="label">Target Mean</div><div class="metric">{target_profile.get("mean", 0):.4f}</div></div>\n'
            f'<div class="metric-card"><div class="label">Target Median</div><div class="metric">{target_profile.get("median", 0):.4f}</div></div>\n'
            f'<div class="metric-card"><div class="label">Std Deviation</div><div class="metric">{target_profile.get("std", 0):.4f}</div></div>\n'
            f'<div class="metric-card"><div class="label">Range (Min - Max)</div><div class="metric" style="font-size:18px;">{target_profile.get("min", 0):.2f} &ndash; {target_profile.get("max", 0):.2f}</div></div>\n'
            '</div>\n'
            '</div>\n'
        )

    # --------------------------------------------------------
    # 11. Error & Confusion Analysis Section (Issues G, H)
    # --------------------------------------------------------
    error_segments_list = result.get("error_segments", [])
    err_seg_html = ""
    if error_segments_list:
        err_seg_rows = []
        for es in error_segments_list:
            err_seg_rows.append(
                f"<tr><td><strong>{safe(es.get('segment_area'))}</strong></td>"
                f"<td>{safe(es.get('error_rate'))}</td><td>{safe(es.get('baseline_error_rate'))}</td>"
                f"<td>{safe(es.get('evidence'))}</td></tr>"
            )
        err_seg_html = (
            '<div class="section-title">Error Segment Discovery (Failure Concentration)</div>\n'
            '<div class="card table-wrap">\n'
            '<table>\n'
            '<thead><tr><th>Failure Subgroup</th><th>Subgroup Error Rate</th><th>Overall Baseline</th><th>Diagnostic Finding</th></tr></thead>\n'
            '<tbody>' + "".join(err_seg_rows) + '</tbody>\n'
            '</table>\n'
            '</div>\n'
        )

    if task_type == "regression":
        reg_errs_all = result.get("prediction_errors", [])
        total_reg_errs = len(reg_errs_all)
        disp_reg_errs = min(15, total_reg_errs)
        pred_err_rows = []
        for err in reg_errs_all[:15]:
            row_num = record_row(err)
            act = err.get("actual", "")
            pred = err.get("predicted", "")
            abs_err = number(err.get("absolute_error", 0))
            pct_disp = err.get("percentage_error_display")
            if not pct_disp:
                raw_pct = err.get("percentage_error")
                pct_disp = f"{number(raw_pct):.2f}%" if raw_pct is not None else "N/A"
            res = number(err.get("residual", 0))
            pred_err_rows.append(
                f"<tr><td>Row {safe(row_num)}</td><td>{act}</td><td>{pred}</td>"
                f"<td>{abs_err:.4f}</td><td>{safe(pct_disp)}</td><td>{res:+.4f}</td></tr>\n"
            )

        reg_title_text = f"Largest Prediction Errors (Displaying Top {disp_reg_errs} of {total_reg_errs} Residuals; full list in CSV export)" if total_reg_errs > 15 else "Largest Prediction Errors (Top Residuals)"

        error_section = (
            '<div class="section-title">Regression Error &amp; Residual Analysis</div>\n'
            '<div class="card">\n'
            '<strong>Diagnostic Summary:</strong>\n'
            f'<div class="sub" style="margin-top:6px; font-size:14px;">{safe(result.get("main_error", result.get("error_analysis", {}).get("main_error", "No major regression issue detected across evaluated partitions.")))}</div>\n'
            '</div>\n'
            + err_seg_html
            + f'<div class="section-title">{reg_title_text}</div>\n'
            + '<div class="card table-wrap">\n'
            + '<table>\n'
            + '<thead><tr><th>Row</th><th>Actual</th><th>Predicted</th><th>Absolute Error</th><th>Error %</th><th>Residual (y - ŷ)</th></tr></thead>\n'
            + '<tbody>' + ("".join(pred_err_rows) or '<tr><td colspan="6">No prediction errors.</td></tr>') + '</tbody>\n'
            + '</table>\n'
            + '</div>\n'
        )
    else:
        pattern_rows = []
        for p in result.get("error_patterns", []):
            pattern_type = p.get("pattern_type", "Confusion Pattern")
            p_badge = "badge-high" if "Dominant" in pattern_type or "Frequent" in pattern_type else "badge-info"
            pattern_rows.append(
                f"<tr><td><strong>{safe(p.get('confusion_pair', ''))}</strong></td>"
                f"<td>{p.get('occurrences', 0)}</td>"
                f"<td>{p.get('error_percentage', 0):.1f}%</td>"
                f"<td>{p.get('average_confidence', 0):.1f}%</td>"
                f"<td><span class=\"badge {p_badge}\">{safe(pattern_type)}</span></td></tr>\n"
            )

        pattern_table_html = ""
        if pattern_rows:
            pattern_table_html = (
                '<div class="section-title">Misclassification Confusion Patterns</div>\n'
                '<div class="card table-wrap">\n'
                '<table>\n'
                '<thead><tr><th>Confusion Pair</th><th>Misclassified Instances</th><th>Error Share</th><th>Avg Confidence</th><th>Classification</th></tr></thead>\n'
                '<tbody>' + "".join(pattern_rows) + '</tbody>\n'
                '</table>\n'
                '</div>\n'
            )

        cm = result.get("confusion_matrix")
        class_names = result.get("class_names", [])
        cm_html = ""
        if cm and isinstance(cm, list) and class_names:
            header_cols = "".join(f"<th>Pred: {safe(c)}</th>" for c in class_names)
            matrix_rows = []
            for i, row in enumerate(cm):
                row_label = f"Actual: {safe(class_names[i])}" if i < len(class_names) else f"Row {i}"
                tds = []
                for j, count in enumerate(row):
                    is_diag = (i == j)
                    bg = ' style="background:#ecfdf5; font-weight:700;"' if is_diag else (' style="background:#fef2f2; color:#991b1b; font-weight:700;"' if count > 0 else "")
                    tds.append(f"<td{bg}>{count}</td>")
                matrix_rows.append(f"<tr><td><strong>{row_label}</strong></td>{''.join(tds)}</tr>")

            cm_html = (
                '<div class="section-title">Confusion Matrix</div>\n'
                '<div class="card table-wrap">\n'
                '<table>\n'
                f'<thead><tr><th>Actual \\ Predicted</th>{header_cols}</tr></thead>\n'
                f'<tbody>{"".join(matrix_rows)}</tbody>\n'
                '</table>\n'
                '</div>\n'
            )

        c_metrics = result.get("class_metrics", {})
        c_metric_rows = []
        if c_metrics and isinstance(c_metrics, dict):
            for c_name, m_data in c_metrics.items():
                if isinstance(m_data, dict):
                    c_prec = percent(m_data.get("precision", 0))
                    c_rec = percent(m_data.get("recall", 0))
                    c_f1 = percent(m_data.get("f1_score", m_data.get("f1", 0)))
                    c_supp = m_data.get("support", 0)
                    c_prop = percent(m_data.get("dataset_proportion", m_data.get("percentage_of_dataset", m_data.get("proportion", 0))))
                    c_metric_rows.append(
                        f"<tr><td><strong>Class '{safe(c_name)}'</strong></td>"
                        f"<td>{c_prec:.1f}%</td><td>{c_rec:.1f}%</td><td>{c_f1:.1f}%</td>"
                        f"<td>{c_supp}</td><td>{c_prop:.1f}%</td></tr>\n"
                    )

        class_metrics_html = ""
        if c_metric_rows:
            class_metrics_html = (
                '<div class="section-title">Per-Class Performance Breakdown</div>\n'
                '<div class="card table-wrap">\n'
                '<table>\n'
                '<thead><tr><th>Target Class</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>Test Support</th><th>Dataset Proportion</th></tr></thead>\n'
                '<tbody>' + "".join(c_metric_rows) + '</tbody>\n'
                '</table>\n'
                '</div>\n'
            )

        pred_errors_all = result.get("prediction_errors", [])
        total_err_count = int(result.get("total_test_errors", len(pred_errors_all)))
        disp_err_count = min(15, len(pred_errors_all))
        pred_html = ""
        for err in pred_errors_all[:15]:
            row_num = record_row(err)
            act = err.get("actual", "")
            pred = err.get("predicted", "")
            conf_val = confidence(err)
            e_type = err.get("error_type", "Misclassification")
            pred_html += (
                f"<tr><td>Row {safe(row_num)}</td><td>{safe(act)} &rarr; {safe(pred)}</td>"
                f"<td>{safe(e_type)}</td><td>{conf_val:.1f}%</td></tr>\n"
            )

        cls_title_text = f"Individual Prediction Errors (Displaying first {disp_err_count} of {total_err_count} test errors; complete list in CSV export)" if total_err_count > 15 else "Individual Prediction Errors"

        error_section = (
            pattern_table_html
            + cm_html
            + class_metrics_html
            + err_seg_html
            + f'<div class="section-title">{cls_title_text}</div>\n'
            + '<div class="card">\n'
            + '<div class="note" style="margin-top:0; margin-bottom:14px;">\n'
            + '<strong>Prediction Confidence vs. Root-Cause Confidence:</strong> The "Prediction Confidence" column below reflects the probability the model assigned to its predicted class for that single row. This is distinct from "Root-Cause Confidence", which evaluates how strongly statistical evidence across the entire dataset supports a diagnostic finding.\n'
            + '</div>\n'
            + '<div class="table-wrap">\n'
            + '<table>\n'
            + '<thead><tr><th>Row</th><th>Actual &rarr; Predicted</th><th>Error Type</th><th>Prediction Confidence</th></tr></thead>\n'
            + '<tbody>' + (pred_html or '<tr><td colspan="4">No prediction errors on test set.</td></tr>') + '</tbody>\n'
            + '</table>\n'
            + '</div>\n'
            + '</div>\n'
        )

    # --------------------------------------------------------
    # 12. Risk Drivers Table
    # --------------------------------------------------------
    risk_obj = result.get("risk", {})
    risk_drivers = risk_obj.get("drivers", result.get("risk_drivers", []))
    risk_driver_rows = []
    for d in risk_drivers:
        issue_name = d.get("issue", "Diagnostic Finding")
        ev_text = d.get("evidence", "")
        conf_val = d.get("confidence", 0.8)
        conf_pct = f"{conf_val * 100:.0f}%" if isinstance(conf_val, (int, float)) and conf_val <= 1.0 else f"{conf_val}"
        contrib = int(d.get("contribution", 0))
        risk_driver_rows.append(
            f"<tr><td><strong>{safe(issue_name)}</strong></td>"
            f"<td>{safe(ev_text)}</td>"
            f"<td><span class=\"badge badge-info\">{conf_pct}</span></td>"
            f"<td><strong>+{contrib} pts</strong></td></tr>\n"
        )

    # --------------------------------------------------------
    # 13. Warnings & Diagnostics
    # --------------------------------------------------------
    warnings_list = result.get("warnings", [])
    warnings_html = "".join(f'<div class="error">{safe(w)}</div>\n' for w in warnings_list)
    if not warnings_html:
        warnings_html = '<div class="sub">No diagnostic warnings reported.</div>'

    total_features = int(number(performance.get("feature_count", len(df.columns) - 1)))

    global CURRENT_AI_EXPLAINER
    global CURRENT_AI_REPORT
    global CURRENT_FIX_SCRIPT
    global CURRENT_VISUALIZATIONS

    if CURRENT_AI_EXPLAINER is None:
        CURRENT_AI_EXPLAINER = AIExplainer()

    if CURRENT_AI_REPORT is None:
        CURRENT_AI_REPORT = CURRENT_AI_EXPLAINER.explain(result, quality)
    if CURRENT_FIX_SCRIPT is None:
        CURRENT_FIX_SCRIPT = CURRENT_AI_EXPLAINER.generate_fix_script(result, quality)
    if CURRENT_VISUALIZATIONS is None:
        CURRENT_VISUALIZATIONS = generate_all_visualizations(result)

    vis = CURRENT_VISUALIZATIONS

    # Build interactive chart boxes
    chart_boxes = []
    if "risk_gauge" in vis:
        chart_boxes.append('<div class="chart-box" data-category="risk"><div id="chart-risk-gauge" class="chart-container"></div></div>')
    if "risk_breakdown" in vis:
        chart_boxes.append('<div class="chart-box" data-category="risk"><div id="chart-risk-breakdown" class="chart-container"></div></div>')
    if "performance_metrics" in vis:
        chart_boxes.append('<div class="chart-box" data-category="perf"><div id="chart-perf-metrics" class="chart-container"></div></div>')
    if "cv_stability" in vis:
        chart_boxes.append('<div class="chart-box" data-category="perf"><div id="chart-cv-stability" class="chart-container"></div></div>')
    if "feature_importance" in vis:
        chart_boxes.append('<div class="chart-box" data-category="feats"><div id="chart-feat-importance" class="chart-container"></div></div>')
    if "feature_relationships" in vis:
        chart_boxes.append('<div class="chart-box" data-category="feats"><div id="chart-feat-relationships" class="chart-container"></div></div>')
    if "data_quality_missing" in vis:
        chart_boxes.append('<div class="chart-box" data-category="quality"><div id="chart-missing-vals" class="chart-container"></div></div>')
    if "target_distribution" in vis:
        chart_boxes.append('<div class="chart-box" data-category="quality"><div id="chart-target-dist" class="chart-container"></div></div>')
    if "confusion_matrix" in vis:
        chart_boxes.append('<div class="chart-box" data-category="errors"><div id="chart-confusion-matrix" class="chart-container"></div></div>')
    if "actual_vs_predicted" in vis:
        chart_boxes.append('<div class="chart-box" data-category="errors"><div id="chart-actual-vs-pred" class="chart-container"></div></div>')
    if "residual_plot" in vis:
        chart_boxes.append('<div class="chart-box" data-category="errors"><div id="chart-residual-plot" class="chart-container"></div></div>')
    if "residual_distribution" in vis:
        chart_boxes.append('<div class="chart-box" data-category="errors"><div id="chart-residual-dist" class="chart-container"></div></div>')
    if "verification_ablation" in vis:
        chart_boxes.append('<div class="chart-box" data-category="verif"><div id="chart-verif-ablation" class="chart-container"></div></div>')
    if "remediation_simulation" in vis:
        chart_boxes.append('<div class="chart-box" data-category="verif"><div id="chart-remed-sim" class="chart-container"></div></div>')

    chart_payload_map = {
        "chart-risk-gauge": vis.get("risk_gauge"),
        "chart-risk-breakdown": vis.get("risk_breakdown"),
        "chart-perf-metrics": vis.get("performance_metrics"),
        "chart-cv-stability": vis.get("cv_stability"),
        "chart-feat-importance": vis.get("feature_importance"),
        "chart-feat-relationships": vis.get("feature_relationships"),
        "chart-missing-vals": vis.get("data_quality_missing"),
        "chart-target-dist": vis.get("target_distribution"),
        "chart-confusion-matrix": vis.get("confusion_matrix"),
        "chart-actual-vs-pred": vis.get("actual_vs_predicted"),
        "chart-residual-plot": vis.get("residual_plot"),
        "chart-residual-dist": vis.get("residual_distribution"),
        "chart-verif-ablation": vis.get("verification_ablation"),
        "chart-remed-sim": vis.get("remediation_simulation"),
    }
    clean_map = {k: v for k, v in chart_payload_map.items() if v is not None}
    chart_json = json.dumps(clean_map).replace("</script>", "<\\/script>")

    visual_center_html = (
        '<div class="section-title">📊 Evidence-Driven Visual Diagnostic Center</div>\n'
        '<div class="card" style="padding:16px 20px 24px; margin-bottom:24px;">\n'
        '<div class="chart-tabs">\n'
        '<button class="chart-tab-btn active" type="button" onclick="switchChartTab(\'all\', this)">All Charts</button>\n'
        '<button class="chart-tab-btn" type="button" onclick="switchChartTab(\'verif\', this)">🔬 Verification &amp; Remediation</button>\n'
        '<button class="chart-tab-btn" type="button" onclick="switchChartTab(\'risk\', this)">Risk &amp; Health</button>\n'
        '<button class="chart-tab-btn" type="button" onclick="switchChartTab(\'perf\', this)">Performance &amp; CV</button>\n'
        '<button class="chart-tab-btn" type="button" onclick="switchChartTab(\'feats\', this)">Feature Impact</button>\n'
        '<button class="chart-tab-btn" type="button" onclick="switchChartTab(\'quality\', this)">Data Quality</button>\n'
        '<button class="chart-tab-btn" type="button" onclick="switchChartTab(\'errors\', this)">Error &amp; Diagnostics</button>\n'
        '</div>\n'
        '<div class="chart-grid">\n'
        + "".join(chart_boxes) + "\n"
        + '</div>\n'
        + '</div>\n'
        + f'<script>\nvar DIAGNOSTIC_CHARTS = {chart_json};\n'
        + """
document.addEventListener("DOMContentLoaded", function() {
    if (typeof Plotly !== "undefined" && typeof DIAGNOSTIC_CHARTS !== "undefined") {
        for (var elemId in DIAGNOSTIC_CHARTS) {
            var el = document.getElementById(elemId);
            if (el && DIAGNOSTIC_CHARTS[elemId]) {
                var c = DIAGNOSTIC_CHARTS[elemId];
                Plotly.newPlot(elemId, c.data || [], c.layout || {}, c.config || {responsive: true});
            }
        }
    }
});
window.addEventListener("resize", function() {
    var chartContainers = document.querySelectorAll(".chart-container");
    chartContainers.forEach(function(el) {
        if (typeof Plotly !== "undefined") {
            Plotly.Plots.resize(el);
        }
    });
});
</script>\n"""
    )

    # --------------------------------------------------------
    # 4b. Automated Root-Cause Verification & Closed-Loop Remediation Matrix
    # --------------------------------------------------------
    verif_data = result.get("verification_engine", {})
    candidate_experiments = verif_data.get("candidate_experiments", result.get("verification_experiments", []))
    remed_sim = verif_data.get("remediation_simulation", result.get("remediation_simulation", {}))

    verif_cards_html = ""
    if candidate_experiments and isinstance(candidate_experiments, list):
        for exp in candidate_experiments:
            cand_name = safe(exp.get("candidate_feature", "Candidate Feature"))
            metric_label = safe(exp.get("metric_name", "Weighted F1" if task_type == "classification" else "R2 Score"))
            base_val = number(exp.get("baseline_metric", 0.0))
            abl_val = number(exp.get("ablated_metric", 0.0))
            abl_delta = number(exp.get("ablation_delta", 0.0))
            abl_delta_pct = number(exp.get("ablation_delta_pct", 0.0))
            perm_val = number(exp.get("permuted_metric", 0.0))
            perm_delta = number(exp.get("permutation_delta", 0.0))
            perm_delta_pct = number(exp.get("permutation_delta_pct", 0.0))
            pert_val = number(exp.get("perturbed_metric", base_val))
            noise_delta = number(exp.get("noise_delta", 0.0))
            noise_delta_pct = number(exp.get("noise_delta_pct", 0.0))
            flip_pct = number(exp.get("prediction_flip_rate_pct", 0.0))
            ctrl_feat = safe(exp.get("control_feature", "None"))
            ctrl_val = number(exp.get("control_ablation_metric", 0.0))
            ctrl_delta = number(exp.get("control_delta", 0.0))
            ratings = exp.get("evidence_ratings", {})
            abl_rating = safe(ratings.get("ablation", "neutral"))
            perm_rating = safe(ratings.get("permutation", "neutral"))
            noise_rating = safe(ratings.get("noise", exp.get("noise_sensitivity", "robust")))
            ctrl_rating = safe(ratings.get("control", "inconclusive"))
            verdict = safe(exp.get("verdict", "PARTIAL / INTERACTIVE SIGNAL"))
            ev_score = int(number(exp.get("evidence_score", 0)))
            summary_text = safe(exp.get("summary", ""))

            decomp = exp.get("score_decomposition", {})
            abl_pts = int(decomp.get("ablation_points", 0))
            perm_pts = int(decomp.get("permutation_points", 0))
            ctrl_pts = int(decomp.get("control_points", 0))
            stab_pts = int(decomp.get("stability_points", 0))
            const_pts = int(decomp.get("consistency_points", 0))
            decomp_str = f"Ablation: {abl_pts}/35 · Permutation: {perm_pts}/35 · Control: {ctrl_pts}/15 · Stability: {stab_pts}/10 · Consistency: {const_pts}/5"

            if "VERIFIED" in verdict:
                verdict_badge_class = "badge-good"
            elif "NO MEASURABLE" in verdict:
                verdict_badge_class = "badge-info"
            elif "BRITTLENESS" in verdict:
                verdict_badge_class = "badge-high"
            else:
                verdict_badge_class = "badge-medium"

            if ctrl_rating == "passed":
                ctrl_badge_class = "badge-good"
            elif ctrl_rating == "inconclusive":
                ctrl_badge_class = "badge-medium"
            else:
                ctrl_badge_class = "badge-high"

            verif_cards_html += (
                f'<div class="verif-card">\n'
                f'<div class="verif-header">\n'
                f'<div><strong style="font-size:16px; color:#0f172a;">Candidate: <code>{cand_name}</code></strong> '
                f'<span class="badge {verdict_badge_class}" style="margin-left:8px;">{verdict}</span></div>\n'
                f'<div style="text-align:right;">'
                f'<div><span style="font-size:13px; font-weight:700; color:#475569;">Evidence Score: </span>'
                f'<strong style="font-size:18px; color:{"#2563eb" if ev_score > 30 else "#475569"};">{ev_score}/100</strong></div>\n'
                f'<div style="font-size:11px; color:#64748b; font-weight:600; margin-top:2px;">{decomp_str}</div>'
                f'</div>\n'
                f'</div>\n'
                f'<div class="verif-metric-grid">\n'
                f'<div class="verif-box"><div class="verif-box-label">1. Baseline ({metric_label})</div><div class="verif-box-val">{base_val:.4f}</div><div class="verif-box-sub">Full model</div></div>\n'
                f'<div class="verif-box"><div class="verif-box-label">2. Retrained Ablation</div><div class="verif-box-val">{abl_val:.4f}</div><div class="verif-box-sub" style="color:{"#dc2626" if abl_delta < 0 else "#166534"}; font-weight:700;">delta: {abl_delta:+.4f} ({abl_delta_pct:+.1f}%)</div></div>\n'
                f'<div class="verif-box"><div class="verif-box-label">3. Permutation Test</div><div class="verif-box-val">{perm_val:.4f}</div><div class="verif-box-sub" style="color:{"#dc2626" if perm_delta < 0 else "#166534"}; font-weight:700;">delta: {perm_delta:+.4f} ({perm_delta_pct:+.1f}%)</div></div>\n'
                f'<div class="verif-box"><div class="verif-box-label">4. Measurement Stability (10% σ Jitter)</div><div class="verif-box-val">{pert_val:.4f}</div><div class="verif-box-sub" style="color:{"#dc2626" if noise_delta < -0.05 or flip_pct >= 15 else "#166534"}; font-weight:700;">delta: {noise_delta:+.4f} (flip: {flip_pct:.1f}%)</div></div>\n'
                f'<div class="verif-box"><div class="verif-box-label">5. Control ({ctrl_feat})</div><div class="verif-box-val">{ctrl_val:.4f}</div><div class="verif-box-sub">delta: {ctrl_delta:+.4f}</div></div>\n'
                f'</div>\n'
                f'<div class="verif-evidence-row">\n'
                f'<span><strong>4 Controlled Trials:</strong></span>'
                f'<span>Ablation: <span class="badge badge-info">{abl_rating}</span></span>'
                f'<span>Permutation: <span class="badge badge-info">{perm_rating}</span></span>'
                f'<span>Measurement Stability: <span class="badge badge-info">{noise_rating}</span></span>'
                f'<span>Control Test: <span class="badge {ctrl_badge_class}">{ctrl_rating}</span></span>'
                f'</div>\n'
                f'<div style="font-size:13px; color:#334155; line-height:1.5; background:#f8fafc; padding:10px 14px; border-radius:8px; border:1px solid #e2e8f0;">{summary_text}</div>\n'
                f'</div>\n'
            )

    remediation_sim_html = ""
    if remed_sim and remed_sim.get("status") == "Success":
        res_verdict = safe(remed_sim.get("resolution_verdict", "Evaluated"))
        proof_summary = safe(remed_sim.get("proof_summary", ""))
        metric_lbl = safe(remed_sim.get("metric_name", "Metric"))
        b_train = number(remed_sim.get("baseline", {}).get("train_score", 0.0))
        b_test = number(remed_sim.get("baseline", {}).get("test_score", 0.0))
        b_gap = number(remed_sim.get("baseline", {}).get("generalization_gap", 0.0))
        r_train = number(remed_sim.get("remediated", {}).get("train_score", 0.0))
        r_test = number(remed_sim.get("remediated", {}).get("test_score", 0.0))
        r_gap = number(remed_sim.get("remediated", {}).get("generalization_gap", 0.0))
        d_test = number(remed_sim.get("deltas", {}).get("test_metric_delta", 0.0))
        d_gap = number(remed_sim.get("deltas", {}).get("generalization_gap_reduction", 0.0))

        if "BOTH IMPROVED" in res_verdict or "PERFORMANCE IMPROVED" in res_verdict:
            res_badge_class = "badge-good"
        elif "PARTIALLY RESOLVED" in res_verdict:
            res_badge_class = "badge-info"
        elif "BASELINE PREFERRED" in res_verdict:
            res_badge_class = "badge-high"
        else:
            res_badge_class = "badge-medium"

        remediation_sim_html = (
            f'<div class="remed-card">\n'
            f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:14px; padding-bottom:10px; border-bottom:1px solid #dcfce7;">\n'
            f'<div><strong style="font-size:18px; color:#14532d;">🔄 Closed-Loop Remediation Simulation &amp; Proof of Fix</strong></div>\n'
            f'<div><span class="badge {res_badge_class}" style="font-size:13px; padding:6px 12px;">{res_verdict}</span></div>\n'
            f'</div>\n'
            f'<div class="verif-metric-grid">\n'
            f'<div class="verif-box"><div class="verif-box-label">Baseline Test ({metric_lbl})</div><div class="verif-box-val">{b_test:.4f}</div><div class="verif-box-sub">Train: {b_train:.4f} | Gap: {b_gap:.4f}</div></div>\n'
            f'<div class="verif-box"><div class="verif-box-label">Remediated Test ({metric_lbl})</div><div class="verif-box-val" style="color:#15803d;">{r_test:.4f}</div><div class="verif-box-sub">Train: {r_train:.4f} | Gap: {r_gap:.4f}</div></div>\n'
            f'<div class="verif-box"><div class="verif-box-label">Change in Held-Out Performance</div><div class="verif-box-val" style="color:{"#15803d" if d_test >= 0 else "#dc2626"};">{d_test:+.4f}</div><div class="verif-box-sub">Held-out test delta</div></div>\n'
            f'<div class="verif-box"><div class="verif-box-label">Train–Evaluation Gap Reduction</div><div class="verif-box-val" style="color:{"#15803d" if d_gap >= 0 else "#dc2626"};">{d_gap:+.4f}</div><div class="verif-box-sub">Generalization gap delta</div></div>\n'
            f'</div>\n'
            f'<div style="font-size:14px; color:#14532d; line-height:1.5;"><strong>Empirical Proof:</strong> {proof_summary}</div>\n'
            f'</div>\n'
        )

    verification_matrix_html = (
        '<div class="section-title">🔬 Automated Root-Cause Verification &amp; Closed-Loop Remediation Matrix</div>\n'
        + (remediation_sim_html if remediation_sim_html else '')
        + (
            '<div class="card" style="margin-bottom:24px;">\n'
            '<div class="note" style="margin-top:0; margin-bottom:16px;">\n'
            '<strong>Controlled Counterfactual Verification:</strong> Rather than relying only on observational correlations, the engine runs targeted ablation retraining, test-time permutation shuffling, measurement stability testing, and baseline control tests on candidate features. Controlled ablation and permutation experiments measure empirical model reliance under the tested interventions; they do not establish real-world causal relationships. The Evidence Score reflects experimental proof strength with transparent, configurable diagnostic thresholds.\n'
            '</div>\n'
            + (verif_cards_html or '<div class="sub">No candidate features required ablation testing.</div>')
            + '</div>\n'
            if verif_cards_html else ''
        )
    )

    ai_provider_name = CURRENT_AI_REPORT.get("provider", CURRENT_AI_EXPLAINER.get_active_provider_name())
    ai_summary = CURRENT_AI_REPORT.get("executive_summary", "")
    risk_assessment = CURRENT_AI_REPORT.get("risk_assessment", {})
    readiness = risk_assessment.get("deployment_readiness", "Requires Review")
    roadmap = CURRENT_AI_REPORT.get("remediation_roadmap", [])

    roadmap_html = "".join(
        f'<div class="roadmap-item"><div class="roadmap-num">{idx}</div><div>{safe(step)}</div></div>\n'
        for idx, step in enumerate(roadmap, 1)
    )

    ai_executive_html = (
        '<div class="section-title">🤖 AI Executive Root-Cause Summary &amp; Remediation Plan</div>\n'
        '<div class="ai-card">\n'
        '<div class="ai-header">\n'
        '<div class="ai-title">🤖 AI Reliability &amp; Root-Cause Audit</div>\n'
        f'<div><span class="badge" style="background:#4338ca; color:#ffffff; font-weight:700;">⚡ {safe(ai_provider_name)}</span> '
        f'<span class="badge badge-good" style="margin-left:6px;">Status: {safe(readiness)}</span></div>\n'
        '</div>\n'
        f'<div class="ai-summary-text">{safe(ai_summary)}</div>\n'
        + (f'<div style="font-weight:700; color:#312e81; margin-top:14px; margin-bottom:6px;">Recommended Remediation Roadmap:</div>\n<div class="roadmap-list">\n{roadmap_html}</div>\n' if roadmap_html else '')
        + '<div style="margin-top:20px; display:flex; gap:10px; flex-wrap:wrap;">\n'
        + '<button class="btn btn-primary" type="button" style="background:#4f46e5; border-color:#4f46e5;" onclick="var d=document.getElementById(\'code-drawer\'); d.style.display=(d.style.display===\'none\'?\'block\':\'none\');">⚡ View Auto-Remediation Python Script</button>\n'
        + '<a class="btn btn-secondary" href="/download/remediation_script.py" style="border-color:#c7d2fe; color:#312e81;">Download fix_pipeline.py</a>\n'
        + '</div>\n'
        + '<div id="code-drawer" class="code-drawer" style="display:none;">\n'
        + '<div class="code-drawer-header">\n'
        + '<span>Python Preprocessing &amp; Model Remediation Script</span>\n'
        + '<button id="copy-code-btn" type="button" class="btn btn-secondary" style="padding:4px 10px; font-size:12px;" onclick="copyFixCode()">Copy Code</button>\n'
        + '</div>\n'
        + f'<pre id="fix-script-code" class="code-pre">{safe(CURRENT_FIX_SCRIPT)}</pre>\n'
        + '</div>\n'
        '</div>\n'
    )


    copilot_html = (
        '<div class="section-title">💬 Interactive AI Diagnostic Copilot</div>\n'
        '<div class="copilot-card">\n'
        '<div class="copilot-header">\n'
        '<div><strong style="font-size:16px;">Diagnostic Copilot</strong> &mdash; Grounded in Empirical Dataset Evidence</div>\n'
        f'<span class="badge" style="background:rgba(255,255,255,0.2); color:white;">{safe(ai_provider_name)}</span>\n'
        '</div>\n'
        '<div id="chat-history" class="chat-history">\n'
        '<div class="chat-bubble ai">\n'
        f'Hello! I am your AI Diagnostic Copilot. I have analyzed the model diagnostics for <strong>{safe(CURRENT_FILENAME)}</strong> on target <strong>{safe(CURRENT_TARGET)}</strong>. What questions do you have about the root causes, feature impacts, or remediation strategy?\n'
        '</div>\n'
        '</div>\n'
        '<div class="chips-row">\n'
        '<span style="font-size:12px; font-weight:700; color:#64748b; align-self:center;">Suggested:</span>\n'
        '<button class="prompt-chip" type="button" onclick="askCopilot(\'Why did the model perform poorly?\')">Why did the model perform poorly?</button>\n'
        '<button class="prompt-chip" type="button" onclick="askCopilot(\'Which feature causes the highest risk?\')">Which feature causes highest risk?</button>\n'
        '<button class="prompt-chip" type="button" onclick="askCopilot(\'What is the recommended fix strategy?\')">Recommended fix strategy?</button>\n'
        '<button class="prompt-chip" type="button" onclick="askCopilot(\'Is this model ready for production deployment?\')">Production readiness?</button>\n'
        '</div>\n'
        '<div class="chat-input-row">\n'
        '<input id="copilot-input" class="chat-input" type="text" placeholder="Ask a question about the diagnostic results (e.g. why is recall low, which feature to remove?)..." onkeypress="if(event.key===\'Enter\') askCopilot();">\n'
        '<button class="btn btn-primary" type="button" style="background:#4f46e5; border-color:#4f46e5;" onclick="askCopilot()">Send</button>\n'
        '</div>\n'
        '</div>\n'
    )

    html = (
        page_start("Dashboard - AI Model Root-Cause Analyzer")
        + '<div class="btn-row" style="margin-bottom:24px;">\n'
        + '<a class="btn btn-secondary" href="/back-to-preview">&larr; Back to Preview</a>\n'
        + '<a class="btn btn-secondary" href="/back-to-upload">Upload New Dataset</a>\n'
        + '<a class="btn btn-primary" href="/download/analysis.json">Download JSON</a>\n'
        + '<a class="btn btn-secondary" href="/download/summary.csv">Download Summary CSV</a>\n'
        + '<a class="btn btn-secondary" href="/download/prediction-errors.csv">Download Errors CSV</a>\n'
        + '<a class="btn btn-secondary" href="/download/remediation_script.py" style="background:#4338ca; color:#ffffff; border-color:#4338ca;">⚡ Download Auto-Fix .py</a>\n'
        + '</div>\n'
        + f'<div class="success">Analyzed <strong>{safe(CURRENT_FILENAME)}</strong> using target <strong>{safe(CURRENT_TARGET)}</strong> with <strong>{safe(mode_label(task_type))}</strong> analysis.<br>'
        + f'<span style="font-size:13px; font-weight:normal; color:#14532d;">{safe(task_reason)}</span></div>\n'
        + ai_executive_html
        + visual_center_html
        + verification_matrix_html
        + model_selection_card_html
        + performance_html

        + model_comp_html
        + target_stats_html
        + '<div class="section-title">Model Stability &amp; Cross-Validation Folds</div>\n'
        + '<div class="grid">\n'
        + cv_html
        + '</div>\n'
        + stability_cards_html
        + '<div class="section-title">Diagnostic Signals &amp; Root-Cause Hypotheses</div>\n'
        + '<div class="card">\n'
        + (root_causes_html or '<div class="sub">No diagnostic bottlenecks identified.</div>')
        + '</div>\n'
        + '<div class="section-title">Priority Findings</div>\n'
        + '<div class="card table-wrap">\n'
        + '<table>\n'
        + '<thead><tr><th>Priority</th><th>Finding</th><th>Evidence</th><th>Evidence Confidence</th></tr></thead>\n'
        + '<tbody>' + ("".join(p_rows) or '<tr><td colspan="4">No critical or high priority findings.</td></tr>') + '</tbody>\n'
        + '</table>\n'
        + '</div>\n'
        + '<div class="section-title">Feature Impact &amp; Importance</div>\n'
        + '<div class="card">\n'
        + '<div class="info-box" style="margin-bottom:16px;">\n'
        + '<strong>Predictive Utility vs. Causality:</strong> Feature importance measures predictive usefulness in the model for this dataset. It does <em>not</em> prove causal necessity, positive/negative relationship direction, or business importance. Directional relationships are separately evaluated below.\n'
        + '</div>\n'
        + '<div class="table-wrap">\n'
        + '<table>\n'
        + '<thead><tr><th>Rank</th><th>Feature</th><th>Importance Score</th><th>Relative Share</th><th>Model Influence Tier</th><th>Observed Tendency</th></tr></thead>\n'
        + '<tbody>' + (feature_html or '<tr><td colspan="6">No feature impact available.</td></tr>') + '</tbody>\n'
        + '</table>\n'
        + '</div>\n'
        + '</div>\n'
        + (
            '<div class="section-title">Feature &rarr; Target Empirical Relationships</div>\n'
            '<div class="card">\n'
            + '<div style="font-size:13px; color:#64748b; margin-bottom:14px;">Observed empirical target behavior across quantile ranges and top categories. Note: These indicate observed historical distributions and not proven causal mechanisms.</div>\n'
            + rel_cards_html
            + '</div>\n'
            if rel_cards_html else ''
        )
        + (
            '<div class="section-title">Subgroup &amp; Segment Analysis</div>\n'
            '<div class="card table-wrap">\n'
            '<table>\n'
            '<thead><tr><th>Subgroup Segment</th><th>Population</th><th>Target Outcome</th><th>Important Difference</th></tr></thead>\n'
            '<tbody>' + "".join(segment_rows) + '</tbody>\n'
            '</table>\n'
            '</div>\n'
            if segment_rows else ''
        )
        + error_section
        + '<div class="section-title">Dataset Health &amp; Discovery</div>\n'
        + '<div class="grid">\n'
        + f'<div class="card metric-card"><div class="label">Total Rows</div><div class="metric">{len(df):,}</div></div>\n'
        + f'<div class="card metric-card"><div class="label">Usable Predictors</div><div class="metric">{total_features}</div></div>\n'
        + f'<div class="card metric-card"><div class="label">Quality Status</div><div class="metric">{safe(quality.get("overall_quality", "GOOD"))}</div></div>\n'
        + f'<div class="card metric-card"><div class="label">Missing Cells</div><div class="metric">{missing:,}</div></div>\n'
        + f'<div class="card metric-card"><div class="label">Duplicate Rows</div><div class="metric">{int(number(quality.get("duplicate_rows", 0))):,}</div></div>\n'
        + f'<div class="card metric-card"><div class="label">Potential Outliers</div><div class="metric">{outliers:,}</div><div class="sub">1.5 &times; IQR Fence</div></div>\n'
        + '</div>\n'
        + '<div class="card" style="margin-top:16px;">\n'
        + f'<div style="font-weight:700; color:#1e3a8a; margin-bottom:4px;">Quality Assessment: <span class="badge badge-info">{safe(quality.get("overall_quality", "GOOD"))}</span></div>\n'
        + f'<div style="font-size:14px; color:#334155; margin-bottom:10px;">{safe(quality.get("quality_reason", "Dataset exhibits healthy structural integrity."))}</div>\n'
        + (f'<div style="font-size:13px; color:#64748b; margin-bottom:8px;"><strong>Duplicate Handling:</strong> {safe(quality.get("duplicate_handling_note", ""))}</div>\n' if quality.get("duplicate_handling_note") else '')
        + '<div style="font-size:12px; color:#64748b;"><strong>Outlier Advisory:</strong> Numerical outliers are detected via standard interquartile range (IQR) boundaries. Extreme values are flagged as potential outliers for operational verification and are not automatically considered bad data.</div>\n'
        + '</div>\n'
        + '<div class="section-title">Prescriptive Recommendations</div>\n'
        + '<div class="card">\n'
        + (recommendations_html or '<div class="sub">No recommendations reported.</div>')
        + '</div>\n'
        + '<div class="section-title">Evidence-Driven Risk Assessment</div>\n'
        + '<div class="card">\n'
        + '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">\n'
        + f'<div><span class="badge {risk_class}" style="font-size:14px; padding:6px 14px;">{safe(risk)} RISK</span><div class="metric" style="margin-top:8px;">{risk_score}/100</div></div>\n'
        + f'<div style="font-size:13px; color:#64748b;"><strong>Calculation Method:</strong> {safe(risk_obj.get("method", "Evidence-weighted diagnostic aggregation"))}</div>\n'
        + '</div>\n'
        + (
            '<div class="table-wrap">\n'
            '<table>\n'
            '<thead><tr><th>Contributing Diagnostic Domain</th><th>Observed Evidence</th><th>Evidence Confidence</th><th>Risk Impact</th></tr></thead>\n'
            '<tbody>' + "".join(risk_driver_rows) + '</tbody>\n'
            '</table>\n'
            '</div>\n'
            if risk_driver_rows else '<div class="sub">No elevated diagnostic risk drivers identified. Baseline operational risk is minimal.</div>\n'
        )
        + '</div>\n'
        + '<div class="section-title">Warnings &amp; Diagnostics</div>\n'
        + '<div class="card">\n'
        + warnings_html
        + '</div>\n'
        + copilot_html
        + page_end()
    )
    return html


# ============================================================
# HTTP REQUEST HANDLER
# ============================================================

class AppHandler(BaseHTTPRequestHandler):

    def send_html(self, content, status=200):
        payload = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_download(self, content_type, filename, payload):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    # ========================================================
    # GET
    # ========================================================

    def do_GET(self):
        global CURRENT_ERROR

        path = urllib.parse.urlparse(self.path).path

        if path == "/":
            err = CURRENT_ERROR
            CURRENT_ERROR = None
            self.send_html(upload_page(err or ""))
            return

        if path == "/preview":
            if CURRENT_DATAFRAME is None:
                self.send_response(302)
                self.send_header("Location", "/")
                self.end_headers()
                return

            err = CURRENT_ERROR
            CURRENT_ERROR = None
            self.send_html(
                preview_page(
                    CURRENT_DATAFRAME,
                    CURRENT_FILENAME,
                    CURRENT_MODE,
                    target=CURRENT_TARGET,
                    error=err,
                )
            )
            return

        if path == "/dashboard":
            if CURRENT_DATAFRAME is None or CURRENT_RESULT is None:
                self.send_response(302)
                self.send_header("Location", "/")
                self.end_headers()
                return

            self.send_html(dashboard_page())
            return

        if path == "/back-to-preview":
            if CURRENT_DATAFRAME is None:
                self.send_response(302)
                self.send_header("Location", "/")
                self.end_headers()
                return

            self.send_response(302)
            self.send_header("Location", "/preview")
            self.end_headers()
            return

        if path == "/back-to-upload":
            reset_state()
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
            return

        if path == "/download/dataset.csv":
            if CURRENT_DATAFRAME is None:
                self.send_html("<h1>No dataset loaded.</h1>", 404)
                return

            buffer = io.StringIO()
            CURRENT_DATAFRAME.to_csv(buffer, index=False)
            self.send_download(
                "text/csv; charset=utf-8",
                "analyzed_dataset.csv",
                buffer.getvalue().encode("utf-8-sig")
            )
            return

        if path == "/download/summary.csv":
            if CURRENT_RESULT is None:
                self.send_html("<h1>No analysis available.</h1>", 404)
                return

            performance = CURRENT_RESULT.get("model_performance", {})
            cv = CURRENT_RESULT.get("cross_validation", {})
            task_type = get_task_type(CURRENT_RESULT)

            if task_type == "regression":
                reg_m = CURRENT_RESULT.get("regression_metrics", {})
                rows = [
                    {"metric": "selected_model", "value": CURRENT_RESULT.get("selected_model", "")},
                    {"metric": "mae", "value": number(reg_m.get("mae", performance.get("mae", 0)))},
                    {"metric": "mse", "value": number(reg_m.get("mse", performance.get("mse", 0)))},
                    {"metric": "rmse", "value": number(reg_m.get("rmse", performance.get("rmse", 0)))},
                    {"metric": "r2", "value": number(reg_m.get("r2", performance.get("r2", 0)))},
                    {"metric": "cv_average", "value": number(cv.get("average_score", 0))},
                    {"metric": "cv_std", "value": f"{cv.get('standard_deviation', 0):.4f}"},
                    {"metric": "total_test_errors", "value": CURRENT_RESULT.get("total_test_errors", len(CURRENT_RESULT.get("prediction_errors", [])))},
                    {"metric": "overall_risk", "value": CURRENT_RESULT.get("overall_risk", "")},
                    {"metric": "risk_score", "value": CURRENT_RESULT.get("risk_score", 0)},
                ]
            else:
                rows = [
                    {"metric": "selected_model", "value": CURRENT_RESULT.get("selected_model", "")},
                    {"metric": "accuracy", "value": f"{percent(performance.get('accuracy', 0)):.2f}%"},
                    {"metric": "precision", "value": f"{percent(performance.get('precision', 0)):.2f}%"},
                    {"metric": "recall", "value": f"{percent(performance.get('recall', 0)):.2f}%"},
                    {"metric": "f1_score", "value": f"{percent(performance.get('f1_score', 0)):.2f}%"},
                    {"metric": "cv_average", "value": f"{percent(cv.get('average_score', 0)):.2f}%"},
                    {"metric": "cv_std", "value": f"{percent(cv.get('standard_deviation', 0)):.2f}%" if cv.get('standard_deviation') is not None else "0.00%"},
                    {"metric": "total_test_errors", "value": CURRENT_RESULT.get("total_test_errors", len(CURRENT_RESULT.get("prediction_errors", [])))},
                    {"metric": "high_confidence_error_count", "value": CURRENT_RESULT.get("high_confidence_errors_count", len(CURRENT_RESULT.get("error_analysis", {}).get("high_confidence_errors", [])))},
                    {"metric": "overall_risk", "value": CURRENT_RESULT.get("overall_risk", "")},
                    {"metric": "risk_score", "value": CURRENT_RESULT.get("risk_score", 0)},
                ]
                for c_name, c_data in CURRENT_RESULT.get("class_metrics", {}).items():
                    rows.append({"metric": f"class_{c_name}_precision", "value": f"{percent(c_data.get('precision', 0)):.2f}%"})
                    rows.append({"metric": f"class_{c_name}_recall", "value": f"{percent(c_data.get('recall', 0)):.2f}%"})
                    rows.append({"metric": f"class_{c_name}_f1_score", "value": f"{percent(c_data.get('f1_score', c_data.get('f1', 0))):.2f}%"})
                    rows.append({"metric": f"class_{c_name}_support", "value": c_data.get("support", 0)})
                    rows.append({"metric": f"class_{c_name}_dataset_proportion", "value": f"{percent(c_data.get('dataset_proportion', c_data.get('percentage_of_dataset', 0))):.2f}%"})

            buffer = io.StringIO()
            pd.DataFrame(rows).to_csv(buffer, index=False)
            self.send_download(
                "text/csv; charset=utf-8",
                "analysis_summary.csv",
                buffer.getvalue().encode("utf-8-sig")
            )
            return

        if path == "/download/prediction-errors.csv":
            if CURRENT_RESULT is None:
                self.send_html("<h1>No analysis available.</h1>", 404)
                return

            task_type = get_task_type(CURRENT_RESULT)
            rows = []

            for item in CURRENT_RESULT.get("prediction_errors", []):
                actual = item.get("actual", "")
                predicted = item.get("predicted", "")
                if task_type == "regression":
                    rows.append({
                        "row": record_row(item),
                        "actual": actual,
                        "predicted": predicted,
                        "absolute_error": item.get("absolute_error", 0),
                        "percentage_error": item.get("percentage_error", 0),
                        "residual": item.get("residual", 0),
                    })
                else:
                    rows.append({
                        "row": record_row(item),
                        "actual": actual,
                        "predicted": predicted,
                        "error_type": error_type(item),
                        "confidence_percent": f"{confidence(item):.2f}",
                    })

            buffer = io.StringIO()
            if task_type == "regression":
                cols = ["row", "actual", "predicted", "absolute_error", "percentage_error", "residual"]
            else:
                cols = ["row", "actual", "predicted", "error_type", "confidence_percent"]
            pd.DataFrame(rows, columns=cols).to_csv(buffer, index=False)
            self.send_download(
                "text/csv; charset=utf-8",
                "prediction_errors.csv",
                buffer.getvalue().encode("utf-8-sig")
            )
            return

        if path in {"/download/analysis.json", "/download-json"}:
            if CURRENT_RESULT is None or CURRENT_DATAFRAME is None:
                self.send_html("<h1>No analysis available.</h1>", 404)
                return

            payload = {
                "dataset": {
                    "filename": CURRENT_FILENAME,
                    "rows": len(CURRENT_DATAFRAME),
                    "columns": list(CURRENT_DATAFRAME.columns),
                    "target": CURRENT_TARGET,
                    "analysis_type": CURRENT_MODE,
                },
                "analysis_type": CURRENT_MODE,
                "data_quality": CURRENT_DATA_QUALITY,
                "analysis": CURRENT_RESULT,
            }

            content = json.dumps(payload, indent=2, default=str).encode("utf-8")
            self.send_download(
                "application/json; charset=utf-8",
                "full_analysis.json",
                content
            )
            return

        if path in {"/download/remediation_script.py", "/download/fix_pipeline.py"}:
            global CURRENT_FIX_SCRIPT
            global CURRENT_AI_EXPLAINER
            if CURRENT_FIX_SCRIPT is None:
                if CURRENT_RESULT is not None:
                    if CURRENT_AI_EXPLAINER is None:
                        CURRENT_AI_EXPLAINER = AIExplainer()
                    CURRENT_FIX_SCRIPT = CURRENT_AI_EXPLAINER.generate_fix_script(CURRENT_RESULT, CURRENT_DATA_QUALITY)
                else:
                    self.send_html("<h1>No analysis available.</h1>", 404)
                    return
            self.send_download(
                "text/x-python; charset=utf-8",
                "remediation_pipeline.py",
                CURRENT_FIX_SCRIPT.encode("utf-8")
            )
            return

        self.send_html("<h1>Not Found</h1>", 404)

    # ========================================================
    # POST
    # ========================================================

    def do_POST(self):
        global CURRENT_DATAFRAME
        global CURRENT_RESULT
        global CURRENT_DATA_QUALITY
        global CURRENT_FILENAME
        global CURRENT_TARGET
        global CURRENT_MODE
        global CURRENT_ERROR
        global CURRENT_AI_EXPLAINER
        global CURRENT_AI_REPORT
        global CURRENT_FIX_SCRIPT

        if self.path == "/api/copilot/chat":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length > 0 else b""
                question = ""
                content_type = self.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    try:
                        data = json.loads(body.decode("utf-8", errors="replace"))
                        question = data.get("question", "")
                    except Exception:
                        pass
                if not question:
                    fields = urllib.parse.parse_qs(body.decode("utf-8", errors="replace"))
                    q_list = fields.get("question", [""])
                    question = q_list[0].strip() if q_list else ""

                if not question:
                    raise ValueError("Please provide a question.")

                if CURRENT_RESULT is None:
                    raise ValueError("No active dataset analysis available. Please run an analysis first.")

                if CURRENT_AI_EXPLAINER is None:
                    CURRENT_AI_EXPLAINER = AIExplainer()

                reply = CURRENT_AI_EXPLAINER.ask_copilot(question, CURRENT_RESULT, CURRENT_DATA_QUALITY)
                payload = {
                    "status": "success",
                    "question": question,
                    "reply": reply,
                    "provider": CURRENT_AI_EXPLAINER.get_active_provider_name()
                }
                payload_bytes = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload_bytes)))
                self.end_headers()
                self.wfile.write(payload_bytes)
                return
            except Exception as exc:
                err_payload = json.dumps({"status": "error", "reply": f"Copilot Error: {str(exc)}"}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(err_payload)))
                self.end_headers()
                self.wfile.write(err_payload)
                return

        if self.path == "/upload":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("No upload was received.")
                if length > 100 * 1024 * 1024:
                    raise ValueError("File is too large. Maximum size is 100 MB.")

                body = self.rfile.read(length)
                content_type = self.headers.get("Content-Type", "")

                fields, files = parse_multipart(body, content_type)

                uploaded = files.get("dataset")
                if not uploaded:
                    raise ValueError("Please choose a CSV or Excel file.")

                filename = uploaded["filename"]
                content = uploaded["content"]

                df = load_uploaded_dataframe(filename, content)

                if df.empty:
                    raise ValueError("The uploaded dataset is empty.")

                if len(df.columns) < 2:
                    raise ValueError(
                        "The dataset must contain at least one feature column and one target column."
                    )

                mode_val = fields.get("mode", "auto")
                if isinstance(mode_val, list):
                    mode_val = mode_val[0] if mode_val else "auto"
                mode = str(mode_val).strip().lower()

                if mode not in {"classification", "regression", "auto"}:
                    mode = "auto"

                CURRENT_DATAFRAME = df
                CURRENT_FILENAME = filename
                CURRENT_TARGET = None
                CURRENT_RESULT = None
                CURRENT_DATA_QUALITY = None
                CURRENT_MODE = mode
                CURRENT_ERROR = None

                self.send_response(302)
                self.send_header("Location", "/preview")
                self.end_headers()
                return

            except Exception as exc:
                CURRENT_ERROR = str(exc)
                self.send_html(upload_page(CURRENT_ERROR), 400)
                return

        if self.path == "/analyze":
            try:
                if CURRENT_DATAFRAME is None:
                    self.send_response(302)
                    self.send_header("Location", "/")
                    self.end_headers()
                    return

                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                fields = urllib.parse.parse_qs(body.decode("utf-8", errors="replace"))

                target_list = fields.get("target", [""])
                target = target_list[0].strip() if target_list else ""

                mode_list = fields.get("mode", [CURRENT_MODE])
                mode = mode_list[0].strip().lower() if mode_list else CURRENT_MODE

                if mode not in {"classification", "regression", "auto"}:
                    mode = "auto"

                if not target:
                    raise ValueError("Please choose a target column.")

                if target not in CURRENT_DATAFRAME.columns:
                    raise ValueError(f"Target column '{target}' was not found in the dataset.")

                if CURRENT_DATAFRAME[target].nunique(dropna=True) < 2:
                    raise ValueError("The target column must contain at least two distinct values.")

                CURRENT_TARGET = target
                CURRENT_MODE = mode
                CURRENT_RESULT = analyze_model(
                    CURRENT_DATAFRAME,
                    target,
                    analysis_type=mode
                )
                CURRENT_DATA_QUALITY = CURRENT_RESULT.get("data_quality", analyze_data_quality(CURRENT_DATAFRAME))
                CURRENT_AI_REPORT = None
                CURRENT_FIX_SCRIPT = None
                CURRENT_ERROR = None

                self.send_response(302)
                self.send_header("Location", "/dashboard")
                self.end_headers()
                return

            except Exception as exc:
                CURRENT_ERROR = str(exc)
                self.send_response(302)
                self.send_header("Location", "/preview")
                self.end_headers()
                return

        self.send_html("<h1>Not Found</h1>", 404)


# ============================================================
# APPLICATION ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("       AI MODEL ROOT-CAUSE ANALYZER")
    print("=" * 60)
    print()
    print(f"Application running at: http://{HOST}:{PORT}")
    print()
    print("Upload a CSV or Excel dataset and select its analysis mode.")
    print()
    print("Press CTRL+C to stop the application.")
    print()

    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nApplication stopped.")
    finally:
        server.server_close()