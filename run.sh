#!/usr/bin/env bash
# run.sh - AI Model Root-Cause Analyzer Web Server Runner

set -e

# Detect virtual environment
if [ -d ".venv" ]; then
    if [ -f ".venv/Scripts/python.exe" ]; then
        PYTHON=".venv/Scripts/python.exe"
    elif [ -f ".venv/bin/python" ]; then
        PYTHON=".venv/bin/python"
    else
        PYTHON="python"
    fi
else
    PYTHON="python3"
fi

echo "============================================================"
echo " Starting AI Model Root-Cause Analyzer on http://localhost:8000"
echo "============================================================"
$PYTHON app.py
