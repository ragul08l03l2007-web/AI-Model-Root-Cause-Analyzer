#!/usr/bin/env bash
# test.sh - Run full acceptance, model selection, and diagnostic test suite

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
echo " 1. Checking Environment"
echo "============================================================"
$PYTHON test_environment.py

echo ""
echo "============================================================"
echo " 2. Running Model Selection Verification Tests"
echo "============================================================"
$PYTHON test_model_selection.py

echo ""
echo "============================================================"
echo " 3. Running Multi-Dataset Generalization Tests"
echo "============================================================"
$PYTHON test_generalization.py

echo ""
echo "============================================================"
echo " 4. Running Acceptance Verification (Issues A-Q)"
echo "============================================================"
$PYTHON test_acceptance_issues_a_to_q.py

echo ""
echo "============================================================"
echo " 5. Running Web Server Endpoints & Integration Tests"
echo "============================================================"
$PYTHON test_server_endpoints.py

echo ""
echo "============================================================"
echo " 6. Running Multi-Dataset Diagnostic Validation Tests"
echo "============================================================"
$PYTHON test_multi_dataset_validation.py

echo ""
echo "============================================================"
echo " 7. Running Diagnostic Deduplication & Grouping Tests"
echo "============================================================"
$PYTHON test_diagnostic_deduplication.py

echo ""
echo "============================================================"
echo " 8. Running Universal Dataset Sensitivity Test Suite"
echo "============================================================"
$PYTHON test_universal_datasets.py

echo ""
echo "============================================================"
echo " ALL TESTS PASSED SUCCESSFULLY!"
echo "============================================================"


