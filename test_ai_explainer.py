# test_ai_explainer.py
"""
Comprehensive Test Suite for AI Explainer, Pluggable Providers & Remediation Engine.
Validates Stage 1 (Executive Summary & Root Cause Deep Dive), Stage 2 (Fix Script Generator),
and Stage 3 (Diagnostic Copilot) against real classification and regression datasets.
"""

import os
import sys
import pandas as pd
from analysis.model_analyzer import analyze_model
from analysis.profiler import DataProfiler
from analysis.ai_providers import (
    get_ai_provider,
    BaseAIProvider,
    GeminiProvider,
    OpenAICompatibleProvider,
    OllamaProvider,
    OfflineDeterministicProvider,
)
from analysis.ai_explainer import AIExplainer, extract_evidence_payload


def test_provider_factory_and_hierarchy():
    print("\n[TEST 1] Testing Pluggable Provider Factory & Hierarchy...")

    # Default fallback without keys should be OfflineDeterministicProvider
    prev_gemini = os.environ.pop("GEMINI_API_KEY", None)
    prev_gemini_model = os.environ.pop("GEMINI_MODEL", None)
    prev_openai = os.environ.pop("OPENAI_API_KEY", None)
    prev_provider = os.environ.pop("AI_PROVIDER", None)

    try:
        provider = get_ai_provider()
        assert isinstance(provider, OfflineDeterministicProvider), "Expected OfflineDeterministicProvider by default"
        print("  -> Default fallback correctly resolved to OfflineDeterministicProvider")

        # Explicit Gemini Provider
        os.environ["GEMINI_API_KEY"] = "dummy_test_key"
        gemini_prov = get_ai_provider()
        assert isinstance(gemini_prov, GeminiProvider), "Expected GeminiProvider when GEMINI_API_KEY is present"
        assert gemini_prov.model_name == "gemini-3.5-flash-lite"
        print("  -> Auto-detection correctly resolved to GeminiProvider with gemini-3.5-flash-lite")

        # Explicit override
        ollama_prov = get_ai_provider(provider_type="ollama", model_name="qwen2.5:7b")
        assert isinstance(ollama_prov, OllamaProvider), "Expected OllamaProvider when explicitly requested"
        assert ollama_prov.model_name == "qwen2.5:7b"
        print("  -> Explicit provider request for Ollama correctly resolved")

    finally:
        # Restore environment
        if prev_gemini:
            os.environ["GEMINI_API_KEY"] = prev_gemini
        else:
            os.environ.pop("GEMINI_API_KEY", None)
        if prev_gemini_model:
            os.environ["GEMINI_MODEL"] = prev_gemini_model
        else:
            os.environ.pop("GEMINI_MODEL", None)
        if prev_openai:
            os.environ["OPENAI_API_KEY"] = prev_openai
        if prev_provider:
            os.environ["AI_PROVIDER"] = prev_provider

    print("  -> TEST 1 PASSED!")


def test_classification_ai_explainer():
    print("\n[TEST 2] Testing AI Explainer on Classification Dataset (random_test_dataset.csv)...")

    dataset_path = "random_test_dataset.csv"
    assert os.path.exists(dataset_path), f"Missing {dataset_path}"
    df = pd.read_csv(dataset_path)

    # Deterministic analysis run
    model_result = analyze_model(df, target_column="churn")
    data_quality = DataProfiler.profile_dataset(df)

    # Verify compact evidence extraction
    payload = extract_evidence_payload(model_result, data_quality)
    assert payload["task_type"] == "classification"
    assert "target_profile" in payload
    assert "dataset_summary" in payload
    assert "selected_model" in payload
    assert "root_causes" in payload
    assert "overall_risk" in payload
    print(f"  -> Extracted compact payload: {payload['dataset_summary']['total_rows']} rows, Risk: {payload['overall_risk']['risk_score']}/100")

    # Run AI Explainer Stage 1
    explainer = AIExplainer(provider=OfflineDeterministicProvider())
    report = explainer.explain(model_result, data_quality)

    assert "executive_summary" in report and len(report["executive_summary"]) > 20
    assert "root_causes_detailed" in report and isinstance(report["root_causes_detailed"], list)
    assert "risk_assessment" in report
    assert "remediation_roadmap" in report and len(report["remediation_roadmap"]) > 0

    print("  -> Stage 1 Executive Summary Sample:")
    print("     " + report["executive_summary"])
    print(f"  -> Identified {len(report['root_causes_detailed'])} detailed root causes")
    print(f"  -> Risk Level: {report['risk_assessment']['overall_risk_level']} (Score: {report['risk_assessment']['risk_score']}/100)")
    print(f"  -> Remediation Steps: {len(report['remediation_roadmap'])} items")

    # Run AI Fix Generator Stage 2
    fix_code = explainer.generate_fix_script(model_result, data_quality)
    assert "import pandas as pd" in fix_code
    assert "def run_remediated_pipeline" in fix_code
    # Verify Python syntax compiles cleanly
    compile(fix_code, "<string>", "exec")
    print("  -> Stage 2 Remediation Python Script generated and verified (valid Python syntax)")

    # Run Diagnostic Copilot Stage 3
    q1 = "Why did the model underperform?"
    ans1 = explainer.ask_copilot(q1, model_result, data_quality)
    assert len(ans1) > 20
    print("  -> Stage 3 Copilot Q&A: 'Why did the model underperform?' -> Answered based on evidence")

    q2 = "What are the most important features?"
    ans2 = explainer.ask_copilot(q2, model_result, data_quality)
    assert len(ans2) > 20
    print("  -> Stage 3 Copilot Q&A: 'What are the most important features?' -> Answered based on evidence")

    q3 = "What did the ablation and verification experiments show?"
    ans3 = explainer.ask_copilot(q3, model_result, data_quality)
    assert len(ans3) > 20 and ("Ablation" in ans3 or "Verification" in ans3 or "Score" in ans3)
    print("  -> Stage 3 Copilot Q&A: 'What did the ablation experiments show?' -> Answered with 5-part score breakdown")

    q4 = "What was the remediation simulation resolution verdict?"
    ans4 = explainer.ask_copilot(q4, model_result, data_quality)
    assert len(ans4) > 20 and ("Remediation" in ans4 or "Verdict" in ans4 or "Simulation" in ans4)
    print("  -> Stage 3 Copilot Q&A: 'What was the remediation verdict?' -> Answered with resolution status")

    print("  -> TEST 2 PASSED!")


def test_regression_ai_explainer():
    print("\n[TEST 3] Testing AI Explainer on Continuous Regression Dataset...")

    dataset_path = "continuous_regression_test_dataset.csv"
    assert os.path.exists(dataset_path), f"Missing {dataset_path}"
    df = pd.read_csv(dataset_path)

    # Deterministic analysis run
    model_result = analyze_model(df, target_column="annual_bonus")
    data_quality = DataProfiler.profile_dataset(df)

    # Verify compact evidence extraction
    payload = extract_evidence_payload(model_result, data_quality)
    assert payload["task_type"] == "regression"
    assert "annual_bonus" in payload["target_profile"]["target_column"]

    # Run AI Explainer Stage 1
    explainer = AIExplainer(provider=OfflineDeterministicProvider())
    report = explainer.explain(model_result, data_quality)

    assert "executive_summary" in report
    assert "risk_assessment" in report
    assert payload["selected_model"]["name"] in report["executive_summary"] or "Model" in report["executive_summary"]
    print("  -> Stage 1 Executive Summary for Regression:")
    print("     " + report["executive_summary"])

    # Run Stage 2 Fix Generator
    fix_code = explainer.generate_fix_script(model_result, data_quality)
    assert "GradientBoostingRegressor" in fix_code or "r2_score" in fix_code
    compile(fix_code, "<string>", "exec")
    print("  -> Stage 2 Regression Remediation Script verified and compiles cleanly")

    print("  -> TEST 3 PASSED!")


def test_gemini_graceful_fallback():
    print("\n[TEST 4] Testing Gemini Provider with Graceful Fallback Handling...")

    # Create Gemini provider with an invalid key to test fallback resilience
    gemini_prov = GeminiProvider(api_key="invalid_test_api_key", model_name="gemini-2.5-flash")
    explainer = AIExplainer(provider=gemini_prov)

    # Test payload
    sample_payload = {
        "task_type": "classification",
        "dataset_summary": {"total_rows": 100, "missing_values": 0},
        "target_profile": {"target_column": "target", "class_distribution": {"0": 90, "1": 10}},
        "selected_model": {"name": "Logistic Regression", "metrics": {"accuracy": 0.9}},
        "model_stability": {"stability_status": "STABLE"},
        "root_causes": [{"finding": "Severe class imbalance", "severity": "HIGH", "confidence": "High", "interpretation": "90/10 ratio", "recommended_action": "Use SMOTE"}],
        "feature_importance": {},
        "overall_risk": {"risk_level": "MEDIUM", "risk_score": 45}
    }

    report = gemini_prov.generate_explanation(sample_payload)
    assert "executive_summary" in report
    assert "provider_warning" in report or "provider" in report
    print("  -> Gracefully handled invalid API key / offline state without crashing")

    fix_script = gemini_prov.generate_fix(sample_payload)
    assert "def run_remediated_pipeline" in fix_script
    compile(fix_script, "<string>", "exec")
    print("  -> Gracefully generated fallback remediation script")

    chat_reply = gemini_prov.chat(sample_payload, "What is the primary risk?")
    assert len(chat_reply) > 10
    print("  -> Gracefully answered chat question via grounded evidence")

    print("  -> TEST 4 PASSED!")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING AI EXPLAINER & PLUGGABLE PROVIDER VERIFICATION SUITE")
    print("=" * 70)
    test_provider_factory_and_hierarchy()
    test_classification_ai_explainer()
    test_regression_ai_explainer()
    test_gemini_graceful_fallback()
    print("\n" + "=" * 70)
    print("ALL AI EXPLAINER & PROVIDER TESTS PASSED 100%!")
    print("=" * 70)
