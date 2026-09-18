# analysis/ai_providers.py
"""
Pluggable AI Provider Architecture for AI Model Root-Cause Analyzer.
Supports Google Gemini, OpenAI-compatible APIs (OpenAI, DeepSeek, Groq, OpenRouter),
Local Ollama, and a zero-dependency deterministic Offline Fallback Provider.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import os
import json
import urllib.request
import urllib.error
import re


def _load_dotenv_if_present():
    """Lightweight built-in .env parser (no extra dependencies needed)."""
    env_paths = [".env", os.path.join(os.path.dirname(__file__), "..", ".env")]
    for p in env_paths:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass


_load_dotenv_if_present()


class BaseAIProvider(ABC):
    """Abstract Base Class for AI Diagnostic & Remediation Providers."""

    def __init__(self, model_name: str = "default", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or ""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the human-readable name of the provider."""
        pass

    @abstractmethod
    def generate_explanation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receives a sanitized diagnostic evidence payload and produces:
        - executive_summary (str)
        - root_causes_detailed (List[Dict[str, Any]])
        - risk_assessment (Dict[str, Any])
        - remediation_roadmap (List[str])
        """
        pass

    @abstractmethod
    def generate_fix(self, payload: Dict[str, Any]) -> str:
        """
        Generates a complete, executable Python data-remediation and training script
        tailored specifically to the detected issues.
        """
        pass

    @abstractmethod
    def chat(
        self,
        payload: Dict[str, Any],
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Answers developer questions strictly grounded in the diagnostic evidence payload.
        """
        pass


class OfflineDeterministicProvider(BaseAIProvider):
    """
    Built-in, zero-dependency offline fallback engine.
    Synthesizes rich, professional executive reports and runnable remediation scripts
    directly from statistical evidence when no API key or network is available.
    """

    @property
    def provider_name(self) -> str:
        return "Deterministic Offline Engine (Local Fallback)"

    def generate_explanation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        task_type = payload.get("task_type", "classification")
        target_info = payload.get("target_profile", {})
        model_name = payload.get("selected_model", {}).get("name", "Predictive Model")
        metrics = payload.get("selected_model", {}).get("metrics", {})
        root_causes = payload.get("root_causes", [])
        risk_level = payload.get("overall_risk", {}).get("risk_level", "LOW")
        risk_score = payload.get("overall_risk", {}).get("risk_score", 0)
        data_summary = payload.get("dataset_summary", {})
        total_rows = data_summary.get("total_rows", 0)
        target_col = target_info.get("target_column", "target")
        verif_exps = payload.get("verification_experiments", [])
        remed_sim = payload.get("remediation_simulation", {})

        # Build Executive Summary
        if not root_causes and not verif_exps:
            summary = (
                f"The diagnostic audit confirmed that the {model_name} on target '{target_col}' "
                f"({total_rows} records) is performing in a healthy regime with an overall risk score of {risk_score}/100. "
                "No critical data leakage, high-severity class distortion, or catastrophic overfitting was detected."
            )
        else:
            top_findings = ", ".join([rc.get("finding", "") for rc in root_causes[:2]]) if root_causes else "Feature reliance verified through controlled trials"
            verif_phrase = ""
            if verif_exps:
                verified_feats = [e.get("candidate") for e in verif_exps if e.get("verdict") == "VERIFIED MODEL RELIANCE"]
                refuted_feats = [e.get("candidate") for e in verif_exps if e.get("verdict") == "NO MEASURABLE MODEL RELIANCE"]
                if verified_feats:
                    verif_phrase = f" Controlled empirical experiments verified strong operational model reliance on '{', '.join(verified_feats)}'."
                if refuted_feats:
                    verif_phrase += f" Interventions confirmed no measurable reliance on '{', '.join(refuted_feats)}'."

            remed_phrase = ""
            if remed_sim and remed_sim.get("resolution_verdict") != "None":
                remed_phrase = f" Closed-loop remediation simulation verdict: {remed_sim.get('resolution_verdict')}."

            summary = (
                f"The diagnostic audit identified {len(root_causes)} primary issue(s) affecting {model_name} "
                f"trained on '{target_col}' ({total_rows} observations). The model is operating under "
                f"a {risk_level} risk level (Score: {risk_score}/100). The most critical vulnerabilities are: {top_findings}.{verif_phrase}{remed_phrase} "
                "Targeted data engineering and hyperparameter adjustments are required before production deployment."
            )

        # Build Detailed Root-Cause Analyses
        detailed_rcs = []
        for rc in root_causes:
            detailed_rcs.append({
                "category": rc.get("category", "General"),
                "finding": rc.get("finding", ""),
                "severity": rc.get("severity", "MEDIUM"),
                "confidence": rc.get("confidence", "Medium"),
                "why_it_matters": rc.get("interpretation", "Impacts model reliability and generalization."),
                "evidence_summary": rc.get("evidence", "Statistical threshold breach detected."),
                "business_risk": rc.get("impact", "Reduced predictive accuracy and elevated false predictions."),
                "recommended_action": rc.get("recommended_action", "Apply data transformation and re-evaluate.")
            })

        # Append structured verification findings if available
        for exp in verif_exps:
            cand = exp.get("candidate", exp.get("candidate_feature"))
            score = exp.get("evidence_score", 0)
            verdict = exp.get("verdict", "")
            decomp = exp.get("score_decomposition", {})
            if verdict == "VERIFIED MODEL RELIANCE":
                detailed_rcs.append({
                    "category": "Experimental Verification",
                    "finding": f"Empirical Model Reliance Verified for '{cand}' (Score: {score}/100)",
                    "severity": "HIGH" if score >= 80 else "MEDIUM",
                    "confidence": "High",
                    "why_it_matters": (
                        f"The model exhibits strong empirical reliance on '{cand}' under the tested dataset, split, and interventions "
                        f"(ablation delta: {exp.get('ablation_delta', 0):+.4f}, permutation delta: {exp.get('permutation_delta', 0):+.4f})."
                    ),
                    "evidence_summary": (
                        f"Evidence Score: {score}/100 [Ablation: {decomp.get('ablation_points', decomp.get('ablation_evidence_pts', 0))}/35, "
                        f"Permutation: {decomp.get('permutation_points', decomp.get('permutation_evidence_pts', 0))}/35, "
                        f"Control: {decomp.get('control_points', decomp.get('control_specificity_pts', 0))}/15, "
                        f"Stability: {decomp.get('stability_points', decomp.get('measurement_stability_pts', 0))}/10, "
                        f"Consistency: {decomp.get('consistency_points', decomp.get('experimental_consistency_pts', 0))}/5]. "
                        f"Measurement stability test (10% σ jitter) produced {exp.get('prediction_flip_rate_pct', 0.0):.1f}% prediction flips (stable)."
                    ),
                    "business_risk": (
                        f"Operational consideration: Because the selected model exhibits strong empirical reliance on '{cand}', "
                        f"production monitoring should track changes in its distribution and data-collection process if deployed. "
                        f"The current experiments do not establish that such shifts will degrade production performance."
                    ),
                    "recommended_action": (
                        f"If deployed, monitor distribution drift and data-quality changes in '{cand}' over time, audit data collection integrity, "
                        f"and evaluate alternative model configurations if reducing feature concentration is operationally necessary."
                    )
                })
            elif verdict == "NO MEASURABLE MODEL RELIANCE":
                detailed_rcs.append({
                    "category": "Experimental Verification",
                    "finding": f"No Measurable Model Reliance on '{cand}' (Score: 0/100)",
                    "severity": "LOW",
                    "confidence": "High",
                    "why_it_matters": "Under the tested split and interventions, removing or permuting this feature produced no measurable change in the evaluation metric.",
                    "evidence_summary": f"Ablation Delta: {exp.get('ablation_delta', 0):+.4f}, Permutation Delta: {exp.get('permutation_delta', 0):+.4f}, Control Delta: {exp.get('control_delta', 0):+.4f}.",
                    "business_risk": "No measurable operational risk detected under current model evaluation partitions.",
                    "recommended_action": "No measurable model reliance detected under current testing; retain or re-evaluate based on domain requirements and future data/model changes."
                })

        # Risk Assessment
        risk_assessment = {
            "overall_risk_level": risk_level,
            "risk_score": risk_score,
            "deployment_readiness": "Ready for Staging" if risk_score < 30 else ("Requires Review" if risk_score < 60 else "Block Deployment"),
            "critical_vulnerabilities_count": sum(1 for rc in detailed_rcs if rc.get("severity") in ("HIGH", "CRITICAL")),
            "governance_note": "Grounded directly in deterministic statistical tests and controlled counterfactual interventions."
        }

        # Remediation Roadmap (Distinguishing Completed Diagnostics from Recommended Future Actions)
        roadmap = []
        seen_actions = set()

        # Add domain-specific future engineering actions from general root causes
        for rc in detailed_rcs:
            if rc.get("category") in ("Diagnostic Observation", "Experimental Verification"):
                continue
            act = rc.get("recommended_action", "").strip()
            if act and act not in seen_actions:
                seen_actions.add(act)
                roadmap.append(f"Step {len(roadmap) + 1} [{rc.get('severity', 'MEDIUM')}]: {act}")

        # Add forward-looking future actions for verified model reliance candidates
        for exp in verif_exps:
            cand = exp.get("candidate", exp.get("candidate_feature"))
            verdict = exp.get("verdict", "")
            if verdict == "VERIFIED MODEL RELIANCE":
                future_actions = [
                    f"Audit data collection integrity and pipeline stability specifically for '{cand}'.",
                    f"Evaluate alternative model configurations, feature selection, or regularization techniques if reducing feature concentration is operationally necessary.",
                    f"If deployed, monitor distribution drift and data-quality changes in '{cand}' over time.",
                    f"Use the completed measurement-stability result as a baseline reference when evaluating realistic production measurement-error scenarios.",
                    f"Evaluate model behavior under realistic production error and distribution-shift scenarios when representative deployment data becomes available."
                ]
                for act in future_actions:
                    if act not in seen_actions:
                        seen_actions.add(act)
                        roadmap.append(f"Step {len(roadmap) + 1} [MEDIUM]: {act}")

        if not roadmap:
            roadmap.append("Step 1 [LOW]: If deployed, monitor distribution drift and data-quality changes in primary features and target distributions over time.")

        return {
            "provider": self.provider_name,
            "executive_summary": summary,
            "root_causes_detailed": detailed_rcs,
            "risk_assessment": risk_assessment,
            "remediation_roadmap": roadmap
        }

    def generate_fix(self, payload: Dict[str, Any]) -> str:
        task_type = payload.get("task_type", "classification")
        target_info = payload.get("target_profile", {})
        target_col = target_info.get("target_column", "target")
        model_name = payload.get("selected_model", {}).get("name", "Logistic Regression")
        root_causes = payload.get("root_causes", [])
        data_summary = payload.get("dataset_summary", {})

        # Identify required fixes
        has_missing = data_summary.get("missing_values", 0) > 0
        has_imbalance = any("imbalance" in rc.get("finding", "").lower() or "disparity" in rc.get("finding", "").lower() for rc in root_causes)
        has_leakage = any("leakage" in rc.get("finding", "").lower() for rc in root_causes)
        leakage_cols = []
        for rc in root_causes:
            if "leakage" in rc.get("finding", "").lower():
                m = re.search(r"'(.*?)'", rc.get("finding", ""))
                if m:
                    leakage_cols.append(m.group(1))

        code_lines = [
            "# ====================================================================",
            "# AUTO-GENERATED ML REMEDIATION PIPELINE",
            "# Grounded in AI Model Root-Cause Diagnostic Evidence",
            "# ====================================================================",
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.model_selection import train_test_split, cross_val_score",
            "from sklearn.pipeline import Pipeline",
            "from sklearn.compose import ColumnTransformer",
            "from sklearn.impute import SimpleImputer",
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder",
        ]

        if task_type == "regression":
            code_lines.extend([
                "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor",
                "from sklearn.linear_model import Ridge",
                "from sklearn.metrics import root_mean_squared_error, r2_score",
            ])
        else:
            code_lines.extend([
                "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier",
                "from sklearn.linear_model import LogisticRegression",
                "from sklearn.metrics import classification_report, f1_score",
            ])

        code_lines.extend([
            "",
            "def run_remediated_pipeline(dataset_path: str):",
            "    print('[1/5] Loading and inspecting dataset...')",
            "    df = pd.read_csv(dataset_path)",
            f"    target_col = '{target_col}'",
            "",
            "    if target_col not in df.columns:",
            "        raise ValueError(f'Target column {target_col} not found in dataset')",
            "",
        ])

        if leakage_cols:
            code_lines.extend([
                "    # [2/5] REMEDIATION: Remove identified target-leakage features",
                f"    leakage_features = {leakage_cols}",
                "    existing_leakage = [c for c in leakage_features if c in df.columns]",
                "    if existing_leakage:",
                "        print(f'Dropping leakage columns: {existing_leakage}')",
                "        df = df.drop(columns=existing_leakage)",
                "",
            ])
        else:
            code_lines.extend([
                "    # [2/5] Deduplicate observations if present",
                "    initial_len = len(df)",
                "    df = df.drop_duplicates()",
                "    if len(df) < initial_len:",
                "        print(f'Removed {initial_len - len(df)} duplicate records.')",
                "",
            ])

        code_lines.extend([
            "    # [3/5] Separate features and target",
            "    X = df.drop(columns=[target_col])",
            "    y = df[target_col]",
            "",
            "    # Identify feature types",
            "    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()",
            "    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()",
            "",
            "    # Build robust transformers",
            "    num_transformer = Pipeline(steps=[",
            "        ('imputer', SimpleImputer(strategy='median')),",
            "        ('scaler', StandardScaler())",
            "    ])",
            "    cat_transformer = Pipeline(steps=[",
            "        ('imputer', SimpleImputer(strategy='most_frequent')),",
            "        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))",
            "    ])",
            "",
            "    preprocessor = ColumnTransformer(transformers=[",
            "        ('num', num_transformer, num_cols),",
            "        ('cat', cat_transformer, cat_cols)",
            "    ])",
            "",
            "    # [4/5] Train/Test Split",
        ])

        if task_type == "classification":
            stratify_arg = "stratify=y" if len(target_info.get("class_distribution", {})) >= 2 else "None"
            code_lines.extend([
                f"    X_train, X_test, y_train, y_test = train_test_split(",
                f"        X, y, test_size=0.2, random_state=42, {stratify_arg}",
                f"    )",
                "",
                "    # [5/5] Assemble Remediated Model Pipeline",
                f"    # Addressing class balance and regularization",
                "    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)",
                "    pipeline = Pipeline(steps=[('prep', preprocessor), ('clf', model)])",
                "",
                "    print('Training model pipeline...')",
                "    pipeline.fit(X_train, y_train)",
                "    y_pred = pipeline.predict(X_test)",
                "",
                "    print('\\n--- REMEDIATED MODEL EVALUATION ---')",
                "    print(classification_report(y_test, y_pred))",
                "    print(f'Weighted F1 Score: {f1_score(y_test, y_pred, average=\"weighted\"):.4f}')",
            ])
        else:
            code_lines.extend([
                "    X_train, X_test, y_train, y_test = train_test_split(",
                "        X, y, test_size=0.2, random_state=42",
                "    )",
                "",
                "    # [5/5] Assemble Remediated Regression Pipeline",
                "    model = GradientBoostingRegressor(n_estimators=100, random_state=42)",
                "    pipeline = Pipeline(steps=[('prep', preprocessor), ('reg', model)])",
                "",
                "    print('Training model pipeline...')",
                "    pipeline.fit(X_train, y_train)",
                "    y_pred = pipeline.predict(X_test)",
                "",
                "    print('\\n--- REMEDIATED REGRESSION EVALUATION ---')",
                "    print(f'Test R^2 Score : {r2_score(y_test, y_pred):.4f}')",
                "    print(f'Test RMSE      : {root_mean_squared_error(y_test, y_pred):.4f}')",
            ])

        code_lines.extend([
            "    return pipeline",
            "",
            "if __name__ == '__main__':",
            "    import sys",
            "    dataset_file = sys.argv[1] if len(sys.argv) > 1 else 'dataset.csv'",
            "    run_remediated_pipeline(dataset_file)",
        ])

        return "\n".join(code_lines)

    def chat(
        self,
        payload: Dict[str, Any],
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        q_lower = question.lower()
        root_causes = payload.get("root_causes", [])
        metrics = payload.get("selected_model", {}).get("metrics", {})
        model_name = payload.get("selected_model", {}).get("name", "Model")
        feat_imp = payload.get("feature_importance", {})
        task_type = payload.get("task_type", "classification")
        verif_exps = payload.get("verification_experiments", [])
        remed_sim = payload.get("remediation_simulation", {})

        if any(w in q_lower for w in ["ablation", "permutation", "verif", "experiment", "evidence score", "reliance", "noise", "jitter", "control"]):
            if not verif_exps:
                return "No targeted verification experiments were conducted for this model run."
            lines = ["**Empirical Root-Cause Verification & Intervention Trials**:"]
            for exp in verif_exps:
                cand = exp.get("candidate")
                verdict = exp.get("verdict")
                score = exp.get("evidence_score", 0)
                decomp = exp.get("score_decomposition", {})
                lines.append(f"\n- **Candidate '{cand}'** → Verdict: `{verdict}` (Evidence Score: {score}/100)")
                lines.append(f"  • Ablation Drop: {exp.get('ablation_delta', 0):+.4f} (Score Contribution: {decomp.get('ablation_evidence_pts', 0)}/35 pts)")
                lines.append(f"  • Permutation Drop: {exp.get('permutation_delta', 0):+.4f} (Score Contribution: {decomp.get('permutation_evidence_pts', 0)}/35 pts)")
                lines.append(f"  • Control ({exp.get('control_feature')}): {exp.get('control_delta', 0):+.4f} (Score Contribution: {decomp.get('control_specificity_pts', 0)}/15 pts)")
                lines.append(f"  • Measurement Stability (10% σ Jitter): Flip Rate = {exp.get('prediction_flip_rate_pct', 0.0)}% (Score Contribution: {decomp.get('measurement_stability_pts', 0)}/10 pts)")
                lines.append(f"  • Consistency Contribution: {decomp.get('experimental_consistency_pts', 0)}/5 pts")
            return "\n".join(lines)

        elif any(w in q_lower for w in ["remediat", "simulation", "fix", "resolution", "pipeline"]):
            if not remed_sim or remed_sim.get("resolution_verdict") == "None":
                return f"Closed-loop remediation simulated standard data hygiene and regularized modeling for {model_name}."
            verdict = remed_sim.get("resolution_verdict")
            t_delta = remed_sim.get("test_metric_delta", 0.0)
            g_red = remed_sim.get("generalization_gap_reduction", 0.0)
            return (
                f"**Closed-Loop Remediation Simulation Results**:\n"
                f"- **Resolution Verdict**: `{verdict}`\n"
                f"- **Held-Out Test Metric Delta**: {t_delta:+.4f}\n"
                f"- **Generalization Gap Reduction**: {g_red:+.4f}\n"
                f"The automated fix pipeline was simulated end-to-end to verify resolution before suggesting deployment."
            )

        elif "why" in q_lower or "poor" in q_lower or "fail" in q_lower or "cause" in q_lower:
            if not root_causes:
                return (
                    f"According to the diagnostic evidence, the {model_name} did not exhibit major failure patterns. "
                    f"Overall risk score is {payload.get('overall_risk', {}).get('risk_score', 0)}/100."
                )
            reasons = "\n".join([f"- **{rc.get('finding')}** (Severity: {rc.get('severity')}): {rc.get('interpretation')}" for rc in root_causes])
            return f"Based on the empirical diagnostic run, here are the root causes behind model behavior:\n\n{reasons}"

        elif "feature" in q_lower or "important" in q_lower:
            if not feat_imp:
                return "Feature importance calculation did not find strong dominant single features."
            top_feats = list(feat_imp.items())[:5]
            feat_text = "\n".join([f"- **{k}**: Importance score {v.get('importance', 0) if isinstance(v, dict) else v}" for k, v in top_feats])
            return f"Top predictive features identified by the model analyzer:\n\n{feat_text}"

        elif "metric" in q_lower or "performance" in q_lower or "accuracy" in q_lower or "f1" in q_lower or "r2" in q_lower:
            metric_text = "\n".join([f"- **{k.upper()}**: {v}" for k, v in metrics.items()])
            return f"Evaluation metrics for selected {model_name}:\n\n{metric_text}"

        else:
            verif_count = len(verif_exps)
            return (
                f"**Diagnostic Summary for {model_name}**:\n"
                f"- Task Type: {task_type.capitalize()}\n"
                f"- Risk Score: {payload.get('overall_risk', {}).get('risk_score', 0)}/100 ({payload.get('overall_risk', {}).get('risk_level', 'LOW')})\n"
                f"- Active Root Causes Detected: {len(root_causes)}\n"
                f"- Verification Experiments Conducted: {verif_count}\n"
                f"You can ask me about ablation results, evidence score breakdowns, stability tests, or remediation resolution!"
            )


class GeminiProvider(BaseAIProvider):
    """
    Google Gemini API Provider.
    Calls the Gemini REST API with structured prompting, JSON schema enforcement,
    and automatic fallback if the network or API key fails.
    Optimized for high throughput and low-latency responses.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.5-flash-lite",
        fallback_provider: Optional[BaseAIProvider] = None
    ):
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        super().__init__(model_name=model_name, api_key=key)
        self.fallback_provider = fallback_provider or OfflineDeterministicProvider()

    @property
    def provider_name(self) -> str:
        return f"Google Gemini ({self.model_name})"

    def _call_gemini_api(
        self,
        prompt: str,
        system_instruction: str = "",
        response_json: bool = False,
        max_tokens: int = 1024
    ) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Please provide a key or set GEMINI_API_KEY environment variable.")

        candidate_models = [self.model_name]
        for fallback_m in ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-flash-latest"]:
            if fallback_m not in candidate_models:
                candidate_models.append(fallback_m)

        last_error = None
        for m in candidate_models:
            model_clean = m if m.startswith("models/") else f"models/{m}"
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_clean}:generateContent?key={self.api_key}"

            request_data: Dict[str, Any] = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ]
            }

            if system_instruction:
                request_data["system_instruction"] = {
                    "parts": [{"text": system_instruction}]
                }

            generation_config: Dict[str, Any] = {
                "temperature": 0.1,
                "maxOutputTokens": max_tokens,
            }
            if response_json:
                generation_config["responseMimeType"] = "application/json"

            request_data["generationConfig"] = generation_config

            req = urllib.request.Request(
                url,
                data=json.dumps(request_data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )

            try:
                with urllib.request.urlopen(req, timeout=12) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    candidates = result.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts:
                            self.model_name = m
                            return parts[0].get("text", "")
                    raise RuntimeError(f"Unexpected response structure from Gemini API: {result}")
            except urllib.error.HTTPError as e:
                err_msg = e.read().decode("utf-8")
                last_error = RuntimeError(f"Gemini API HTTP Error ({e.code}) for {m}: {err_msg}")
                if e.code in (400, 404, 503):
                    continue
                raise last_error
            except Exception as e:
                last_error = RuntimeError(f"Gemini API connection error ({m}): {str(e)}")
                continue

        if last_error:
            raise last_error
        raise RuntimeError("Failed to call Gemini API across candidate models.")

    def generate_explanation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        system_instruction = (
            "You are a Senior Machine Learning Reliability and Quality Assurance Engineer.\n"
            "MANDATORY SCIENTIFIC & CAUSALITY GUIDELINES:\n"
            "1. DISTINGUISH OBSERVED EVIDENCE FROM HYPOTHESIS / OPERATIONAL RISK:\n"
            "   - State observed empirical evidence (ablation drop, permutation drop, control test comparison, 10% measurement jitter stability).\n"
            "   - Frame potential future vulnerabilities (e.g. data drift, schema corruption) strictly as operational considerations or hypotheses, NOT as experimentally proven facts unless proven by an actual drift/error trial.\n"
            "   - If counterfactual noise/jitter testing demonstrated low flip rate and minimal drop, report that the model showed measurement stability under the tested 10% jitter, rather than asserting high vulnerability.\n"
            "   - For operational monitoring recommendations, use conditional deployment phrasing: 'If deployed, monitor distribution drift and data-quality changes in [feature] over time.' Do not assume a production environment exists.\n"
            "2. DISTINGUISH COMPLETED EXPERIMENTS FROM FUTURE REMEDIATION ROADMAP:\n"
            "   - The payload contains 'completed_experiments' (retrained_feature_ablation, test_time_permutation, measurement_stability, control_feature_test, closed_loop_remediation).\n"
            "   - COMPLETED experiments are already finished evidence. Do NOT imply that completed experiments still need to be performed or that they established general production robustness (e.g. do NOT say 'Review completed measurement-stability results to establish baseline robustness parameters for deployment').\n"
            "   - For measurement stability, state: 'Use the completed measurement-stability result as a baseline reference when evaluating realistic production measurement-error scenarios.'\n"
            "   - FUTURE roadmap items must focus strictly on forward-looking engineering and deployment actions:\n"
            "     • Audit data collection integrity and pipeline stability specifically for [feature].\n"
            "     • Evaluate alternative model configurations, feature selection, or regularization techniques if reducing feature concentration is operationally necessary.\n"
            "     • If deployed, monitor distribution drift and data-quality changes in [feature] over time.\n"
            "     • Use the completed measurement-stability result as a baseline reference when evaluating realistic production measurement-error scenarios.\n"
            "     • Evaluate model behavior under realistic production error and distribution-shift scenarios when representative deployment data becomes available.\n"
            "3. STRICTLY PROHIBIT REAL-WORLD CAUSAL CLAIMS:\n"
            "   - Feature importance = predictive usefulness/influence within the model on this dataset.\n"
            "   - Ablation = effect of removing a feature and retraining under the tested setup.\n"
            "   - Permutation = effect of disrupting the feature-target relationship at test time.\n"
            "   - Measurement jitter = robustness/sensitivity to tested perturbation.\n"
            "   - Control experiment = comparison against another feature/intervention.\n"
            "   - None of these alone establishes real-world causality. Never claim a feature causes the target in reality (e.g. do NOT say 'annual_income causes customer segment').\n"
            "   - Use scientifically precise phrasing: 'The model exhibits strong empirical reliance on [feature] under the tested dataset, split, and interventions.'\n"
            "4. NO MEASURABLE MODEL RELIANCE:\n"
            "   - For features showing no measurable change under ablation or permutation, state: 'Under the tested split and interventions, removing or permuting this feature produced no measurable change in the selected evaluation metric.' Do not call them irrelevant, useless, or unneeded in reality.\n"
            "5. Return valid JSON matching the requested schema."
        )

        prompt = (
            "Analyze this deterministic ML diagnostic evidence payload and produce a structured root-cause report in JSON:\n\n"
            f"```json\n{json.dumps(payload, indent=2)}\n```\n\n"
            "Return a JSON object with exactly these top-level keys:\n"
            "- executive_summary: A 2-3 sentence executive synthesis explaining model behavior, observed reliance, and key findings.\n"
            "- root_causes_detailed: List of objects with [finding, category, severity, confidence, why_it_matters, evidence_summary, business_risk, recommended_action].\n"
            "- risk_assessment: Object with [overall_risk_level, risk_score, deployment_readiness, critical_vulnerabilities_count, governance_note].\n"
            "- remediation_roadmap: List of sequential step-by-step strings to fix the data and model pipeline.\n"
        )

        try:
            raw_response = self._call_gemini_api(prompt, system_instruction, response_json=True, max_tokens=1024)
            # Clean potential backticks
            cleaned = raw_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            parsed = json.loads(cleaned.strip())
            parsed["provider"] = self.provider_name
            return parsed
        except Exception as e:
            fallback_res = self.fallback_provider.generate_explanation(payload)
            fallback_res["provider_warning"] = f"Gemini API unavailable ({str(e)}). Returned deterministic offline report."
            return fallback_res

    def generate_fix(self, payload: Dict[str, Any]) -> str:
        system_instruction = (
            "You are an expert ML Data Engineering Copilot. Given a deterministic ML diagnostic report, "
            "write complete, clean, runnable Python/Pandas/Scikit-learn remediation code. "
            "Include proper imputers, encoders, and model pipelines specifically targeting detected issues."
        )

        prompt = (
            "Given this diagnostic evidence payload:\n"
            f"```json\n{json.dumps(payload, indent=2)}\n```\n\n"
            "Generate a complete, self-contained Python remediation script `fix_pipeline.py`. "
            "Return ONLY the executable python code."
        )

        try:
            raw_response = self._call_gemini_api(prompt, system_instruction, response_json=False, max_tokens=1500)
            cleaned = raw_response.strip()
            if cleaned.startswith("```python"):
                cleaned = cleaned[9:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            return cleaned.strip()
        except Exception as e:
            return self.fallback_provider.generate_fix(payload)

    def chat(
        self,
        payload: Dict[str, Any],
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        system_instruction = (
            "You are the AI Model Diagnostic Copilot. Provide a direct, high-signal, concise response "
            "(2-3 short bullet points max unless asked for details). Answer strictly from the provided statistical evidence and controlled experiment trials."
        )

        # Compact context for ultra-fast generation
        compact_context = {
            "task_type": payload.get("task_type"),
            "target": payload.get("target_profile", {}).get("target_column"),
            "selected_model": payload.get("selected_model", {}).get("name"),
            "metrics": payload.get("selected_model", {}).get("metrics"),
            "overall_risk_score": payload.get("overall_risk", {}).get("risk_score"),
            "risk_level": payload.get("overall_risk", {}).get("risk_level"),
            "top_root_causes": [
                f"{rc.get('finding')} [{rc.get('severity')}] - {rc.get('interpretation')}"
                for rc in payload.get("root_causes", [])[:3]
            ],
            "top_features": list(payload.get("feature_importance", {}).items())[:4],
            "completed_experiments": payload.get("completed_experiments", []),
            "verification_experiments": payload.get("verification_experiments", [])[:3],
            "remediation_simulation": payload.get("remediation_simulation", {})
        }

        history_text = ""
        if chat_history:
            for item in chat_history[-3:]:
                role = item.get("role", "user")
                content = item.get("content", "")
                history_text += f"{role.upper()}: {content}\n"

        prompt = (
            f"SUMMARY EVIDENCE & VERIFICATION TRIALS:\n{json.dumps(compact_context, indent=1)}\n\n"
            f"{history_text}"
            f"QUESTION: {question}\n\n"
            "Answer clearly and concisely:"
        )

        try:
            return self._call_gemini_api(prompt, system_instruction, response_json=False, max_tokens=400).strip()
        except Exception:
            return self.fallback_provider.chat(payload, question, chat_history)


class OpenAICompatibleProvider(BaseAIProvider):
    """
    OpenAI-compatible API Provider (OpenAI, DeepSeek, Groq, OpenRouter, vLLM).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o-mini",
        fallback_provider: Optional[BaseAIProvider] = None
    ):
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        super().__init__(model_name=model_name, api_key=key)
        self.base_url = base_url.rstrip("/")
        self.fallback_provider = fallback_provider or OfflineDeterministicProvider()

    @property
    def provider_name(self) -> str:
        return f"OpenAI-Compatible ({self.model_name})"

    def _call_chat_completions(self, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.2
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

    def generate_explanation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        system_instruction = (
            "You are a Senior Machine Learning Reliability and Quality Assurance Engineer.\n"
            "MANDATORY SCIENTIFIC & CAUSALITY GUIDELINES:\n"
            "1. DISTINGUISH OBSERVED EVIDENCE FROM HYPOTHESIS / OPERATIONAL RISK:\n"
            "   - State observed empirical evidence (ablation drop, permutation drop, control test comparison, 10% measurement jitter stability).\n"
            "   - Frame potential future vulnerabilities (e.g. data drift, schema corruption) strictly as operational considerations or hypotheses, NOT as experimentally proven facts unless proven by an actual drift/error trial.\n"
            "   - If counterfactual noise/jitter testing demonstrated low flip rate and minimal drop, report that the model showed measurement stability under the tested 10% jitter, rather than asserting high vulnerability.\n"
            "   - For operational monitoring recommendations, use conditional deployment phrasing: 'If deployed, monitor distribution drift and data-quality changes in [feature] over time.' Do not assume a production environment exists.\n"
            "2. DISTINGUISH COMPLETED EXPERIMENTS FROM FUTURE REMEDIATION ROADMAP:\n"
            "   - The payload contains 'completed_experiments' (retrained_feature_ablation, test_time_permutation, measurement_stability, control_feature_test, closed_loop_remediation).\n"
            "   - COMPLETED experiments are already finished evidence. Do NOT imply that completed experiments still need to be performed or that they established general production robustness (e.g. do NOT say 'Review completed measurement-stability results to establish baseline robustness parameters for deployment').\n"
            "   - For measurement stability, state: 'Use the completed measurement-stability result as a baseline reference when evaluating realistic production measurement-error scenarios.'\n"
            "   - FUTURE roadmap items must focus strictly on forward-looking engineering and deployment actions:\n"
            "     • Audit data collection integrity and pipeline stability specifically for [feature].\n"
            "     • Evaluate alternative model configurations, feature selection, or regularization techniques if reducing feature concentration is operationally necessary.\n"
            "     • If deployed, monitor distribution drift and data-quality changes in [feature] over time.\n"
            "     • Use the completed measurement-stability result as a baseline reference when evaluating realistic production measurement-error scenarios.\n"
            "     • Evaluate model behavior under realistic production error and distribution-shift scenarios when representative deployment data becomes available.\n"
            "3. STRICTLY PROHIBIT REAL-WORLD CAUSAL CLAIMS:\n"
            "   - Feature importance = predictive usefulness/influence within the model on this dataset.\n"
            "   - Ablation = effect of removing a feature and retraining under the tested setup.\n"
            "   - Permutation = effect of disrupting the feature-target relationship at test time.\n"
            "   - Measurement jitter = robustness/sensitivity to tested perturbation.\n"
            "   - Control experiment = comparison against another feature/intervention.\n"
            "   - None of these alone establishes real-world causality. Never claim a feature causes the target in reality (e.g. do NOT say 'annual_income causes customer segment').\n"
            "   - Use scientifically precise phrasing: 'The model exhibits strong empirical reliance on [feature] under the tested dataset, split, and interventions.'\n"
            "4. NO MEASURABLE MODEL RELIANCE:\n"
            "   - For features showing no measurable change under ablation or permutation, state: 'Under the tested split and interventions, removing or permuting this feature produced no measurable change in the selected evaluation metric.' Do not call them irrelevant, useless, or unneeded in reality.\n"
            "5. Return valid JSON matching the requested schema."
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Analyze this diagnostic payload and return JSON with keys [executive_summary, root_causes_detailed, risk_assessment, remediation_roadmap]:\n{json.dumps(payload)}"}
        ]
        try:
            content = self._call_chat_completions(messages, json_mode=True)
            parsed = json.loads(content)
            parsed["provider"] = self.provider_name
            return parsed
        except Exception:
            return self.fallback_provider.generate_explanation(payload)

    def generate_fix(self, payload: Dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": "You are an ML Engineer. Generate only clean executable Python remediation code."},
            {"role": "user", "content": f"Generate fix_pipeline.py for this diagnostic:\n{json.dumps(payload)}"}
        ]
        try:
            content = self._call_chat_completions(messages, json_mode=False)
            return content.replace("```python", "").replace("```", "").strip()
        except Exception:
            return self.fallback_provider.generate_fix(payload)

    def chat(
        self,
        payload: Dict[str, Any],
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        messages = [{"role": "system", "content": f"Ground answers on this evidence and controlled verification trials:\n{json.dumps(payload)}"}]
        if chat_history:
            messages.extend(chat_history[-4:])
        messages.append({"role": "user", "content": question})
        try:
            return self._call_chat_completions(messages, json_mode=False).strip()
        except Exception:
            return self.fallback_provider.chat(payload, question, chat_history)


class OllamaProvider(BaseAIProvider):
    """
    Local Ollama Provider (100% private, free, runs on localhost:11434).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3.2",
        fallback_provider: Optional[BaseAIProvider] = None
    ):
        super().__init__(model_name=model_name, api_key="")
        self.base_url = base_url.rstrip("/")
        self.fallback_provider = fallback_provider or OfflineDeterministicProvider()

    @property
    def provider_name(self) -> str:
        return f"Local Ollama ({self.model_name})"

    def _generate(self, prompt: str, system: str = "") -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": 0.2}
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")

    def generate_explanation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        system_instruction = (
            "You are a Senior Machine Learning Reliability and Quality Assurance Engineer.\n"
            "MANDATORY SCIENTIFIC & CAUSALITY GUIDELINES:\n"
            "1. DISTINGUISH OBSERVED EVIDENCE FROM HYPOTHESIS / OPERATIONAL RISK: If deployed, monitor distribution drift and data-quality changes in input features over time.\n"
            "2. DISTINGUISH COMPLETED EXPERIMENTS FROM FUTURE REMEDIATION ROADMAP: Completed experiments are evidence. Use completed measurement-stability results as a baseline reference. Future actions focus on data audits, alternative model configurations, and production monitoring if deployed.\n"
            "3. STRICTLY PROHIBIT REAL-WORLD CAUSAL CLAIMS: Use 'The model exhibits strong empirical reliance on [feature] under tested interventions.'\n"
            "4. Return valid JSON with [executive_summary, root_causes_detailed, risk_assessment, remediation_roadmap]."
        )
        prompt = f"Analyze this ML diagnostic JSON and return a JSON explanation:\n{json.dumps(payload)}"
        try:
            res = self._generate(prompt, system=system_instruction)
            return json.loads(res)
        except Exception:
            return self.fallback_provider.generate_explanation(payload)

    def generate_fix(self, payload: Dict[str, Any]) -> str:
        prompt = f"Generate executable Python remediation code for this diagnostic:\n{json.dumps(payload)}"
        try:
            res = self._generate(prompt)
            return res.replace("```python", "").replace("```", "").strip()
        except Exception:
            return self.fallback_provider.generate_fix(payload)

    def chat(
        self,
        payload: Dict[str, Any],
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        prompt = f"Evidence:\n{json.dumps(payload)}\n\nQuestion: {question}"
        try:
            return self._generate(prompt)
        except Exception:
            return self.fallback_provider.chat(payload, question, chat_history)


def get_ai_provider(
    provider_type: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> BaseAIProvider:
    """
    Factory function to obtain an AI provider instance.
    Auto-detects available credentials if provider_type is omitted.
    """
    ptype = (provider_type or os.environ.get("AI_PROVIDER", "")).lower().strip()

    if ptype == "gemini" or (not ptype and (api_key or os.environ.get("GEMINI_API_KEY"))):
        mname = model_name or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        return GeminiProvider(api_key=api_key, model_name=mname)

    elif ptype in ("openai", "groq", "deepseek") or (not ptype and os.environ.get("OPENAI_API_KEY")):
        mname = model_name or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAICompatibleProvider(api_key=api_key, model_name=mname)

    elif ptype == "ollama":
        mname = model_name or os.environ.get("OLLAMA_MODEL", "llama3.2")
        return OllamaProvider(model_name=mname)

    # Default fallback
    return OfflineDeterministicProvider()
