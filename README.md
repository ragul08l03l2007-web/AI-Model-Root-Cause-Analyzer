# AI Model Root-Cause Analyzer & Diagnostic Copilot 🔍🤖

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Engine: Scikit-Learn](https://img.shields.io/badge/Engine-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Visualization: Plotly](https://img.shields.io/badge/Visualization-Plotly.js-purple.svg)](https://plotly.com/)
[![AI: Google Gemini](https://img.shields.io/badge/AI-Google%20Gemini%20Flash-4285F4.svg)](https://ai.google.dev/)
[![Tests Passing](https://img.shields.io/badge/Tests-100%25%20Passed-brightgreen.svg)](#-test-verification-suite)

An enterprise-grade, evidence-driven machine learning diagnostic and verification platform. It automatically ingests tabular datasets, trains and benchmarks multiple candidate models, runs targeted counterfactual experiments (ablation, permutation, jitter, control tests), mathematically proves root causes of predictive behavior via a **Multi-Experiment Evidence Graph (DAG)**, simulates closed-loop remediations, renders interactive **Plotly** visualizations (including Sankey lineage traces), and provides screen-oriented AI explanations, runnable auto-fix code, and live Copilot Q&A via **Google Gemini**.

---

## 🏛️ 4-Tier Evidence-Driven Architecture

The platform strictly separates **mathematical fact-finding**, **controlled experimentation**, **visual representation**, and **AI interpretation** to eliminate hallucinations and ensure 100% grounded diagnostics:

```
                              INPUT DATASET (CSV / Excel)
                                           │
                                           ▼
              ┌─────────────────────────────────────────────────────────┐
              │             🧮 TIER 1: DETERMINISTIC CORE               │
              │  • Universal Data Profiler & Quality Auditor            │
              │  • Target Analyzer & Leakage Detector                   │
              │  • Multi-Model Benchmark & Cross-Validation Engine      │
              │  • Error Analyzer & Subgroup Disparity Evaluator        │
              │  • Canonical Diagnostic Engine & Evidence Bus           │
              └────────────────────────────┬────────────────────────────┘
                                           │ (Observational Candidates)
                                           ▼
              ┌─────────────────────────────────────────────────────────┐
              │        🧪 TIER 2: VERIFICATION & EVIDENCE GRAPH         │
              │  • Retrained Feature Ablation (35 pts)                  │
              │  • Test-Time Feature Permutation (35 pts)               │
              │  • Negative Baseline Control Comparison (15 pts)        │
              │  • 10% σ Measurement Jitter Stability (10 pts)          │
              │  • Cross-Experiment Consistency Check (5 pts)           │
              │  • Evidence Graph DAG (11 Node Types, 7 Edge Relations) │
              │  • Closed-Loop Remediation Simulation (In-Memory Fix)   │
              └────────────────────────────┬────────────────────────────┘
                                           │ (Validated Graphs & Metrics)
                                           ▼
              ┌─────────────────────────────────────────────────────────┐
              │        📊 TIER 3: EVIDENCE VISUALIZATION (Plotly)       │
              │  • Multi-Experiment Evidence Graph (Interactive Sankey) │
              │  • Risk Severity Gauges & Authoritative Driver Bars     │
              │  • Model Metrics & Fold Stability Distribution          │
              │  • Feature Importance Rankings & Influence Tiers        │
              │  • Data Quality (Missingness, Outliers, Target Spread)  │
              │  • 2D Confusion Matrix Heatmap / Regression Residuals   │
              └────────────────────────────┬────────────────────────────┘
                                           │ (Sanitized Evidence & Calculation Context)
                                           ▼
              ┌─────────────────────────────────────────────────────────┐
              │         🤖 TIER 4: GROUNDED AI REASONING LAYER          │
              │  • Stage 1: Executive Summary & Root-Cause Audit        │
              │  • Stage 2: Executable Auto-Fix Python Script Generator │
              │  • Stage 3: Screen-Aware Diagnostic Copilot (Live Chat) │
              │  • Indexes ALL Features & On-Screen Calculation Formulas│
              │  • Providers: Gemini Flash, OpenAI, Ollama, Offline     │
              └────────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
              ┌─────────────────────────────────────────────────────────┐
              │           🌐 INTERACTIVE WEB DASHBOARD & UI             │
              │     (http://localhost:8000 — Zero Server-Side Lag)      │
              └─────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features & Capabilities

### 1. 🧮 Comprehensive Deterministic Machine Learning Engine
- **Universal Target Analysis**: Automatically detects binary classification, multiclass classification (with Balanced Accuracy `mean(recall_i)`), and continuous regression with intelligent type coercion.
- **Multi-Model Benchmark & Selection**: Trains and evaluates multiple candidate architectures (Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, Linear Regression, Ridge) and selects champion models based on validation stability.
- **Canonical 8-Field Evidence Chains**: Every detected diagnostic candidate is backed by concrete empirical proof:
  `Candidate ID` → `Category` → `Severity` → `Confidence` → `Evidence` → `Interpretation` → `Operational Impact` → `Recommended Action`.
- **Target Leakage Discovery**: Detects high mutual information, quasi-perfect correlation, and identity leaks before models overfit.
- **Evidence-Weighted Risk Score (`0–100`)**: Transparently aggregates vulnerability penalties across Data Quality, Generalization, Class Imbalance, and Reliance Concentration.

---

### 2. 🧪 Multi-Experiment Evidence Graph & Verification Engine
- **5-Part Decomposed Scoring Rubric (100 pts Max)**:
  - **Retrained Feature Ablation (35 pts)**: Retrains the model without the feature on identical splits to evaluate retraining impact.
  - **Test-Time Permutation (35 pts)**: Randomly disrupts feature values at inference time without retraining.
  - **Negative Baseline Control Specificity (15 pts)**: Runs control interventions on low-importance features to verify candidate specificity.
  - **Measurement Stability (10 pts)**: Evaluates prediction flip rates under $10\%\ \sigma$ Gaussian measurement noise.
  - **Cross-Experiment Consistency (5 pts)**: Confirms both ablation and permutation independently validate the diagnostic hypothesis.
- **Evidence Graph Directed Acyclic Graph (DAG)**:
  - **11 Node Types**: `candidate`, `feature_importance`, `ablation_experiment`, `permutation_experiment`, `stability_experiment`, `control_experiment`, `evidence_fusion`, `verdict`, `intervention`, `remediation_result`, `resolution`.
  - **7 Edge Relations**: `supported_by`, `tested_by`, `compared_against`, `contributes_to`, `produces`, `triggers`, `evaluated_as`.
  - **Graph Integrity Engine**: Automatic validation ensuring zero orphan nodes, cycles, or broken references.
- **Closed-Loop Remediation Simulation**:
  - Automatically simulates standard data hygiene, imputation, scaling, and regularization in-memory.
  - Evaluates generalization gap reduction and held-out test improvement before recommending deployment.

---

### 3. 📊 Evidence-Driven Visualization Center (Plotly.js)
Rendered with interactive category tabs (*All Charts*, *Evidence Graph*, *Risk & Health*, *Performance & CV*, *Feature Impact*, *Data Quality*, *Error & Diagnostics*):

| Category | Chart Generated | Diagnostic Value |
|---|---|---|
| **Evidence Graph** | **Interactive Lineage Sankey Diagram** | Visualizes full end-to-end evidence lineage from candidate discovery through ablation/permutation experiments to verdict and resolution. |
| **Risk & Health** | **Radial Risk Severity Gauge** (`0–100`)<br>**Domain Risk Breakdown Bar** | Instantly reveals operational vulnerability with color-coded safety tiers and exact authoritative driver point contributions (`+4 pts Data Quality`, etc.). |
| **Model Performance** | **Metrics Bar Chart**<br>**Fold-by-Fold Cross-Validation Stability** | Tracks Accuracy, Balanced Accuracy, Precision, Recall, F1 (or R², RMSE, MAE) alongside fold-by-fold stability and mean reference lines. |
| **Feature Impact** | **Horizontal Feature Importance Ranking**<br>**Empirical Association Bar** | Color-codes feature tiers (Very Strong, Strong, Moderate, Low) with relative share percentages and non-causal relationship summaries. |
| **Data Quality** | **Missing Values % by Column**<br>**Target Class Distribution Donut / Spread** | Includes 5% (Warning) and 20% (Critical) threshold lines, class imbalance flags, and continuous quartile spreads. |
| **Error Diagnostics** | **2D Confusion Matrix Heatmap** (Classification) | Interactive actual vs predicted grid with cell observation counts, normalized percentages, and error rates. |
| **Regression Diagnostics** | **Actual vs Predicted Scatter ($y = \hat{y}$)**<br>**Residual Plot** & **Residual Histogram** | Plots residuals against predicted values with a zero-error baseline and tests error normality/skew. |

---

### 4. 🤖 Screen-Oriented AI Diagnostic & Remediation Engine
- **Complete Feature Catalogue (`feature_catalogue`)**:
  - Indexes **100% of dataset features** with data types, missingness, outliers, importance, relative share %, empirical directions, subgroup error involvement, and experimental trials.
  - Answers developer questions about **any random feature** or generates side-by-side **multi-feature comparisons**.
- **On-Screen Calculation Glossary (`calculation_glossary`)**:
  - Grounded formulas and live values for Balanced Accuracy (`mean(recall_i)`), Weighted F1, Risk Score driver formulas, Evidence Fusion (35+35+15+10+5), and Cross-Validation stability.
- **Stage 1 — AI Executive Summary**: Synthesizes high-level reliability audits, operational risk summaries, and sequential remediation roadmaps.
- **Stage 2 — Auto-Fix Script Generator**: Produces custom, executable `scikit-learn` Python remediation scripts (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `RandomForestClassifier` / `GradientBoostingRegressor`) addressing detected vulnerabilities.
- **Stage 3 — Interactive Diagnostic Copilot**: In-dashboard chat drawer answering queries (*"Why did the model fail?"*, *"Tell me about monthly_charges"*, *"How is balanced accuracy computed?"*) strictly grounded in the dataset's evidence.
- **Pluggable Architecture**:
  - ⚡ **Google Gemini**: Native REST API with ultra-fast latency (`gemini-3.5-flash-lite`, ~0.9s response time).
  - 🌐 **OpenAI-Compatible**: Seamless connection to OpenAI, DeepSeek, Groq, OpenRouter, and vLLM.
  - 🏠 **Local Ollama**: Run on-device privacy-first models (`llama3.2`, `qwen2.5:7b`).
  - 🛡️ **Offline Deterministic Provider**: Built-in zero-dependency fallback engine that generates grounded summaries and fix scripts with no external keys or internet connection.

---

## 📁 Repository Structure

```
AI-Model-Root-Cause-Analyzer/
├── analysis/
│   ├── ai_explainer.py          # Stage 1-3 AI Explainer, Fix Generator & Copilot logic
│   ├── ai_providers.py          # Pluggable Providers (Gemini, OpenAI, Ollama, Offline)
│   ├── data_quality.py          # Structural data auditor & missingness profiler
│   ├── diagnostic_engine.py     # Canonical evidence aggregator & risk scoring engine
│   ├── error_analyzer.py        # Prediction error patterns & confusion matrix builder
│   ├── evidence.py              # Canonical diagnostic evidence schema & evidence bus
│   ├── evidence_graph.py        # Multi-experiment evidence graph data structures & validation
│   ├── feature_analyzer.py      # Feature importance & empirical relationship analyzer
│   ├── leakage_detector.py      # Target leakage & high-correlation detector
│   ├── model_analyzer.py        # Master pipeline coordinator (analyze_model)
│   ├── model_engine.py          # Multi-model training, evaluation & CV engine
│   ├── preprocessor.py          # Robust encoding, scaling & imputation transformer
│   ├── profiler.py              # Dataset statistical profiler & column inference
│   ├── target_analyzer.py       # Universal target analyzer & task detector
│   ├── verification_engine.py   # Counterfactual experiments (ablation, permutation, jitter, control)
│   └── visualization.py        # Evidence-driven Plotly chart generator & Sankey diagrams
├── app.py                       # High-performance web application & dashboard server
├── dashboard.py                 # Pure-Python diagnostic dashboard renderer
├── report_generator.py          # CLI text & JSON report generator
├── test_acceptance_issues_a_to_q.py  # 17-point full diagnostic acceptance test suite
├── test_ai_explainer.py         # AI explainer, provider hierarchy, feature catalog & copilot tests
├── test_dashboard_correctness_pass.py # Balanced accuracy & risk driver consistency tests
├── test_evidence_graph.py       # Multi-Experiment Evidence Graph test suite (11 nodes, 7 edges, Sankey)
├── test_server_endpoints.py     # Live server routes, downloads & API test suite
├── test_verification_engine.py  # Feature ablations, permutations, noise jitter, controls & remediation
├── test_visualization.py        # Plotly chart structure & JSON serialization test suite
├── .env.example                 # Environment configuration template
├── requirements.txt             # Project Python dependencies
└── README.md                    # Project documentation
```

---

## ⚡ Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ragul08l03l2007-web/AI-Model-Root-Cause-Analyzer.git
cd AI-Model-Root-Cause-Analyzer
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Configure AI API Key (Optional)
Copy `.env.example` to `.env` and configure your API key (if omitted, the system seamlessly falls back to the **Offline Deterministic Engine**):
```ini
# .env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

### 4. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to **`http://localhost:8000`**.

---

## 🧪 Test Verification Suite

The repository contains an exhaustive test suite covering all mathematical algorithms, edge cases, providers, visualizations, evidence graphs, and server endpoints:

```bash
# 1. Run Multi-Experiment Evidence Graph Test Suite
python test_evidence_graph.py

# 2. Run Screen-Aware AI Explainer & Copilot Test Suite
python test_ai_explainer.py

# 3. Run Dashboard Consistency & Multiclass Verification Suite
python test_dashboard_correctness_pass.py

# 4. Run Full Server Routes & Copilot API Test Suite
python test_server_endpoints.py

# 5. Run Complete 17-Point Acceptance Criteria Suite (Issues A through Q)
python test_acceptance_issues_a_to_q.py
```

---

## 📥 Export & Artifacts Generated

From the web dashboard, users can immediately export:
- **`analysis.json`**: Complete structured evidence chain, multi-experiment graph nodes/edges, and mathematical metrics.
- **`summary.csv`**: Feature-level impact summary, relative importance, and influence tiers.
- **`prediction-errors.csv`**: Granular observation-level errors, prediction confidences, and error types.
- **`remediation_script.py`**: Runnable Python remediation pipeline ready for production deployment.
- **High-Resolution Visuals**: Direct one-click download for every Plotly visual chart.

---

## 📄 License

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute it for research, academic, or commercial machine learning projects.
