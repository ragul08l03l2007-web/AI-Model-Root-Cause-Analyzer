# AI Model Root-Cause Analyzer & Diagnostic Copilot 🔍🤖

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Engine: Scikit-Learn](https://img.shields.io/badge/Engine-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Visualization: Plotly](https://img.shields.io/badge/Visualization-Plotly.js-purple.svg)](https://plotly.com/)
[![AI: Google Gemini](https://img.shields.io/badge/AI-Google%20Gemini%20Flash-4285F4.svg)](https://ai.google.dev/)
[![Tests Passing](https://img.shields.io/badge/Tests-100%25%20Passed-brightgreen.svg)](#-test-verification-suite)

An enterprise-grade, evidence-driven machine learning diagnostic platform. It automatically ingests tabular datasets, trains and compares multiple candidate models, mathematically pinpoints root causes of predictive failure (data leakage, generalization gaps, class imbalances, outlier distortions, feature over-reliance), visualizes evidence with interactive **Plotly** charts, and generates executive summaries, auto-fix Python scripts, and real-time copilot guidance using **Google Gemini**.

---

## 🏛️ 3-Tier Unidirectional Architecture

The platform strictly separates **mathematical fact-finding**, **visual representation**, and **AI interpretation** to eliminate hallucinations and ensure 100% grounded diagnostics:

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
                                           │ (Deterministic Evidence & Metrics)
                                           ▼
              ┌─────────────────────────────────────────────────────────┐
              │        📊 TIER 2: EVIDENCE VISUALIZATION (Plotly)       │
              │  • Risk Severity Gauges & Domain Breakdown Bars         │
              │  • Model Metrics & Fold Stability Distribution          │
              │  • Feature Importance Rankings & Influence Tiers        │
              │  • Data Quality (Missingness, Outliers, Target Donut)   │
              │  • 2D Confusion Matrix Heatmap / Regression Residuals   │
              └────────────────────────────┬────────────────────────────┘
                                           │ (Sanitized Evidence JSON Payload)
                                           ▼
              ┌─────────────────────────────────────────────────────────┐
              │         🤖 TIER 3: GROUNDED AI REASONING LAYER          │
              │  • Stage 1: Executive Summary & Root-Cause Audit        │
              │  • Stage 2: Executable Auto-Fix Python Script Generator │
              │  • Stage 3: Interactive Diagnostic Copilot (Live Chat)  │
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
- **Universal Target Analysis**: Automatically detects classification (binary or multiclass) and continuous regression, with intelligent fallback and type coercion.
- **Multi-Model Evaluation & Selection**: Trains and benchmarks multiple candidate architectures (Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, Linear Regression, Ridge) and selects the champion model using cross-validation generalization stability.
- **Canonical 8-Field Evidence Chains**: Every detected diagnostic candidate is backed by concrete empirical proof:
  `Candidate ID` → `Category` → `Severity` → `Confidence` → `Evidence` → `Interpretation` → `Operational Impact` → `Recommended Action`.
- **Target Leakage Discovery**: Detects high mutual information, quasi-perfect correlation, and identity leaks before models overfit.
- **Evidence-Weighted Risk Score (`0–100`)**: Transparently aggregates vulnerability points across Data Quality, Generalization, Class Imbalance, and Feature Dominance.

---

### 2. 📊 Evidence-Driven Visualization Center (Plotly.js)
Rendered with interactive category tabs (*All Charts*, *Risk & Health*, *Performance & CV*, *Feature Impact*, *Data Quality*, *Error & Diagnostics*):

| Category | Chart Generated | Diagnostic Value |
|---|---|---|
| **Risk & Health** | **Radial Risk Severity Gauge** (`0–100`)<br>**Domain Risk Breakdown Bar** | Instantly reveals operational vulnerability with color-coded safety tiers (Low/Moderate/High) and exact point contributions. |
| **Model Performance** | **Metrics Bar Chart**<br>**Fold-by-Fold Cross-Validation Stability** | Tracks Accuracy, Balanced Acc, Precision, Recall, F1 (or R², RMSE, MAE) alongside fold-by-fold stability and mean reference lines. |
| **Feature Impact** | **Horizontal Feature Importance Ranking**<br>**Empirical Association Bar** | Color-codes feature tiers (Very Strong, Strong, Moderate, Low) with relative share percentages and non-causal relationship summaries. |
| **Data Quality** | **Missing Values % by Column**<br>**Target Class Distribution Donut / Spread** | Includes 5% (Warning) and 20% (Critical) threshold lines, class imbalance flags, and continuous quartile spreads. |
| **Error Diagnostics** | **2D Confusion Matrix Heatmap** (Classification) | Interactive actual vs predicted grid with cell observation counts, normalized percentages, and error rates. |
| **Regression Diagnostics** | **Actual vs Predicted Scatter ($y = \hat{y}$)**<br>**Residual Plot** & **Residual Histogram** | Plots residuals against predicted values with a zero-error baseline and tests error normality/skew. |

---

### 3. 🤖 Pluggable AI Diagnostic & Remediation Engine
- **Stage 1 — AI Executive Summary**: Synthesizes high-level reliability audits, operational risk summaries, and sequential remediation roadmaps.
- **Stage 2 — Auto-Fix Script Generator**: Produces custom, executable `scikit-learn` Python remediation scripts (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `RandomForestClassifier` / `GradientBoostingRegressor`) addressing detected vulnerabilities.
- **Stage 3 — Interactive Diagnostic Copilot**: In-dashboard chat drawer answering queries (*"Why did the model fail?"*, *"Which feature causes the highest risk?"*, *"How do I fix class imbalance?"*) strictly grounded in the dataset's evidence.
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
│   ├── feature_analyzer.py      # Feature importance & empirical relationship analyzer
│   ├── leakage_detector.py      # Target leakage & high-correlation detector
│   ├── model_analyzer.py        # Master pipeline coordinator (analyze_model)
│   ├── model_engine.py          # Multi-model training, evaluation & CV engine
│   ├── preprocessor.py          # Robust encoding, scaling & imputation transformer
│   ├── profiler.py              # Dataset statistical profiler & column inference
│   ├── target_analyzer.py       # Universal target analyzer & task detector
│   └── visualization.py        # Evidence-driven Plotly chart generator
├── app.py                       # High-performance web application & dashboard server
├── dashboard.py                 # Pure-Python diagnostic dashboard renderer
├── report_generator.py          # CLI text & JSON report generator
├── test_acceptance_issues_a_to_q.py  # 17-point full diagnostic acceptance test suite
├── test_ai_explainer.py         # AI explainer, provider hierarchy & copilot test suite
├── test_server_endpoints.py     # Live server routes, downloads & API test suite
├── test_visualization.py        # Plotly chart structure & JSON serialization test suite
├── .env.example                 # Environment configuration template
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
pip install -r requirements.txt # or install core packages:
pip install pandas scikit-learn numpy plotly matplotlib
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

The repository contains an exhaustive test suite covering all mathematical algorithms, edge cases, providers, visualizations, and server endpoints:

```bash
# 1. Run Evidence-Driven Visualization Test Suite
python test_visualization.py

# 2. Run AI Explainer & Provider Test Suite
python test_ai_explainer.py

# 3. Run Full Server Routes & Copilot API Test Suite
python test_server_endpoints.py

# 4. Run Complete 17-Point Acceptance Criteria Suite (Issues A through Q)
python test_acceptance_issues_a_to_q.py
```

---

## 📥 Export & Artifacts Generated

From the web dashboard, users can immediately export:
- **`analysis.json`**: Complete structured evidence chain and mathematical metrics.
- **`summary.csv`**: Feature-level impact summary, relative importance, and influence tiers.
- **`prediction-errors.csv`**: Granular observation-level errors, prediction confidences, and error types.
- **`remediation_script.py`**: Runnable Python remediation pipeline ready for production deployment.
- **High-Resolution PNGs**: Direct one-click download for every Plotly visual chart.

---

## 📄 License

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute it for research, academic, or commercial machine learning projects.
