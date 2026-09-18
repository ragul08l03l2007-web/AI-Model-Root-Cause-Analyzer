# report_generator.py

def generate_report(data_quality, model_result):
    """
    Generate a professional diagnostic report from model_analyzer results,
    supporting both classification and regression workflows dynamically.
    Clearly distinguishes Observation -> Evidence -> Interpretation -> Potential Root Cause -> Impact -> Recommendation.
    """
    performance = model_result.get("model_performance", {})
    cross_validation = model_result.get("cross_validation", {})
    stability = model_result.get("model_stability", {})
    task_type = str(model_result.get("task_type", "classification")).lower()

    missing_dict = data_quality.get("missing_values", {})
    total_missing = sum(missing_dict.values()) if isinstance(missing_dict, dict) else int(missing_dict or 0)

    outliers_dict = data_quality.get("outliers", {})
    total_outliers = sum(outliers_dict.values()) if isinstance(outliers_dict, dict) else int(outliers_dict or 0)

    print()
    print("=" * 70)
    print("         AI MODEL ROOT-CAUSE DIAGNOSTIC & INVESTIGATION REPORT")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Dataset Health & Discovery
    # --------------------------------------------------------
    print()
    print("1. DATASET HEALTH & PROFILE")
    print("-" * 70)
    print(f"Dataset Size       : {data_quality.get('total_rows', 0)} rows x {len(data_quality.get('numeric_features', [])) + len(data_quality.get('categorical_features', []))} features")
    print(f"Data Quality Status: {data_quality.get('overall_quality', 'GOOD')} (Score: {data_quality.get('quality_score', 0)}/100)")
    if "quality_reason" in data_quality:
        print(f"Quality Reason     : {data_quality.get('quality_reason')}")
    print(f"Missing Values     : {total_missing}")
    print(f"Duplicate Rows     : {data_quality.get('duplicate_rows', 0)}")
    if "duplicate_handling_note" in data_quality:
        print(f"Duplicate Policy   : {data_quality.get('duplicate_handling_note')}")
    print(f"Potential Outliers : {total_outliers} (IQR-detected extreme values)")

    # --------------------------------------------------------
    # 2. Target Analysis & Distribution
    # --------------------------------------------------------
    target_prof = model_result.get("target_profile", {})
    if target_prof:
        print()
        print("2. TARGET PROFILE & CLASS DISTRIBUTION")
        print("-" * 70)
        print(f"Target Column      : {target_prof.get('target_column', 'N/A')}")
        print(f"Target Type        : {target_prof.get('target_type', task_type)}")
        if task_type == "classification":
            c_dist = target_prof.get("class_distribution", {})
            c_pcts = target_prof.get("class_percentages", {})
            for cls_name, cnt in c_dist.items():
                pct = c_pcts.get(cls_name, 0.0)
                print(f"  Class '{cls_name}' : {cnt} observations ({pct:.2f}% of dataset)")
            if "minority_to_majority_ratio" in target_prof:
                print(f"Imbalance Ratio    : {target_prof.get('minority_to_majority_ratio', 1.0):.2f} (Minority-to-Majority)")
            if "imbalance_explanation" in target_prof:
                print(f"Distribution Note  : {target_prof.get('imbalance_explanation')}")
        else:
            print(f"Mean: {target_prof.get('mean', 0.0):.4f} | Median: {target_prof.get('median', 0.0):.4f} | Std: {target_prof.get('std', 0.0):.4f}")
            print(f"Range: {target_prof.get('min', 0.0):.4f} to {target_prof.get('max', 0.0):.4f}")

    # --------------------------------------------------------
    # 3. Model Performance & Multi-Model Evaluation
    # --------------------------------------------------------
    model_selection = model_result.get("model_selection", {})
    selected_name = model_selection.get("selected_model", model_result.get("selected_model", performance.get("selected_model", "N/A")))
    criterion_text = model_selection.get("criterion", "Cross-Validation Performance")
    sel_explanation = model_selection.get("selection_explanation", "")

    print()
    print("3. MODEL SELECTION & PERFORMANCE EVALUATION")
    print("-" * 70)
    print(f"Selected Model     : {selected_name} [SELECTED]")
    print(f"Selection Criterion: {criterion_text}")
    print(f"Selection Basis    : Cross-Validation Generalization (f1_weighted / R^2 across folds)")
    if sel_explanation:
        print(f"Selection Rationale: {sel_explanation}")
    print(f"Task Type          : {task_type.capitalize()}")
    print()
    print("Test Set Performance (Held-Out Split):")

    if task_type == "regression":
        reg_m = model_result.get("regression_metrics", {})
        mae = reg_m.get("mae", performance.get("mae", 0.0))
        mse = reg_m.get("mse", performance.get("mse", 0.0))
        rmse = reg_m.get("rmse", performance.get("rmse", 0.0))
        r2 = reg_m.get("r2", performance.get("r2", 0.0))
        cv_avg = cross_validation.get("average_score", 0.0)
        cv_std = cross_validation.get("standard_deviation", 0.0)

        print(f"  Test MAE         : {mae:.4f}")
        print(f"  Test MSE         : {mse:.4f}")
        print(f"  Test RMSE        : {rmse:.4f}")
        print(f"  Test R^2 Score   : {r2:.4f}")
        print(f"  CV Mean (R^2)    : {cv_avg:.4f} (+/- {cv_std:.4f})")
    else:
        acc = performance.get("accuracy", 0.0)
        prec = performance.get("precision", 0.0)
        rec = performance.get("recall", 0.0)
        f1 = performance.get("f1_score", 0.0)
        cv_avg = cross_validation.get("average_score", 0.0)
        cv_std = cross_validation.get("standard_deviation", 0.0)

        print(f"  Test Accuracy    : {acc:.1%}" if isinstance(acc, float) else f"  Test Accuracy    : {acc}")
        print(f"  Test Precision   : {prec:.1%}" if isinstance(prec, float) else f"  Test Precision   : {prec}")
        print(f"  Test Recall      : {rec:.1%}" if isinstance(rec, float) else f"  Test Recall      : {rec}")
        print(f"  Test F1 Score    : {f1:.1%}" if isinstance(f1, float) else f"  Test F1 Score    : {f1}")
        print(f"  CV Mean (F1)     : {cv_avg:.1%} (+/- {cv_std:.1%})" if isinstance(cv_avg, float) and isinstance(cv_std, float) else f"  CV Mean (F1)     : {cv_avg}")

    # Stability & Overfitting
    if stability:
        print()
        print("Model Stability & Overfitting Assessment:")
        print(f"  CV Stability     : {stability.get('stability_status', 'STABLE')} ({stability.get('stability_explanation', '')})")
        print(f"  Overfitting Check: {stability.get('overfitting_diagnostic', 'None')} ({stability.get('overfitting_explanation', '')})")

    # --------------------------------------------------------
    # 4. Diagnostic Signals & Root-Cause Hypotheses
    # --------------------------------------------------------
    print()
    print("4. DIAGNOSTIC SIGNALS & ROOT-CAUSE HYPOTHESES")
    print("-" * 70)
    print("Disclaimer: Controlled ablation and permutation experiments measure empirical model reliance")
    print("under the tested interventions. They do not establish real-world causal relationships.")
    print()

    structured_rc = model_result.get("root_causes_structured", [])
    if structured_rc:
        for index, rc in enumerate(structured_rc, start=1):
            sev = rc.get("severity", "MEDIUM")
            cat = rc.get("category", "Root Cause Candidate")
            conf = rc.get("confidence", "High")
            status = rc.get("diagnostic_status", "SIGNAL DETECTED")
            verif_score = rc.get("verification_score", "")

            print(f"{index}. [{cat}] {rc.get('finding', rc.get('root_cause', ''))}")
            print(f"   Diagnostic Status: {status}" + (f" ({verif_score})" if verif_score and verif_score != "N/A" else ""))
            print(f"   Severity         : {sev}")
            print(f"   Signal           : {rc.get('signal', rc.get('title', ''))}")
            print(f"   Initial Evidence : {rc.get('initial_evidence', rc.get('evidence', ''))}")
            print(f"   Hypothesis       : {rc.get('hypothesis', rc.get('potential_explanation', ''))}")
            print(f"   Interpretation   : {rc.get('interpretation', '')}")
            if rc.get("verification_evidence"):
                print(f"   Verif. Evidence  : {rc.get('verification_evidence')}")
            print(f"   Impact           : {rc.get('impact', '')}")
            print(f"   Confidence       : {conf} (based on multi-source evidence)")
            print(f"   Recommended      : {rc.get('recommended_action', '')}")
            print()
    else:
        root_causes = model_result.get("root_causes", [])
        if root_causes:
            for index, cause in enumerate(root_causes, start=1):
                print(f"{index}. {cause}")
        else:
            print("No major root causes or diagnostic bottlenecks identified.")

    # --------------------------------------------------------
    # 5. Feature Importance & Observed Relationships
    # --------------------------------------------------------
    feature_impact = model_result.get("feature_impact", {})
    if feature_impact:
        print()
        print("5. FEATURE IMPORTANCE & EMPIRICAL RELATIONSHIPS")
        print("-" * 70)
        print("Disclaimer: Feature importance measures predictive usefulness in the model for this dataset.")
        print("It does not prove causal necessity, positive/negative relationship direction, or business importance.")
        print()
        for feat, data in list(feature_impact.items())[:6]:
            if isinstance(data, dict):
                imp = data.get("importance", 0.0)
                share = data.get("relative_share_pct", 0.0)
                tier = data.get("influence_tier", "Moderate Model Influence")
                direct = data.get("direction", "")
                print(f"- {feat:<20}: Score {imp:.4f} ({share:.1f}% share | {tier})")
                if direct:
                    print(f"  Observed Direction: {direct}")
            else:
                print(f"- {feat:<20}: Score {float(data):.4f}")

    # --------------------------------------------------------
    # 6. Overall Risk Assessment & Recommendations
    # --------------------------------------------------------
    print()
    print("6. OVERALL RISK & ACTIONABLE RECOMMENDATIONS")
    print("-" * 70)
    print(f"Overall Risk Level : {model_result.get('overall_risk', 'UNKNOWN')} (Score: {model_result.get('risk_score', 0)}/100)")
    print()
    recommendations = model_result.get("recommendations", [])
    if recommendations:
        for index, rec in enumerate(recommendations, start=1):
            print(f"{index}. {rec}")
    else:
        print("No specific recommendations required for this healthy model/dataset.")

    print()
    print("=" * 70)
    print("                        ANALYSIS COMPLETE")
    print("=" * 70)