import os
import json
import unittest
import pandas as pd
import numpy as np

from analysis.model_analyzer import analyze_model
from analysis.derivations import DerivationEngine


class TestDerivationEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create small test classification dataframe
        np.random.seed(42)
        n = 100
        cls.clf_df = pd.DataFrame({
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n) * 2 + 1,
            "feature3": np.random.choice([0, 1, 2], size=n),
            "target": np.random.choice(["ClassA", "ClassB"], size=n, p=[0.7, 0.3])
        })
        # Add some missing values and outliers
        cls.clf_df.loc[0:4, "feature1"] = np.nan
        cls.clf_df.loc[5, "feature2"] = 100.0

        # Create small test regression dataframe
        cls.reg_df = pd.DataFrame({
            "feat_a": np.random.randn(n),
            "feat_b": np.random.randn(n) * 5,
            "feat_c": np.random.uniform(10, 50, n),
            "target_num": np.random.randn(n) * 10 + 50
        })

    def test_classification_derivations(self):
        result = analyze_model(self.clf_df, "target")
        self.assertIn("derivations", result)
        derivations = result["derivations"]

        # Check required categories
        expected_categories = [
            "risk_score",
            "target_profile",
            "model_performance",
            "generalization",
            "cross_validation",
            "features",
            "data_quality",
            "confidence_score",
            "error_analysis"
        ]
        for cat in expected_categories:
            self.assertIn(cat, derivations, f"Missing category: {cat}")
            cat_data = derivations[cat]

            if cat == "features":
                self.assertIsInstance(cat_data, dict)
                self.assertGreater(len(cat_data), 0)
                for f_name, f_d in cat_data.items():
                    self.assertIn("title", f_d)
                    self.assertIn("formula_latex", f_d)
                    self.assertIn("formula_text", f_d)
                    self.assertIn("variables", f_d)
                    self.assertIn("calculation_steps", f_d)
                    self.assertIn("output_value", f_d)
            else:
                self.assertIn("title", cat_data)
                self.assertIn("formula_latex", cat_data)
                self.assertIn("formula_text", cat_data)
                self.assertIn("variables", cat_data)
                self.assertIn("calculation_steps", cat_data)
                self.assertIn("output_value", cat_data)
                self.assertIn("interpretation", cat_data)

        # Verify JSON serializability
        try:
            json_str = json.dumps(derivations)
            self.assertGreater(len(json_str), 100)
        except Exception as e:
            self.fail(f"Derivations dictionary failed JSON serialization: {e}")

    def test_regression_derivations(self):
        result = analyze_model(self.reg_df, "target_num")
        self.assertIn("derivations", result)
        derivations = result["derivations"]

        self.assertIn("model_performance", derivations)
        perf_deriv = derivations["model_performance"]
        self.assertIn("Regression", perf_deriv["title"])
        self.assertIn("variables", perf_deriv)
        self.assertIn("R_squared", perf_deriv["variables"])

        # Check generalization gap derivation for regression
        gen_deriv = derivations["generalization"]
        self.assertIn("Generalization", gen_deriv["title"])

        # Verify JSON serializability
        json_str = json.dumps(derivations)
        self.assertGreater(len(json_str), 100)

    def test_features_derivation_structure(self):
        result = analyze_model(self.clf_df, "target")
        feat_data = result["derivations"]["features"]
        self.assertIsInstance(feat_data, dict)
        self.assertGreater(len(feat_data), 0)
        
        # Verify that feature derivations contain valid fields
        for feat_name, f_deriv in feat_data.items():
            self.assertIn("feature_name", f_deriv)
            self.assertEqual(f_deriv["feature_name"], feat_name)
            self.assertIn("variables", f_deriv)
            self.assertIn("calculation_steps", f_deriv)
            self.assertIn("Relative_Share", f_deriv["variables"])


if __name__ == "__main__":
    unittest.main()
