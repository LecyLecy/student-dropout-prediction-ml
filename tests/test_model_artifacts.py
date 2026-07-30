import json
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ModelArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads(
            (PROJECT_ROOT / "models" / "model_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        cls.feature_config = json.loads(
            (PROJECT_ROOT / "app" / "feature_config.json").read_text(
                encoding="utf-8"
            )
        )
        cls.pipelines = joblib.load(
            PROJECT_ROOT / "models" / "model_pipelines.pkl"
        )
        cls.processed_data = pd.read_csv(
            PROJECT_ROOT / "data" / "processed" / "processed.csv"
        )

    def test_feature_contract_matches_across_artifacts(self):
        metadata_features = self.metadata["mvp_features"]
        configured_features = self.feature_config["features"]
        data_features = [
            column
            for column in self.processed_data.columns
            if column != "Target"
        ]

        self.assertEqual(metadata_features, configured_features)
        self.assertEqual(metadata_features, data_features)

    def test_all_declared_models_are_available(self):
        self.assertEqual(
            set(self.metadata["available_models"]),
            set(self.pipelines.keys()),
        )
        self.assertIn(self.metadata["best_model"], self.pipelines)

    def test_every_pipeline_returns_valid_binary_probabilities(self):
        sample = self.processed_data[self.metadata["mvp_features"]].iloc[[0]]

        for model_name, pipeline in self.pipelines.items():
            with self.subTest(model=model_name):
                probabilities = pipeline.predict_proba(sample)[0]
                self.assertEqual(probabilities.shape, (2,))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue((probabilities >= 0).all())
                self.assertTrue((probabilities <= 1).all())
                self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
