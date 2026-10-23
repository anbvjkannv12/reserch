import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.pytorch_transformer_artifacts import (
    BtcTransformerClassifier,
    TransformerConfig,
    load_artifact_pair,
    predict_latest_probability,
    save_artifact_pair,
)


class PyTorchTransformerArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.feature_names = [f"feature_{index}" for index in range(53)]
        self.config = TransformerConfig(
            feature_count=53, sequence_length=24, d_model=16, ff_dim=32
        )
        self.model = BtcTransformerClassifier(self.config)
        self.metadata = {
            "feature_columns": self.feature_names,
            "scaler_mean": [0.0] * 53,
            "scaler_std": [1.0] * 53,
            "sequence_length": 24,
            "selected_threshold": 0.5,
            "target_definition": "close[t + 24] > close[t]",
            "training_coverage": {"fit_start": "2024-01-01", "fit_end": "2024-01-31"},
        }

    def test_classifier_returns_one_logit_for_each_sequence(self):
        """The classifier must preserve batch size and emit one logit per sequence."""
        model = BtcTransformerClassifier(
            TransformerConfig(feature_count=53, sequence_length=24, d_model=16, ff_dim=32)
        )

        logits = model(torch.zeros((3, 24, 53), dtype=torch.float32))

        self.assertEqual(tuple(logits.shape), (3,))

    def test_artifact_pair_round_trips_state_and_metadata(self):
        """A saved pair must restore the exact model tensors and inference schema."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            base_path = Path(temporary_directory) / "btc_24h_fold1"

            save_artifact_pair(base_path, self.model, self.config, self.metadata)
            restored_model, restored_metadata = load_artifact_pair(
                base_path.with_suffix(".pth"), base_path.with_suffix(".pkl")
            )

        self.assertEqual(restored_metadata["feature_columns"], self.feature_names)
        self.assertTrue(
            torch.equal(
                restored_model.state_dict()["head.1.weight"],
                self.model.state_dict()["head.1.weight"],
            )
        )

    def test_inference_rejects_reordered_feature_columns(self):
        """Reordered inputs must not silently produce a prediction with wrong features."""
        rows = pd.DataFrame(np.zeros((24, 53)), columns=self.feature_names[::-1])

        with self.assertRaisesRegex(ValueError, "feature column order"):
            predict_latest_probability(self.model, self.metadata, rows)

    def test_inference_returns_probability_and_thresholded_direction(self):
        """A valid latest feature window must produce a bounded probability and decision."""
        rows = pd.DataFrame(np.zeros((24, 53)), columns=self.feature_names)

        result = predict_latest_probability(self.model, self.metadata, rows)

        self.assertGreaterEqual(result["up_probability"], 0.0)
        self.assertLessEqual(result["up_probability"], 1.0)
        self.assertIsInstance(result["predict_up"], bool)


if __name__ == "__main__":
    unittest.main()
