import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.lstm_experiment_utils import add_features, feature_columns
from src.pytorch_transformer_artifacts import (
    load_artifact_pair,
    predict_latest_probability,
)
from src.train_btc_24h_pytorch_transformer import run_training


class Btc24hPyTorchTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.csv_path = self.root / "btc_hourly.csv"
        self.output_dir = self.root / "outputs"
        hours = pd.date_range("2024-01-01", periods=1400, freq="h", tz="UTC")
        trend = 100.0 + np.linspace(0.0, 25.0, len(hours)) + np.sin(np.arange(len(hours)) / 5)
        raw = pd.DataFrame(
            {
                "open_time": hours,
                "open": trend - 0.1,
                "high": trend + 0.5,
                "low": trend - 0.5,
                "close": trend,
                "volume": 1000 + np.arange(len(hours)) % 50,
                "quote_asset_volume": (1000 + np.arange(len(hours)) % 50) * trend,
                "number_of_trades": 100 + np.arange(len(hours)) % 20,
                "taker_buy_base": 450 + np.arange(len(hours)) % 30,
                "taker_buy_quote": (450 + np.arange(len(hours)) % 30) * trend,
            }
        )
        raw.to_csv(self.csv_path, index=False)
        self.feature_rows = add_features(raw, prediction_horizon=24)[feature_columns(
            add_features(raw, prediction_horizon=24)
        )].tail(24)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_smoke_training_writes_a_loadable_final_pair(self):
        """One small chronological run must emit an inference-ready final pair."""
        output_dir = run_training(
            csv_path=self.csv_path,
            output_dir=self.output_dir,
            folds=1,
            epochs=1,
            batch_size=32,
            test_size=240,
            inner_valid_size=240,
            max_train_size=None,
            d_model=16,
            ff_dim=32,
        )

        model, metadata = load_artifact_pair(
            output_dir / "checkpoints" / "btc_24h_final.pth",
            output_dir / "checkpoints" / "btc_24h_final.pkl",
        )
        result = predict_latest_probability(model, metadata, self.feature_rows)

        self.assertIn("up_probability", result)
        self.assertTrue((output_dir / "fold_metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
