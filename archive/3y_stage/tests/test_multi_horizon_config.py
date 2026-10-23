import unittest

import numpy as np

from src.lstm_experiment_utils import (
    cost_aware_target,
    find_best_threshold,
    resolve_model_config,
)


class MultiHorizonConfigTests(unittest.TestCase):
    def test_cost_aware_target_marks_neutral_returns(self):
        returns = np.array([-0.003, -0.001, 0.0, 0.001, 0.003])

        result = cost_aware_target(returns, transaction_cost=0.002)

        np.testing.assert_array_equal(result, np.array([-1, 0, 0, 0, 1]))

    def test_model_config_is_selected_by_symbol_and_horizon(self):
        configs = {
            "BTC": {"4": {"sequence_length": 12, "lstm_units": 32}},
            "default": {"sequence_length": 24, "lstm_units": 64},
        }

        config = resolve_model_config(configs, "BTC", 4, defaults=configs["default"])

        self.assertEqual(config["sequence_length"], 12)
        self.assertEqual(config["lstm_units"], 32)

    def test_threshold_is_selected_from_validation_candidates(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        proba = np.array([0.41, 0.44, 0.48, 0.52, 0.56, 0.59])
        returns = np.array([-0.01, -0.01, 0.01, 0.01, 0.01, 0.01])

        threshold, score = find_best_threshold(
            y_true,
            proba,
            returns,
            objective="f1_then_sharpe",
            thresholds=[0.45, 0.55, 0.60],
        )

        self.assertIn(threshold, {0.45, 0.55, 0.60})
        self.assertNotEqual(threshold, 0.5)
        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
