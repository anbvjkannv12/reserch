# Multi-Horizon Experiment Guide

The shared experiment functions now support 4, 12, and 24 hour targets, per-symbol and per-horizon model settings, validation-only threshold tuning, and alternate CSV mappings for extended history.

## Extended data mapping

Keep the original three-year files unchanged and place the expanded files in `data/raw/`. Pass the selected files explicitly:

```python
coin_files_5y = {
    "BTC": "btc_5y.csv",
    "ETH": "eth_5y.csv",
    "SOL": "sol_5y.csv",
    "XRP": "xrp_5y.csv",
}
```

Use `max_train_size=17520` for the five-year experiment. The outer test window remains the latest five 720-row folds, so the baseline and expanded-data runs use the same ending period.

## Per-symbol and per-horizon settings

The `configs` mapping is keyed by symbol, horizon, and model:

```python
configs = {
    "BTC": {
        "4": {
            "lstm": {"sequence_length": 12, "lstm_units": 32},
            "transformer": {"sequence_length": 24, "d_model": 32, "ff_dim": 64},
        },
        "12": {
            "lstm": {"sequence_length": 24, "lstm_units": 64},
            "transformer": {"sequence_length": 48, "d_model": 64, "ff_dim": 128},
        },
    }
}
```

Missing entries use the shared defaults. This allows each coin and horizon to be tuned independently without changing the training code.

## Threshold policy

Classification runs tune thresholds on the chronological inner validation window using candidates from `0.35` to `0.65` in `0.01` increments. The default objective maximizes validation F1, then uses validation strategy Sharpe as the tie-breaker when F1 is within `0.01` of the best value. Outer test predictions only apply the selected threshold.

Every classification run writes `threshold_choices.csv` and records the policy in `manifest.json`.

## Cost-aware labels

The reusable `cost_aware_target` helper returns `-1`, `0`, and `1` for down, neutral, and up moves outside a supplied transaction-cost band. It is intentionally not part of the current binary multi-horizon training run; use it for the follow-up three-class experiment after the best horizon and model are selected.
