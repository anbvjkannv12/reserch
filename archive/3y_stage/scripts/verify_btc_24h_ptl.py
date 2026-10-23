"""Check that the .ptl gives the same up-probability as the original .pth inference path."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.jit.mobile import _load_for_lite_interpreter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pytorch_transformer_artifacts import load_artifact_pair, predict_latest_probability  # noqa: E402

CHECKPOINTS = ROOT / "models" / "btc_24h_pytorch_transformer" / "checkpoints"
TOLERANCE = 1e-5
WINDOWS = 200


def main() -> None:
    spec = json.loads((CHECKPOINTS / "btc_24h_final.ptl.json").read_text(encoding="utf-8"))
    columns, seq_len = spec["feature_columns"], spec["input"]["shape"][1]
    model, metadata = load_artifact_pair(CHECKPOINTS / "btc_24h_final.pth", CHECKPOINTS / "btc_24h_final.pkl")
    # The lite interpreter on this Windows torch build cannot open non-ASCII paths.
    ascii_tmp_dir = Path.home() / "ptl_export_tmp"
    ascii_tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_ptl = ascii_tmp_dir / "btc_24h_final.ptl"
    shutil.copy2(CHECKPOINTS / "btc_24h_final.ptl", tmp_ptl)
    lite = _load_for_lite_interpreter(str(tmp_ptl))

    # Any real feature rows work for an equivalence test; these have the same 53 columns.
    features = pd.read_csv(ROOT / "data" / "features" / "btc_24h.csv", usecols=columns)[columns]
    ends = np.linspace(seq_len, len(features), WINDOWS, dtype=int)

    worst = 0.0
    for end in ends:
        rows = features.iloc[end - seq_len:end]
        expected = predict_latest_probability(model, metadata, rows)["up_probability"]
        raw = torch.from_numpy(rows.to_numpy(dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            actual = float(lite(raw)[0])
        worst = max(worst, abs(actual - expected))

    batch = torch.from_numpy(np.stack([features.iloc[e - seq_len:e].to_numpy(dtype=np.float32) for e in ends[:4]]))
    with torch.no_grad():
        batch_out = lite(batch)
        single_out = torch.stack([lite(batch[i:i + 1])[0] for i in range(4)])
    batch_diff = float((batch_out - single_out).abs().max())

    print(f"windows={len(ends)} max|ptl - pth|={worst:.2e} batch_vs_single={batch_diff:.2e}")
    if worst > TOLERANCE or batch_diff > TOLERANCE:
        raise SystemExit("FAIL: .ptl output differs from the original model")
    print("PASS")


if __name__ == "__main__":
    main()
