"""Export the BTC 24h Transformer (.pth) to a PyTorch Lite (.ptl) model.

The .ptl takes RAW (unscaled) features of shape (batch, 24, 53) and returns the
up-probability, because scaling and sigmoid are baked in. See docs/ptl_guide.md.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pytorch_transformer_artifacts import load_artifact_pair  # noqa: E402

CHECKPOINTS = ROOT / "models" / "btc_24h_pytorch_transformer" / "checkpoints"
PTH = CHECKPOINTS / "btc_24h_final.pth"
PKL = CHECKPOINTS / "btc_24h_final.pkl"
PTL = CHECKPOINTS / "btc_24h_final.ptl"
SPEC = CHECKPOINTS / "btc_24h_final.ptl.json"


class ProbabilityModel(torch.nn.Module):
    """Raw features in, up-probability out: (x - mean) / std -> model -> sigmoid."""

    def __init__(self, model: torch.nn.Module, mean: list[float], std: list[float]) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, raw_sequences: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model((raw_sequences - self.mean) / self.std))


def main() -> None:
    model, metadata = load_artifact_pair(PTH, PKL)  # eval() mode
    if any(s <= 0 for s in metadata["scaler_std"]):
        raise ValueError("scaler_std must be positive")
    wrapper = ProbabilityModel(model, metadata["scaler_mean"], metadata["scaler_std"]).eval()

    # The Transformer "fast path" uses fused kernels that the mobile runtime may not support.
    torch.backends.mha.set_fastpath_enabled(False)
    seq_len, n_features = model.config.sequence_length, model.config.feature_count
    example = torch.zeros(1, seq_len, n_features, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)
    try:
        mobile_model = optimize_for_mobile(traced)
        mobile_optimization = "optimize_for_mobile"
    except RuntimeError as exc:
        # This Windows torch build may not include XNNPACK. The lite interpreter can still
        # load a module saved directly for mobile; verification below is the authority.
        if "XNNPACK" not in str(exc):
            raise
        print(f"warning: optimize_for_mobile skipped because XNNPACK is unavailable: {exc}")
        mobile_model = traced
        mobile_optimization = "none - optimize_for_mobile skipped because XNNPACK is unavailable"
    ascii_tmp_dir = Path.home() / "ptl_export_tmp"
    ascii_tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_ptl = ascii_tmp_dir / PTL.name
    mobile_model._save_for_lite_interpreter(str(tmp_ptl))
    shutil.move(str(tmp_ptl), PTL)

    spec = {
        "model_file": PTL.name,
        "source_checkpoint": PTH.name,
        "source_checkpoint_sha256": hashlib.sha256(PTH.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "mobile_optimization": mobile_optimization,
        "input": {"shape": ["batch", seq_len, n_features], "dtype": "float32",
                  "scaling": "none - pass RAW feature values; scaling is inside the model",
                  "row_order": "oldest to newest; the last row is the most recent closed 1h candle"},
        "output": {"shape": ["batch"], "meaning": "probability that close[t + 24] > close[t]"},
        "threshold": float(metadata["selected_threshold"]),
        "feature_columns": list(metadata["feature_columns"]),
        "target_definition": metadata["target_definition"],
        "training_coverage": metadata["training_coverage"],
    }
    SPEC.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {PTL} ({PTL.stat().st_size} bytes) and {SPEC.name}")


if __name__ == "__main__":
    main()
