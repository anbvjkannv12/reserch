# BTC 24h PyTorch Transformer

此訓練流程建立 BTC 未來 24 小時漲跌方向分類的 PyTorch Transformer。
它是研究用方向訊號，不構成投資建議或保證交易獲利。

## 安裝

請使用具 PyTorch 的 Python 環境：

```powershell
pip install -r requirements-pytorch-transformer.txt
```

## 訓練

完整訓練使用 5 個 chronological walk-forward folds、每 fold 最多 15 epochs：

```powershell
python src/train_btc_24h_pytorch_transformer.py --epochs 15 --output-dir models/btc_24h_pytorch_transformer
```

快速驗證會以一個 fold、1 epoch 建立獨立 smoke-test 輸出：

```powershell
python src/train_btc_24h_pytorch_transformer.py --smoke-test --output-dir models/btc_24h_pytorch_transformer
```

完整訓練會輸出：

- `checkpoints/btc_24h_foldN.pth`：每個 walk-forward fold 的 PyTorch `state_dict` checkpoint。
- `checkpoints/btc_24h_foldN.pkl`：對應的特徵順序、scaler、threshold 與資料範圍。
- `checkpoints/btc_24h_final.pth`／`.pkl`：適合後續推論的最終模型配對。
- `fold_metrics.csv`、`metrics_summary.csv`、`predictions.csv`、`training_history.csv`、`threshold_choices.csv` 與 `manifest.json`：可重現的評估紀錄。

`.pth` 是 PyTorch checkpoint，可使用 `torch.load(..., weights_only=True)` 載入；`.pkl` 僅儲存推論 metadata，不含模型權重。

## 推論

輸入必須是已完成特徵工程的 DataFrame，且欄位名稱與欄位順序必須完全符合 checkpoint metadata。至少需要最近 24 筆小時資料。

```python
from pathlib import Path

import pandas as pd

from src.pytorch_transformer_artifacts import (
    load_artifact_pair,
    predict_latest_probability,
)

checkpoint_dir = Path("models/btc_24h_pytorch_transformer/checkpoints")
model, metadata = load_artifact_pair(
    checkpoint_dir / "btc_24h_final.pth",
    checkpoint_dir / "btc_24h_final.pkl",
)

# feature_rows 必須是資料工程完成後、僅包含 metadata['feature_columns'] 的 DataFrame。
result = predict_latest_probability(model, metadata, feature_rows)
print(result["up_probability"], result["predict_up"])
```

若欄位缺失、額外欄位、順序不符、包含非數值／無限值，或少於 24 筆資料，推論函式會拒絕執行，以免錯誤資料被靜默送入模型。
