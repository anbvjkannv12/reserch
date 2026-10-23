# Week 5 四幣 Transformer Encoder 結果

更新日期：2026-06-30

## 實驗設定

- 任務：下一小時 Up/Down 分類與下一小時 log return 回歸。
- 輸入：過去 24 小時、每小時 53 個共用特徵。
- 驗證：5-fold chronological walk-forward；每 fold 720 筆測試資料、24-row outer/inner gap，最多 5,760 筆 outer-training rows。
- 防洩漏：特徵 scaler 與回歸 target scaler 僅以 inner-training rows 擬合；Early Stopping 僅觀察時間上較後、但仍早於 outer test 的 inner-validation window。
- 架構：1 個 Transformer Encoder block、`d_model=64`、4 heads、FFN 128、dropout 0.2、learned positional embedding、Global Average Pooling。
- 訓練：Adam（learning rate 0.001）、batch size 64、最多 15 epochs、patience 3。
- 環境：Windows CPU、Python 3.12、TensorFlow 2.21.0。
- 實作 Notebook：`notebooks/week5_transformer.ipynb`，已完整執行並保留各 cell 輸出。

## 分類結果

主指標為 F1。表內交易結果未計入手續費，僅供模型間一致比較。

| 幣種 | 模型 | Accuracy | F1 | Log loss | 策略報酬 | Sharpe | 最大回撤 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | XGBoost | **0.5283** | 0.5242 | **0.6919** | 0.0996 | 3.4311 | -0.0795 |
| BTC | LSTM | 0.5281 | **0.5543** | 0.6921 | **0.1244** | **3.8182** | **-0.0657** |
| BTC | Transformer | 0.5136 | 0.5377 | 0.6933 | 0.0247 | 0.9014 | -0.1072 |
| ETH | XGBoost | 0.5086 | 0.5411 | 0.6931 | -0.1429 | -3.0819 | -0.2576 |
| ETH | LSTM | **0.5186** | **0.5433** | **0.6931** | **0.1175** | **1.7735** | **-0.1724** |
| ETH | Transformer | 0.5014 | 0.5380 | 0.6955 | -0.0751 | -1.6756 | -0.2394 |
| SOL | XGBoost | 0.5114 | 0.5260 | **0.6941** | **0.1338** | **2.0014** | **-0.1750** |
| SOL | LSTM | 0.5008 | 0.5080 | 0.6986 | 0.0287 | 0.3486 | -0.1865 |
| SOL | Transformer | 0.5097 | **0.5444** | 0.6970 | -0.0250 | -0.5293 | -0.2626 |
| XRP | XGBoost | 0.5119 | 0.4920 | 0.6954 | -0.1292 | -2.4651 | -0.2174 |
| XRP | LSTM | **0.5186** | **0.5089** | **0.6944** | -0.0198 | -0.3724 | -0.2006 |
| XRP | Transformer | 0.5014 | 0.4815 | 0.6961 | **-0.0143** | **-0.2485** | **-0.1590** |

粗體依主要分類品質與交易風險分開標示，不代表已找到可實際交易的模型。

### 分類判讀

- BTC、ETH：LSTM 的整體結果最佳；Transformer 的 F1 尚可，但 accuracy、log loss 與交易指標未改善。
- SOL：Transformer 得到三個學習模型中最高 F1（0.5444），但 recall 偏高且 precision 較低，log loss 與交易結果也較差；不能只看 F1 就宣稱全面勝出。
- XRP：Transformer 的回撤較小，但分類品質未勝 LSTM，三個模型的策略報酬皆為負。
- 四幣沒有同一個模型全面領先，保留三者的 OOF 預測進行 Week 6 Stacking 是合理下一步。

## 回歸結果

| 幣種 | 模型 | MAE | RMSE | Directional accuracy | Correlation | R² |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BTC | Train-mean baseline | **0.002411** | **0.003549** | 0.5053 | 0.0000 | -0.0003 |
| BTC | Transformer | 0.002419 | 0.003568 | **0.5117** | 0.0070 | -0.0111 |
| ETH | Train-mean baseline | **0.004597** | **0.006880** | 0.4803 | 0.0000 | -0.0017 |
| ETH | Transformer | 0.004629 | 0.006917 | **0.5142** | 0.0032 | -0.0127 |
| SOL | Train-mean baseline | **0.005602** | **0.007780** | 0.4894 | 0.0000 | -0.0009 |
| SOL | Transformer | 0.005619 | 0.007801 | **0.5033** | -0.0028 | -0.0065 |
| XRP | Train-mean baseline | **0.004738** | **0.007011** | **0.5183** | 0.0000 | -0.0014 |
| XRP | Transformer | 0.004780 | 0.007033 | 0.5050 | 0.0087 | -0.0075 |

Transformer 在 BTC、ETH、SOL 的 directional accuracy 高於同 fold 的 train-mean baseline，但四幣 MAE、RMSE 皆未勝 baseline，且 R² 全為負。這延續 Week 3–4 的結論：目前特徵對方向有少量訊號，對下一小時報酬幅度沒有穩定可泛化的預測力。

## 可重現執行方式

在專題根目錄執行：

```powershell
C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m venv .venv-transformer
.\.venv-transformer\Scripts\python.exe -m pip install -r requirements-week5-transformer.txt
.\.venv-transformer\Scripts\python.exe src\train_week5_transformer.py --dry-run --coins BTC --folds 1
.\.venv-transformer\Scripts\python.exe src\train_week5_transformer.py --smoke-test --coins BTC
.\.venv-transformer\Scripts\python.exe src\train_week5_transformer.py
```

正式輸出位於 `models/week5_transformer/`：

- `manifest.json`：模型、環境、切分及防洩漏設定。
- `classification/`：fold metrics、summary、training history 與 14,400 筆 OOF predictions。
- `regression/`：fold metrics、summary、training history 與 14,400 筆 OOF predictions。
- `smoke_test/`：1 幣、1 fold、1 epoch 的端到端測試產物，不納入正式結論。

Notebook 會依序執行環境檢查、四幣資料稽核、模型結構摘要、dry run、分類訓練、回歸訓練及結果完整性檢查。若只想閱讀本週實作與既有輸出，可直接開啟已執行的 `notebooks/week5_transformer.ipynb`，不需重新訓練。

## 結論

Week 5 驗收完成：四幣 Transformer 分類與回歸皆完成 5-fold walk-forward，設定與輸出可重現，也已保留 Week 6 Stacking 所需的逐筆 OOF 預測。Transformer 沒有全面取代 XGBoost 或 LSTM；最值得保留的互補訊號是 SOL 分類 F1，以及 BTC／ETH／SOL 回歸方向正確率，但回歸幅度預測仍應視為負結果。
