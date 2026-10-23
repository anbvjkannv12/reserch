# 四幣下一小時 Log Return 回歸比較

更新日期：2026-06-30

## 實驗問題

使用目前 53 個共用特徵，預測下一小時：

`future_log_return = log(close[t+1] / close[t])`

所有正式模型皆使用 5-fold chronological walk-forward validation、每 fold 720 筆測試資料、24-row gap 及最多 5,760 筆訓練資料。主要指標為 MAE、RMSE；directional accuracy 與 correlation 僅作補充。

## 平均結果

| 幣種 | 模型 | MAE | RMSE | Directional accuracy | Correlation | 判讀 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| BTC | Train-mean baseline | **0.002411** | **0.003550** | 0.5053 | 0.0000 | 誤差最低 |
| BTC | XGBRegressor | 0.002425 | 0.003570 | 0.5158 | -0.0046 | 方向略高，幅度未勝 baseline |
| BTC | LSTM 64 units | 0.002431 | 0.003575 | 0.4942 | 0.0061 | 未勝 baseline |
| BTC | LSTM 32 units＋clipping | 0.002415 | 0.003559 | 0.5158 | 0.0248 | 最佳學習模型，仍未勝 baseline |
| BTC | Transformer Encoder | 0.002419 | 0.003568 | 0.5117 | 0.0070 | 未勝 baseline |
| ETH | Train-mean baseline | **0.004595** | **0.006879** | 0.5014 | 0.0000 | 誤差最低 |
| ETH | XGBRegressor | 0.004609 | 0.006886 | 0.5086 | 0.0358 | 學習模型中誤差最低，但未勝 baseline |
| ETH | LSTM 64 units | 0.004624 | 0.006913 | **0.5153** | 0.0228 | 方向略高，幅度未改善 |
| ETH | LSTM 32 units＋clipping | 0.004610 | 0.006905 | 0.5122 | 0.0145 | 較原始 LSTM 改善，仍未勝 baseline |
| ETH | Transformer Encoder | 0.004629 | 0.006917 | 0.5142 | 0.0032 | 方向略高，幅度未勝 baseline |
| SOL | Train-mean baseline | 0.005601 | 0.007780 | 0.4917 | 0.0000 | 幾乎與 XGBoost 相同 |
| SOL | XGBRegressor | **0.005596** | **0.007779** | **0.5194** | 0.0460 | 僅極小幅勝出，R² 仍為負 |
| SOL | LSTM 64 units | 0.005675 | 0.007888 | 0.5056 | -0.0204 | 未勝 baseline |
| SOL | LSTM 32 units＋clipping | 0.005623 | 0.007828 | 0.5072 | -0.0003 | 較原始 LSTM 改善，仍未勝 baseline |
| SOL | Transformer Encoder | 0.005619 | 0.007801 | 0.5033 | -0.0028 | 未勝 baseline |
| XRP | Train-mean baseline | **0.004738** | **0.007011** | **0.5183** | 0.0000 | 誤差與方向皆最佳 |
| XRP | XGBRegressor | 0.004766 | 0.007076 | 0.4919 | -0.0119 | 未勝 baseline |
| XRP | LSTM 64 units | 0.004765 | 0.007079 | 0.5114 | 0.0307 | 未勝 baseline |
| XRP | LSTM 32 units＋clipping | 0.004779 | 0.007076 | 0.5078 | -0.0114 | 未勝 baseline |
| XRP | Transformer Encoder | 0.004780 | 0.007033 | 0.5050 | 0.0087 | 未勝 baseline |

註：原始 LSTM Notebook 的 train-mean baseline 使用 inner-training mean；本表 baseline 數字採用完整 outer-training mean，以便和 XGBoost 及控制實驗使用一致定義。差異極小，不影響結論。

## Baseline 檢查

- `zero_return` 和 `train_mean` 幾乎相同，說明下一小時報酬的條件平均值非常接近 0。
- `naive_last_return` 在四幣皆明顯較差，RMSE 約為 train-mean baseline 的 1.4 倍，表示上一小時報酬幅度不能直接延續到下一小時。
- SOL XGBRegressor 的 MAE 僅比 train-mean baseline 低約 0.09%，RMSE 改善也接近 0；mean R² 仍為負，因此不能視為穩定的數值預測能力。

## 控制實驗

控制實驗將 LSTM 從 64 units 降為 32 units，Dense hidden layer 從 32 降為 16，並使用 inner-training fold 的第 1／99 百分位裁切訓練目標。所有測試指標仍以未裁切的真實 log return 計算。

較小模型改善 BTC、ETH、SOL 的 MAE，但沒有任何幣種穩定超越 train-mean baseline。這表示原始結果不只是模型過大或少數極端值造成；主要限制仍是目前特徵對下一小時報酬幅度的訊號不足。

## 最終結論

目前資料與特徵較適合預測下一小時漲跌方向，不適合精確預測下一小時報酬幅度。回歸任務已完成 XGBoost、LSTM、Transformer 與多個 naive baselines 的正式驗證，可以保留為報告中的負結果；不建議繼續投入大量時間微調單一模型。

後續 Transformer 與 Stacking 可保留回歸輸出以符合雙任務研究規格，但模型選擇必須和 zero/train-mean baseline 比較。若仍無法穩定改善，最終系統應以分類結果作為主要成果，回歸結果作為限制與未來工作。

## 產物

- `models/week3_naive_xgboost/log_return_regression/`
- `models/week4_lstm/log_return_regression/`
- `models/week4_lstm/log_return_regression_control_32_clipped/`
- `models/week5_transformer/regression/`
- `src/train_log_return_xgboost.py`
- `src/train_log_return_lstm_control.py`
- `src/train_week5_transformer.py`
