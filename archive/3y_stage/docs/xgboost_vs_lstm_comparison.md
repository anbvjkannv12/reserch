# 四幣 XGBoost vs. LSTM 總比較

更新日期：2026-06-29

## 比較基準

本表比較 Week 3 XGBoost 與 Week 4 基準 LSTM 的下一小時漲跌分類結果。兩者皆採用四幣共用特徵、5-fold chronological walk-forward validation、每 fold 720 筆測試資料，以及訓練集與測試集之間的 24-row gap。

為維持可比性，主表使用 Week 4 原始設定的 LSTM（24 小時序列、64 units、threshold 0.5），不直接以事後挑選的最佳延伸實驗取代基準結果。

## 模型指標

| 幣種 | 模型 | Accuracy | F1 | Log loss | 策略報酬 | Sharpe | 最大回撤 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | XGBoost | 0.5283 | 0.5242 | 0.6919 | 0.0996 | 3.4311 | -0.0795 |
| BTC | LSTM | 0.5281 | 0.5543 | 0.6921 | 0.1244 | 3.8182 | -0.0657 |
| ETH | XGBoost | 0.5086 | 0.5411 | 0.6931 | -0.1429 | -3.0819 | -0.2576 |
| ETH | LSTM | 0.5186 | 0.5433 | 0.6931 | 0.1175 | 1.7735 | -0.1724 |
| SOL | XGBoost | 0.5114 | 0.5260 | 0.6941 | 0.1338 | 2.0014 | -0.1750 |
| SOL | LSTM | 0.5008 | 0.5080 | 0.6986 | 0.0287 | 0.3486 | -0.1865 |
| XRP | XGBoost | 0.5119 | 0.4920 | 0.6954 | -0.1292 | -2.4651 | -0.2174 |
| XRP | LSTM | 0.5186 | 0.5089 | 0.6944 | -0.0198 | -0.3724 | -0.2006 |

## LSTM 相對 XGBoost 的差異

正值代表 LSTM 較高；最大回撤的差值為正時，代表回撤更接近 0、風險較低。

| 幣種 | Accuracy 差異 | F1 差異 | 策略報酬差異 | Sharpe 差異 | 最大回撤差異 | 基準模型判讀 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| BTC | -0.0003 | +0.0301 | +0.0248 | +0.3871 | +0.0137 | Accuracy 幾乎持平，LSTM 的分類平衡與交易指標較佳 |
| ETH | +0.0100 | +0.0022 | +0.2604 | +4.8554 | +0.0852 | LSTM 明顯較佳，策略報酬由負轉正 |
| SOL | -0.0106 | -0.0180 | -0.1051 | -1.6528 | -0.0115 | XGBoost 較佳，基準 LSTM 不適合作為 SOL 首選 |
| XRP | +0.0067 | +0.0169 | +0.1094 | +2.0927 | +0.0168 | LSTM 有改善，但兩者策略報酬仍為負 |

## 總結

- BTC：基準 LSTM 與 XGBoost 的 accuracy 幾乎相同，但 LSTM 的 F1、策略報酬、Sharpe 與最大回撤較佳。
- ETH：基準 LSTM 是較明確的選擇，分類結果小幅改善，交易結果則大幅優於 XGBoost。
- SOL：XGBoost 優於基準 LSTM；Week 4 延伸實驗的 32-unit LSTM 雖有改善，仍應將 XGBoost 保留為主要 benchmark。
- XRP：基準 LSTM 優於 XGBoost，但策略報酬仍為負；延伸實驗顯示 48 小時序列較有潛力。

目前較合理的模型保留策略是：BTC、ETH、XRP 保留 LSTM 作為後續融合候選，SOL 同時保留 XGBoost 與 32-unit LSTM。所有交易結果目前均未納入手續費，不能直接解讀為實際可交易績效。

## 資料來源

- `models/week3_naive_xgboost/metrics_summary.csv`
- `models/week4_lstm/metrics_summary.csv`
- `models/week4_lstm/week4_lstm_final_experiment_conclusion.md`
