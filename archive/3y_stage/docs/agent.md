# 熱門幣預測專題執行手冊

> **（2026-09-30 註記）本文件為 3 年資料階段（2026-06～08）的歷史紀錄，已不再更新。**現行研究範圍以根目錄 [`management.md`](../../../management.md) 為準，進度見 [`時程.md`](../時程.md)，最後階段工作見 [`final_stage_work_plan.md`](final_stage_work_plan.md)。

> 建立日期：2026-06-02  
> 規劃期間：2026-06-02 至 2026-07-27  
> 研究範圍：BTC、ETH、SOL、XRP 熱門幣種價格預測

## 0. 本次討論結論與下次起點（2026-08-05）

### 已完成的整理

- `[x]` 已完成 XGBoost、LSTM、Transformer 在 1／4／12／24 小時的分類結果整理。
- `[x]` 已完成逐幣種最佳模型表與三種模型平均比較表，文件位置：`docs/final_model_comparison_and_conclusion.md`。
- `[x]` 已決定專題主要成果採用「漲跌方向分類」。
- `[x]` 已決定將報酬率回歸定位為負結果與研究限制，不再作為主要成果，也不優先追加回歸調參。

### 下次請從這裡開始

下一階段以 **12 小時分類最佳化** 為主軸，並保留 1 小時作為對照組：

1. 先確認 12h 的 `Up / Down` 類別比例、標籤定義與目前 baseline。
2. 以時間序列 walk-forward validation 重跑 XGBoost、LSTM、Transformer，確保 12h 結果不是單一切分造成的偶然提升。
3. 以 F1、Balanced Accuracy、MCC、ROC-AUC、Log Loss 同時評估；不要只追求 Accuracy。
4. 優先調整特徵、類別權重與分類 threshold，再進行模型超參數調整。
5. 12h 與 1h 的結果確認穩定後，再考慮機率校準、soft voting／stacking、SHAP 與交易回測。
6. 4h 與 24h 保留為補充比較，不需要先投入大量調參時間。

目前的工作假設是：12h 值得優先深入，但尚未證明全面優於 1h；1h LSTM 仍是重要對照。即使後續分類結果提升，也應描述為有限的方向預測訊號，不宣稱穩定獲利能力。

## 1. 專題範圍

### 1.1 本階段目標

以 BTC、ETH、SOL、XRP 的 1 小時 K 線資料與過去 24 小時滑動視窗，建立可重現的熱門幣預測流程，輸出下一小時：

1. `Up / Down` 價格方向分類。
2. 對數收益率（log return）回歸預測。

最終比較 XGBoost、LSTM、Transformer Encoder 與 Stacking 融合模型，加入模型解釋與交易回測，整理成可展示及可寫入成果報告的圖表與結論。

### 1.2 不在工作範圍內

本階段不處理以下內容：

- 穩定幣脫鉤風險預警。
- USDC/USDT、TUSD/USDT 監控。
- Isolation Forest 異常偵測。
- 動態預警門檻與風險分數。
- 熱門幣與穩定幣之跨市場傳導分析。

## 2. 狀態標記

| 標記 | 意義 |
| --- | --- |
| `[x]` | 已完成 |
| `[-]` | 進行中 |
| `[ ]` | 待處理 |
| `[!]` | 阻塞，需要先排除問題 |

## 3. 現況盤點

### 3.1 已有資料

| 幣種 | CSV 檔案 | 筆數 | 已知狀況 |
| --- | --- | ---: | --- |
| BTC | `btc_3y.csv` | 26,253 | 無空值、無重複時間戳；時間範圍比其他幣種偏移一小時 |
| ETH | `eth_3years.csv` | 26,253 | 無空值、無重複時間戳 |
| SOL | `sol_3years.csv` | 26,253 | 無空值、無重複時間戳 |
| XRP | `xrp_3y_hourly.csv` | 26,253 | 無空值、無重複時間戳 |

四份資料皆為小時級 OHLCV 與交易資訊，每份資料目前各缺少 27 個小時。正式訓練前，需先統一 UTC 時間軸、缺值處理與四幣共同可用時段。

### 3.2 已有實驗

- `[x]` 已蒐集 BTC、ETH、SOL、XRP 約三年的小時級資料。
- `[x]` BTC notebook 已有較完整的缺值處理、時間特徵與技術指標雛形。
- `[x]` 已有 XGBoost、Random Forest、LSTM 與回測實驗雛形。
- `[x]` BTC、ETH、SOL、XRP 已統一使用 53 個共用特徵與相同的分類標籤。
- `[x]` 下一小時 log return 回歸已納入 Week 3 XGBoost 與 Week 4 LSTM 的正式 walk-forward 比較；目前各模型皆未穩定超越簡單基準。
- `[x]` 已完成 Transformer Encoder：四幣分類與回歸皆採 24 小時序列、5-fold walk-forward 與 24-row gap，並保留逐筆 OOF 預測。
- `[ ]` 尚未完成 OOF Stacking。
- `[ ]` 尚未完成完整 SHAP 分析與四幣差異比較。
- `[x]` Week 2–4 正式流程的分類標籤已統一為下一小時 `Up / Down`；舊 notebook 仍保留不同規格，不作正式比較依據。
- `[x]` TensorFlow/protobuf 相依性問題已排除，Week 4 使用 TensorFlow 2.10.1 完成四幣實驗。

## 4. 統一實驗規格

### 4.1 輸入與標籤

- 資料頻率：`1 hour`。
- 序列視窗：過去 `24 hours`。
- 分類標籤：下一小時收盤價高於目前收盤價為 `Up`，否則為 `Down`。
- 回歸標籤：下一小時對數收益率 `log(close[t+1] / close[t])`。
- 幣種：BTC、ETH、SOL、XRP。

### 4.2 共用特徵

- OHLCV：open、high、low、close、volume。
- 報酬與動能：log return、RSI、MACD。
- 通道與波動：Bollinger Bands、ATR、rolling volatility、high-low range。
- 時間依賴：lag features。
- 時間週期：小時與星期的週期性編碼。

### 4.3 資料切分原則

- 依時間順序切分資料，不可使用隨機切分。
- 使用 TimeSeriesSplit 與 walk-forward validation。
- 訓練集與驗證集之間加入 gap，降低高度相鄰樣本造成的評估偏差。
- Scaler 僅能在訓練集上 `fit`，再套用到驗證集與測試集。
- 特徵工程不得使用任何未來資訊。
- 每次實驗需記錄資料期間、特徵集合、模型參數與結果。

### 4.4 下一階段實驗規格

- 主要 horizon：`12h`。
- 對照 horizon：`1h`。
- 補充 horizon：`4h`、`24h`，只做必要比較。
- 主要任務：分類；回歸保留既有負結果，不列為下一階段調參主線。
- 選模不可只看 Accuracy，至少同時記錄 F1、Balanced Accuracy、MCC、ROC-AUC 與 Log Loss。
- Threshold tuning 與機率校準只能使用 validation 資料，outer test 僅做一次最終評估。

## 5. 八週進度追蹤

| 狀態 | 週次 | 日期 | 主要工作 | 驗收成果 |
| --- | --- | --- | --- | --- |
| `[-]` | 第 1 週 | 2026-06-02 至 2026-06-08 | 已盤點四幣資料、確認 UTC 時序與缺漏；尚缺獨立乾淨資料輸出、共同時段對齊及正式品質摘要 | 可重現的四幣乾淨資料與品質摘要 |
| `[x]` | 第 2 週 | 2026-06-09 至 2026-06-15 | 已建立 53 個共用特徵、下一小時分類與回歸標籤，四幣皆由同一流程產生 | 四幣使用同一套特徵與雙任務標籤 |
| `[x]` | 第 3 週 | 2026-06-16 至 2026-06-22 | 已完成 naive baseline、XGBoost 分類與回歸、5-fold walk-forward validation、內層調參及 24-row gap | 四幣 XGBoost 分類與回歸比較表 |
| `[x]` | 第 4 週 | 2026-06-23 至 2026-06-29 | 已修正 TensorFlow 環境，完成四幣 LSTM 分類與回歸、訓練曲線、分類延伸實驗及一組回歸控制實驗 | 四幣 LSTM 結果與訓練曲線 |
| `[x]` | 第 5 週 | 2026-06-30 至 2026-07-06 | 已建立 Transformer Encoder，完成四幣分類與回歸 5-fold walk-forward，沿用 24 小時序列輸入及 leakage controls | 四幣 Transformer 結果、OOF 預測與可重現設定 |
| `[ ]` | 第 6 週 | 2026-07-07 至 2026-07-13 | 新增 CatBoost／TCN 分類與回歸；複驗 LSTM 12h 分類；完成 Transformer 24h 分類機率校準；建立 OOF Stacking | 六項候選實驗、校準前後比較、單一模型與融合模型效能比較 |
| `[ ]` | 第 7 週 | 2026-07-14 至 2026-07-20 | 加入 SHAP 解釋、交易回測、手續費設定與四幣差異分析 | 圖表、特徵重要性、回測摘要 |
| `[ ]` | 第 8 週 | 2026-07-21 至 2026-07-27 | 重跑實驗、整理展示流程、更新成果報告可直接使用的表格與圖 | 可展示版本與最終檢查清單 |

### 第 6 週詳細計劃（2026-07-07 至 2026-07-13）

| 日期 | 工作項目 | 執行內容 | 驗收成果 |
| --- | --- | --- | --- |
| 07-07 | 共用流程與基準固定 | 固定 4／12／24h 標籤、53 個特徵、5-fold walk-forward、每 fold 720 筆 test、24-row gap；建立 CatBoost／TCN 共用輸出介面 | 可重現設定與 smoke test |
| 07-08 | CatBoost classification | 四幣、三個 horizon 完成分類訓練；保留 fold metrics、OOF probabilities 與特徵重要性 | CatBoost 分類結果與基準比較 |
| 07-09 | CatBoost regression | 四幣、三個 horizon 完成 log-return 回歸；比較 train-mean baseline | CatBoost 回歸 MAE、RMSE、R² 與方向正確率 |
| 07-10 | TCN classification | 使用 causal／dilated 1D convolution 與 residual blocks，完成四幣、三個 horizon 分類 | TCN 分類結果、訓練曲線與 OOF probabilities |
| 07-11 | TCN regression | 沿用相同序列、fold 與 leakage controls 完成 log-return 回歸 | TCN 回歸結果、訓練曲線與 OOF predictions |
| 07-12 | 候選模型複驗與校準 | 複驗 LSTM 12h 分類；複驗 Transformer 24h 分類，使用 inner validation 比較 Platt scaling 與 isotonic calibration，outer test 僅做一次最終評估 | LSTM 12h 正式結果；Transformer 24h 校準前後 Log Loss、Brier Score、ECE 與可靠度圖 |
| 07-13 | 比較與融合 | 對齊各模型 OOF predictions，建立分類 Logistic Regression／回歸 Ridge Stacking，整理第 6 週報告 | 單一模型、校準模型與融合模型完整比較表 |

第 6 週模型通過條件：

- 分類：主要檢查 Balanced Accuracy、MCC、F1、ROC-AUC 與 Log Loss；相較同 fold 現有最佳模型平均提升至少 1 個百分點，且至少 4／5 folds 同方向改善。
- 機率校準：校準後 Log Loss、Brier Score、ECE 至少兩項改善；校準器只可由 inner-training／validation 擬合，不可使用 outer test。
- 回歸：MAE／RMSE 必須勝過 train-mean baseline，且 R² 大於 0；方向正確率只作補充，不可單獨作為通過依據。
- 所有模型均需輸出 `fold_metrics.csv`、`metrics_summary.csv`、`predictions.csv` 與可重現的 manifest；神經網路另保留 `training_history.csv`。

## 6. 評估與驗收

### 6.1 分類任務

- 主要指標：F1-score。
- 補充指標：Precision、Recall、Accuracy、混淆矩陣。

### 6.2 回歸任務

- MAE。
- RMSE。

### 6.3 交易回測

- 交易次數。
- 勝率。
- 累積報酬。
- Sharpe Ratio。
- 手續費假設。

### 6.4 最終交付成果

- `[ ]` 四幣資料品質摘要。
- `[x]` 四幣 XGBoost 與 LSTM 分類效能比較表（`docs/xgboost_vs_lstm_comparison.md`）。
- `[x]` 四幣 XGBoost、LSTM 與 naive baselines 回歸效能比較表（`docs/log_return_regression_comparison.md`）。
- `[x]` 納入 Transformer 的四幣單一模型完整效能比較表（`docs/week5_transformer_comparison.md`）。
- `[ ]` 單一模型與 Stacking 比較圖。
- `[ ]` SHAP 特徵重要性圖。
- `[ ]` 交易回測圖與摘要。
- `[ ]` 四幣差異分析結論。
- `[ ]` 可重現的執行說明與最終檢查清單。

## 7. 每週更新紀錄

每週至少更新一次，並同步調整第 5 節的狀態標記。

### 更新模板

```markdown
### YYYY-MM-DD

- 本週完成：
  - 
- 實驗結果：
  - 
- 問題與阻塞：
  - 
- 下一步：
  - 
```

### 2026-06-02

- 本週完成：
  - 讀取初審報告，將本階段範圍限縮為熱門幣預測。
  - 盤點 BTC、ETH、SOL、XRP 的 CSV 與既有 notebook。
  - 確認四份 CSV 各有 26,253 筆資料，無空值與重複時間戳。
- 實驗結果：
  - 四份資料皆各缺少 27 個小時。
  - BTC 時間範圍相較其他三幣偏移一小時。
- 問題與阻塞：
  - 現有 notebook 的標籤定義尚未統一。
  - TensorFlow/protobuf 環境問題需在 LSTM 階段前處理。
- 下一步：
  - 統一四幣 UTC 時間軸與缺值補齊策略。
  - 固定資料切分規則，產出資料品質摘要。

### 2026-06-29

- 本週完成：
  - TensorFlow 環境已固定為 2.10.1，完成 BTC、ETH、SOL、XRP 的 LSTM 分類訓練。
  - 沿用 Week 3 的 5-fold walk-forward、每 fold 720 筆測試資料及 24-row gap。
  - 完成 12／24／48 小時序列、32-unit LSTM 與分類門檻調整三組延伸實驗。
  - 新增 `docs/xgboost_vs_lstm_comparison.md`，整理四幣 XGBoost 與基準 LSTM 的公平比較。
  - 完成四幣下一小時 log return 回歸：XGBRegressor、64-unit LSTM、zero/train-mean/last-return baselines。
  - 完成一組 32-unit LSTM＋訓練 fold 1%／99% target clipping 控制實驗。
  - 新增 `docs/log_return_regression_comparison.md`，整理回歸模型正式結論。
- 實驗結果：
  - BTC 的 XGBoost 與 LSTM accuracy 幾乎相同，但 LSTM 的 F1、策略報酬、Sharpe 與最大回撤較佳。
  - ETH 的基準 LSTM 明顯優於 XGBoost，尤其策略報酬由負轉正。
  - SOL 的基準 LSTM 不如 XGBoost；延伸實驗顯示 32-unit LSTM 較適合 SOL。
  - XRP 的基準 LSTM 改善分類及交易指標，但策略報酬仍為負；48 小時序列延伸實驗表現最佳。
- 問題與阻塞：
  - 回歸流程已完成，但 XGBoost 與兩組 LSTM 都未穩定超越 train-mean／zero-return baseline，顯示目前特徵對下一小時報酬幅度的可泛化訊號不足。
  - Week 1 仍缺四幣共同時段對齊後的乾淨資料輸出與獨立品質摘要。
- 下一步：
  - Week 5 建立 Transformer Encoder，沿用相同 fold 與 leakage controls；回歸任務保留為研究結果，但不再追加 LSTM 調參。

### 2026-06-30

- 本週完成：
  - 新增 `src/train_week5_transformer.py`，建立包含 learned positional embedding、Multi-Head Attention、FFN 與 residual connection 的 Transformer Encoder。
  - 新增並完整執行 `notebooks/week5_transformer.ipynb`，依序保留資料稽核、模型摘要、dry run、分類／回歸訓練及結果表輸出。
  - 建立 `notebooks/調整後(預測4,12,24小時)/`，準備 XGBoost、LSTM、Transformer 三份多 horizon 實驗 Notebook；三種模型皆已通過 BTC 4 小時 1-fold smoke test。
  - 四幣分類與回歸皆完成 5-fold walk-forward；每 fold 使用 720 筆測試資料、24-row outer/inner gap 及 24 小時序列。
  - 特徵 scaler、回歸 target scaler 與 Early Stopping 均限制於 inner-training／validation，未接觸 outer test。
  - 輸出 14,400 筆分類 OOF predictions 與 14,400 筆回歸 OOF predictions，供 Week 6 Stacking 使用。
  - 新增 `docs/week5_transformer_comparison.md`，完成 XGBoost、LSTM、Transformer 三模型比較與可重現指令。
- 實驗結果：
  - 分類方面，BTC、ETH 整體仍以 LSTM 較佳；SOL Transformer 的 F1 為 0.5444，但 log loss 與交易指標未改善；XRP Transformer 未勝 LSTM。
  - 回歸方面，Transformer 在 BTC、ETH、SOL 的方向正確率高於 train-mean baseline，但四幣 MAE／RMSE 皆未勝 baseline，R² 皆為負。
  - Transformer 提供部分互補訊號，但沒有全面取代 XGBoost 或 LSTM，適合進入 OOF Stacking 驗證。
- 問題與阻塞：
  - Windows 原生 TensorFlow 2.21 僅使用 CPU；本次完整訓練約 7 分鐘，不影響 Week 5 交付。
  - 分類交易摘要尚未計入手續費，須等 Week 7 正式回測後才能解讀實際可交易性。
  - Week 1 仍缺四幣共同時段對齊後的乾淨資料輸出與獨立品質摘要。
- 下一步：
  - Week 6 依 `symbol + fold + open_time` 對齊 XGBoost、LSTM、Transformer 的 OOF predictions；分類使用 Logistic Regression、回歸使用 Ridge Regression，且 meta-model 仍需遵守時間序列切分。

### 2026-08-05

- 本次完成：
  - 整理 XGBoost、LSTM、Transformer 的 1／4／12／24 小時分類結果。
  - 建立逐幣種最佳模型表與三模型平均比較表，整理於 `docs/final_model_comparison_and_conclusion.md`。
  - 核對並修正依 F1 選模的結果：24h ETH 應選 XGBoost。
- 最終判斷：
  - 分類比回歸更適合作為專題主要成果。
  - 12h 分類值得作為下一階段主軸，但目前不應直接宣稱全面優於 1h。
  - 1h LSTM 保留為重要對照；4h 與 24h 作補充分析。
  - 4／12／24h 的 36 組回歸模型皆未勝過 train-mean baseline；1h 僅 SOL-XGBoost 有極小 MAE 改善，且 R² 仍接近或低於 0。
- 下一步：
  - 先檢查 12h 類別比例與 baseline。
  - 以 walk-forward validation 重跑 12h XGBoost、LSTM、Transformer，並保留 1h 對照。
  - 優先嘗試特徵改善、class weight、threshold tuning，再做超參數調整。
  - 用 F1、Balanced Accuracy、MCC、ROC-AUC、Log Loss 評估；確認穩定後再做校準、stacking、SHAP 與交易回測。

## 8. 規劃依據

- 熱門幣種沿用初審報告定義：BTC、ETH、SOL、XRP。
- 八週時程自 2026-06-02 起算。
- 初審報告附錄沒有可直接沿用的實際甘特排程，因此重新規劃。
- 本文件僅追蹤熱門幣預測工作，不追蹤穩定幣脫鉤分支。
