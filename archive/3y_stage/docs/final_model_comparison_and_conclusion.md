# 加密貨幣價格預測：最終模型比較與主結論

更新日期：2026-08-05  
分析範圍：BTC、ETH、SOL、XRP；預測 horizon 為 1、4、12、24 小時；模型包含 XGBoost、LSTM、Transformer。

## 一、最終主結論

本研究的主要成果應定位為「加密貨幣未來漲跌方向分類」，而不是精確的報酬率回歸預測。

分類結果顯示，不同模型在部分幣種與預測時間尺度上能取得略高於隨機猜測的方向辨識能力；其中表現較穩定的組合主要出現在 1 小時與 12 小時 horizon，代表目前特徵對「上漲或下跌方向」的資訊較容易被模型捕捉。整體 Accuracy 與 F1 多數仍落在接近 0.50 的區間，因此研究結論應採取「具有有限但不穩定的方向預測訊號」，不宜宣稱已建立穩定獲利的交易策略。

相較之下，回歸任務對未來報酬率大小的預測效果大多不佳。4、12、24 小時的 36 組學習模型組合均未在 MAE 上勝過 train-mean baseline；1 小時也只有 SOL 的 XGBoost 出現非常小的 MAE 改善，且 R² 仍接近或低於 0，無法視為穩定的預測能力。因此，回歸結果應作為本研究的重要負結果與限制，而不是主要成果。

## 二、分類最終比較表

下表每一列代表一個「幣種 × 預測時間」案例；模型欄位採用該案例中 F1 最高的學習模型。括號內依序為 `Accuracy / F1 / Log Loss`。數值來自各模型的 `metrics_summary.csv`，不包含 naive baseline。

| Horizon | 幣種 | 最佳模型（依 F1） | Accuracy | F1 | Log Loss | 判讀 |
|---|---|---|---:|---:|---:|---|
| 1h | BTC | LSTM | 0.5281 | 0.5543 | 0.6921 | 目前 1h 分類中較完整的結果 |
| 1h | ETH | LSTM | 0.5186 | 0.5433 | 0.6931 | 略高於隨機，但優勢有限 |
| 1h | SOL | Transformer | 0.5097 | 0.5444 | 0.6970 | F1 較高，但 Accuracy 未明顯提升 |
| 1h | XRP | LSTM | 0.5186 | 0.5089 | 0.6944 | 方向訊號較弱 |
| 4h | BTC | XGBoost | 0.5128 | 0.5243 | 0.7021 | 有輕微訊號，但信心校準不佳 |
| 4h | ETH | XGBoost | 0.5103 | 0.5412 | 0.7016 | F1 略高，整體仍偏弱 |
| 4h | SOL | Transformer | 0.5061 | 0.5090 | 0.7031 | 接近隨機 |
| 4h | XRP | LSTM | 0.5206 | 0.4823 | 0.6988 | Accuracy 與 F1 不一致，需保守解讀 |
| 12h | BTC | Transformer | 0.5197 | 0.5672 | 0.7043 | F1 是所有案例中較突出者 |
| 12h | ETH | XGBoost | 0.5303 | 0.5610 | 0.7093 | Accuracy 與 F1 同時較高，但 Log Loss 偏高 |
| 12h | SOL | Transformer | 0.5167 | 0.4958 | 0.7055 | 分類能力有限 |
| 12h | XRP | Transformer | 0.5325 | 0.4471 | 0.6915 | Accuracy 不能單獨代表有效方向辨識 |
| 24h | BTC | Transformer | 0.5350 | 0.5486 | 0.7022 | 24h 中相對較好的案例 |
| 24h | ETH | XGBoost | 0.4756 | 0.4730 | 0.7359 | 低於隨機，表現不佳 |
| 24h | SOL | Transformer | 0.5242 | 0.4411 | 0.7179 | Accuracy 尚可，但 F1 偏低 |
| 24h | XRP | Transformer | 0.5272 | 0.4102 | 0.7196 | F1 偏低，不能視為穩定成果 |

## 三、三種模型的整體平均比較

下表為四個幣種的平均值，用來觀察模型家族在各 horizon 的整體趨勢。平均值適合做總體比較，但不能取代前一節的逐幣種結果，因為不同幣種的類別比例與市場特性並不相同。

| Horizon | 模型 | 平均 Accuracy | 平均 F1 | 平均 Log Loss | 整體觀察 |
|---|---|---:|---:|---:|---|
| 1h | XGBoost | 0.5151 | 0.5208 | 0.6936 | Accuracy 略高，整體接近隨機 |
| 1h | LSTM | 0.5165 | 0.5286 | 0.6945 | 平均 F1 最高 |
| 1h | Transformer | 0.5065 | 0.5254 | 0.6955 | F1 尚可，但 Accuracy 較低 |
| 4h | XGBoost | 0.5053 | 0.4997 | 0.7042 | 整體偏弱 |
| 4h | LSTM | 0.5134 | 0.4875 | 0.7009 | Accuracy 較高，但 F1 較低 |
| 4h | Transformer | 0.4983 | 0.4669 | 0.7054 | 接近或低於隨機 |
| 12h | XGBoost | 0.5178 | 0.5017 | 0.7129 | 平均 Accuracy 較高 |
| 12h | LSTM | 0.5166 | 0.4719 | 0.7070 | F1 偏低 |
| 12h | Transformer | 0.5057 | 0.4826 | 0.7112 | 個別案例突出，平均值普通 |
| 24h | XGBoost | 0.4879 | 0.4449 | 0.7331 | 整體表現不佳 |
| 24h | LSTM | 0.4924 | 0.4289 | 0.7119 | 整體表現不佳 |
| 24h | Transformer | 0.5167 | 0.4463 | 0.7310 | Accuracy 較高，但 F1 仍弱 |

### 表格使用方式

- 若報告需要選出代表模型：可使用 **1h BTC-LSTM、12h BTC-Transformer、12h ETH-XGBoost、24h BTC-Transformer** 作為代表案例。
- 若以 Accuracy 選最佳模型，部分案例會改變；因此正文應先聲明本研究以 F1 作為主要選模指標，Accuracy 作為輔助指標。
- Log Loss 多數接近或高於 0.693，表示模型機率預測的信心與校準仍有限；不要只用 Accuracy 或 F1 宣稱模型具有強預測能力。

## 四、分類結果的模型與 horizon 比較

### 1 小時

1h 是目前最適合呈現的分類時間尺度之一。BTC 與 ETH 由 LSTM 取得較佳 F1，SOL 由 Transformer 取得較佳 F1；不過各模型的 Accuracy 仍只比 0.50 高出有限幅度，結果應描述為「捕捉到弱方向訊號」。

### 4 小時

4h 的結果整體偏弱，XGBoost 在 BTC 與 ETH 的 F1 表現相對較好，LSTM 在 XRP 的 Accuracy 較高。此 horizon 適合作為模型比較與限制討論，不適合當作最主要的亮點。

### 12 小時

12h 是分類結果中較值得深入討論的 horizon。BTC 的 Transformer 與 ETH 的 XGBoost 同時呈現相對較高的 F1 與 Accuracy；但 Log Loss 偏高，代表預測機率仍不夠可靠。這些結果支持「分類比回歸更適合目前資料與特徵」，但仍不足以支持穩定交易獲利的結論。

### 24 小時

24h 的模型差異較大。BTC 的 Transformer 表現相對最好，但 ETH 的 XGBoost 結果低於隨機，SOL 與 XRP 則出現 Accuracy 尚可而 F1 偏低的情況。這顯示長 horizon 的方向預測不穩定，可能受到類別比例、錯誤類型與市場 regime 變化影響。

## 五、回歸結果：負結果與限制

| 預測時間 | 檢查範圍 | 與 train-mean baseline 的結果 | 最終判定 |
|---|---|---|---|
| 1h | XGBoost、LSTM、LSTM control、Transformer；4 幣種 | 僅 SOL-XGBoost 出現極小 MAE 改善；R² 仍接近或低於 0 | 沒有穩定回歸能力 |
| 4h | XGBoost、LSTM、Transformer；4 幣種 | 12 組模型全部未勝過 baseline | 負結果 |
| 12h | XGBoost、LSTM、Transformer；4 幣種 | 12 組模型全部未勝過 baseline | 負結果 |
| 24h | XGBoost、LSTM、Transformer；4 幣種 | 12 組模型全部未勝過 baseline | 負結果 |

回歸結果的研究意義不是「模型完全沒有價值」，而是說在目前資料、特徵、標籤定義與驗證設定下，模型較難準確預測報酬率的連續數值。這個負結果可以用來支持研究問題的回答：對本專題而言，方向分類比報酬率回歸更適合作為主要任務。

## 六、可直接放入報告的結論段落

> 綜合 XGBoost、LSTM 與 Transformer 在不同預測時間尺度的實驗結果，本研究選擇以漲跌方向分類作為主要成果。分類模型在部分幣種與 horizon 上的 Accuracy 與 F1 略高於隨機基準，其中 1 小時與 12 小時的結果較具討論價值；然而整體提升幅度有限，且 Log Loss 多數接近隨機基準，顯示模型只捕捉到弱而不穩定的市場方向訊號。因此，本研究不將分類模型解讀為已具備穩定獲利能力的交易系統。
>
> 另一方面，報酬率回歸結果大多未能勝過 train-mean baseline。4、12、24 小時的實驗中，所有模型組合皆未在 MAE 上取得改善；1 小時僅有單一 SOL-XGBoost 案例出現極小改善，但 R² 仍接近或低於 0。這表示在目前的特徵與建模設定下，模型較適合判斷未來報酬率的方向，而不適合精確估計報酬率大小。故本研究將回歸結果列為負結果與研究限制，並以分類模型的跨模型、跨幣種與跨 horizon 比較作為主要研究貢獻。

## 七、資料來源檔案

- 1h 分類：`models/week3_naive_xgboost/metrics_summary.csv`、`models/week4_lstm/metrics_summary.csv`、`models/week5_transformer/classification/metrics_summary.csv`
- 4/12/24h 分類：`models/調整後(預測4,12,24小時)/xgboost/classification/metrics_summary.csv`、`models/調整後(預測4,12,24小時)/lstm/classification/metrics_summary.csv`、`models/調整後(預測4,12,24小時)/transformer/classification/metrics_summary.csv`
- 回歸比較：`models/week3_naive_xgboost/log_return_regression/metrics_summary.csv`、`models/week4_lstm/log_return_regression/metrics_summary.csv`、`models/week5_transformer/regression/metrics_summary.csv`，以及 4/12/24h 對應的 regression 資料夾。
