# 任務：shop#3 (DHC) 真復活預測模型（第一版）

用前一階段產出的特徵，訓練 XGBoost 二元分類模型，輸出 AUC 系列指標與 SHAP 特徵重要性。
這是第一版，**用 XGBoost 預設參數，不調超參**，目標是先拿到可用結果。

---

## 輸入

- `./output/train_features.csv`（198,370 列）
- `./output/test_features.csv`（89,454 列）
- 兩檔欄位：`sample_id, label, split, [F1~F6 特徵欄位...]`

---

## 輸出（存到 ./output/model/）

1. `xgb_model.json`（XGBoost 原生格式，可重載）
2. `metrics.md`（評估指標：train/test 分別）
3. `shap_importance.csv`（全局特徵重要性，按 mean |SHAP| 排序）
4. `shap_summary.png`（beeswarm 圖）
5. `shap_bar.png`（特徵重要性長條圖）
6. `feature_importance_gain.csv`（XGBoost 內建 gain importance，與 SHAP 對照）
7. `pred_distribution.png`（test 集預測機率直方圖，依 label 分色）
8. `modeling_report.md`（整體流程與結果說明）

工具：xgboost + scikit-learn + shap + matplotlib（**不要用 plotly**）

---

## Step 1：載入與特徵準備

```python
import pandas as pd
import numpy as np

train = pd.read_csv('./output/train_features.csv', encoding='utf-8-sig')
test  = pd.read_csv('./output/test_features.csv',  encoding='utf-8-sig')

# 非特徵欄位（不進模型）
DROP_COLS = ['sample_id', 'label', 'split', 'member_id',
             't0_date', 't0_trades_group_code']
# 注意：member_id / t0_date / t0_trades_group_code 若在特徵檔裡，務必排除
#       這些是識別碼，進模型會造成嚴重洩漏

feature_cols = [c for c in train.columns if c not in DROP_COLS]
```

**重要**：print 出最終 feature_cols，人工確認沒有任何識別碼、日期、label 衍生欄位混進去。

---

## Step 2：類別特徵編碼

類別特徵（hist_main_channel, hist_main_payment, t0_channel, t0_payment,
t0_shipping, register_source, gender, country_code 等）：

採用 **XGBoost 原生 categorical 支援**（最簡單，避免 one-hot 維度爆炸）：

```python
# 找出類別欄位
cat_cols = train[feature_cols].select_dtypes(include=['object']).columns.tolist()

# 缺失值已在特徵階段填 'unknown'，這裡轉成 category dtype
for col in cat_cols:
    train[col] = train[col].astype('category')
    # test 的 category 必須對齊 train 的 categories（避免未見類別出錯）
    test[col] = pd.Categorical(test[col], categories=train[col].cat.categories)

X_train = train[feature_cols]
y_train = train['label']
X_test  = test[feature_cols]
y_test  = test['label']
```

數值特徵的 NaN **不要填補**，XGBoost 原生支援缺失值。

---

## Step 3：訓練（預設參數）

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    enable_categorical=True,   # 啟用原生類別支援
    tree_method='hist',        # 配合 categorical
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    # 其餘全部用預設，不調超參
)

model.fit(X_train, y_train)
```

> 第一版**不設 scale_pos_weight**（y=1 約 62%，非極端不平衡，先看預設表現）。
> 若後續發現 test recall 太低，再考慮加權，但那是第二輪的事。

---

## Step 4：評估（train / test 分別算）

```python
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    brier_score_loss, confusion_matrix, classification_report
)

for split_name, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    auc_roc = roc_auc_score(y, proba)
    auc_pr  = average_precision_score(y, proba)
    brier   = brier_score_loss(y, proba)
    prec    = precision_score(y, pred)
    rec     = recall_score(y, pred)
    f1      = f1_score(y, pred)
    cm      = confusion_matrix(y, pred)
    # print 全部
```

**必出指標**：

| 指標 | 為什麼重要 |
|------|-----------|
| AUC-ROC | 整體區分能力 |
| AUC-PR | 不平衡下更可靠 |
| Brier Score | 機率校準品質（你要做中機率分層，這個關鍵） |
| Precision / Recall / F1 | 在 0.5 門檻下的實際表現 |
| Confusion Matrix | 看 FP / FN 結構 |

**特別注意**：因為前一階段已知 train y=1 (67%) 與 test y=1 (56%) 差 10.6pp，
要**同時看 train 和 test 指標的差距**。若 train AUC 遠高於 test AUC，
代表 distribution shift 或 overfitting，要在 report 中討論。

---

## Step 5：SHAP 特徵重要性

```python
import shap

# TreeExplainer 對樹模型是精確解
explainer = shap.TreeExplainer(model)

# 用 test 集計算（若太慢可抽樣 1-2 萬筆）
X_shap = X_test.sample(n=min(20000, len(X_test)), random_state=42)
shap_values = explainer.shap_values(X_shap)

# 全局重要性：mean |SHAP|
import numpy as np
global_imp = pd.DataFrame({
    'feature': feature_cols,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)
global_imp.to_csv('./output/model/shap_importance.csv',
                  index=False, encoding='utf-8-sig')

# 圖
shap.summary_plot(shap_values, X_shap, show=False)  # beeswarm
# 存成 shap_summary.png

shap.summary_plot(shap_values, X_shap, plot_type='bar', show=False)
# 存成 shap_bar.png
```

> 若 SHAP 計算太慢（>5 分鐘），把 X_shap 抽樣降到 5000 筆。
> 對「全局重要性排序」來說，5000 筆通常已足夠穩定。

同時輸出 XGBoost 內建的 gain importance 作對照：

```python
gain_imp = pd.DataFrame({
    'feature': feature_cols,
    'gain': model.feature_importances_
}).sort_values('gain', ascending=False)
gain_imp.to_csv('./output/model/feature_importance_gain.csv',
                index=False, encoding='utf-8-sig')
```

---

## Step 6：預測機率分佈圖

```python
import matplotlib.pyplot as plt

proba_test = model.predict_proba(X_test)[:, 1]
plt.figure(figsize=(8, 5))
plt.hist(proba_test[y_test == 1], bins=50, alpha=0.5, label='真復活 (y=1)')
plt.hist(proba_test[y_test == 0], bins=50, alpha=0.5, label='路過 (y=0)')
plt.axvline(0.4, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0.6, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('預測真復活機率')
plt.ylabel('樣本數')
plt.title('Test 集預測機率分佈')
plt.legend()
# 存 pred_distribution.png
```

> 0.4 / 0.6 兩條虛線標出中機率客群區間（你做行銷介入的目標群）。
> 這張圖直接呼應 Proposal Q3 的三層分群。

---

## modeling_report.md 內容要求

1. `# 真復活預測模型報告（第一版）`
2. `## 1. 模型設定`
   - XGBoost 預設參數、enable_categorical、random_state
   - 明確寫「第一版未調超參、未加類別權重」
3. `## 2. 特徵與編碼`
   - 進模型的特徵數、排除了哪些識別碼欄位
   - 類別特徵用原生 categorical、數值 NaN 保留
4. `## 3. 評估結果`（表格，train/test 並列）
   | 指標 | train | test |
   - AUC-ROC, AUC-PR, Brier, Precision, Recall, F1
   - train/test 兩個混淆矩陣
5. `## 4. SHAP 特徵重要性`
   - Top 15 特徵表（mean |SHAP|）
   - 嵌入 shap_summary.png / shap_bar.png
   - 文字解讀 Top 5 特徵的方向性（值高 → 機率上升或下降）
   - 對照 SHAP 排序 vs gain 排序是否一致
6. `## 5. 預測機率分佈`
   - 嵌入 pred_distribution.png
   - 中機率 (0.4-0.6) 客群的樣本數與佔比
7. `## 6. 結果討論`
   - train vs test 指標差距（呼應 10.6pp distribution shift）
   - sleep_days / t0_buys_familiar_product / F5 交互特徵是否如預期重要
   - 第一版的限制
8. `## 7. 下一步`
   - 機率校準（Isotonic / Platt）→ 讓中機率分層更可信
   - 對照模型（Logistic Regression / Random Forest）
   - 超參調整、scale_pos_weight 實驗
   - 子群分析（Lost vs Sealed）

寫作風格：markdown 表格 + 程式碼區塊；指標數字直接填入；圖用相對路徑嵌入。

---

## 程式碼結構

- 模組化：load_data / encode / train / evaluate / shap_analysis / plot / write_report
- 每步 print 一行 summary
- 圖檔存檔時用 `plt.savefig(path, dpi=120, bbox_inches='tight')` 後 `plt.close()`
- 相對路徑、可重跑
- 模型存成 `xgb_model.json`（`model.save_model()`）

---

## 合理性檢查（跑完自我驗證）

- [ ] feature_cols 不含任何識別碼 / 日期 / label 衍生欄
- [ ] test AUC-ROC 落在 0.65~0.85 之間（太高要懷疑洩漏，太低要檢查特徵）
- [ ] train AUC 與 test AUC 差距 < 0.1（差太多 = overfit 或 shift）
- [ ] SHAP Top 特徵符合商業直覺（sleep_days、F5 交互特徵預期在前段）
- [ ] 若 test AUC > 0.9，**高度懷疑特徵洩漏**，回頭檢查特徵工程的因果截止點
