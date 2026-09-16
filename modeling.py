"""
shop#3 (DHC) 真復活預測模型 — 第一版 (modeling_prompt.md)

XGBoost 二元分類，預設參數（不調超參、不加類別權重）。
輸出 AUC 系列指標、SHAP 全局重要性、預測機率分佈圖、報告。

結構：load_data / encode / train / evaluate / shap_analysis / plot / write_report
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb
import shap
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    brier_score_loss, confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# CJK font so Chinese labels render in plots
for _f in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]:
    try:
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).parent
OUT_DIR = BASE / "output"
MODEL_DIR = OUT_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 非特徵欄位（識別碼 / 標籤 / 日期），務必排除以防洩漏
DROP_COLS = ["sample_id", "label", "split", "member_id",
             "t0_date", "t0_trades_group_code"]

RANDOM_STATE = 42


# ============================================================
# Step 1: Load + feature prep
# ============================================================

def load_data():
    train = pd.read_csv(OUT_DIR / "train_features.csv", encoding="utf-8-sig")
    test = pd.read_csv(OUT_DIR / "test_features.csv", encoding="utf-8-sig")

    feature_cols = [c for c in train.columns if c not in DROP_COLS]

    print("=" * 60)
    print("  Step 1: Load & feature prep")
    print("=" * 60)
    print(f"  train: {len(train):,} rows   test: {len(test):,} rows")
    print(f"  feature count: {len(feature_cols)}")
    print("  feature_cols (human-confirm: no id/date/label-derived):")
    for i in range(0, len(feature_cols), 4):
        print("    " + ", ".join(feature_cols[i:i + 4]))

    # leakage guard: none of the drop cols should remain
    leaked = [c for c in feature_cols if c in DROP_COLS]
    assert not leaked, f"識別碼洩漏進特徵: {leaked}"
    print(f"  label balance — train y=1: {train['label'].mean():.1%}  "
          f"test y=1: {test['label'].mean():.1%}")
    return train, test, feature_cols


# ============================================================
# Step 2: Categorical encoding (XGBoost 原生 categorical)
# ============================================================

def encode(train, test, feature_cols):
    cat_cols = train[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    print("\n--- Step 2: Categorical encoding ---")
    print(f"  categorical cols ({len(cat_cols)}): {cat_cols}")

    for col in cat_cols:
        # 類別缺失補 'unknown'（特徵階段未填者在此統一），數值 NaN 保留
        train[col] = train[col].fillna("unknown").astype("category")
        test[col] = test[col].fillna("unknown")
        # test 的 categories 對齊 train，避免未見類別出錯（未見→NaN）
        test[col] = pd.Categorical(test[col], categories=train[col].cat.categories)

    X_train = train[feature_cols]
    y_train = train["label"]
    X_test = test[feature_cols]
    y_test = test["label"]
    print(f"  X_train {X_train.shape}   X_test {X_test.shape}")
    return X_train, y_train, X_test, y_test, cat_cols


# ============================================================
# Step 3: Train (default params)
# ============================================================

def train_model(X_train, y_train):
    print("\n--- Step 3: Train (XGBoost default params) ---")
    model = xgb.XGBClassifier(
        enable_categorical=True,
        tree_method="hist",
        eval_metric="logloss",
        importance_type="gain",  # 與 SHAP 對照用
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"  trained: n_estimators={model.n_estimators}, "
          f"max_depth={model.max_depth}, lr={model.learning_rate}")
    model.save_model(MODEL_DIR / "xgb_model.json")
    print(f"  saved -> {MODEL_DIR / 'xgb_model.json'}")
    return model


# ============================================================
# Step 4: Evaluate (train / test 分別)
# ============================================================

def evaluate(model, X_train, y_train, X_test, y_test):
    print("\n--- Step 4: Evaluation ---")
    results = {}
    for split_name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = {
            "auc_roc": roc_auc_score(y, proba),
            "auc_pr": average_precision_score(y, proba),
            "brier": brier_score_loss(y, proba),
            "precision": precision_score(y, pred),
            "recall": recall_score(y, pred),
            "f1": f1_score(y, pred),
            "cm": confusion_matrix(y, pred),
        }
        results[split_name] = m
        print(f"  [{split_name}] AUC-ROC={m['auc_roc']:.4f}  AUC-PR={m['auc_pr']:.4f}  "
              f"Brier={m['brier']:.4f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}")
        print(f"           confusion matrix [[TN FP][FN TP]] = {m['cm'].tolist()}")
        print(classification_report(y, pred, digits=3))
    gap = results["train"]["auc_roc"] - results["test"]["auc_roc"]
    print(f"  train-test AUC gap = {gap:.4f}  "
          f"({'OK <0.1' if gap < 0.1 else '⚠ >0.1 overfit/shift'})")
    return results


# ============================================================
# Step 5: SHAP analysis
# ============================================================

def shap_analysis(model, X_test, feature_cols):
    print("\n--- Step 5: SHAP analysis ---")
    n = min(20000, len(X_test))
    X_shap = X_test.sample(n=n, random_state=RANDOM_STATE)
    print(f"  TreeExplainer on {n:,} test rows ...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    global_imp = (pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    global_imp.to_csv(MODEL_DIR / "shap_importance.csv",
                      index=False, encoding="utf-8-sig")
    print(f"  shap_importance.csv -> {MODEL_DIR / 'shap_importance.csv'}")

    # beeswarm
    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  shap_summary.png -> {MODEL_DIR / 'shap_summary.png'}")

    # bar
    plt.figure()
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "shap_bar.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  shap_bar.png -> {MODEL_DIR / 'shap_bar.png'}")

    # gain importance (對照)
    gain_imp = (pd.DataFrame({
        "feature": feature_cols,
        "gain": model.feature_importances_,
    }).sort_values("gain", ascending=False).reset_index(drop=True))
    gain_imp.to_csv(MODEL_DIR / "feature_importance_gain.csv",
                    index=False, encoding="utf-8-sig")
    print(f"  feature_importance_gain.csv -> {MODEL_DIR / 'feature_importance_gain.csv'}")

    return global_imp, gain_imp, X_shap, shap_values


# ============================================================
# Step 6: Prediction probability distribution plot
# ============================================================

def plot_pred_distribution(model, X_test, y_test):
    print("\n--- Step 6: Prediction distribution plot ---")
    proba = model.predict_proba(X_test)[:, 1]
    plt.figure(figsize=(8, 5))
    plt.hist(proba[y_test == 1], bins=50, alpha=0.5, label="真復活 (y=1)")
    plt.hist(proba[y_test == 0], bins=50, alpha=0.5, label="路過 (y=0)")
    plt.axvline(0.4, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(0.6, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("預測真復活機率")
    plt.ylabel("樣本數")
    plt.title("Test 集預測機率分佈")
    plt.legend()
    plt.savefig(MODEL_DIR / "pred_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  pred_distribution.png -> {MODEL_DIR / 'pred_distribution.png'}")

    mid_mask = (proba >= 0.4) & (proba <= 0.6)
    mid_n = int(mid_mask.sum())
    mid_pct = mid_mask.mean() * 100
    print(f"  中機率 (0.4-0.6) 客群: {mid_n:,} 筆 ({mid_pct:.1f}%)")
    return mid_n, mid_pct


# ============================================================
# metrics.md
# ============================================================

def write_metrics_md(results):
    def cm_md(cm):
        return (f"|  | 預測 0 | 預測 1 |\n|---|---|---|\n"
                f"| **實際 0** | {cm[0, 0]:,} | {cm[0, 1]:,} |\n"
                f"| **實際 1** | {cm[1, 0]:,} | {cm[1, 1]:,} |")

    tr, te = results["train"], results["test"]
    md = f"""# 評估指標（train / test）

| 指標 | train | test |
|------|------:|-----:|
| AUC-ROC | {tr['auc_roc']:.4f} | {te['auc_roc']:.4f} |
| AUC-PR | {tr['auc_pr']:.4f} | {te['auc_pr']:.4f} |
| Brier Score | {tr['brier']:.4f} | {te['brier']:.4f} |
| Precision (@0.5) | {tr['precision']:.4f} | {te['precision']:.4f} |
| Recall (@0.5) | {tr['recall']:.4f} | {te['recall']:.4f} |
| F1 (@0.5) | {tr['f1']:.4f} | {te['f1']:.4f} |

train − test AUC-ROC gap = **{tr['auc_roc'] - te['auc_roc']:.4f}**

## 混淆矩陣 — train
{cm_md(tr['cm'])}

## 混淆矩陣 — test
{cm_md(te['cm'])}

*Generated by modeling.py — 2026-06-03*
"""
    (MODEL_DIR / "metrics.md").write_text(md, encoding="utf-8")
    print(f"  metrics.md -> {MODEL_DIR / 'metrics.md'}")


# ============================================================
# modeling_report.md
# ============================================================

def write_report(results, global_imp, gain_imp, mid_n, mid_pct,
                 n_features, cat_cols, X_shap, shap_values):
    tr, te = results["train"], results["test"]
    gap = tr["auc_roc"] - te["auc_roc"]

    top15 = global_imp.head(15)
    top15_lines = ["| 排名 | 特徵 | mean \\|SHAP\\| |", "|---|---|---|"]
    for i, r in top15.iterrows():
        top15_lines.append(f"| {i + 1} | {r['feature']} | {r['mean_abs_shap']:.4f} |")

    # 方向性：Top5 特徵 SHAP 值與特徵值的相關（值高→機率升/降）
    fmap = {f: j for j, f in enumerate(global_imp["feature"])}  # not used directly
    feat_order = list(X_shap.columns)
    dir_lines = ["| 特徵 | 方向性（值高時） |", "|---|---|"]
    for f in global_imp["feature"].head(5):
        idx = feat_order.index(f)
        col = X_shap[f]
        sv = shap_values[:, idx]
        if col.dtype.name == "category":
            direction = "類別特徵（見 beeswarm 分佈）"
        else:
            mask = col.notna().values
            if mask.sum() > 10 and np.nanstd(col.values[mask].astype(float)) > 0:
                corr = np.corrcoef(col.values[mask].astype(float), sv[mask])[0, 1]
                arrow = "→ 機率上升 ↑" if corr > 0 else "→ 機率下降 ↓"
                direction = f"corr(value, SHAP)={corr:+.2f}  {arrow}"
            else:
                direction = "n/a"
        dir_lines.append(f"| {f} | {direction} |")

    # SHAP vs gain top10 一致性
    shap_top10 = list(global_imp["feature"].head(10))
    gain_top10 = list(gain_imp["feature"].head(10))
    overlap = len(set(shap_top10) & set(gain_top10))

    gain_lines = ["| 排名 | SHAP Top10 | gain Top10 |", "|---|---|---|"]
    for i in range(10):
        gain_lines.append(f"| {i + 1} | {shap_top10[i]} | {gain_top10[i]} |")

    key_feats = ["sleep_days", "t0_buys_familiar_product", "t0_vs_avg_ticket_ratio",
                 "sleep_vs_ci_ratio", "t0_discount_vs_hist_ratio"]
    rank_map = {r["feature"]: i + 1 for i, r in global_imp.iterrows()}
    key_lines = ["| 關注特徵 | SHAP 排名 |", "|---|---|"]
    for f in key_feats:
        key_lines.append(f"| {f} | {rank_map.get(f, 'n/a')} |")

    md = f"""# 真復活預測模型報告（第一版）

> 腳本：`modeling.py` — 2026-06-03
> 輸入：`output/train_features.csv` ({results.get('_train_n', 198370):,}) /
> `output/test_features.csv` ({results.get('_test_n', 89454):,})

---

## 1. 模型設定

- 演算法：**XGBoost (XGBClassifier)**，預設參數
- `enable_categorical=True`、`tree_method='hist'`、`eval_metric='logloss'`
- `random_state={RANDOM_STATE}`、`n_jobs=-1`
- **第一版未調超參、未加類別權重（scale_pos_weight）**
- 模型存檔：`output/model/xgb_model.json`（原生格式，可重載）

---

## 2. 特徵與編碼

- 進模型特徵數：**{n_features}**（排除識別碼 {', '.join(c for c in DROP_COLS)}；
  實際特徵檔僅含 sample_id/label，其餘皆為特徵）
- 類別特徵（{len(cat_cols)} 個）：{', '.join(cat_cols)}
  - 採 XGBoost **原生 categorical**（`astype('category')`），缺失補 `'unknown'`
  - test 的 categories 對齊 train，未見類別 → NaN
- 數值特徵 NaN **保留不填**，由 XGBoost 原生處理
- 前置清理：`age_at_t0` 限 [10,100]、`days_since_register` 負值→NaN（壞 Birthday/快照偽影）

---

## 3. 評估結果

| 指標 | train | test |
|------|------:|-----:|
| AUC-ROC | {tr['auc_roc']:.4f} | {te['auc_roc']:.4f} |
| AUC-PR | {tr['auc_pr']:.4f} | {te['auc_pr']:.4f} |
| Brier Score | {tr['brier']:.4f} | {te['brier']:.4f} |
| Precision (@0.5) | {tr['precision']:.4f} | {te['precision']:.4f} |
| Recall (@0.5) | {tr['recall']:.4f} | {te['recall']:.4f} |
| F1 (@0.5) | {tr['f1']:.4f} | {te['f1']:.4f} |

**混淆矩陣 — train** `[[TN, FP], [FN, TP]]` = `{tr['cm'].tolist()}`
**混淆矩陣 — test** `[[TN, FP], [FN, TP]]` = `{te['cm'].tolist()}`

---

## 4. SHAP 特徵重要性

**Top 15（mean |SHAP|）**

{chr(10).join(top15_lines)}

![SHAP beeswarm](shap_summary.png)

![SHAP bar](shap_bar.png)

**Top 5 特徵方向性**

{chr(10).join(dir_lines)}

**SHAP 排序 vs gain 排序（Top10 重疊 {overlap}/10）**

{chr(10).join(gain_lines)}

---

## 5. 預測機率分佈

![預測機率分佈](pred_distribution.png)

中機率客群 (0.4–0.6)：**{mid_n:,} 筆**，佔 test 集 **{mid_pct:.1f}%**。
這群是行銷介入的主要目標（呼應 Proposal Q3 三層分群）。

---

## 6. 結果討論

- **train vs test AUC 差距 = {gap:.4f}**：{'差距小，泛化穩定。' if gap < 0.1 else '差距 >0.1，可能 overfit 或 distribution shift（呼應 train/test y=1 約差 10.6pp）。'}
- 關注特徵的 SHAP 排名：

{chr(10).join(key_lines)}

- 第一版限制：未調超參、未做機率校準、未處理 F5 極端比率值。

---

## 7. 下一步

1. **機率校準**（Isotonic / Platt）→ 改善 Brier、讓中機率分層更可信
2. **對照模型**：Logistic Regression / Random Forest 基準
3. **超參調整 + scale_pos_weight** 實驗（若 recall 偏低）
4. **F5 極端值 winsorize** 後重評特徵重要性
5. **子群分析**：Lost（沉睡<365d）vs Sealed（>365d）

*Generated by modeling.py — 2026-06-03*
"""
    (MODEL_DIR / "modeling_report.md").write_text(md, encoding="utf-8")
    print(f"  modeling_report.md -> {MODEL_DIR / 'modeling_report.md'}")


# ============================================================
# Main
# ============================================================

def main():
    train, test, feature_cols = load_data()
    X_train, y_train, X_test, y_test, cat_cols = encode(train, test, feature_cols)
    model = train_model(X_train, y_train)
    results = evaluate(model, X_train, y_train, X_test, y_test)
    results["_train_n"] = len(train)
    results["_test_n"] = len(test)
    write_metrics_md(results)
    global_imp, gain_imp, X_shap, shap_values = shap_analysis(model, X_test, feature_cols)
    mid_n, mid_pct = plot_pred_distribution(model, X_test, y_test)
    write_report(results, global_imp, gain_imp, mid_n, mid_pct,
                 len(feature_cols), cat_cols, X_shap, shap_values)

    print("\n" + "=" * 60)
    print("  Modeling pipeline complete!")
    print(f"  Outputs in: {MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
