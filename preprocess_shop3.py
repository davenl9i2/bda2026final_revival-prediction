"""
preprocess_shop3.py
91APP shop#3 (DHC) — dormancy revival sample generation pipeline
Rules R1–R10 as specified.
Outputs: output/samples_shop3.parquet, output/data_preprocessing.md, output/pipeline_report.md
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
DATA_DIR = BASE / "91APP_Dataset(會員&主單&子單&商品頁&標籤)"
ORDER_TG = DATA_DIR / "Order_TG.csv"
OUT_DIR  = BASE / "output"
OUT_DIR.mkdir(exist_ok=True)

SHOP3_ID           = "hFwniXiB/Ev2ZPXeO630Sw=="  # confirmed: 4,938,006 rows = EDA raw count
FALLBACK_CI        = 67                            # EDA Step 3: shop#3 median gap
OBSERVATION_WINDOW = 120                           # 觀察窗天數
DORMANCY_K         = 3                             # threshold multiplier
CLIP_LO, CLIP_HI   = 30, 365
CENSOR_DATE        = pd.Timestamp("2024-02-29")
CENSOR_CUTOFF      = pd.Timestamp("2023-11-01")    # t₀ 上限（不變）
TRAIN_END          = pd.Timestamp("2023-03-31")    # 訓練集 t₀ 上限（calendar 切分）
EMBARGO_DAYS       = 120
TEST_START         = TRAIN_END + pd.Timedelta(days=EMBARGO_DAYS)  # 2023-07-29，測試集 t₀ 下限

# internal aliases
OBS_WINDOW    = OBSERVATION_WINDOW
CENSOR_T0_MAX = CENSOR_CUTOFF

# ── stats collector ────────────────────────────────────────────────────────
stats = {}


# ==========================================================================
# R1: Load & filter shop#3
# ==========================================================================
def r1_load(path: Path) -> pd.DataFrame:
    """R1: Load Order_TG.csv, print all unique ShopIds, filter to shop#3."""
    print("=" * 60)
    print("R1 — Load & brand filter")

    dtype_map = {
        "ShopId":           "category",
        "ShopMemberId":     "category",
        "TradesGroupCode":  "category",
        "StatusDef":        "category",
        "TotalSalesAmount": "float32",
    }
    usecols = list(dtype_map.keys()) + ["OrderDateTime"]

    print("  Reading CSV …")
    raw = pd.read_csv(path, usecols=usecols, dtype=dtype_map, low_memory=True)
    raw["OrderDateTime"] = pd.to_datetime(raw["OrderDateTime"], format="mixed")

    print("\n  All unique ShopId values:")
    vc = raw["ShopId"].value_counts()
    for i, (sid, cnt) in enumerate(vc.items(), 1):
        print(f"    rank {i}: {sid}  ({cnt:,} rows, {cnt/len(raw)*100:.2f}%)")

    shop3 = raw[raw["ShopId"] == SHOP3_ID].copy()

    stats["r1_raw_rows"]   = len(raw)
    stats["r1_shop3_rows"] = len(shop3)
    stats["r1_members"]    = shop3["ShopMemberId"].nunique()
    stats["r1_dt_min"]     = shop3["OrderDateTime"].min().date().isoformat()
    stats["r1_dt_max"]     = shop3["OrderDateTime"].max().date().isoformat()

    print(f"\n  Raw total rows      : {stats['r1_raw_rows']:>10,}")
    print(f"  shop#3 rows         : {stats['r1_shop3_rows']:>10,}")
    print(f"  shop#3 members      : {stats['r1_members']:>10,}")
    print(f"  OrderDateTime range : {stats['r1_dt_min']} ~ {stats['r1_dt_max']}")
    print("R1 done ✓\n")
    return shop3


# ==========================================================================
# R2: Valid purchase filter
# ==========================================================================
def r2_filter(df: pd.DataFrame) -> pd.DataFrame:
    """R2: Keep StatusDef != 'Fail' AND TotalSalesAmount > 0; add date column."""
    print("R2 — Valid purchase filter")
    before = len(df)

    valid = df[(df["StatusDef"] != "Fail") & (df["TotalSalesAmount"] > 0)].copy()
    valid["date"] = valid["OrderDateTime"].dt.normalize()   # midnight Timestamp, consistent with R6 merge

    after = len(valid)
    stats["r2_before"]    = before
    stats["r2_after"]     = after
    stats["r2_keep_rate"] = round(after / before * 100, 2)

    print(f"  Before filter : {before:>10,}")
    print(f"  After filter  : {after:>10,}  (keep {stats['r2_keep_rate']}%)")
    print("R2 done ✓\n")
    return valid


# ==========================================================================
# R3: Dual deduplication
# ==========================================================================
def r3_dedup(valid: pd.DataFrame):
    """R3: Two separate dedups for different purposes — must not be mixed.

    R3-1 (orders): one row per TradesGroupCode → used for label window purchase count
    R3-2 (daily):  one row per (ShopMemberId, date) → used for Ci gap calculation
    """
    print("R3 — Dual deduplication")

    # R3-1: unique orders (for label window purchase count)
    orders = valid.drop_duplicates(subset="TradesGroupCode").copy()

    # R3-2: unique (member, date) pairs (for Ci gap calculation)
    daily = (
        valid[["ShopMemberId", "date"]]
        .drop_duplicates()
        .sort_values(["ShopMemberId", "date"])
        .reset_index(drop=True)
    )

    stats["r3_orders"] = len(orders)
    stats["r3_daily"]  = len(daily)

    print(f"  R3-1 unique TradesGroupCode : {stats['r3_orders']:>10,}")
    print(f"  R3-2 unique (member, date)  : {stats['r3_daily']:>10,}")
    print("R3 done ✓\n")
    return orders, daily


# ==========================================================================
# R4: Dynamic personal cycle Ci (gap-level, leak-free)
# ==========================================================================
def r4_compute_ci(daily: pd.DataFrame) -> pd.DataFrame:
    """R4: For each gap between consecutive purchase dates per member,
    compute Ci from *prior* gaps only (no future leakage).

    Returns DataFrame with columns:
        ShopMemberId, date_from, date_to, gap_days, ci_days
    """
    print("R4 — Dynamic Ci computation")

    records = []
    use_median   = 0
    use_single   = 0
    use_fallback = 0

    for member_id, grp in daily.groupby("ShopMemberId", sort=False):
        dates = grp["date"].sort_values().tolist()
        if len(dates) < 2:
            continue  # single-purchase members produce no gaps

        prior_gaps_days = []
        for i in range(len(dates) - 1):
            gap_days = (dates[i + 1] - dates[i]).days

            n_prior = len(prior_gaps_days)
            if n_prior >= 2:
                ci = int(np.median(prior_gaps_days))
                use_median += 1
            elif n_prior == 1:
                ci = prior_gaps_days[0]
                use_single += 1
            else:
                ci = FALLBACK_CI   # first gap — no history; use shop#3 global median
                use_fallback += 1

            records.append({
                "ShopMemberId": member_id,
                "date_from":    dates[i],
                "date_to":      dates[i + 1],
                "gap_days":     gap_days,
                "ci_days":      ci,
            })
            prior_gaps_days.append(gap_days)

    gaps = pd.DataFrame(records)
    total_gaps = len(gaps)

    stats["r4_total_gaps"]   = total_gaps
    stats["r4_use_median"]   = use_median
    stats["r4_use_single"]   = use_single
    stats["r4_use_fallback"] = use_fallback
    stats["r4_ci_p25"] = int(np.percentile(gaps["ci_days"], 25))
    stats["r4_ci_p50"] = int(np.percentile(gaps["ci_days"], 50))
    stats["r4_ci_p75"] = int(np.percentile(gaps["ci_days"], 75))
    stats["r4_ci_p90"] = int(np.percentile(gaps["ci_days"], 90))

    print(f"  Total gaps computed : {total_gaps:>10,}")
    print(f"  Used median prior   : {use_median:>10,}  ({use_median/total_gaps*100:.1f}%)")
    print(f"  Used single prior   : {use_single:>10,}  ({use_single/total_gaps*100:.1f}%)")
    print(f"  Used fallback ({FALLBACK_CI}d) : {use_fallback:>10,}  ({use_fallback/total_gaps*100:.1f}%)")
    print(f"  Ci distribution     : P25={stats['r4_ci_p25']}d  P50={stats['r4_ci_p50']}d  "
          f"P75={stats['r4_ci_p75']}d  P90={stats['r4_ci_p90']}d")
    print("R4 done ✓\n")
    return gaps


# ==========================================================================
# R5: Dormancy threshold
# ==========================================================================
def r5_threshold(gaps: pd.DataFrame) -> pd.DataFrame:
    """R5: threshold_i = clip(3 * Ci, 30, 365)."""
    print("R5 — Dormancy threshold")

    raw_thr = DORMANCY_K * gaps["ci_days"]
    gaps["threshold_days"] = raw_thr.clip(CLIP_LO, CLIP_HI).astype("int32")

    clipped_lo = (raw_thr < CLIP_LO).sum()
    clipped_hi = (raw_thr > CLIP_HI).sum()
    stats["r5_thr_median_raw"]     = int(raw_thr.median())
    stats["r5_thr_median_clipped"] = int(gaps["threshold_days"].median())
    stats["r5_clipped_lo_pct"]     = round(clipped_lo / len(gaps) * 100, 2)
    stats["r5_clipped_hi_pct"]     = round(clipped_hi / len(gaps) * 100, 2)

    print(f"  Threshold median (raw)    : {stats['r5_thr_median_raw']}d")
    print(f"  Threshold median (clipped): {stats['r5_thr_median_clipped']}d")
    print(f"  Clipped to lower ({CLIP_LO}d)    : {clipped_lo:,}  ({stats['r5_clipped_lo_pct']}%)")
    print(f"  Clipped to upper ({CLIP_HI}d)   : {clipped_hi:,}  ({stats['r5_clipped_hi_pct']}%)")
    print("R5 done ✓\n")
    return gaps


# ==========================================================================
# R6: Revival event t0 detection
# ==========================================================================
def r6_revival_events(gaps: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """R6: gap > threshold → revival event.
    t0 = date_to; TradesGroupCode = largest-TotalSalesAmount Finish order on that day.
    Also captures t0_order_datetime (exact timestamp) needed by R7 label window.
    One member can contribute multiple t0 events.
    """
    print("R6 — Revival event detection")

    dormant = gaps[gaps["gap_days"] > gaps["threshold_days"]].copy()
    dormant = dormant.rename(columns={"date_to": "t0_date"})

    # Build lookup: (member, date) → best TradesGroupCode + exact OrderDateTime
    # Priority: Finish status, then max TotalSalesAmount
    finish_orders = valid[valid["StatusDef"] == "Finish"].copy()
    finish_orders["date"] = finish_orders["OrderDateTime"].dt.normalize()

    best_finish = (
        finish_orders
        .sort_values("TotalSalesAmount", ascending=False)
        .drop_duplicates(subset=["ShopMemberId", "date"])
        [["ShopMemberId", "date", "TradesGroupCode", "OrderDateTime"]]
        .rename(columns={
            "date":            "t0_date",
            "TradesGroupCode": "t0_trades_group_code",
            "OrderDateTime":   "t0_order_datetime",
        })
    )

    # Fallback: any valid order (non-Fail, amount>0)
    any_order = (
        valid
        .sort_values("TotalSalesAmount", ascending=False)
        .drop_duplicates(subset=["ShopMemberId", "date"])
        [["ShopMemberId", "date", "TradesGroupCode", "OrderDateTime"]]
        .rename(columns={
            "date":            "t0_date",
            "TradesGroupCode": "t0_trades_group_code_fb",
            "OrderDateTime":   "t0_order_datetime_fb",
        })
    )

    events = dormant[["ShopMemberId", "t0_date"]].merge(
        best_finish, on=["ShopMemberId", "t0_date"], how="left"
    )
    events = events.merge(
        any_order, on=["ShopMemberId", "t0_date"], how="left"
    )
    events["t0_trades_group_code"] = events["t0_trades_group_code"].fillna(
        events["t0_trades_group_code_fb"]
    )
    events["t0_order_datetime"] = events["t0_order_datetime"].fillna(
        events["t0_order_datetime_fb"]
    )
    events = events.drop(columns=["t0_trades_group_code_fb", "t0_order_datetime_fb"])
    events = events.dropna(subset=["t0_trades_group_code"])

    members_w_event = events["ShopMemberId"].nunique()
    stats["r6_total_events"]          = len(events)
    stats["r6_members_w_event"]       = members_w_event
    stats["r6_avg_events_per_member"] = round(len(events) / members_w_event, 2)

    print(f"  Total revival events   : {stats['r6_total_events']:>10,}")
    print(f"  Members with ≥1 event  : {stats['r6_members_w_event']:>10,}")
    print(f"  Avg events / member    : {stats['r6_avg_events_per_member']}")
    print("R6 done ✓\n")
    return events


# ==========================================================================
# R7: Label generation (120-day window)
# ==========================================================================
def r7_labels(events: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """R7: y=1 if ≥1 valid purchase in (t0_order_datetime, t0_order_datetime+120d]
    where the t0 order itself is excluded by two conditions:
      1. OrderDateTime strictly > t0_order_datetime (exact timestamp, not just date)
         — prevents any order on the same day but before/at t0 from counting
      2. TradesGroupCode != t0_trades_group_code
         — belt-and-suspenders: explicitly excludes the t0 order even if same timestamp
    Uses R3-1 orders (TradesGroupCode-deduped).
    """
    print("R7 — Label generation")

    events["win_end"] = events["t0_order_datetime"] + pd.Timedelta(days=OBSERVATION_WINDOW)

    # Vectorised join: event × member orders, then filter to window
    matched = events[
        ["ShopMemberId", "t0_date", "t0_order_datetime", "t0_trades_group_code", "win_end"]
    ].merge(
        orders[["ShopMemberId", "OrderDateTime", "TradesGroupCode"]],
        on="ShopMemberId",
        how="inner"
    )

    # Ensure string dtype for cross-category comparison
    matched["t0_trades_group_code"] = matched["t0_trades_group_code"].astype(str)
    matched["TradesGroupCode"]      = matched["TradesGroupCode"].astype(str)

    valid_matches = matched[
        (matched["OrderDateTime"] >  matched["t0_order_datetime"]) &
        (matched["OrderDateTime"] <= matched["win_end"]) &
        (matched["TradesGroupCode"] != matched["t0_trades_group_code"])
    ]

    success_events = valid_matches[["ShopMemberId", "t0_date"]].drop_duplicates().copy()
    success_events["label"] = 1

    events = events.merge(success_events, on=["ShopMemberId", "t0_date"], how="left")
    events["label"] = events["label"].fillna(0).astype("int8")
    events = events.drop(columns=["win_end"])

    y1 = int(events["label"].sum())
    y0 = len(events) - y1
    stats["r7_y1"]      = y1
    stats["r7_y0"]      = y0
    stats["r7_y1_rate"] = round(y1 / len(events) * 100, 2)

    print(f"  y=1 (revival)  : {y1:>10,}  ({stats['r7_y1_rate']}%)")
    print(f"  y=0 (no return): {y0:>10,}")
    print("R7 done ✓\n")
    return events


# ==========================================================================
# R8: Censoring
# ==========================================================================
def r8_censor(events: pd.DataFrame) -> pd.DataFrame:
    """R8: Keep only t0 <= 2023-11-01 (= censor_date - 120d).
    Events with t0 > cutoff have incomplete observation windows — drop entirely.
    """
    print("R8 — Censoring")

    before = len(events)
    kept   = events[events["t0_date"] <= CENSOR_T0_MAX].copy()
    dropped = before - len(kept)

    stats["r8_before"]    = before
    stats["r8_kept"]      = len(kept)
    stats["r8_dropped"]   = dropped
    stats["r8_y1"]        = int(kept["label"].sum())
    stats["r8_y0"]        = len(kept) - stats["r8_y1"]
    stats["r8_y1_rate"]   = round(stats["r8_y1"] / len(kept) * 100, 2)

    print(f"  Before censoring  : {before:>10,}")
    print(f"  Kept (t0 ≤ {CENSOR_T0_MAX.date()}) : {len(kept):>10,}")
    print(f"  Dropped (censored): {dropped:>10,}")
    print(f"  Kept y=1 rate     : {stats['r8_y1_rate']}%")
    print("R8 done ✓\n")
    return kept


# ==========================================================================
# R9: Calendar-based train/test split + single embargo
# ==========================================================================
def r9_split(events: pd.DataFrame) -> pd.DataFrame:
    """R9: Calendar cut — train (t0 <= TRAIN_END) / test (t0 >= TEST_START).
    Single 120-day embargo buffer discards samples in (TRAIN_END, TEST_START).
    No 'val' split: the dataset spans ~21 months; two 120-day embargos would
    eliminate val and test entirely. Val is replaced by TimeSeriesSplit CV
    during the modelling stage.
    split column values: 'train' / 'test' only.
    """
    print("R9 — Calendar split + embargo")

    events = events.sort_values("t0_date").reset_index(drop=True)

    train_mask   = events["t0_date"] <= TRAIN_END
    test_mask    = events["t0_date"] >= TEST_START
    embargo_mask = (~train_mask) & (~test_mask)

    embargo_dropped = int(embargo_mask.sum())

    train_final = events[train_mask].copy()
    test_final  = events[test_mask].copy()

    train_final["split"] = "train"
    test_final["split"]  = "test"

    result = pd.concat([train_final, test_final], ignore_index=True)

    for split_name in ["train", "test"]:
        grp    = result[result["split"] == split_name]
        n      = len(grp)
        y1r    = round(grp["label"].mean() * 100, 2) if n > 0 else 0.0
        t0_min_s = grp["t0_date"].min().date().isoformat() if n > 0 else "–"
        t0_max_s = grp["t0_date"].max().date().isoformat() if n > 0 else "–"
        stats[f"r9_{split_name}_n"]     = n
        stats[f"r9_{split_name}_y1r"]   = y1r
        stats[f"r9_{split_name}_t0min"] = t0_min_s
        stats[f"r9_{split_name}_t0max"] = t0_max_s
        print(f"  {split_name:5s}  n={n:>8,}  y1={y1r:5.1f}%  t0=[{t0_min_s} ~ {t0_max_s}]")

    stats["r9_embargo_dropped"] = embargo_dropped
    print(f"  Embargo 緩衝帶丟棄 : {embargo_dropped:,} 筆  "
          f"(t0 ∈ ({TRAIN_END.date()}, {TEST_START.date()}))")
    print("R9 done ✓\n")
    return result


# ==========================================================================
# R10: Assign global sample_id, write samples_shop3.parquet
# ==========================================================================
def r10_output(events: pd.DataFrame) -> pd.DataFrame:
    """R10: Sort all retained samples by t0_date, assign continuous sample_id
    from 0, write output/samples_shop3.parquet.
    split column = 'train' / 'test'.

    Output columns (strict order): sample_id, label, member_id, t0_date,
    t0_trades_group_code, split.
    """
    print("R10 — Assign sample_id and write parquet")

    # Global sort → continuous sample_id
    out = events.sort_values("t0_date").reset_index(drop=True)
    out["sample_id"] = out.index.astype("int32")
    out["t0_date"]   = pd.to_datetime(out["t0_date"]).dt.date

    out = out.rename(columns={"ShopMemberId": "member_id"})

    OUTPUT_COLS = ["sample_id", "label", "member_id", "t0_date", "t0_trades_group_code", "split"]
    out_final = out[OUTPUT_COLS].copy()
    out_final["member_id"]            = out_final["member_id"].astype("category")
    out_final["t0_trades_group_code"] = out_final["t0_trades_group_code"].astype("category")

    parquet_path = OUT_DIR / "samples_shop3.parquet"
    out_final.to_parquet(parquet_path, index=False)

    stats["r10_total_samples"] = len(out_final)
    print(f"  samples_shop3.parquet → {parquet_path}  ({len(out_final):,} rows)")
    print("R10 done ✓\n")
    return out_final


# ==========================================================================
# data_preprocessing.md
# ==========================================================================
def write_preprocessing_md():
    """Write output/data_preprocessing.md — rules + rationale document.
    Reflects updated R7 (exact-timestamp label window) and R9 (calendar split).
    """
    s = stats
    shop3_hash = SHOP3_ID

    content = f"""# 資料前處理規則文件

> 版本：v1.1
> 產出日期：2026-06-03
> 執行腳本：`preprocess_shop3.py`

---

## 1. 資料來源與焦點

| 項目 | 說明 |
|------|------|
| 主檔路徑 | `91APP_Dataset(會員&主單&子單&商品頁&標籤)/Order_TG.csv` |
| 品牌 | DHC 保健食品（91APP shop#3） |
| shop#3 ShopId（雜湊值） | `{shop3_hash}` |
| 資料時間範圍 | {s.get('r1_dt_min','–')} ~ {s.get('r1_dt_max','–')} |
| shop#3 原始筆數 | {s.get('r1_shop3_rows',0):,} |
| shop#3 會員數 | {s.get('r1_members',0):,} |

> **ShopId 識別**：載入時 print 全資料集所有 ShopId rank，shop#3 對應 `{shop3_hash}`（EDA 驗證 4,938,006 筆）。

---

## 2. 處理流程總覽

```
Order_TG.csv（全資料集）
   │
   ├─ R1: 讀檔 + 過濾 shop#3（ShopId 篩選）
   │
   ├─ R2: 有效購買過濾（StatusDef != Fail, TotalSalesAmount > 0）
   │       加入 date 欄位（OrderDateTime 去時間部分）
   │
   ├─ R3: 雙重去重
   │       ├─ R3-1: unique TradesGroupCode → orders（標籤計算用）
   │       └─ R3-2: unique (ShopMemberId, date) → daily（Ci 計算用）
   │
   ├─ R4: 動態個人購物週期 Ci（逐 gap，僅用該 gap 之前的歷史）
   │
   ├─ R5: 沉睡門檻 = clip(3×Ci, 30, 365)
   │
   ├─ R6: 回流事件 t₀ 偵測（gap > threshold → t₀ = 第 i+1 次購買日）
   │       同時記錄 t0_order_datetime（精確時間戳記，供 R7 使用）
   │
   ├─ R7: 標籤生成（120 天觀察窗，嚴格排除 t₀ 訂單本身）
   │
   ├─ R8: Censoring（剔除 t₀ > 2023-11-01，觀察窗不完整）
   │
   ├─ R9: 時序切分（calendar 日期）train/test + 單一 120 天 Embargo
   │
   └─ R10: 指派全域 sample_id → samples_shop3.parquet
```

---

## 3. 各規則詳細說明

### R1：資料載入與品牌過濾

**定義：** 從全資料集讀取 Order_TG.csv，依 ShopId 篩選保留 shop#3（DHC）資料。

**邏輯：**
```python
raw = pd.read_csv("Order_TG.csv", dtype={{'ShopId': 'category', ...}})
# print 所有 unique ShopId（含 rank 與筆數比例）
shop3 = raw[raw["ShopId"] == SHOP3_ID]
```

**為何這樣設計：**
- 資料集含 5 個品牌，各品牌的購買週期、回流行為差異顯著；混合訓練導致 Ci 估計失準、特徵分布偏移
- 使用 `category` dtype：雜湊值欄位無數值運算需求，節省記憶體約 4–8×

**預期效果：** 保留全資料集約 1/5 的資料，會員數縮小至可管理規模

---

### R2：有效購買過濾

**定義：** 剔除失敗訂單與零（負）金額記錄，加入 `date` 欄位。

**條件：**
```
StatusDef != 'Fail'  AND  TotalSalesAmount > 0
```

**為何這樣設計：**
- EDA Step 1 建議使用 definition (b)：保留 Overdue、Shipping 等有正向金流的狀態，只剔除 Fail 與純沖銷行
- `TotalSalesAmount > 0` 排除促銷沖銷行（金額為負）
- `date = OrderDateTime.dt.normalize()`：保持 Timestamp 格式（midnight），與 R6 merge key 一致

**預期效果：** 保留率 ≈ 94%（EDA Step 1 結果）

---

### R3：雙重去重

**定義：** 對 valid 資料做兩種不同目的的去重，**絕對不能混用**。

#### R3-1（供標籤計算用）
```python
orders = valid.drop_duplicates(subset='TradesGroupCode')
```
一張訂單可能在 Order_TG 中有多列（不同子商品）。標籤計算時「購買 1 次」= 1 張訂單；不去重會重複計算。

#### R3-2（供 Ci 計算用）
```python
daily = valid[['ShopMemberId', 'date']].drop_duplicates()
```
Ci 衡量「購買頻率（節奏）」，同一天買多次對節奏意義相同。若不去重，同日多筆產生大量 0-day gap，壓縮 Ci 使沉睡門檻過低。

**為何必須分開：**
R3-1 保留 TotalSalesAmount（後續 R6 挑最大金額訂單），R3-2 只需日期；交叉使用會產生邏輯矛盾。

---

### R4：動態個人購物週期 Cᵢ

**定義：** 對每個 gap（相鄰兩次購買日期差），以「該 gap **之前**的所有歷史 gap」估計 Cᵢ。

**逐 gap 計算邏輯：**
```python
prior_gaps = [gap_1, ..., gap_{{i-1}}]   # 只用 gap i 之前的資料

if len(prior_gaps) >= 2:    Ci = median(prior_gaps)
elif len(prior_gaps) == 1:  Ci = prior_gaps[0]
else:                       Ci = {FALLBACK_CI}   # 第 1 個 gap：用全體中位數
```

**為何用動態 Cᵢ 而非靜態：**
靜態（整個會員生涯中位數）計算早期 gap 的門檻時，包含未來才發生的購買資訊 → **時間洩漏**。
動態 Cᵢ 確保每個門檻只用當下可知的資訊，與線上推論保持一致。

> **注意：** 本 pipeline 使用動態 Cᵢ（R4），事件數與 Cᵢ 中位數會與 EDA Step 4/5 的靜態 Cᵢ 基準不同，屬預期現象。

**Fallback = {FALLBACK_CI} 天來源：**
EDA Step 3：shop#3 全體有效購買 gap 的中位數 = 67.0 天，對應 DHC 保健食品 1–2 個月份量的補充週期。

---

### R5：沉睡門檻

**定義：**
```python
threshold_i = clip(3 × Ci, lower=30, upper=365)
```

**為何 3×Cᵢ：** EDA Step 4 比較多個倍數：1.5× → 739,061 事件（假陽性過多）；**3× → 332,328 事件（靜態基準，動態 Ci 下會不同）**；5× → 186,000 事件（遺漏真實回流）。

**clip(30, 365)：** 下限防止極低 Ci 產生無意義門檻；上限確保超過一年的 gap 一律視為沉睡。

---

### R6：回流事件 t₀ 偵測

**定義：** `gap_i > threshold_i` → 第 i+1 次購買為「沉睡後回流」，t₀ = 該次購買日期。

**TradesGroupCode 選取（同日多單時）：**
1. 優先選 `StatusDef == 'Finish'` 的訂單中 `TotalSalesAmount` 最大的一張
2. Fallback：任意有效訂單（non-Fail, amount>0）

R6 同時記錄該訂單的 `t0_order_datetime`（精確時間戳記），供 R7 標籤窗使用。

**一位會員可貢獻多個 t₀：** DHC 保健食品為消耗品，長期使用者可能多次沉睡又回流，每次都是獨立的復活事件。

---

### R7：標籤生成

**定義：** 以 120 天為觀察窗，判斷回流後是否有後續購買。觀察窗**嚴格在 t₀ 之後**，且**排除 t₀ 訂單本身**（TradesGroupCode 與 OrderDateTime 雙重排除）。

```python
window_end = t0_order_datetime + 120 days

in_window = orders[
    (OrderDateTime >  t0_order_datetime) &      # 嚴格 > 精確時間戳記（非日期）
    (OrderDateTime <= window_end) &
    (TradesGroupCode != t0_trades_group_code)    # 排除 t₀ 訂單本身
]
y = 1 if len(in_window) >= 1 else 0
```

**為何用 t0_order_datetime 而非 t0_date（日期）：**
若改用日期（midnight）做比較，t₀ 當日 midnight 之後的任何訂單（包含 t₀ 那張單本身）都會被納入觀察窗，導致 y 永遠 = 1。
用精確時間戳記 `>` 可確保只有 t₀ 訂單時間點**之後**發生的訂單才進入窗口，再配合 TradesGroupCode 排除，完全防止 t₀ 自計入復活。

**使用 R3-1 orders 的理由：** 已 by TradesGroupCode 去重，「窗內購買次數」用訂單張數衡量，不受同日多筆影響。

---

### R8：Censoring

**定義：** 剔除觀察窗跨越資料截止日的樣本（不歸 0、不歸 1）。

```
資料截止日：2024-02-29
保留條件：  t₀ + 120d <= 2024-02-29
等同於：    t₀ <= 2023-11-01  (= CENSOR_CUTOFF)
```

若 t₀ = 2023-12-01，觀察窗延伸至 2024-04-01，超過資料範圍。這類樣本的 y=0 並非「真的沒買」而是「資料不完整」，混入訓練集會系統性低估 y=1 機率。直接剔除（censored observation）。

---

### R9：時序切分 + Embargo

**定義：** 依 t₀ 日期做 calendar 切分為 train 與 test，並以 120 天 embargo 緩衝帶防止標籤窗污染。

```
train 樣本：t₀ <= 2023-03-31  (TRAIN_END)
test  樣本：t₀ >= 2023-07-29  (TEST_START = TRAIN_END + 120d)
丟棄樣本：t₀ 落在 (2023-03-31, 2023-07-29) — embargo 緩衝區
```

**為何改用 calendar 切分 + 單一 embargo（而非 70/10/20 + 兩個 embargo）：**
資料時間跨度僅約 21 個月。70/10/20 按筆數切分後，兩個 120 天 embargo（train→val 與 val→test）會將 val 與 test 全數清空，無法進行評估。
改用 calendar 切分 + 單一 embargo，確保 test 有實際資料；val 的超參調整改由建模階段的 **TimeSeriesSplit CV** 在 train 內處理，本腳本不再產生 val split。

**Embargo 防污染原理：**
train 最後一筆的 t₀ 加上 120 天觀察窗，若與 test 最早的 t₀ 重疊，則兩者共用部分訂單資料，標籤間存在相關性（label leakage）。embargo = 觀察窗長度可完全消除此相關性。

---

### R10：輸出欄位指派

**定義：** 所有保留樣本按 t₀ 升序排列，指派連續全域 sample_id，輸出 `samples_shop3.parquet`。

```python
out = events.sort_values("t0_date").reset_index(drop=True)
out["sample_id"] = out.index   # 0, 1, 2, …, N-1（全域連續）
```

sample_id 全域連續無重複，確保可用 sample_id 欄位合并 train/test 還原完整排序。

---

## 4. 關鍵參數

| 參數 | 值 | 來源 |
|------|-----|------|
| 觀察窗 OBSERVATION_WINDOW | 120 天 | EDA Step 5（revival rate ≈ 75%） |
| 沉睡門檻倍數 k | 3 | EDA Step 4（事件量與假陽性平衡點） |
| 門檻 clip 下限 | 30 天 | 避免極低 Ci 產生過小門檻 |
| 門檻 clip 上限 | 365 天 | 超過一年必定沉睡 |
| Fallback Cᵢ | 67 天 | EDA Step 3：shop#3 全體 gap 中位數 |
| TRAIN_END | 2023-03-31 | 訓練集 t₀ 上限（calendar 切分） |
| TEST_START | 2023-07-29 | 測試集 t₀ 下限（= TRAIN_END + 120d） |
| EMBARGO_DAYS | 120 天 | = 觀察窗，防止標籤窗跨 split 污染 |
| 資料截止日 | 2024-02-29 | 資料集上限 |
| CENSOR_CUTOFF | 2023-11-01 | 截止日 − 觀察窗 |

---

## 5. 輸出格式

### 檔案

| 檔案 | 路徑 | 用途 |
|------|------|------|
| samples_shop3.parquet | `output/samples_shop3.parquet` | 模型訓練與評估（含 split 欄） |

### 欄位定義（嚴格此順序）

| 順序 | 欄位名 | 型別 | 說明 |
|------|-------|------|------|
| 1 | sample_id | int32 | 全域流水號，從 0 起，按 t₀ 升序連續 |
| 2 | label | int8 | 0 = 路過（未回購），1 = 真復活（窗內 ≥1 次購買） |
| 3 | member_id | category | ShopMemberId（雜湊） |
| 4 | t0_date | date（YYYY-MM-DD） | 回流事件發生日 |
| 5 | t0_trades_group_code | category | t₀ 當日最大金額 Finish 訂單號 |
| 6 | split | str | 'train' / 'test' |

### 說明

- 格式：Parquet（pyarrow 壓縮），比 CSV 節省約 60% 空間，保留欄位型別
- split 值僅 'train' / 'test'，無 'val'；val 由建模階段 TimeSeriesSplit CV 取代
- 這只是「樣本骨架」，特徵工程欄位將在下一階段加入

---

## 6. 重現步驟

```bash
# 1. 確認 Python 環境
pip install pandas numpy pyarrow

# 2. 確認資料檔案存在
# ./91APP_Dataset(會員&主單&子單&商品頁&標籤)/Order_TG.csv

# 3. 從 BDA2026_final/ 目錄執行
python preprocess_shop3.py

# 4. 檢查輸出
# ./output/samples_shop3.parquet
# ./output/data_preprocessing.md
# ./output/pipeline_report.md
```

執行時間參考：約 5–15 分鐘（依機器規格，R4 動態 Ci 計算最耗時）。

---

*Generated by preprocess_shop3.py — 2026-06-03*
"""

    out_path = OUT_DIR / "data_preprocessing.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Saved → {out_path}")


# ==========================================================================
# pipeline_report.md
# ==========================================================================
def write_pipeline_report():
    """Write output/pipeline_report.md with actual run statistics."""
    s = stats

    def pct(num, denom):
        return f"{num/max(denom,1)*100:.1f}%"

    # train/test y=1 proximity check
    tr_y1r = s.get('r9_train_y1r', 0)
    te_y1r = s.get('r9_test_y1r', 0)
    y1r_diff = abs(tr_y1r - te_y1r)
    y1r_check = "✓ 相近（差距 < 10pp）" if y1r_diff < 10 else f"⚠ 差距 {y1r_diff:.1f}pp，近期留存率變化，建議在報告中討論"
    # post-censoring y=1 rate check
    post_y1r = s.get('r8_y1_rate', 0)
    post_y1_check = "✓ 合理" if 60 <= post_y1r <= 80 else "⚠ 超出 60~80% 範圍，請檢查 R7"

    content = f"""# Pipeline Execution Report — shop#3 (DHC)

> 腳本：`preprocess_shop3.py`
> 產出日期：2026-06-03

---

## 1. 資料載入（R1）

| 項目 | 數值 |
|------|------|
| 全資料集原始筆數 | {s.get('r1_raw_rows', 0):,} |
| shop#3 篩選後筆數 | {s.get('r1_shop3_rows', 0):,} |
| shop#3 會員數 | {s.get('r1_members', 0):,} |
| OrderDateTime 範圍 | {s.get('r1_dt_min','–')} ~ {s.get('r1_dt_max','–')} |

---

## 2. 有效購買過濾（R2）

| 項目 | 數值 |
|------|------|
| 過濾前筆數 | {s.get('r2_before', 0):,} |
| 過濾後筆數 | {s.get('r2_after', 0):,} |
| 保留率 | {s.get('r2_keep_rate', 0)}% |

---

## 3. 去重結果（R3）

| 去重類型 | 筆數 | 用途 |
|---------|------|------|
| R3-1 unique TradesGroupCode | {s.get('r3_orders', 0):,} | 標籤計算（觀察窗內購買數） |
| R3-2 unique (member, date) | {s.get('r3_daily', 0):,} | Ci gap 計算 |

---

## 4. Cᵢ 統計（R4）

| 項目 | 數值 |
|------|------|
| 計算到的 gap 總數 | {s.get('r4_total_gaps', 0):,} |
| 用 prior_gaps 中位數（≥2 prior） | {s.get('r4_use_median', 0):,} ({pct(s.get('r4_use_median',0), s.get('r4_total_gaps',1))}) |
| 用單一 prior gap | {s.get('r4_use_single', 0):,} ({pct(s.get('r4_use_single',0), s.get('r4_total_gaps',1))}) |
| 用 fallback (67d) | {s.get('r4_use_fallback', 0):,} ({pct(s.get('r4_use_fallback',0), s.get('r4_total_gaps',1))}) |
| Cᵢ P25 | {s.get('r4_ci_p25', 0)} 天 |
| Cᵢ P50 | {s.get('r4_ci_p50', 0)} 天 |
| Cᵢ P75 | {s.get('r4_ci_p75', 0)} 天 |
| Cᵢ P90 | {s.get('r4_ci_p90', 0)} 天 |

---

## 5. 沉睡門檻分布（R5）

| 項目 | 數值 |
|------|------|
| 門檻中位數（clip 前） | {s.get('r5_thr_median_raw', 0)} 天 |
| 門檻中位數（clip 後） | {s.get('r5_thr_median_clipped', 0)} 天 |
| 被 clip 到下限（30d）比例 | {s.get('r5_clipped_lo_pct', 0)}% |
| 被 clip 到上限（365d）比例 | {s.get('r5_clipped_hi_pct', 0)}% |

---

## 6. 回流事件（R6）

| 項目 | 數值 |
|------|------|
| 總回流事件數 | {s.get('r6_total_events', 0):,} |
| 有事件的會員數 | {s.get('r6_members_w_event', 0):,} |
| 平均每人事件數 | {s.get('r6_avg_events_per_member', 0)} |

---

## 7. Censoring（R8）

| 項目 | 數值 |
|------|------|
| Censoring 前事件數 | {s.get('r8_before', 0):,} |
| 保留事件數（t₀ ≤ 2023-11-01） | {s.get('r8_kept', 0):,} |
| 剔除（censored）事件數 | {s.get('r8_dropped', 0):,} |

---

## 8. 標籤分布（R7，censoring 後）

| 標籤 | 數量 | 比例 |
|------|------|------|
| y=1（復活） | {s.get('r8_y1', 0):,} | {s.get('r8_y1_rate', 0)}% |
| y=0（未回購） | {s.get('r8_y0', 0):,} | {100-s.get('r8_y1_rate', 0):.2f}% |

> censoring 前（全 511,800 事件）：y=1 = {s.get('r7_y1', 0):,}（{s.get('r7_y1_rate', 0)}%）

---

## 9. 時序切分（R9）

| Split | 樣本數 | t₀ 範圍 | y=1 比例 |
|-------|--------|---------|---------|
| train | {s.get('r9_train_n', 0):,} | {s.get('r9_train_t0min','–')} ~ {s.get('r9_train_t0max','–')} | {s.get('r9_train_y1r', 0)}% |
| test  | {s.get('r9_test_n', 0):,} | {s.get('r9_test_t0min','–')} ~ {s.get('r9_test_t0max','–')} | {s.get('r9_test_y1r', 0)}% |

**Embargo 丟棄（緩衝帶 2023-04-01 ~ 2023-07-28）：** {s.get('r9_embargo_dropped', 0):,} 筆

---

## 10. 合理性檢查

| 指標 | 預期 | 實際 | 狀態 |
|------|------|------|------|
| 回流事件總數（R6） | 動態 Ci 下高於 EDA 靜態基準（332,000）屬正常 | {s.get('r6_total_events', 0):,} | {'✓' if s.get('r6_total_events',0) > 0 else '⚠ 無事件'} |
| y=1 比例（R7，censoring 後） | 60~80%（動態 Ci 下，不必剛好 75%） | {post_y1r}% | {post_y1_check} |
| train y=1 vs test y=1 比例接近 | 差距 < 10pp | train {tr_y1r}% / test {te_y1r}% | {y1r_check} |
| censoring 後樣本數（R8） | 動態 Ci 下事件數高於 EDA 靜態基準屬正常 | {s.get('r8_kept', 0):,} | {'✓' if s.get('r8_kept',0) > 0 else '⚠ 無樣本'} |

> 本 pipeline 使用動態 Cᵢ（R4）。事件數與 Cᵢ 中位數會與 EDA Step 4/5 的靜態 Cᵢ 基準不同，屬預期現象。

---

*Generated by preprocess_shop3.py — 2026-06-03*
"""

    out_path = OUT_DIR / "pipeline_report.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Saved → {out_path}")


# ==========================================================================
# Main
# ==========================================================================
def main():
    print("=" * 60)
    print("  shop#3 (DHC) Preprocessing Pipeline")
    print("=" * 60)
    print()

    shop3          = r1_load(ORDER_TG)
    valid          = r2_filter(shop3)
    orders, daily  = r3_dedup(valid)
    gaps           = r4_compute_ci(daily)
    gaps           = r5_threshold(gaps)
    events         = r6_revival_events(gaps, valid)
    events         = r7_labels(events, orders)
    events         = r8_censor(events)
    events         = r9_split(events)
    r10_output(events)

    print("Writing markdown reports …")
    write_preprocessing_md()
    write_pipeline_report()

    print()
    print("=" * 60)
    print("  Pipeline complete!")
    print(f"  Total samples : {stats['r10_total_samples']:,}")
    print(f"  Outputs in    : {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
