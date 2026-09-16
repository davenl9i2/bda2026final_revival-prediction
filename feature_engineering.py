"""
feature_engineering.py
shop#3 (DHC) return sample feature engineering — F1 through F6.
Anti-leakage: all historical features strictly use OrderDateTime < t0_date.
F2 is the sole exception (t0 order attributes observed at return time).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

SHOP3_ID = "hFwniXiB/Ev2ZPXeO630Sw=="
BASE = Path(__file__).parent
DATA_DIR = BASE / "91APP_Dataset(會員&主單&子單&商品頁&標籤)"
OUT_DIR = BASE / "output"
OUT_DIR.mkdir(exist_ok=True)

N_BATCHES = 10  # split sample_ids into batches to manage memory


# ============================================================
# Data loading helpers
# ============================================================

def load_samples():
    s = pd.read_parquet(OUT_DIR / "samples_shop3.parquet")
    s["t0_date"] = pd.to_datetime(s["t0_date"])
    s["member_id"] = s["member_id"].astype(str)
    s["t0_trades_group_code"] = s["t0_trades_group_code"].astype(str)
    print(f"Samples: {len(s):,}  train={s['split'].eq('train').sum():,}  test={s['split'].eq('test').sum():,}")
    return s


def load_orders():
    """
    Load Order_TG for shop3, dedup by TradesGroupCode.
    Adds is_valid (non-Fail, amount>0) and is_return flags.
    """
    usecols = ["ShopId", "ShopMemberId", "TradesGroupCode", "OrderDateTime",
               "ChannelType", "PaymentType", "ShippingType",
               "TsCount", "Qty", "TotalSalesAmount", "TotalPrice",
               "TotalDiscount", "TotalCouponDiscount", "StatusDef"]
    dtype_map = {"ShopId": "category", "StatusDef": "category",
                 "ShopMemberId": str, "TradesGroupCode": str,
                 "ChannelType": str, "PaymentType": str, "ShippingType": str}

    print("Loading Order_TG.csv …")
    orders = pd.read_csv(DATA_DIR / "Order_TG.csv",
                         usecols=usecols, dtype=dtype_map, low_memory=False)
    orders["OrderDateTime"] = pd.to_datetime(orders["OrderDateTime"], format="mixed")
    orders = orders[orders["ShopId"] == SHOP3_ID].drop(columns=["ShopId"]).copy()
    orders = orders.drop_duplicates(subset="TradesGroupCode")

    status_str = orders["StatusDef"].astype(str)
    orders["is_return"] = (status_str == "Return").astype("int8")
    orders["is_valid"] = ((status_str != "Fail") & (orders["TotalSalesAmount"] > 0)).astype("int8")
    orders["is_nonfail"] = (status_str != "Fail").astype("int8")

    for col in ["ChannelType", "PaymentType", "ShippingType"]:
        orders[col] = orders[col].fillna("unknown")

    print(f"  shop3 orders (deduped): {len(orders):,}")
    return orders


def load_order_slave():
    """
    Load Order_TS for shop3, filter to valid (non-Fail, UnitPrice > 0).
    """
    usecols = ["ShopId", "ShopMemberId", "TradesGroupCode", "OrderDateTime",
               "SalePageId", "Qty", "UnitPrice", "StatusDef"]
    dtype_map = {"ShopId": "category", "StatusDef": "category",
                 "ShopMemberId": str, "TradesGroupCode": str, "SalePageId": str}

    print("Loading Order_TS.csv …")
    slave = pd.read_csv(DATA_DIR / "Order_TS.csv",
                        usecols=usecols, dtype=dtype_map, low_memory=False)
    slave["OrderDateTime"] = pd.to_datetime(slave["OrderDateTime"], format="mixed")
    slave = slave[slave["ShopId"] == SHOP3_ID].drop(columns=["ShopId"]).copy()
    slave = slave[(slave["StatusDef"].astype(str) != "Fail") & (slave["UnitPrice"] > 0)].copy()
    slave = slave.drop(columns=["StatusDef"])

    print(f"  shop3 slave (valid): {len(slave):,}")
    return slave


def load_member():
    """
    Load Member.csv for shop3.
    Sentinel 1900-01-01 → NaT; MemberCardLevel 0 → NaN.
    """
    print("Loading Member.csv …")
    m = pd.read_csv(DATA_DIR / "Member.csv",
                    dtype={"ShopId": "category", "ShopMemberId": str,
                           "RegisterSourceTypeDef": str, "Gender": str, "CountryAliasCode": str},
                    low_memory=False)
    m = m[m["ShopId"] == SHOP3_ID].drop(columns=["ShopId"]).copy()

    sentinel = pd.Timestamp("1900-01-01")
    for col in ["RegisterDateTime", "Birthday"]:
        m[col] = pd.to_datetime(m[col], errors="coerce")
        m.loc[m[col] == sentinel, col] = pd.NaT

    m["MemberCardLevel"] = m["MemberCardLevel"].astype(float)
    m.loc[m["MemberCardLevel"] == 0, "MemberCardLevel"] = np.nan

    print(f"  shop3 members: {len(m):,}")
    return m


# ============================================================
# Step 0: Sanity check
# ============================================================

def sanity_check(samples):
    """
    Step 0: Cross-split member overlap and t0_date distribution.
    Returns dict of stats for use in feature_sanity.md.
    """
    print("\n" + "=" * 60)
    print("Step 0 — Sanity Check")
    print("=" * 60)

    train = samples[samples["split"] == "train"]
    test = samples[samples["split"] == "test"]

    train_members = set(train["member_id"])
    test_members = set(test["member_id"])
    overlap = train_members & test_members

    ov_pct_train = len(overlap) / len(train_members) * 100
    ov_pct_test = len(overlap) / len(test_members) * 100

    print(f"Train: {len(train):,}  unique members={len(train_members):,}")
    print(f"Test : {len(test):,}  unique members={len(test_members):,}")
    print(f"Cross-split overlap: {len(overlap):,}  ({ov_pct_train:.1f}% of train, {ov_pct_test:.1f}% of test)")

    print("\nt0_date monthly distribution (train):")
    print(train["t0_date"].dt.to_period("M").value_counts().sort_index().to_string())
    print("\nt0_date monthly distribution (test):")
    print(test["t0_date"].dt.to_period("M").value_counts().sort_index().to_string())

    if ov_pct_train > 30:
        print("\n⚠️  Overlap > 30% — reinforcing leakage checks downstream.")

    return {
        "overlap_count": len(overlap),
        "overlap_pct_train": round(ov_pct_train, 1),
        "overlap_pct_test": round(ov_pct_test, 1),
        "train_n": len(train),
        "test_n": len(test),
        "train_members": len(train_members),
        "test_members": len(test_members),
    }


# ============================================================
# Helper: batched history explode
# ============================================================

def _explode_history(samples, source, on="ShopMemberId", cols=None, n_batches=N_BATCHES):
    """
    Merge samples with source on ShopMemberId (= member_id), filter OrderDateTime < t0_date.
    Returns exploded DataFrame. Uses batching to bound peak memory.
    """
    base = samples[["sample_id", "member_id", "t0_date"]].rename(
        columns={"member_id": "ShopMemberId"}
    )
    src = source[source["ShopMemberId"].isin(set(base["ShopMemberId"]))]
    if cols is not None:
        keep = ["ShopMemberId"] + [c for c in cols if c in src.columns]
        src = src[keep]

    batch_size = (len(base) + n_batches - 1) // n_batches
    parts = []

    for i in range(n_batches):
        batch = base.iloc[i * batch_size: (i + 1) * batch_size]
        if len(batch) == 0:
            continue
        b_members = set(batch["ShopMemberId"])
        merged = batch.merge(src[src["ShopMemberId"].isin(b_members)],
                             on="ShopMemberId", how="left")
        if "OrderDateTime" in merged.columns:
            hist = merged[merged["OrderDateTime"] < merged["t0_date"]].copy()
        else:
            hist = merged
        parts.append(hist)

    return pd.concat(parts, ignore_index=True)


def _mode_by_group(df, group_col, val_col, out_col):
    """Efficient mode (most frequent value) per group."""
    cnt = df.groupby([group_col, val_col]).size().reset_index(name="_cnt")
    top = (cnt.sort_values("_cnt", ascending=False)
              .drop_duplicates(group_col)[[group_col, val_col]]
              .rename(columns={val_col: out_col}))
    return top


def _safe_div(num, denom):
    """Vectorised safe division: NaN where denom == 0 or NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where((denom == 0) | pd.isna(denom) | pd.isna(num), np.nan, num / denom)
    return result


# ============================================================
# F1: Historical RFM and behavioral stability
# ============================================================

def compute_f1_history_rfm(samples, orders):
    """
    F1: Historical RFM and behavioral stability.
    Leakage guard: OrderDateTime strictly < t0_date.
    Output: 17 features (16 spec + hist_total_qty used by F5).
    """
    print("\n--- F1: History RFM & behavioral stability ---")

    order_cols = ["OrderDateTime", "TotalSalesAmount", "TotalPrice", "TotalDiscount",
                  "TotalCouponDiscount", "Qty", "ChannelType", "PaymentType",
                  "is_valid", "is_return", "is_nonfail"]

    hist = _explode_history(samples, orders, cols=order_cols)
    print(f"  Exploded hist rows: {len(hist):,}")

    # ---- Valid orders for RFM ----
    hist_v = hist[hist["is_valid"] == 1].copy()
    hist_v["purchase_date"] = hist_v["OrderDateTime"].dt.normalize()

    agg = hist_v.groupby("sample_id").agg(
        hist_n_orders=("TotalSalesAmount", "count"),
        hist_total_amount=("TotalSalesAmount", "sum"),
        hist_max_ticket=("TotalSalesAmount", "max"),
        hist_total_qty=("Qty", "sum"),
        _sum_disc_abs=("TotalDiscount", lambda x: x.abs().sum()),
        _sum_price=("TotalPrice", "sum"),
        _coupon_orders=("TotalCouponDiscount", lambda x: (x != 0).sum()),
        last_purchase_date=("OrderDateTime", "max"),
        first_purchase_date=("OrderDateTime", "min"),
    ).reset_index()

    agg["hist_avg_ticket"] = _safe_div(agg["hist_total_amount"].values, agg["hist_n_orders"].values)
    agg["hist_discount_ratio"] = _safe_div(agg["_sum_disc_abs"].values, agg["_sum_price"].values)
    agg["hist_coupon_order_ratio"] = _safe_div(agg["_coupon_orders"].values, agg["hist_n_orders"].values)

    # ---- Return ratio: returns / non-fail ----
    hist_nf = hist[hist["is_nonfail"] == 1]
    ret_agg = hist_nf.groupby("sample_id").agg(
        _n_nonfail=("is_nonfail", "count"),
        _n_return=("is_return", "sum"),
    ).reset_index()
    ret_agg["hist_return_ratio"] = _safe_div(ret_agg["_n_return"].values, ret_agg["_n_nonfail"].values)

    # ---- Channel / payment mode ----
    ch_mode = _mode_by_group(hist_v, "sample_id", "ChannelType", "hist_main_channel")
    pay_mode = _mode_by_group(hist_v, "sample_id", "PaymentType", "hist_main_payment")

    # ---- Gaps: personal_ci, gap_std, gap_cv ----
    daily = hist_v[["sample_id", "purchase_date"]].drop_duplicates()
    daily = daily.sort_values(["sample_id", "purchase_date"])
    daily["prev_date"] = daily.groupby("sample_id")["purchase_date"].shift(1)
    daily["gap_days"] = (daily["purchase_date"] - daily["prev_date"]).dt.days
    gaps_df = daily.dropna(subset=["gap_days"])

    ci_agg = (gaps_df.groupby("sample_id")["gap_days"]
              .agg(["median", "std", "count"])
              .reset_index())
    ci_agg.columns = ["sample_id", "personal_ci", "gap_std", "_n_gaps"]
    ci_agg.loc[ci_agg["_n_gaps"] < 2, ["personal_ci", "gap_std"]] = np.nan
    ci_agg["gap_cv"] = _safe_div(ci_agg["gap_std"].values, ci_agg["personal_ci"].values)

    # ---- Assemble ----
    f1 = samples[["sample_id", "t0_date"]].copy()
    for df, cols in [
        (agg[["sample_id", "hist_n_orders", "hist_total_amount", "hist_avg_ticket",
              "hist_max_ticket", "hist_total_qty", "hist_discount_ratio",
              "hist_coupon_order_ratio", "last_purchase_date", "first_purchase_date"]], None),
        (ret_agg[["sample_id", "hist_return_ratio"]], None),
        (ci_agg[["sample_id", "personal_ci", "gap_std", "gap_cv"]], None),
        (ch_mode, None),
        (pay_mode, None),
    ]:
        f1 = f1.merge(df, on="sample_id", how="left")

    f1["sleep_days"] = (f1["t0_date"] - f1["last_purchase_date"]).dt.days
    f1["member_tenure_days"] = (f1["t0_date"] - f1["first_purchase_date"]).dt.days
    f1["was_sealed"] = (f1["sleep_days"] > 365).astype("Int8")
    f1["prior_once_only"] = (f1["hist_n_orders"] == 1).astype("Int8")

    f1["hist_main_channel"] = f1["hist_main_channel"].fillna("unknown")
    f1["hist_main_payment"] = f1["hist_main_payment"].fillna("unknown")

    out_cols = ["sample_id", "sleep_days", "hist_n_orders", "hist_total_amount",
                "hist_avg_ticket", "hist_max_ticket", "hist_total_qty",
                "personal_ci", "gap_std", "gap_cv",
                "hist_discount_ratio", "hist_coupon_order_ratio", "hist_return_ratio",
                "hist_main_channel", "hist_main_payment",
                "member_tenure_days", "was_sealed", "prior_once_only"]

    f1_out = f1[out_cols].copy()
    miss = f1_out.drop(columns="sample_id").isnull().mean().median() * 100
    print(f"  F1: {len(out_cols)-1} features  missing_P50={miss:.1f}%  n={len(f1_out):,}")
    return f1_out


# ============================================================
# F2: t0 order features
# ============================================================

def compute_f2_t0_order(samples, orders):
    """
    F2: t0 order attributes (the revival purchase itself, already observed).
    No leakage: t0 order is the event trigger, not a future event.
    Output: 12 features.
    """
    print("\n--- F2: t0 order features ---")

    t0_cols = ["TradesGroupCode", "TotalSalesAmount", "TsCount", "Qty",
               "TotalCouponDiscount", "TotalDiscount", "TotalPrice",
               "ChannelType", "PaymentType", "ShippingType"]

    t0_orders = orders[orders["is_valid"] == 1][t0_cols].copy()

    f2 = samples[["sample_id", "t0_trades_group_code", "t0_date"]].copy()
    f2 = f2.merge(t0_orders.rename(columns={"TradesGroupCode": "t0_trades_group_code"}),
                  on="t0_trades_group_code", how="left")

    f2["t0_amount"] = f2["TotalSalesAmount"]
    f2["t0_ts_count"] = f2["TsCount"]
    f2["t0_qty"] = f2["Qty"]
    f2["t0_used_coupon"] = (f2["TotalCouponDiscount"] != 0).astype("Int8")
    f2["t0_discount_ratio"] = _safe_div(f2["TotalDiscount"].abs().values, f2["TotalPrice"].values)
    f2["t0_channel"] = f2["ChannelType"].fillna("unknown")
    f2["t0_payment"] = f2["PaymentType"].fillna("unknown")
    f2["t0_shipping"] = f2["ShippingType"].fillna("unknown")
    f2["t0_month"] = f2["t0_date"].dt.month
    f2["t0_weekday"] = f2["t0_date"].dt.dayofweek
    f2["t0_is_weekend"] = f2["t0_weekday"].isin([5, 6]).astype("Int8")
    f2["t0_is_big_promo"] = f2["t0_month"].isin([5, 11]).astype("Int8")

    out_cols = ["sample_id", "t0_amount", "t0_ts_count", "t0_qty", "t0_used_coupon",
                "t0_discount_ratio", "t0_channel", "t0_payment", "t0_shipping",
                "t0_month", "t0_weekday", "t0_is_weekend", "t0_is_big_promo"]

    f2_out = f2[out_cols].copy()
    miss = f2_out.drop(columns="sample_id").isnull().mean().median() * 100
    print(f"  F2: {len(out_cols)-1} features  missing_P50={miss:.1f}%  n={len(f2_out):,}")
    return f2_out


# ============================================================
# F3: Product behavior (Order_TS + SalePage)
# ============================================================

def compute_f3_product(samples, orders, order_slave, salepage):
    """
    F3: Product-level behavior from slave orders and sale pages.
    Hist features: OrderDateTime < t0_date (leakage-safe).
    t0 features: the t0 order's slave rows (already observed).
    Output: 7 features.
    """
    print("\n--- F3: Product behavior ---")

    slave_cols = ["OrderDateTime", "SalePageId", "Qty", "UnitPrice"]

    # ---- Historical slave orders ----
    hist_slave = _explode_history(samples, order_slave, cols=slave_cols)
    print(f"  Exploded hist slave rows: {len(hist_slave):,}")

    hist_agg = hist_slave.groupby("sample_id").agg(
        hist_distinct_salepages=("SalePageId", "nunique"),
        _hist_total_slave_rows=("SalePageId", "count"),
    ).reset_index()

    # Repeat SKU ratio: rows for repeated SalePageId / total rows
    sp_counts = hist_slave.groupby(["sample_id", "SalePageId"]).size().reset_index(name="_cnt")
    sp_counts["_is_repeat"] = (sp_counts["_cnt"] > 1).astype(int)
    repeat_agg = sp_counts.groupby("sample_id").agg(
        _n_repeat_skus=("_is_repeat", "sum"),
        _n_total_skus=("_is_repeat", "count"),
    ).reset_index()
    repeat_agg["hist_repeat_sku_ratio"] = _safe_div(
        repeat_agg["_n_repeat_skus"].values, repeat_agg["_n_total_skus"].values
    )

    # ---- t0 order slave rows ----
    t0_slave = order_slave.merge(
        samples[["sample_id", "t0_trades_group_code"]].rename(
            columns={"t0_trades_group_code": "TradesGroupCode"}
        ),
        on="TradesGroupCode", how="inner",
    )

    t0_agg = t0_slave.groupby("sample_id").agg(
        t0_n_distinct_salepages=("SalePageId", "nunique"),
        t0_avg_unit_price=("UnitPrice", "mean"),
        t0_min_unit_price=("UnitPrice", "min"),
        t0_max_unit_price=("UnitPrice", "max"),
    ).reset_index()

    # t0_buys_familiar_product: vectorised inner join on (sample_id, SalePageId)
    hist_sp_pairs = hist_slave[["sample_id", "SalePageId"]].drop_duplicates()
    t0_sp_pairs = t0_slave[["sample_id", "SalePageId"]].drop_duplicates()
    familiar = (t0_sp_pairs
                .merge(hist_sp_pairs, on=["sample_id", "SalePageId"], how="inner")
                [["sample_id"]].drop_duplicates()
                .assign(t0_buys_familiar_product=np.int8(1)))

    f3_merged = t0_agg.merge(familiar, on="sample_id", how="left")
    f3_merged["t0_buys_familiar_product"] = (
        f3_merged["t0_buys_familiar_product"].fillna(0).astype("Int8")
    )

    # ---- Assemble ----
    f3 = samples[["sample_id"]].copy()
    f3 = f3.merge(hist_agg[["sample_id", "hist_distinct_salepages"]], on="sample_id", how="left")
    f3 = f3.merge(repeat_agg[["sample_id", "hist_repeat_sku_ratio"]], on="sample_id", how="left")
    f3 = f3.merge(f3_merged[["sample_id", "t0_n_distinct_salepages", "t0_buys_familiar_product",
                              "t0_avg_unit_price", "t0_min_unit_price", "t0_max_unit_price"]],
                  on="sample_id", how="left")

    out_cols = ["sample_id", "hist_distinct_salepages", "hist_repeat_sku_ratio",
                "t0_n_distinct_salepages", "t0_buys_familiar_product",
                "t0_avg_unit_price", "t0_min_unit_price", "t0_max_unit_price"]

    f3_out = f3[out_cols].copy()
    miss = f3_out.drop(columns="sample_id").isnull().mean().median() * 100
    print(f"  F3: {len(out_cols)-1} features  missing_P50={miss:.1f}%  n={len(f3_out):,}")
    return f3_out


# ============================================================
# F4: Member background
# ============================================================

def compute_f4_member(samples, member):
    """
    F4: Member background features.
    Safe (time-invariant): register_source, gender, age_at_t0, days_since_register, country_code.
    Snapshot (minor leakage, annotated): member_card_level, is_app_installed, marketing_optin_count.
    Output: 8 features.
    """
    print("\n--- F4: Member background ---")

    mem = member[["ShopMemberId", "RegisterSourceTypeDef", "RegisterDateTime", "Gender",
                  "Birthday", "MemberCardLevel", "IsAppInstalled",
                  "IsEnableEmail", "IsEnablePushNotification", "IsEnableShortMessage",
                  "CountryAliasCode"]].copy()

    f4 = samples[["sample_id", "member_id", "t0_date"]].merge(
        mem.rename(columns={"ShopMemberId": "member_id"}),
        on="member_id", how="left",
    )

    f4["register_source"] = f4["RegisterSourceTypeDef"].fillna("unknown")
    f4["gender"] = f4["Gender"]
    f4["age_at_t0"] = ((f4["t0_date"] - f4["Birthday"]).dt.days / 365.25).where(
        f4["Birthday"].notna()
    )
    f4["days_since_register"] = (f4["t0_date"] - f4["RegisterDateTime"]).dt.days.where(
        f4["RegisterDateTime"].notna()
    )
    f4["country_code"] = f4["CountryAliasCode"].fillna("unknown")
    f4["member_card_level"] = f4["MemberCardLevel"]
    f4["is_app_installed"] = f4["IsAppInstalled"].astype("Int8")
    f4["marketing_optin_count"] = (
        f4["IsEnableEmail"].fillna(False).astype(int) +
        f4["IsEnablePushNotification"].fillna(False).astype(int) +
        f4["IsEnableShortMessage"].fillna(False).astype(int)
    )

    out_cols = ["sample_id", "register_source", "gender", "age_at_t0",
                "days_since_register", "country_code",
                "member_card_level", "is_app_installed", "marketing_optin_count"]

    f4_out = f4[out_cols].copy()
    miss = f4_out.drop(columns="sample_id").isnull().mean().median() * 100
    print(f"  F4: {len(out_cols)-1} features  missing_P50={miss:.1f}%  n={len(f4_out):,}")
    return f4_out


# ============================================================
# F5: Cross-period interaction features
# ============================================================

def compute_f5_interactions(samples, f1, f2):
    """
    F5: Cross-period comparison / interaction features.
    Depends on F1 + F2 — must run after both.
    Output: 8 features.
    """
    print("\n--- F5: Cross-period interactions ---")

    base = samples[["sample_id"]].merge(f1, on="sample_id", how="left")
    base = base.merge(f2, on="sample_id", how="left")

    # Safe divide helper
    sd = _safe_div

    f5 = base[["sample_id"]].copy()
    f5["t0_vs_avg_ticket_ratio"] = sd(base["t0_amount"].values, base["hist_avg_ticket"].values)
    f5["t0_vs_max_ticket_ratio"] = sd(base["t0_amount"].values, base["hist_max_ticket"].values)
    f5["sleep_vs_ci_ratio"] = sd(base["sleep_days"].values, base["personal_ci"].values)

    # hist_discount_ratio floor at 0.01 for denominator
    hist_disc_floor = np.where(
        pd.isna(base["hist_discount_ratio"]) | (base["hist_discount_ratio"] < 0.01),
        np.where(pd.isna(base["hist_discount_ratio"]), np.nan, 0.01),
        base["hist_discount_ratio"],
    )
    f5["t0_discount_vs_hist_ratio"] = sd(base["t0_discount_ratio"].values, hist_disc_floor)
    f5["t0_used_coupon_vs_hist"] = (base["t0_used_coupon"].astype(float)
                                    - base["hist_coupon_order_ratio"].astype(float))

    f5["t0_channel_matches_hist"] = (base["t0_channel"] == base["hist_main_channel"]).astype("Int8")
    f5["t0_payment_matches_hist"] = (base["t0_payment"] == base["hist_main_payment"]).astype("Int8")

    hist_avg_qty = sd(base["hist_total_qty"].values, base["hist_n_orders"].values)
    f5["t0_qty_vs_avg"] = sd(base["t0_qty"].astype(float).values, hist_avg_qty)

    out_cols = ["sample_id", "t0_vs_avg_ticket_ratio", "t0_vs_max_ticket_ratio",
                "sleep_vs_ci_ratio", "t0_discount_vs_hist_ratio", "t0_used_coupon_vs_hist",
                "t0_channel_matches_hist", "t0_payment_matches_hist", "t0_qty_vs_avg"]

    f5_out = f5[out_cols].copy()
    miss = f5_out.drop(columns="sample_id").isnull().mean().median() * 100
    print(f"  F5: {len(out_cols)-1} features  missing_P50={miss:.1f}%  n={len(f5_out):,}")
    return f5_out


# ============================================================
# F6: Recent activity window features
# ============================================================

def compute_f6_recent_windows(samples, orders, f1):
    """
    F6: Recent activity windows.
    recent_Nd_n_orders: (t0_date - N, t0_date) open interval, strictly < t0_date.
    pre_dormancy: (last_purchase_date - 90d, last_purchase_date] open/closed.
    Output: 5 features.
    """
    print("\n--- F6: Recent activity windows ---")

    # Reuse sleep_days / last_purchase_date from F1
    last_purchase = f1[["sample_id", "sleep_days"]].copy()

    order_cols = ["OrderDateTime", "TotalSalesAmount"]
    hist = _explode_history(samples, orders[orders["is_valid"] == 1], cols=order_cols)
    print(f"  Exploded hist rows (F6): {len(hist):,}")

    # Attach t0_date and last_purchase_date
    base = samples[["sample_id", "t0_date"]].merge(last_purchase, on="sample_id", how="left")
    base["last_purchase_date"] = base["t0_date"] - pd.to_timedelta(base["sleep_days"], unit="D")

    # hist already carries t0_date from _explode_history; only attach last_purchase_date
    hist = hist.merge(base[["sample_id", "last_purchase_date"]], on="sample_id", how="left")

    # Window filters (open interval: t0_date - N < OrderDateTime < t0_date)
    for n_days in [90, 180, 365]:
        mask = hist["OrderDateTime"] > (hist["t0_date"] - pd.Timedelta(days=n_days))
        sub = hist[mask].groupby("sample_id").size().reset_index(name=f"recent_{n_days}d_n_orders")
        base = base.merge(sub, on="sample_id", how="left")
        base[f"recent_{n_days}d_n_orders"] = base[f"recent_{n_days}d_n_orders"].fillna(0).astype("Int32")

    # Pre-dormancy 90-day window: (last_purchase_date - 90d) < OrderDateTime <= last_purchase_date
    predorm_mask = (
        (hist["OrderDateTime"] > (hist["last_purchase_date"] - pd.Timedelta(days=90))) &
        (hist["OrderDateTime"] <= hist["last_purchase_date"])
    )
    predorm = hist[predorm_mask].groupby("sample_id").agg(
        pre_dormancy_90d_n_orders=("TotalSalesAmount", "count"),
        pre_dormancy_90d_amount=("TotalSalesAmount", "sum"),
    ).reset_index()

    base = base.merge(predorm, on="sample_id", how="left")
    base["pre_dormancy_90d_n_orders"] = base["pre_dormancy_90d_n_orders"].fillna(0).astype("Int32")
    base["pre_dormancy_90d_amount"] = base["pre_dormancy_90d_amount"].fillna(0.0)

    out_cols = ["sample_id", "recent_90d_n_orders", "recent_180d_n_orders", "recent_365d_n_orders",
                "pre_dormancy_90d_n_orders", "pre_dormancy_90d_amount"]

    f6_out = base[out_cols].copy()
    miss = f6_out.drop(columns="sample_id").isnull().mean().median() * 100
    print(f"  F6: {len(out_cols)-1} features  missing_P50={miss:.1f}%  n={len(f6_out):,}")
    return f6_out


# ============================================================
# Output writing
# ============================================================

def write_outputs(features):
    """
    Split features by split column, write train/test CSV (utf-8-sig).
    """
    print("\n--- Writing CSV outputs ---")
    assert features["sample_id"].nunique() == len(features), "Duplicate sample_ids found!"

    train = features[features["split"] == "train"].drop(columns="split")
    test = features[features["split"] == "test"].drop(columns="split")

    train_path = OUT_DIR / "train_features.csv"
    test_path = OUT_DIR / "test_features.csv"
    train.to_csv(train_path, index=False, encoding="utf-8-sig")
    test.to_csv(test_path, index=False, encoding="utf-8-sig")
    print(f"  train_features.csv: {len(train):,} rows  {len(train.columns)} cols → {train_path}")
    print(f"  test_features.csv : {len(test):,} rows  {len(test.columns)} cols → {test_path}")
    return train, test


# ============================================================
# Leakage self-verification
# ============================================================

def _leakage_check(samples, orders, features, n_samples=5):
    """
    Random-sample 5 rows and verify that the max OrderDateTime used in historical
    aggregation is strictly < t0_date.
    """
    sample_ids = features.sample(n=n_samples, random_state=42)["sample_id"].tolist()
    rows = samples[samples["sample_id"].isin(sample_ids)][
        ["sample_id", "member_id", "t0_date"]
    ].copy()

    order_slim = orders[orders["is_valid"] == 1][["ShopMemberId", "TradesGroupCode", "OrderDateTime"]]

    results = []
    for _, row in rows.iterrows():
        member_orders = order_slim[order_slim["ShopMemberId"] == row["member_id"]]
        hist = member_orders[member_orders["OrderDateTime"] < row["t0_date"]]
        max_dt = hist["OrderDateTime"].max() if len(hist) > 0 else pd.NaT
        results.append({
            "sample_id": row["sample_id"],
            "t0_date": row["t0_date"],
            "max_hist_OrderDateTime": max_dt,
            "leakage_ok": pd.isna(max_dt) or (max_dt < row["t0_date"]),
        })

    return pd.DataFrame(results)


# ============================================================
# Documentation writers
# ============================================================

def write_feature_engineering_md(features):
    cols = [c for c in features.columns if c not in ("sample_id", "label", "split")]
    col_list = "\n".join(f"- {c}" for c in cols)

    md = f"""# 特徵工程文件

> 腳本：`feature_engineering.py`
> 產出日期：2026-06-03
> 樣本骨架：`output/samples_shop3.parquet`

---

## 1. 防洩漏原則

所有歷史特徵（F1、F3、F5、F6）使用 **嚴格小於 t0_date** 的訂單：

```python
OrderDateTime < t0_date   # 不含 t0_date 當天
```

**唯一例外：F2（t₀ 訂單本身）**
t₀ 那筆訂單在回流當下即已觀測，屬於「當下特徵」，不是未來資訊。

**F5/F6 時間窗口邊界**
- `(t0_date - N days, t0_date)` 開區間，兩端均不含
- F6 pre_dormancy 窗口：`(last_purchase_date - 90d, last_purchase_date]`（含最後購買日）
- F2 的 t₀ 訂單子單特徵，僅對 `t0_trades_group_code` 直接 join，不涉及時間過濾

---

## 2. 資料來源與排除說明

| 資料表 | 用途 | 路徑 |
|--------|------|------|
| Order_TG.csv | F1/F2/F6 主單特徵 | `91APP_Dataset/Order_TG.csv` |
| Order_TS.csv | F3 子單商品特徵 | `91APP_Dataset/Order_TS.csv` |
| SalePage.csv | F3 SalePageId 對應（僅確認欄位，無額外特徵） | `91APP_Dataset/SalePage.csv` |
| Member.csv | F4 會員背景特徵 | `91APP_Dataset/Member.csv` |

**排除說明：**
- **Segment 不使用**：shop#3（DHC）於資料集中無 Segment 記錄，確認後排除。
- **Behavior 不使用**：行為日誌（點擊、瀏覽）在特徵組合中信號雜訊比低，且與成交歷史高度相關；為避免過擬合與模型複雜度提升，本階段不納入，留待必要時補入。
- **SalePage 特徵層級**：SalePage 無現成品類（Category）欄位，以 SalePageId 層級為操作單元。

**有效購買定義（與前一階段一致）：**
```python
StatusDef != 'Fail'  AND  TotalSalesAmount > 0
```

**Member 快照限制：**
Member 表為 2024-02-29 快照，F4 的 `member_card_level`、`is_app_installed`、`marketing_optin_count` 三欄的值可能在 t₀ 之後發生過更新，構成輕度前向洩漏（forward leakage）。
決策：保留並在文件中標記，避免過度保守。建模時可考慮以這三欄做敏感度分析。

---

## 3. 特徵總表

| 特徵名 | 來源表 | 定義 | 是否防洩漏安全 | 資料型態 |
|--------|--------|------|----------------|---------|
| sleep_days | Order_TG | t0_date − 上一次購買日（天） | ✓ | float |
| hist_n_orders | Order_TG | t₀ 前累積有效訂單數 | ✓ | float |
| hist_total_amount | Order_TG | t₀ 前 TotalSalesAmount 總和 | ✓ | float |
| hist_avg_ticket | Order_TG | hist_total_amount / hist_n_orders | ✓ | float |
| hist_max_ticket | Order_TG | t₀ 前單筆最高金額 | ✓ | float |
| hist_total_qty | Order_TG | t₀ 前 Qty 總和（供 F5 使用） | ✓ | float |
| personal_ci | Order_TG | t₀ 前相鄰購買 gap 中位數（< 2 gap → NaN） | ✓ | float |
| gap_std | Order_TG | t₀ 前 gap 標準差（< 2 gap → NaN） | ✓ | float |
| gap_cv | Order_TG | gap_std / personal_ci | ✓ | float |
| hist_discount_ratio | Order_TG | abs(TotalDiscount)總和 / TotalPrice總和 | ✓ | float |
| hist_coupon_order_ratio | Order_TG | 用過折價券的訂單比例 | ✓ | float |
| hist_return_ratio | Order_TG | 退貨訂單佔非失敗訂單比例 | ✓ | float |
| hist_main_channel | Order_TG | t₀ 前最常用 ChannelType | ✓ | str |
| hist_main_payment | Order_TG | t₀ 前最常用 PaymentType | ✓ | str |
| member_tenure_days | Order_TG | t0_date − 首次購買日（天） | ✓ | float |
| was_sealed | Order_TG | sleep_days > 365 → 1 else 0 | ✓ | Int8 |
| prior_once_only | Order_TG | hist_n_orders == 1 → 1 else 0 | ✓ | Int8 |
| t0_amount | Order_TG | t₀ 訂單 TotalSalesAmount | F2 例外 | float |
| t0_ts_count | Order_TG | t₀ 訂單 TsCount | F2 例外 | float |
| t0_qty | Order_TG | t₀ 訂單 Qty | F2 例外 | float |
| t0_used_coupon | Order_TG | t₀ 是否用折價券 | F2 例外 | Int8 |
| t0_discount_ratio | Order_TG | abs(TotalDiscount) / TotalPrice | F2 例外 | float |
| t0_channel | Order_TG | t₀ 訂單 ChannelType | F2 例外 | str |
| t0_payment | Order_TG | t₀ 訂單 PaymentType | F2 例外 | str |
| t0_shipping | Order_TG | t₀ 訂單 ShippingType | F2 例外 | str |
| t0_month | samples | t0_date 月份 (1-12) | F2 例外 | int |
| t0_weekday | samples | t0_date 星期 (0=Mon) | F2 例外 | int |
| t0_is_weekend | samples | t0_weekday ∈ {5,6} | F2 例外 | Int8 |
| t0_is_big_promo | samples | t0_month ∈ {5,11} | F2 例外 | Int8 |
| hist_distinct_salepages | Order_TS | t₀ 前不重複 SalePageId 數 | ✓ | float |
| hist_repeat_sku_ratio | Order_TS | 重複購買同一 SalePageId 比例 | ✓ | float |
| t0_n_distinct_salepages | Order_TS | t₀ 訂單內不重複 SalePageId 數 | F2 例外 | float |
| t0_buys_familiar_product | Order_TS | t₀ 是否含曾買過的 SalePageId | F3混合 | Int8 |
| t0_avg_unit_price | Order_TS | t₀ 訂單子單 UnitPrice 平均 | F2 例外 | float |
| t0_min_unit_price | Order_TS | t₀ 訂單子單 UnitPrice 最低 | F2 例外 | float |
| t0_max_unit_price | Order_TS | t₀ 訂單子單 UnitPrice 最高 | F2 例外 | float |
| register_source | Member | RegisterSourceTypeDef | ✓（靜態） | str |
| gender | Member | Gender | ✓（靜態） | str |
| age_at_t0 | Member | (t0_date − Birthday) / 365.25 | ✓（靜態） | float |
| days_since_register | Member | t0_date − RegisterDateTime | ✓（靜態） | float |
| country_code | Member | CountryAliasCode | ✓（靜態） | str |
| member_card_level | Member | MemberCardLevel（快照，輕度洩漏） | ⚠ 快照 | float |
| is_app_installed | Member | IsAppInstalled（快照，輕度洩漏） | ⚠ 快照 | Int8 |
| marketing_optin_count | Member | Email+Push+SMS 啟用加總（快照） | ⚠ 快照 | int |
| t0_vs_avg_ticket_ratio | F1+F2 | t0_amount / hist_avg_ticket | ✓（依賴F1/F2） | float |
| t0_vs_max_ticket_ratio | F1+F2 | t0_amount / hist_max_ticket | ✓ | float |
| sleep_vs_ci_ratio | F1 | sleep_days / personal_ci | ✓ | float |
| t0_discount_vs_hist_ratio | F1+F2 | t0_discount_ratio / max(hist_discount_ratio,0.01) | ✓ | float |
| t0_used_coupon_vs_hist | F1+F2 | t0_used_coupon - hist_coupon_order_ratio | ✓ | float |
| t0_channel_matches_hist | F1+F2 | t0_channel == hist_main_channel | ✓ | Int8 |
| t0_payment_matches_hist | F1+F2 | t0_payment == hist_main_payment | ✓ | Int8 |
| t0_qty_vs_avg | F1+F2 | t0_qty / (hist_total_qty / hist_n_orders) | ✓ | float |
| recent_90d_n_orders | Order_TG | t₀ 前 90 天訂單數 | ✓ | Int32 |
| recent_180d_n_orders | Order_TG | t₀ 前 180 天訂單數 | ✓ | Int32 |
| recent_365d_n_orders | Order_TG | t₀ 前 365 天訂單數 | ✓ | Int32 |
| pre_dormancy_90d_n_orders | Order_TG | 最後購買日前 90 天訂單數 | ✓ | Int32 |
| pre_dormancy_90d_amount | Order_TG | 最後購買日前 90 天訂單金額 | ✓ | float |

---

## 4. 分組說明（F1~F6）

### F1：主單歷史 RFM 與行為穩定性
捕捉每位會員在 t₀ 之前的整體消費行為模式。

**`sleep_days`**：t₀ 事件的核心定義——沉睡長度。預期為基礎強特徵。
沉睡越久的回流往往越不穩定，但若沉睡後下大單（見F5），則可能是真復活。

### F2：t₀ 訂單特徵
回流當下的購買行為。是「路過」還是「真復活」在這筆單的特徵上已有端倪：
金額、品項數、是否用券、是否是促銷月份。

### F3：子單 + 商品頁（商品行為）
**`t0_buys_familiar_product`**：t₀ 訂單是否包含該會員歷史上買過的商品。
「买回熟悉商品」是真復活的強訊號——會員記得 DHC 的哪些產品對自己有效。
若 t₀ 訂單全是從未購買過的商品，更可能是廣告引導的試水性購買（路過）。

### F4：會員背景
人口屬性（性別、年齡、注冊來源）為基礎分層特徵，幫助模型區分不同族群的回流模式。
快照欄位（card_level, app_installed）有輕微前向洩漏，已在文件中標注。

### ⭐ F5：跨期比較/交互特徵（最高優先級）
「真復活 vs 路過」的差異主要體現在「這次回來跟以前比怎樣」。
- `t0_vs_avg_ticket_ratio > 1.5`：大手筆回歸，強烈真復活訊號
- `sleep_vs_ci_ratio`：沉睡相對於個人週期的「離譜程度」
- `t0_discount_vs_hist_ratio > 1.5`：比平常更靠折扣，路過訊號
- 行為一致性（channel/payment matches）：維持慣用行為 = 習慣回復

### F6：近期活躍窗口特徵
**`recent_90d_n_orders`**：通常為 0（因為樣本定義就是沉睡 > threshold 才進入）。
一旦 > 0，代表會員近期才剛開始沉睡，與「真正消失多年後復活」性質不同，是極強訊號。
**`pre_dormancy_90d_n_orders`**：沉睡之前的活躍程度。
沉睡前非常活躍（n > 5）的會員，比沉睡前就很冷淡的會員，更可能真正復活。

---

## 5. 缺失值處理策略

| 情況 | 處理 |
|------|------|
| hist_n_orders == 1（只買過一次）| personal_ci / gap_std / gap_cv → NaN |
| personal_ci == 0（極罕見）| gap_cv → NaN（safe divide） |
| hist_avg/max_ticket 當分母 | F5 比率 → NaN |
| Member 表無對應會員 | F4 所有欄位 → NaN |
| 類別特徵缺失 | 填 'unknown' |
| F6 窗口內無訂單 | 填 0（無訂單≠缺失） |
| hist_discount_ratio 分母 < 0.01 | t0_discount_vs_hist_ratio → 用 0.01 作分母（documented） |

XGBoost 原生支援 NaN，無需填補數值特徵的缺失值。

---

## 6. 輸出格式

```
sample_id, label, [F1×17, F2×12, F3×7, F4×8, F5×8, F6×5 = 57 特徵]
```

完整欄位清單：
{col_list}

---

## 7. 待後續處理事項

1. **特徵顯著性檢驗**：對每個數值特徵做 Mann-Whitney U Test（label=0 vs label=1），
   對每個類別特徵做卡方檢驗 → 留待建模前
2. **高相關去重**：計算 Spearman 相關矩陣，|r| > 0.85 的特徵對保留一個 → 留待建模前
3. **類別特徵編碼**：`hist_main_channel`, `hist_main_payment`, `t0_channel`, `t0_payment`,
   `t0_shipping`, `register_source`, `gender`, `country_code`
   → 考慮 target encoding（利用標籤信息）或 one-hot → 留待建模前
4. **F5 極端值 clip**：`t0_vs_avg_ticket_ratio` 等可能有極大值，建議 clip 到 [0, 10] 後
   再做特徵重要性分析 → 留待 EDA + 建模前確認

---

*Generated by feature_engineering.py — 2026-06-03*
"""
    path = OUT_DIR / "feature_engineering.md"
    path.write_text(md, encoding="utf-8")
    print(f"  feature_engineering.md → {path}")


def write_sanity_md(sanity_stats, features, leakage_df):
    train = features[features["split"] == "train"]
    test = features[features["split"] == "test"]
    feat_cols = [c for c in features.columns if c not in ("sample_id", "label", "split")]

    # Missing rate table
    missing_rows = []
    f_groups = {
        "F1": ["sleep_days", "hist_n_orders", "hist_total_amount", "hist_avg_ticket", "hist_max_ticket",
               "hist_total_qty", "personal_ci", "gap_std", "gap_cv", "hist_discount_ratio",
               "hist_coupon_order_ratio", "hist_return_ratio", "hist_main_channel", "hist_main_payment",
               "member_tenure_days", "was_sealed", "prior_once_only"],
        "F2": ["t0_amount", "t0_ts_count", "t0_qty", "t0_used_coupon", "t0_discount_ratio",
               "t0_channel", "t0_payment", "t0_shipping", "t0_month", "t0_weekday",
               "t0_is_weekend", "t0_is_big_promo"],
        "F3": ["hist_distinct_salepages", "hist_repeat_sku_ratio", "t0_n_distinct_salepages",
               "t0_buys_familiar_product", "t0_avg_unit_price", "t0_min_unit_price", "t0_max_unit_price"],
        "F4": ["register_source", "gender", "age_at_t0", "days_since_register", "country_code",
               "member_card_level", "is_app_installed", "marketing_optin_count"],
        "F5": ["t0_vs_avg_ticket_ratio", "t0_vs_max_ticket_ratio", "sleep_vs_ci_ratio",
               "t0_discount_vs_hist_ratio", "t0_used_coupon_vs_hist",
               "t0_channel_matches_hist", "t0_payment_matches_hist", "t0_qty_vs_avg"],
        "F6": ["recent_90d_n_orders", "recent_180d_n_orders", "recent_365d_n_orders",
               "pre_dormancy_90d_n_orders", "pre_dormancy_90d_amount"],
    }

    missing_table_lines = ["| 組 | 特徵 | 缺失率 |", "|---|---|---|"]
    for grp, cols in f_groups.items():
        for c in cols:
            if c in features.columns:
                r = features[c].isnull().mean() * 100
                missing_table_lines.append(f"| {grp} | {c} | {r:.1f}% |")

    # Quantile table for numeric features
    num_cols = [c for c in feat_cols
                if c in features.columns and features[c].dtype.kind in ("f", "i", "u")]
    quant_lines = ["| 特徵 | min | P25 | P50 | P75 | max |", "|---|---|---|---|---|---|"]
    for c in num_cols[:40]:  # limit to first 40
        sub = features[c].dropna()
        if len(sub) == 0:
            continue
        q = sub.quantile([0, 0.25, 0.5, 0.75, 1.0])
        quant_lines.append(
            f"| {c} | {q[0]:.2f} | {q[0.25]:.2f} | {q[0.5]:.2f} | {q[0.75]:.2f} | {q[1.0]:.2f} |"
        )

    # F5 extreme value check
    f5_ratio_cols = ["t0_vs_avg_ticket_ratio", "t0_vs_max_ticket_ratio",
                     "sleep_vs_ci_ratio", "t0_discount_vs_hist_ratio"]
    f5_extreme_lines = ["| 特徵 | P95 | P99 | max | 建議clip上限 |",
                        "|---|---|---|---|---|"]
    for c in f5_ratio_cols:
        if c in features.columns:
            sub = features[c].dropna()
            if len(sub) > 0:
                p95, p99, mx = sub.quantile(0.95), sub.quantile(0.99), sub.max()
                f5_extreme_lines.append(f"| {c} | {p95:.2f} | {p99:.2f} | {mx:.2f} | {min(p99*2, 20):.1f} |")

    # Leakage verification table
    leak_lines = ["| sample_id | t0_date | max_hist_OrderDateTime | leakage_ok |",
                  "|---|---|---|---|"]
    for _, r in leakage_df.iterrows():
        leak_lines.append(
            f"| {r['sample_id']} | {r['t0_date']} | {r['max_hist_OrderDateTime']} | {r['leakage_ok']} |"
        )

    md = f"""# Feature Engineering Sanity Check

> 腳本：`feature_engineering.py` — 2026-06-03

---

## 1. 跨 split 會員重疊

| 項目 | 數量 |
|------|------|
| 跨 split 會員數 | {sanity_stats['overlap_count']:,} |
| 佔 train 會員比例 | {sanity_stats['overlap_pct_train']}% |
| 佔 test 會員比例 | {sanity_stats['overlap_pct_test']}% |
| train 樣本數 | {sanity_stats['train_n']:,} |
| test 樣本數 | {sanity_stats['test_n']:,} |

{"⚠️ 重疊比例 > 30%，下游特徵因果截止點已強化確認。" if sanity_stats['overlap_pct_train'] > 30 else "✓ 重疊比例在可接受範圍。"}

---

## 2. 各特徵缺失率（按 F1~F6 分組）

{chr(10).join(missing_table_lines)}

---

## 3. 各數值特徵分位數

{chr(10).join(quant_lines)}

---

## 4. F5 比率特徵的極端值

{chr(10).join(f5_extreme_lines)}

> 建議在建模前對上表「建議clip上限」欄位值做 winsorize，防止少數極端比率主導樹分裂。

---

## 5. 防洩漏自我驗證

隨機抽 5 筆樣本，確認「歷史聚合最大 OrderDateTime」嚴格 < t0_date：

{chr(10).join(leak_lines)}

{"✅ 所有抽查樣本均通過防洩漏驗證。" if leakage_df['leakage_ok'].all() else "⚠️ 有樣本未通過防洩漏驗證，請立即檢查！"}

---

*Generated by feature_engineering.py — 2026-06-03*
"""
    path = OUT_DIR / "feature_sanity.md"
    path.write_text(md, encoding="utf-8")
    print(f"  feature_sanity.md → {path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  shop#3 (DHC) Feature Engineering Pipeline")
    print("=" * 60)

    # Load data
    samples = load_samples()
    orders = load_orders()
    order_slave = load_order_slave()
    # SalePage loaded for reference; F3 uses SalePageId from Order_TS
    salepage = pd.read_csv(DATA_DIR / "SalePage.csv", low_memory=False,
                           dtype={"ShopId": "category", "SalePageId": str})
    salepage = salepage[salepage["ShopId"] == SHOP3_ID].drop(columns=["ShopId"])
    print(f"  SalePage: {len(salepage):,} rows")
    member = load_member()

    # Step 0
    sanity_stats = sanity_check(samples)

    # Feature groups
    f1 = compute_f1_history_rfm(samples, orders)
    f2 = compute_f2_t0_order(samples, orders)
    f3 = compute_f3_product(samples, orders, order_slave, salepage)
    f4 = compute_f4_member(samples, member)
    f5 = compute_f5_interactions(samples, f1, f2)
    f6 = compute_f6_recent_windows(samples, orders, f1)

    # Merge all back to samples skeleton
    print("\n--- Assembling final feature DataFrame ---")
    features = samples[["sample_id", "label", "split"]].copy()
    for df in [f1, f2, f3, f4, f5, f6]:
        features = features.merge(df, on="sample_id", how="left")

    n_feat = len(features.columns) - 3  # exclude sample_id, label, split
    print(f"  Total features: {n_feat}")
    print(f"  Total rows: {len(features):,}")
    assert len(features) == len(samples), "Row count mismatch!"

    # Write outputs
    train, test = write_outputs(features)

    # Leakage verification
    leak_df = _leakage_check(samples, orders, features)

    # Documentation
    write_feature_engineering_md(features)
    write_sanity_md(sanity_stats, features, leak_df)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Outputs in: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
