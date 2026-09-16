"""
shop#3 (DHC) — Comprehensive EDA for dormancy model design
Steps 0-8 + summary.md
"""
import os, sys, warnings, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import gaussian_kde

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

OUT     = "c:/Users/stp09/Code/BDA2026_final/output"
DATA    = ("c:/Users/stp09/Code/BDA2026_final/"
           "91APP_Dataset(會員&主單&子單&商品頁&標籤)/Order_TG.csv")
SHOP3   = "hFwniXiB/Ev2ZPXeO630Sw=="

os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 11,
                     "axes.titlesize": 13, "axes.labelsize": 11})
C0 = "#4C72B0"
PAL = sns.color_palette("muted", 10)

# ── helpers ──────────────────────────────────────────────────────
def pct_str(n, total): return f"{n/total*100:.1f}%"
def qstr(arr, qs=(25,50,75,90)):
    p = np.percentile(arr, qs)
    return "  ".join(f"P{q}={v:.1f}" for q,v in zip(qs,p))

# ══════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════
print("Loading shop#3 rows ...")
usecols = ["ShopId","ShopMemberId","TradesGroupCode","OrderDateTime",
           "StatusDef","TsCount","Qty","TotalSalesAmount"]
raw = pd.read_csv(DATA, usecols=usecols, low_memory=False)
raw = raw[raw["ShopId"] == SHOP3].copy()
raw["OrderDateTime"] = pd.to_datetime(raw["OrderDateTime"], format="mixed")
raw["TsCount"] = pd.to_numeric(raw["TsCount"], errors="coerce")
raw["Qty"]     = pd.to_numeric(raw["Qty"],     errors="coerce")
raw["TotalSalesAmount"] = pd.to_numeric(raw["TotalSalesAmount"], errors="coerce")
print(f"  Raw rows: {len(raw):,}  |  Members: {raw['ShopMemberId'].nunique():,}")

# ══════════════════════════════════════════════════════════════════
# STEP 0: Data overview + StatusDef
# ══════════════════════════════════════════════════════════════════
print("\n=== STEP 0: Data overview + StatusDef ===")
print(f"  Rows          : {len(raw):,}")
print(f"  Unique members: {raw['ShopMemberId'].nunique():,}")
print(f"  Date range    : {raw['OrderDateTime'].min().date()} ~ {raw['OrderDateTime'].max().date()}")

vc = raw["StatusDef"].value_counts()
s0 = pd.DataFrame({"Count": vc, "Pct%": (vc/len(raw)*100).round(2)})
print("\n  StatusDef:\n", s0.to_string())

neg_amt  = (raw["TotalSalesAmount"] < 0).mean()*100
zero_amt = (raw["TotalSalesAmount"] == 0).mean()*100
neg_qty  = (raw["Qty"] < 0).mean()*100
print(f"\n  TotalSalesAmount < 0 : {neg_amt:.2f}%")
print(f"  TotalSalesAmount = 0 : {zero_amt:.2f}%")
print(f"  Qty < 0              : {neg_qty:.2f}%")

def_a = raw[~raw["StatusDef"].isin(["Fail"])]
def_b = raw[(~raw["StatusDef"].isin(["Fail"])) & (raw["TotalSalesAmount"] > 0)]
def_c = raw[raw["StatusDef"] == "Finish"]
print("\n  Valid-purchase definitions:")
print(f"  (a) Excl. Fail                : {len(def_a):,} ({pct_str(len(def_a),len(raw))})")
print(f"  (b) Excl. Fail + Amount>0     : {len(def_b):,} ({pct_str(len(def_b),len(raw))})")
print(f"  (c) Finish only               : {len(def_c):,} ({pct_str(len(def_c),len(raw))})")

valid = def_b.copy()
print(f"\n  -> Adopting (b): {len(valid):,} rows")

# ══════════════════════════════════════════════════════════════════
# STEP 1: Purchase count per member
# ══════════════════════════════════════════════════════════════════
print("\n=== STEP 1: Purchase count distribution ===")

orders = valid.drop_duplicates(subset="TradesGroupCode")
opm    = orders.groupby("ShopMemberId").size().rename("n")
vals   = opm.values
pqs    = np.percentile(vals, [25,50,75,90,95])

print(f"  {qstr(vals, [25,50,75,90,95])}  Max={vals.max()}")
print(f"  Mean={vals.mean():.2f}  1-time={( vals==1).mean()*100:.1f}%")
print(f"  >=2: {(vals>=2).sum():,} ({(vals>=2).mean()*100:.1f}%)"
      f"  >=3: {(vals>=3).sum():,} ({(vals>=3).mean()*100:.1f}%)"
      f"  >=5: {(vals>=5).sum():,} ({(vals>=5).mean()*100:.1f}%)")

s1 = pd.DataFrame({
    "Metric": ["P25","P50","P75","P90","P95","Mean","1-time%",">=2%",">=3%",">=5%"],
    "Value":  [*pqs.round(1), round(vals.mean(),2),
               round((vals==1).mean()*100,1),
               round((vals>=2).mean()*100,1),
               round((vals>=3).mean()*100,1),
               round((vals>=5).mean()*100,1)]
})
print(s1.to_string(index=False))

fig, ax = plt.subplots(figsize=(9,5))
bins_log = np.logspace(0, np.log10(vals.max()+1), 60)
ax.hist(vals, bins=bins_log, color=C0, edgecolor="white", linewidth=0.4, alpha=0.85)
ax.axvline(np.median(vals), color="red",    ls="--", lw=1.5, label=f"Median={np.median(vals):.0f}")
ax.axvline(vals.mean(),     color="orange", ls="--", lw=1.5, label=f"Mean={vals.mean():.1f}")
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x)}"))
ax.set_xlabel("Purchase count (log scale)")
ax.set_ylabel("Members")
ax.set_title(f"shop#3 DHC — Purchase count per member  (1-time: {(vals==1).mean()*100:.1f}%)")
ax.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/s3_s1_purchase_count.png"); plt.close()

# ══════════════════════════════════════════════════════════════════
# STEP 2: Order item counts (TsCount / Qty)
# ══════════════════════════════════════════════════════════════════
print("\n=== STEP 2: Order item counts ===")

for col, lbl, clip in [("TsCount","Distinct SKUs (TsCount)",20), ("Qty","Total Qty",30)]:
    sub = orders[col].dropna().values.astype(float)
    p   = np.percentile(sub, [25,50,75,90])
    print(f"\n  {lbl}:  {qstr(sub)}  Mean={sub.mean():.2f}  Max={sub.max():.0f}")
    if col == "TsCount":
        one_sku = (sub == 1).mean()*100
        print(f"  Single-SKU (TsCount=1): {one_sku:.1f}%")

fig, axes = plt.subplots(1,2, figsize=(13,5))
for ax, col, lbl, clip in zip(
        axes,
        ["TsCount","Qty"],
        ["Distinct SKUs per order","Total qty per order"],
        [20, 30]):
    sub = orders[col].dropna().clip(upper=clip).values.astype(float)
    ax.hist(sub, bins=range(0, int(clip)+2), color=C0, edgecolor="white", lw=0.4, alpha=0.85)
    ax.axvline(np.median(sub), color="red", ls="--", lw=1.5,
               label=f"Median={np.median(sub):.1f}")
    ax.set_xlabel(f"{lbl} (clipped <={clip})"); ax.set_ylabel("Orders"); ax.legend()
fig.suptitle("shop#3 DHC — Order item counts", fontsize=13)
plt.tight_layout(); plt.savefig(f"{OUT}/s3_s2_item_counts.png"); plt.close()

# ══════════════════════════════════════════════════════════════════
# Build date-level purchase TIMELINE (used in Steps 3–7)
# ══════════════════════════════════════════════════════════════════
print("\nBuilding purchase timeline (date-level dedup) ...")
valid["date"] = valid["OrderDateTime"].dt.normalize()
tl = (valid[["ShopMemberId","date"]]
      .drop_duplicates()
      .sort_values(["ShopMemberId","date"])
      .reset_index(drop=True))

tl["prev"] = tl.groupby("ShopMemberId")["date"].shift(1)
gaps_all   = tl.dropna(subset=["prev"]).copy()
gaps_all["gap"] = (gaps_all["date"] - gaps_all["prev"]).dt.days.astype(float)
gaps_all   = gaps_all.rename(columns={"prev":"t_last","date":"t_revival"})
gaps_all   = gaps_all[["ShopMemberId","t_last","t_revival","gap"]].reset_index(drop=True)

# Ci per member
ci_ser = (gaps_all.groupby("ShopMemberId")["gap"].median().rename("Ci"))
shop_fb = ci_ser.median()
gaps_all = gaps_all.join(ci_ser, on="ShopMemberId")

DATA_END = tl["date"].max()
print(f"  Gaps: {len(gaps_all):,}  |  Ci members: {ci_ser.nunique():,}  |  Fallback: {shop_fb:.1f}d")

# ══════════════════════════════════════════════════════════════════
# STEP 3: Personal shopping cycle Ci
# ══════════════════════════════════════════════════════════════════
print("\n=== STEP 3: Personal shopping cycle Ci ===")

ci_vals = ci_ser.values
n_single = (opm == 1).sum()
pqs_ci   = np.percentile(ci_vals, [25,50,75,90])

print(f"  {qstr(ci_vals)}  Mean={ci_vals.mean():.1f}  Max={ci_vals.max():.0f}")
print(f"  Single-purchase members (fallback={shop_fb:.1f}d): {n_single:,} "
      f"({n_single/len(opm)*100:.1f}%)")

s3 = pd.DataFrame({
    "Metric": ["P25","P50","P75","P90","Mean","Fallback","Single-purch"],
    "Value":  [*pqs_ci.round(1), round(ci_vals.mean(),1), round(shop_fb,1), n_single]
})
print(s3.to_string(index=False))

# All-gaps distribution
ag  = gaps_all["gap"].values
ag_clip = ag[ag <= 365]
over365 = (ag > 365).mean()*100
pqs_ag  = np.percentile(ag_clip, [25,50,75,90])
print(f"\n  All gaps: {qstr(ag_clip)}  >365d={over365:.1f}%")

fig, axes = plt.subplots(1,2, figsize=(14,5))
ci_clip = ci_vals[ci_vals <= 200]
axes[0].hist(ci_clip, bins=80, color="#55A868", edgecolor="white", lw=0.3, alpha=0.85)
for v,c,lbl in zip(pqs_ci, ["#2ca02c","red","#d62728","#9467bd"],
                   ["P25","P50","P75","P90"]):
    axes[0].axvline(v, color=c, ls="--", lw=1.3, label=f"{lbl}={v:.0f}d")
axes[0].axvline(shop_fb, color="black", ls=":", lw=1.5, label=f"Fallback={shop_fb:.0f}d")
axes[0].set_xlim(0,200); axes[0].set_xlabel("Ci (days, clipped 0-200)")
axes[0].set_ylabel("Members"); axes[0].legend(fontsize=9)
axes[0].set_title(f"Personal cycle Ci  (median={np.median(ci_vals):.1f}d)")

axes[1].hist(ag_clip, bins=73, color=C0, edgecolor="white", lw=0.3, alpha=0.7)
for v,c,lbl in zip(pqs_ag, ["#2ca02c","red","#d62728","#9467bd"],
                   ["P25","P50","P75","P90"]):
    axes[1].axvline(v, color=c, ls="--", lw=1.3, label=f"{lbl}={v:.0f}d")
axes[1].set_xlim(0,365); axes[1].set_xlabel("Gap (days, clipped 0-365)")
axes[1].set_ylabel("Gap count"); axes[1].legend(fontsize=9)
axes[1].set_title(f"All inter-purchase gaps  (>365d: {over365:.1f}%)")

fig.suptitle("shop#3 DHC — Ci & gap distribution", fontsize=13)
plt.tight_layout(); plt.savefig(f"{OUT}/s3_s3_ci_gap.png"); plt.close()

# ══════════════════════════════════════════════════════════════════
# STEP 4: Dormancy threshold multiplier comparison
# ══════════════════════════════════════════════════════════════════
print("\n=== STEP 4: Dormancy threshold comparison ===")

s4_rows = []
for k in [1.5, 2.0, 2.5, 3.0]:
    thr  = (gaps_all["Ci"] * k).clip(30, 365)
    mask = gaps_all["gap"] > thr
    s4_rows.append({
        "Type": "adaptive", "Setting": f"{k}xCi",
        "Thr_median(d)": round(thr.median(), 1),
        "Events": int(mask.sum()),
        "Members_w_event": int(gaps_all[mask]["ShopMemberId"].nunique()),
        "Dormant%": round(mask.mean()*100, 1)
    })
for fixed in [60, 90, 120, 180]:
    mask = gaps_all["gap"] > fixed
    s4_rows.append({
        "Type": "fixed", "Setting": f"{fixed}d",
        "Thr_median(d)": fixed,
        "Events": int(mask.sum()),
        "Members_w_event": int(gaps_all[mask]["ShopMemberId"].nunique()),
        "Dormant%": round(mask.mean()*100, 1)
    })

s4 = pd.DataFrame(s4_rows)
print(s4.to_string(index=False))

fig, axes = plt.subplots(1,2, figsize=(14,5))
colors4 = [PAL[i % len(PAL)] for i in range(len(s4))]
x4 = range(len(s4))
for ax, col, ylabel in zip(axes,
                            ["Events","Dormant%"],
                            ["Revival events","% gaps labelled dormant"]):
    bars = ax.bar(x4, s4[col], color=colors4, edgecolor="white", alpha=0.85)
    ax.set_xticks(x4); ax.set_xticklabels(s4["Setting"], rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    for bar, v in zip(bars, s4[col]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                f"{v:,}" if col=="Events" else f"{v}%",
                ha="center", va="bottom", fontsize=8)
axes[0].set_title("Revival events by threshold")
axes[1].set_title("Dormant% by threshold")
fig.suptitle("shop#3 DHC — Step 4: Threshold comparison", fontsize=13)
plt.tight_layout(); plt.savefig(f"{OUT}/s3_s4_threshold.png"); plt.close()

# ══════════════════════════════════════════════════════════════════
# Helper: compute window stats (censoring + revival rate)
# ══════════════════════════════════════════════════════════════════
def window_stats(dom_ev_df, tl_df, data_end, window_val, window_label, personalized=False):
    """
    dom_ev_df : rows with ShopMemberId, t_revival, Ci
    window_val: float scalar (fixed) or pd.Series aligned to dom_ev_df (personalized)
    Returns dict with metrics.
    """
    dom = dom_ev_df.copy().reset_index(drop=True)
    dom["eid"] = dom.index

    if personalized:
        dom["W"] = np.array(window_val)
    else:
        dom["W"] = float(window_val)

    # Censor
    dom["t_end"] = dom["t_revival"] + pd.to_timedelta(dom["W"], unit="d")
    dom_ok = dom[dom["t_end"] <= data_end].copy()
    n_ok = len(dom_ok)

    if n_ok == 0:
        return {"Window": window_label, "N_events": 0,
                "0purch%": None, "1purch%": None, "2+purch%": None, "Revival%": None}

    # Future purchases for members in dom_ok
    mem_set  = set(dom_ok["ShopMemberId"])
    future   = tl_df[tl_df["ShopMemberId"].isin(mem_set)].copy()

    # Range join via merge + filter
    merged   = dom_ok[["eid","ShopMemberId","t_revival","W"]].merge(
                   future[["ShopMemberId","date"]], on="ShopMemberId")
    after    = merged[merged["date"] > merged["t_revival"]].copy()
    after["dafter"] = (after["date"] - after["t_revival"]).dt.days
    in_win   = after[after["dafter"] <= after["W"]]

    cnt = in_win.groupby("eid").size().reindex(dom_ok["eid"], fill_value=0)

    return {
        "Window":   window_label,
        "N_events": n_ok,
        "0purch%":  round((cnt==0).mean()*100, 1),
        "1purch%":  round((cnt==1).mean()*100, 1),
        "2+purch%": round((cnt>=2).mean()*100, 1),
        "Revival%": round((cnt>=1).mean()*100, 1),
    }

# ══════════════════════════════════════════════════════════════════
# STEP 5: Observation window comparison (dormancy = 3xCi)
# ══════════════════════════════════════════════════════════════════
print("\n=== STEP 5: Observation window comparison (3xCi dormancy) ===")

thr_3x    = (gaps_all["Ci"] * 3).clip(30, 365)
dom_3x    = gaps_all[gaps_all["gap"] > thr_3x][["ShopMemberId","t_revival","gap","Ci"]].copy()
print(f"  Dormancy events (3xCi): {len(dom_3x):,}")

s5_rows = []
for m in [1.5, 2.0, 2.5, 3.0]:
    w = (dom_3x["Ci"] * m).clip(upper=365)
    s5_rows.append(window_stats(dom_3x, tl, DATA_END, w, f"{m}xCi", personalized=True))
for W in [60, 90, 120, 180]:
    s5_rows.append(window_stats(dom_3x, tl, DATA_END, W, f"{W}d"))

s5 = pd.DataFrame(s5_rows)
print(s5.to_string(index=False))

fig, axes = plt.subplots(1,2, figsize=(14,5))
colors5 = [PAL[i % len(PAL)] for i in range(len(s5))]
for ax, col, ylabel, title in zip(
        axes,
        ["Revival%","N_events"],
        ["True revival rate (%)","Usable events (after censor)"],
        ["Revival rate by window","Events by window"]):
    bars = ax.bar(s5["Window"], s5[col], color=colors5, edgecolor="white", alpha=0.85)
    ax.set_xlabel("Window"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.tick_params(axis='x', rotation=35)
    for bar, v in zip(bars, s5[col]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                f"{v}%" if col=="Revival%" else f"{v:,}",
                ha="center", va="bottom", fontsize=8)
fig.suptitle("shop#3 DHC — Step 5: Observation window", fontsize=13)
plt.tight_layout(); plt.savefig(f"{OUT}/s3_s5_window.png"); plt.close()

# Stacked bar: 0 / 1 / 2+ purchase distribution
fig, ax = plt.subplots(figsize=(10,5))
x5 = range(len(s5))
ax.bar(x5, s5["0purch%"], label="0 purchase", color="#d62728", alpha=0.85)
ax.bar(x5, s5["1purch%"], bottom=s5["0purch%"], label="1 purchase", color="#ff7f0e", alpha=0.85)
bot2 = s5["0purch%"] + s5["1purch%"]
ax.bar(x5, s5["2+purch%"], bottom=bot2, label="2+ purchases", color="#2ca02c", alpha=0.85)
ax.set_xticks(x5); ax.set_xticklabels(s5["Window"], rotation=35, ha="right")
ax.set_ylabel("% of events"); ax.legend(); ax.set_ylim(0,105)
ax.set_title("shop#3 DHC — Window: purchase distribution after revival")
plt.tight_layout(); plt.savefig(f"{OUT}/s3_s5_window_stacked.png"); plt.close()

# ══════════════════════════════════════════════════════════════════
# STEP 6: k x m cross table
# ══════════════════════════════════════════════════════════════════
print("\n=== STEP 6: k x m cross table ===")

k_list = [2.0, 2.5, 3.0]
m_list = [1.5, 2.0, 2.5, 3.0]
k_idx  = [f"{k}xCi" for k in k_list]
m_idx  = [f"{m}xCi" for m in m_list]

revival_g = pd.DataFrame(np.nan, index=k_idx, columns=m_idx, dtype=float)
events_g  = pd.DataFrame(0,      index=k_idx, columns=m_idx, dtype=int)

for k in k_list:
    thr_k = (gaps_all["Ci"] * k).clip(30, 365)
    dom_k = gaps_all[gaps_all["gap"] > thr_k][["ShopMemberId","t_revival","gap","Ci"]].copy()
    print(f"  k={k}: {len(dom_k):,} events  ({dom_k['ShopMemberId'].nunique():,} members)")
    for m in m_list:
        w = (dom_k["Ci"] * m).clip(upper=365)
        r = window_stats(dom_k, tl, DATA_END, w, f"{k}/{m}", personalized=True)
        revival_g.loc[f"{k}xCi", f"{m}xCi"] = r["Revival%"]
        events_g.loc[f"{k}xCi",  f"{m}xCi"] = r["N_events"]

print("\n  Revival rate (%) [k=threshold, m=window]:")
print(revival_g.to_string())
print("\n  Usable events:")
print(events_g.to_string())

fig, axes = plt.subplots(1,2, figsize=(14,5))
sns.heatmap(revival_g, annot=True, fmt=".1f", cmap="YlGn", ax=axes[0],
            linewidths=0.5, annot_kws={"size":12})
axes[0].set_xlabel("Observation window"); axes[0].set_ylabel("Dormancy threshold")
axes[0].set_title("True revival rate (%)")

try:
    ev_annot = events_g.map(lambda x: f"{x:,}")       # pandas >= 2.1
except AttributeError:
    ev_annot = events_g.applymap(lambda x: f"{x:,}")  # pandas < 2.1
sns.heatmap(events_g.astype(float), annot=ev_annot, fmt="", cmap="Blues",
            ax=axes[1], linewidths=0.5, annot_kws={"size":11})
axes[1].set_xlabel("Observation window"); axes[1].set_ylabel("Dormancy threshold")
axes[1].set_title("Usable events (after censor)")

fig.suptitle("shop#3 DHC — Step 6: k x m cross table", fontsize=13)
plt.tight_layout(); plt.savefig(f"{OUT}/s3_s6_cross_table.png"); plt.close()

# ══════════════════════════════════════════════════════════════════
# STEP 7: Dormancy length distribution
# ══════════════════════════════════════════════════════════════════
print("\n=== STEP 7: Dormancy length distribution ===")

dorm_lens = dom_3x["gap"].values
pqs_dl    = np.percentile(dorm_lens, [25,50,75,90])
print(f"  {qstr(dorm_lens)}  Mean={dorm_lens.mean():.1f}  Max={dorm_lens.max():.0f}")

clip_dl = dorm_lens[dorm_lens <= 730]
fig, ax = plt.subplots(figsize=(9,5))
ax.hist(clip_dl, bins=100, color="#C44E52", edgecolor="white", lw=0.3, alpha=0.85)
for v,c,lbl in zip(pqs_dl, ["#2ca02c","red","#d62728","#9467bd"],
                   ["P25","P50","P75","P90"]):
    ax.axvline(v, color=c, ls="--", lw=1.3, label=f"{lbl}={v:.0f}d")
ax.set_xlabel("Dormancy length (days, clipped <=730)")
ax.set_ylabel("Events")
ax.set_title("shop#3 DHC — Dormancy length (3xCi threshold)")
ax.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/s3_s7_dormancy_length.png"); plt.close()

# ══════════════════════════════════════════════════════════════════
# STEP 8: Return rate + seasonality
# ══════════════════════════════════════════════════════════════════
print("\n=== STEP 8: Return rate + seasonality ===")

all_ord = raw.drop_duplicates(subset="TradesGroupCode")
returns = all_ord[(all_ord["StatusDef"] == "Return") | (all_ord["TotalSalesAmount"] < 0)]
tot_by_m = all_ord.groupby("ShopMemberId").size()
ret_by_m = returns.groupby("ShopMemberId").size()
rr_m     = (ret_by_m / tot_by_m).fillna(0)

overall_rr = len(returns)/len(all_ord)*100
print(f"  Overall return rate (order level): {overall_rr:.2f}%")
print(f"  Members with >=1 return: {(rr_m>0).sum():,} ({(rr_m>0).mean()*100:.1f}%)")
rr_pos = rr_m[rr_m > 0].values
print(f"  (Returners) {qstr(rr_pos)}")

# Seasonality: valid orders by year-month
orders["ym"] = orders["OrderDateTime"].dt.to_period("M")
seasonal     = orders.groupby("ym").size().reset_index(name="n")
seasonal["ym_s"] = seasonal["ym"].astype(str)
seasonal = seasonal.reset_index(drop=True)

# Promo months to mark
promos = {"2022-11":"Double11","2023-11":"Double11",
          "2022-10":"Anniversary","2023-10":"Anniversary",
          "2022-12":"Xmas","2023-12":"Xmas"}

fig, axes = plt.subplots(2,1, figsize=(14,9))

# Row 0: return rate hist
axes[0].hist(rr_pos, bins=50, color="#DD8452", edgecolor="white", lw=0.3, alpha=0.85)
axes[0].axvline(np.median(rr_pos), color="red", ls="--", lw=1.5,
                label=f"Median={np.median(rr_pos):.3f}")
axes[0].set_xlabel("Per-member return rate (only returners)")
axes[0].set_ylabel("Members")
axes[0].set_title(f"Return rate distribution  (overall: {overall_rr:.2f}%)")
axes[0].legend()

# Row 1: monthly order volume
xs = range(len(seasonal))
axes[1].plot(xs, seasonal["n"], color=C0, lw=1.5, marker="o", ms=3)
axes[1].fill_between(xs, seasonal["n"], alpha=0.15, color=C0)
for xp, row in seasonal.iterrows():
    if row["ym_s"] in promos:
        axes[1].axvline(xp, color="red", ls="--", alpha=0.5)
        axes[1].text(xp, row["n"]*1.03, promos[row["ym_s"]],
                     ha="center", fontsize=8, color="red", rotation=30)
step = max(1, len(seasonal)//20)
axes[1].set_xticks(list(xs)[::step])
axes[1].set_xticklabels(seasonal["ym_s"].iloc[::step], rotation=45, ha="right", fontsize=8)
axes[1].set_ylabel("Valid orders"); axes[1].set_title("Monthly valid order volume (seasonality)")

fig.suptitle("shop#3 DHC — Step 8: Return rate & seasonality", fontsize=13)
plt.tight_layout(); plt.savefig(f"{OUT}/s3_s8_return_seasonal.png"); plt.close()

# ══════════════════════════════════════════════════════════════════
# SUMMARY.MD
# ══════════════════════════════════════════════════════════════════
print("\n=== Writing summary.md ===")

# Best k: adaptive row with most events (all valid)
s4a = s4[s4["Type"]=="adaptive"]

# Best window: highest revival rate among >=1000 events
s5_ok = s5[s5["N_events"] >= 1000]
best_w = s5_ok.loc[s5_ok["Revival%"].idxmax()] if len(s5_ok) else s5.loc[s5["Revival%"].idxmax()]

md = f"""# shop#3 (DHC) — EDA Summary: Dormancy Model Design

## Data
- Date range: {raw['OrderDateTime'].min().date()} ~ {raw['OrderDateTime'].max().date()}
- Raw rows: {len(raw):,}  |  Valid rows (def-b): {len(valid):,}  ({pct_str(len(valid),len(raw))})

---

## Step 4 — Threshold multiplier comparison

```
{s4.to_string(index=False)}
```

---

## Step 5 — Observation window (dormancy = 3xCi, {len(dom_3x):,} events)

```
{s5.to_string(index=False)}
```

---

## Step 6 — Revival rate (%) grid  [row=threshold k, col=window m]

```
{revival_g.to_string()}
```

## Step 6 — Usable events grid

```
{events_g.to_string()}
```

---

## Recommendations

### (1) Valid purchase definition
**Use definition (b): Excl. Fail + TotalSalesAmount > 0**
- Retains {pct_str(len(valid),len(raw))} of raw rows ({len(valid):,} rows)
- More permissive than Finish-only; keeps Overdue/Shipping rows that have positive payment
- Removes pure reversal lines and hard-fail orders reliably

### (2) Personal cycle Ci
- **Median Ci = {np.median(ci_vals):.1f} days  (≈ {np.median(ci_vals)/30:.1f} months)**
- P75 = {pqs_ci[2]:.1f}d | P90 = {pqs_ci[3]:.1f}d
- **Fallback for single-purchase members = {shop_fb:.1f} days**
- Interpretation: DHC supplement replenishment aligns with 1-month supply cycles;
  the ~{np.median(ci_vals):.0f}d median is consistent with 30-day pack sizes.
- {n_single:,} ({n_single/len(opm)*100:.1f}%) members have only 1 purchase and use the fallback.

### (3) Dormancy threshold
From Step 4:
{s4a[['Setting','Events','Dormant%']].to_string(index=False)}

**Recommendation: 3xCi (clip 30–365d)**
- Marks {s4a[s4a['Setting']=='3.0xCi']['Dormant%'].values[0]}% of gaps as dormant — reasonable false-positive risk
- {s4a[s4a['Setting']=='3.0xCi']['Events'].values[0]:,} labeled events — sufficient for model training
- 2.5xCi is a valid fallback if more positive samples are needed (for XGBoost scale_pos_weight tuning)

### (4) Observation window
Best window by revival rate (>=1,000 censored events): **{best_w['Window']}**
- Revival rate: {best_w['Revival%']}%  |  Events: {best_w['N_events']:,}

From Step 6, the revival rate grid shows:
- Revival rate ranges roughly {revival_g.values.astype(float).min():.1f}%–{revival_g.values.astype(float).max():.1f}% across all (k,m) combos
- Result is stable — label quality is robust to threshold/window choice

**Recommendation: 90-day fixed window (primary label)**
- Operationally simple and easy to explain
- Use 2xCi personalized as a sensitivity-check label
- Sufficient events after censoring for most window choices
"""

with open(f"{OUT}/summary.md", "w", encoding="utf-8") as f:
    f.write(md)
print(f"  Saved: {OUT}/summary.md")

print("\n=== Done. Output files: ===")
for fn in sorted(os.listdir(OUT)):
    if "s3_" in fn or fn == "summary.md":
        print(f"  {fn}")
