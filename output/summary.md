# shop#3 (DHC) — EDA Summary: Dormancy Model Design

## Data
- Date range: 2022-01-01 ~ 2024-02-29
- Raw rows: 4,938,006  |  Valid rows (def-b): 4,653,006  (94.2%)

---

## Step 4 — Threshold multiplier comparison

```
    Type Setting  Thr_median(d)  Events  Members_w_event  Dormant%
adaptive  1.5xCi           51.8  739061           360907      24.6
adaptive  2.0xCi           69.0  520368           281957      17.3
adaptive  2.5xCi           86.2  410915           248722      13.7
adaptive  3.0xCi          103.5  332328           220610      11.1
   fixed     60d           60.0 1022319           509938      34.1
   fixed     90d           90.0  726002           456205      24.2
   fixed    120d          120.0  536279           396433      17.9
   fixed    180d          180.0  319138           282258      10.6
```

---

## Step 5 — Observation window (dormancy = 3xCi, 332,328 events)

```
Window  N_events  0purch%  1purch%  2+purch%  Revival%
1.5xCi    255894     39.8     39.4      20.7      60.2
2.0xCi    245736     33.1     31.9      35.1      66.9
2.5xCi    235278     29.0     29.3      41.7      71.0
3.0xCi    225924     25.6     26.7      47.8      74.4
   60d    286116     40.4     28.0      31.6      59.6
   90d    262394     31.0     24.4      44.7      69.0
  120d    240169     25.0     20.8      54.1      75.0
  180d    197779     17.6     15.7      66.7      82.4
```

---

## Step 6 — Revival rate (%) grid  [row=threshold k, col=window m]

```
        1.5xCi  2.0xCi  2.5xCi  3.0xCi
2.0xCi    62.6    70.1    74.6    78.2
2.5xCi    61.2    68.4    72.7    76.3
3.0xCi    60.2    66.9    71.0    74.4
```

## Step 6 — Usable events grid

```
        1.5xCi  2.0xCi  2.5xCi  3.0xCi
2.0xCi  423329  406022  387907  371657
2.5xCi  326398  313425  299831  287629
3.0xCi  255894  245736  235278  225924
```

---

## Recommendations

### (1) Valid purchase definition
**Use definition (b): Excl. Fail + TotalSalesAmount > 0**
- Retains 94.2% of raw rows (4,653,006 rows)
- More permissive than Finish-only; keeps Overdue/Shipping rows that have positive payment
- Removes pure reversal lines and hard-fail orders reliably

### (2) Personal cycle Ci
- **Median Ci = 67.0 days  (≈ 2.2 months)**
- P75 = 147.0d | P90 = 271.0d
- **Fallback for single-purchase members = 67.0 days**
- Interpretation: DHC supplement replenishment aligns with 1-month supply cycles;
  the ~67d median is consistent with 30-day pack sizes.
- 429,090 (39.9%) members have only 1 purchase and use the fallback.

### (3) Dormancy threshold
From Step 4:
Setting  Events  Dormant%
 1.5xCi  739061      24.6
 2.0xCi  520368      17.3
 2.5xCi  410915      13.7
 3.0xCi  332328      11.1

**Recommendation: 3xCi (clip 30–365d)**
- Marks 11.1% of gaps as dormant — reasonable false-positive risk
- 332,328 labeled events — sufficient for model training
- 2.5xCi is a valid fallback if more positive samples are needed (for XGBoost scale_pos_weight tuning)

### (4) Observation window
Best window by revival rate (>=1,000 censored events): **180d**
- Revival rate: 82.4%  |  Events: 197,779

From Step 6, the revival rate grid shows:
- Revival rate ranges roughly 60.2%–78.2% across all (k,m) combos
- Result is stable — label quality is robust to threshold/window choice

**Recommendation: 90-day fixed window (primary label)**
- Operationally simple and easy to explain
- Use 2xCi personalized as a sensitivity-check label
- Sufficient events after censoring for most window choices
