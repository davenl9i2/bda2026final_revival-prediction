# 任務：shop#3 (DHC) 回流樣本特徵工程

我要對 shop#3 的回流樣本做特徵工程。樣本骨架已產出，現在要 join 各來源表，
為每筆樣本計算特徵（含交互特徵），並把所有做過的特徵記錄成文件。

---

## 輸入

- `./output/samples_shop3.parquet`
  - 樣本骨架，欄位：`sample_id, label, member_id, t0_date, t0_trades_group_code, split`
  - split 欄只有 'train' / 'test' 兩種值（前一階段已捨棄 val）
- `./Order_TG.csv`（主單，已 shop#3）
- `./OrderSlave.csv`（子單）
- `./SalePage.csv`（商品頁）
- `./Member.csv`（會員）
- 不使用 Segment（shop#3 無此資料）、不使用 Behavior

---

## 輸出（存到 ./output/）

1. `train_features.csv` / `test_features.csv`（依 samples 的 split 欄拆分，**沒有 val**）
2. `feature_engineering.md`（所有特徵的記錄文件）
3. `feature_sanity.md`（Sanity check 結果：跨 split 會員重疊、缺失率、極端值）

CSV 編碼 utf-8-sig。

---

## ⚠️ 最高原則：防洩漏

每筆樣本的特徵，**只能用「該樣本 t0_date 之前」的資料**計算。

唯一例外：**t₀ 那筆訂單本身的屬性**（金額、品項、通路…）可用，因為它在回流當下就已觀測。

任何 t0_date 之後發生的事，一律不可進入特徵。

**F5/F6 時間窗口邊界**：所有「t₀ 前 N 天」的窗口，邊界是 (t0_date - N days, t0_date)，
不含 t0_date 當天的訂單；t₀ 訂單本身只在 F2 系列特徵中使用。

工具：pandas + numpy，**禁止 per-row Python 迴圈**，所有歷史聚合用 merge + groupby。

---

## Step 0：Sanity Check（先做，print 並寫進 feature_sanity.md）

```python
import pandas as pd
df = pd.read_parquet('./output/samples_shop3.parquet')

train = df[df['split'] == 'train']
test  = df[df['split'] == 'test']

# 跨 split 會員重疊
overlap = set(train['member_id']) & set(test['member_id'])
print(f"跨 split 會員: {len(overlap):,}")
print(f"  佔 train 會員: {len(overlap)/train['member_id'].nunique()*100:.1f}%")
print(f"  佔 test 會員 : {len(overlap)/test['member_id'].nunique()*100:.1f}%")

# t0_date 分佈
print(train['t0_date'].astype(str).str[:7].value_counts().sort_index())
print(test ['t0_date'].astype(str).str[:7].value_counts().sort_index())
```

如果跨 split 會員 > 30%，**特別強化下游特徵的因果截止點檢查**。

---

## 前處理（與前一階段一致）

- 有效購買：`StatusDef != 'Fail' AND TotalSalesAmount > 0`
- 主單去重：`drop_duplicates('TradesGroupCode')`
- OrderDateTime → `pd.to_datetime`
- Member 預設值視為缺失：
  - `RegisterDateTime/Birthday == '1900-01-01'` → NaN
  - `MemberCardLevel == 0` → NaN

---

## 特徵清單

### F1：主單歷史 RFM 與行為穩定性

（用 member 在 t0_date 之前的有效訂單）

| 特徵名 | 定義 |
|--------|------|
| sleep_days | t0_date − 上一次購買日（天） |
| hist_n_orders | t₀ 前累積有效訂單數 |
| hist_total_amount | t₀ 前 TotalSalesAmount 總和 |
| hist_avg_ticket | hist_total_amount / hist_n_orders |
| hist_max_ticket | t₀ 前單筆最高金額 |
| personal_ci | t₀ 前相鄰購買 gap 中位數（< 2 個 gap → NaN） |
| gap_std | t₀ 前 gap 標準差（同上 → NaN） |
| gap_cv | gap_std / personal_ci |
| hist_discount_ratio | t₀ 前 abs(TotalDiscount) 總和 / TotalPrice 總和 |
| hist_coupon_order_ratio | t₀ 前用過折價券的訂單比例（TotalCouponDiscount != 0） |
| hist_return_ratio | t₀ 前退貨訂單佔比 |
| hist_main_channel | t₀ 前最常用 ChannelType（眾數） |
| hist_main_payment | t₀ 前最常用 PaymentType（眾數） |
| member_tenure_days | t0_date − 首次購買日（天） |
| was_sealed | sleep_days > 365 → 1 else 0 |
| prior_once_only | hist_n_orders == 1 → 1 else 0 |

### F2：t₀ 訂單特徵

（直接用 t0_trades_group_code 對到該筆訂單）

| 特徵名 | 定義 |
|--------|------|
| t0_amount | 該訂單 TotalSalesAmount |
| t0_ts_count | TsCount（不重複商品數） |
| t0_qty | Qty |
| t0_used_coupon | TotalCouponDiscount != 0 → 1 else 0 |
| t0_discount_ratio | abs(TotalDiscount) / TotalPrice |
| t0_channel | ChannelType |
| t0_payment | PaymentType |
| t0_shipping | ShippingType |
| t0_month | t0_date 月份 (1-12) |
| t0_weekday | t0_date 星期 (0=Mon, 6=Sun) |
| t0_is_weekend | t0_weekday in (5, 6) → 1/0 |
| t0_is_big_promo | t0_month ∈ {5, 11} → 1/0（母親節 / 雙11） |

### F3：子單 + 商品頁（商品行為）

**注意：SalePage 沒有現成「品類」欄位**，以 SalePageId 層級為主：

| 特徵名 | 定義 |
|--------|------|
| hist_distinct_salepages | t₀ 前不重複 SalePageId 數（廣度） |
| hist_repeat_sku_ratio | t₀ 前重複購買同一 SalePageId 的比例（忠誠度） |
| t0_n_distinct_salepages | t₀ 訂單內不重複 SalePageId 數 |
| t0_buys_familiar_product | t₀ 訂單是否含 member t₀ 前買過的 SalePageId → 1/0 |
| t0_avg_unit_price | t₀ 訂單子單 UnitPrice 平均 |
| t0_min_unit_price | t₀ 訂單子單 UnitPrice 最低（試水溫訊號） |
| t0_max_unit_price | t₀ 訂單子單 UnitPrice 最高 |

### F4：會員背景（Member 表）

**安全欄位**（不隨時間變）：

| 特徵名 | 定義 |
|--------|------|
| register_source | RegisterSourceTypeDef |
| gender | Gender（空值保留 NaN） |
| age_at_t0 | (t0_date − Birthday) 換算歲數（Birthday=1900 → NaN） |
| days_since_register | t0_date − RegisterDateTime（1900 → NaN） |
| country_code | CountryAliasCode |

**快照欄位**（潛在輕微洩漏，標註後仍納入）：

| 特徵名 | 定義 |
|--------|------|
| member_card_level | MemberCardLevel（0 → NaN） |
| is_app_installed | IsAppInstalled |
| marketing_optin_count | IsEnableEmail + IsEnablePushNotification + IsEnableShortMessage (0-3) |

> ⚠️ Limitation：Member 表為 2024-02-29 快照，無歷史版本。F4 快照欄位的值是 t₀ 之後也可能更新過，
> 為輕度洩漏。文件中明確標註，但仍納入以避免過度保守。

### ⭐ F5：跨期比較/交互特徵（NEW，預期是最強訊號群）

「真復活 vs 路過」的差異主要體現在「這次回來跟以前比怎樣」，不是單看 t₀ 或單看歷史。

| 特徵名 | 計算 | 商業意義 |
|--------|------|---------|
| t0_vs_avg_ticket_ratio | t0_amount / hist_avg_ticket | >1.5 大手筆回歸（強訊號）；<0.5 試水溫 |
| t0_vs_max_ticket_ratio | t0_amount / hist_max_ticket | >0.8 接近個人巔峰 |
| sleep_vs_ci_ratio | sleep_days / personal_ci | 沉睡的「離譜程度」 |
| t0_discount_vs_hist_ratio | t0_discount_ratio / max(hist_discount_ratio, 0.01) | >1.5 比平常更靠折扣 = 路過訊號 |
| t0_used_coupon_vs_hist | t0_used_coupon - hist_coupon_order_ratio | 這次比平常更愛用券嗎 |
| t0_channel_matches_hist | int(t0_channel == hist_main_channel) | 行為一致性 |
| t0_payment_matches_hist | int(t0_payment == hist_main_payment) | 行為一致性 |
| t0_qty_vs_avg | t0_qty / (hist_total_qty / hist_n_orders) | 這次量比平常大嗎 |

**Safe divide 注意**：
- 分母為 0 或 NaN → 結果填 NaN（不要填 0 也不要填極大值）
- `hist_avg_ticket` 在 `prior_once_only=1` 時可能很可靠但只代表一個觀測
- 比率特徵在 `hist_n_orders < 2` 的會員上可信度低，但保留 NaN 讓 XGBoost 自行處理

### F6：近期活躍窗口特徵（NEW）

捕捉「沉睡前活躍程度」與「沉睡完整性」：

| 特徵名 | 計算 |
|--------|------|
| recent_90d_n_orders | t₀ 前 90 天訂單數（多半 = 0，但 >0 代表近期才開始沉睡） |
| recent_180d_n_orders | t₀ 前 180 天訂單數 |
| recent_365d_n_orders | t₀ 前 365 天訂單數 |
| pre_dormancy_90d_n_orders | 「最後購買日」前 90 天內的訂單數（沉睡前活躍度） |
| pre_dormancy_90d_amount | 「最後購買日」前 90 天訂單金額總和 |

> 「最後購買日」= t0_date − sleep_days（即沉睡的起點）

---

## 計算效率要求

- 樣本 ~28.8 萬筆（train 198,370 + test 89,454），主單/子單數百萬列
- **禁止 per-row Python 迴圈**
- F1/F5/F6 的歷史聚合建議流程：
  1. 把 samples 與該 member 的所有訂單 merge（保留所有歷史訂單，會 explode）
  2. 過濾 `OrderDateTime < t0_date`（嚴格小於，不含當天）
  3. groupby sample_id 做 aggregate
- 若記憶體吃緊，按 sample_id 切 10 個 batch 處理後 concat
- F2/F3 t₀ 訂單特徵：用 t0_trades_group_code 直接 merge，最快

---

## 缺失值處理

- 只買過一次的 member：personal_ci / gap_std / gap_cv / t0_vs_avg_ticket_ratio 等 → 保留 NaN（XGBoost 原生支援）
- 類別特徵缺失 → 填 'unknown'
- F5 的比率特徵分母為 0 → NaN
- 在 feature_sanity.md 統計每個特徵的缺失率與分位數

---

## 輸出規格

把所有特徵 merge 回 samples（以 sample_id 為 key），保留：

```
sample_id, label, split, [所有特徵欄位...]
```

依 split 欄拆成：
- `./output/train_features.csv`（198,370 列）
- `./output/test_features.csv`（89,454 列）

確認兩檔合併後 sample_id 連續無重複。

---

## feature_engineering.md 內容要求

1. `# 特徵工程文件`
2. `## 1. 防洩漏原則`
   - 明確寫出「t0_date 嚴格小於」規則
   - F2 為唯一例外（t₀ 訂單已觀測）
   - F5/F6 時間窗口邊界說明
3. `## 2. 資料來源與排除說明`
   - 用了哪些表、如何 join
   - shop#3 無 Segment 資料、不使用 Behavior 的決策說明
   - Member 為 2024-02-29 快照、F4 快照欄位的 limitation
4. `## 3. 特徵總表`（一張大表）
   欄位：`特徵名 | 來源表 | 定義 | 計算邏輯 | 是否防洩漏安全 | 資料型態 | 缺失率`
5. `## 4. 分組說明（F1~F6）`
   每組寫設計動機。特別說明：
   - F1 `sleep_days` 預期為基礎強特徵
   - F3 `t0_buys_familiar_product` 的商業意義（買熟悉商品 = 真復活訊號）
   - F5 為何是最高優先級的交互特徵群
   - F6 `recent_90d_n_orders` 通常 = 0，但 > 0 時是極強訊號
6. `## 5. 缺失值處理策略`
7. `## 6. 輸出格式`（train/test CSV 欄位清單）
8. `## 7. 待後續處理事項`
   - 特徵顯著性檢驗（Mann-Whitney U、卡方）→ 留待建模前
   - 高相關（|r| > 0.85）特徵去重 → 留待建模前
   - 類別特徵編碼（target encoding vs one-hot）→ 留待建模前

---

## feature_sanity.md 內容要求

```
# Feature Engineering Sanity Check

## 1. 跨 split 會員重疊
| 項目 | 數量 |
|------|------|
| 跨 split 會員數 | ... |
| 佔 train 會員比例 | ...% |
| 佔 test 會員比例 | ...% |

## 2. 各特徵缺失率（表格，按 F1~F6 分組）

## 3. 各數值特徵分位數
| 特徵 | min | P25 | P50 | P75 | max |

## 4. F5 比率特徵的極端值
（檢查 t0_vs_avg_ticket_ratio 等是否有過大值，可能要 clip）

## 5. 防洩漏自我驗證
隨機抽 5 筆樣本，列出該樣本的 t0_date 與「歷史聚合用到的最大 OrderDateTime」，
確認後者嚴格 < 前者。
```

---

## 程式碼結構

- 每個特徵組 (F1~F6) 一個函式，docstring 標明：
  - 對應規則
  - 防洩漏邏輯
  - 預期輸出欄位數
- 主程式：

```python
def main():
    samples = load_samples()
    sanity_check(samples)  # Step 0

    f1 = compute_f1_history_rfm(samples, orders)
    f2 = compute_f2_t0_order(samples, orders)
    f3 = compute_f3_product(samples, orders, order_slave, salepage)
    f4 = compute_f4_member(samples, member)
    f5 = compute_f5_interactions(samples, f1, f2)  # 依賴 F1 + F2
    f6 = compute_f6_recent_windows(samples, orders)

    features = samples[['sample_id','label','split']].merge_all([f1,f2,f3,f4,f5,f6])
    write_outputs(features)
    write_md_documents(features)
```

- 每組算完 print：特徵數、缺失率 P50、樣本數確認
- 相對路徑、可重跑、不要動 samples_shop3.parquet 本身
