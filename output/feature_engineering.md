# 特徵工程文件

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
| t0_is_weekend | samples | t0_weekday ∈ (5, 6) | F2 例外 | Int8 |
| t0_is_big_promo | samples | t0_month ∈ (5, 11) | F2 例外 | Int8 |
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
- sleep_days
- hist_n_orders
- hist_total_amount
- hist_avg_ticket
- hist_max_ticket
- hist_total_qty
- personal_ci
- gap_std
- gap_cv
- hist_discount_ratio
- hist_coupon_order_ratio
- hist_return_ratio
- hist_main_channel
- hist_main_payment
- member_tenure_days
- was_sealed
- prior_once_only
- t0_amount
- t0_ts_count
- t0_qty
- t0_used_coupon
- t0_discount_ratio
- t0_channel
- t0_payment
- t0_shipping
- t0_month
- t0_weekday
- t0_is_weekend
- t0_is_big_promo
- hist_distinct_salepages
- hist_repeat_sku_ratio
- t0_n_distinct_salepages
- t0_buys_familiar_product
- t0_avg_unit_price
- t0_min_unit_price
- t0_max_unit_price
- register_source
- gender
- age_at_t0
- days_since_register
- country_code
- member_card_level
- is_app_installed
- marketing_optin_count
- t0_vs_avg_ticket_ratio
- t0_vs_max_ticket_ratio
- sleep_vs_ci_ratio
- t0_discount_vs_hist_ratio
- t0_used_coupon_vs_hist
- t0_channel_matches_hist
- t0_payment_matches_hist
- t0_qty_vs_avg
- recent_90d_n_orders
- recent_180d_n_orders
- recent_365d_n_orders
- pre_dormancy_90d_n_orders
- pre_dormancy_90d_amount

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
