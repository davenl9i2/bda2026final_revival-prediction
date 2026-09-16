# 資料前處理規則文件

> 版本：v1.1
> 產出日期：2026-06-03
> 執行腳本：`preprocess_shop3.py`

---

## 1. 資料來源與焦點

| 項目 | 說明 |
|------|------|
| 主檔路徑 | `91APP_Dataset(會員&主單&子單&商品頁&標籤)/Order_TG.csv` |
| 品牌 | DHC 保健食品（91APP shop#3） |
| shop#3 ShopId（雜湊值） | `hFwniXiB/Ev2ZPXeO630Sw==` |
| 資料時間範圍 | 2022-01-01 ~ 2024-02-29 |
| shop#3 原始筆數 | 4,938,006 |
| shop#3 會員數 | 1,077,891 |

> **ShopId 識別**：載入時 print 全資料集所有 ShopId rank，shop#3 對應 `hFwniXiB/Ev2ZPXeO630Sw==`（EDA 驗證 4,938,006 筆）。

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
raw = pd.read_csv("Order_TG.csv", dtype={'ShopId': 'category', ...})
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
prior_gaps = [gap_1, ..., gap_{i-1}]   # 只用 gap i 之前的資料

if len(prior_gaps) >= 2:    Ci = median(prior_gaps)
elif len(prior_gaps) == 1:  Ci = prior_gaps[0]
else:                       Ci = 67   # 第 1 個 gap：用全體中位數
```

**為何用動態 Cᵢ 而非靜態：**
靜態（整個會員生涯中位數）計算早期 gap 的門檻時，包含未來才發生的購買資訊 → **時間洩漏**。
動態 Cᵢ 確保每個門檻只用當下可知的資訊，與線上推論保持一致。

> **注意：** 本 pipeline 使用動態 Cᵢ（R4），事件數與 Cᵢ 中位數會與 EDA Step 4/5 的靜態 Cᵢ 基準不同，屬預期現象。

**Fallback = 67 天來源：**
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
