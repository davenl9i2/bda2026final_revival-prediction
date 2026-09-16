# Feature Engineering Sanity Check

> 腳本：`feature_engineering.py` — 2026-06-03

---

## 1. 跨 split 會員重疊

| 項目 | 數量 |
|------|------|
| 跨 split 會員數 | 26,231 |
| 佔 train 會員比例 | 17.5% |
| 佔 test 會員比例 | 30.0% |
| train 樣本數 | 198,370 |
| test 樣本數 | 89,454 |

✓ 重疊比例在可接受範圍。

---

## 2. 各特徵缺失率（按 F1~F6 分組）

| 組 | 特徵 | 缺失率 |
|---|---|---|
| F1 | sleep_days | 0.2% |
| F1 | hist_n_orders | 0.2% |
| F1 | hist_total_amount | 0.2% |
| F1 | hist_avg_ticket | 0.2% |
| F1 | hist_max_ticket | 0.2% |
| F1 | hist_total_qty | 0.2% |
| F1 | personal_ci | 45.5% |
| F1 | gap_std | 45.5% |
| F1 | gap_cv | 45.5% |
| F1 | hist_discount_ratio | 0.2% |
| F1 | hist_coupon_order_ratio | 0.2% |
| F1 | hist_return_ratio | 0.0% |
| F1 | hist_main_channel | 0.0% |
| F1 | hist_main_payment | 0.0% |
| F1 | member_tenure_days | 0.2% |
| F1 | was_sealed | 0.0% |
| F1 | prior_once_only | 0.0% |
| F2 | t0_amount | 1.3% |
| F2 | t0_ts_count | 1.3% |
| F2 | t0_qty | 1.3% |
| F2 | t0_used_coupon | 0.0% |
| F2 | t0_discount_ratio | 1.3% |
| F2 | t0_channel | 0.0% |
| F2 | t0_payment | 0.0% |
| F2 | t0_shipping | 0.0% |
| F2 | t0_month | 0.0% |
| F2 | t0_weekday | 0.0% |
| F2 | t0_is_weekend | 0.0% |
| F2 | t0_is_big_promo | 0.0% |
| F3 | hist_distinct_salepages | 0.0% |
| F3 | hist_repeat_sku_ratio | 91.7% |
| F3 | t0_n_distinct_salepages | 0.0% |
| F3 | t0_buys_familiar_product | 0.0% |
| F3 | t0_avg_unit_price | 0.0% |
| F3 | t0_min_unit_price | 0.0% |
| F3 | t0_max_unit_price | 0.0% |
| F4 | register_source | 0.0% |
| F4 | gender | 26.8% |
| F4 | age_at_t0 | 13.3% |
| F4 | days_since_register | 0.0% |
| F4 | country_code | 0.0% |
| F4 | member_card_level | 0.0% |
| F4 | is_app_installed | 0.0% |
| F4 | marketing_optin_count | 0.0% |
| F5 | t0_vs_avg_ticket_ratio | 1.5% |
| F5 | t0_vs_max_ticket_ratio | 1.5% |
| F5 | sleep_vs_ci_ratio | 45.5% |
| F5 | t0_discount_vs_hist_ratio | 1.5% |
| F5 | t0_used_coupon_vs_hist | 0.2% |
| F5 | t0_channel_matches_hist | 0.0% |
| F5 | t0_payment_matches_hist | 0.0% |
| F5 | t0_qty_vs_avg | 1.5% |
| F6 | recent_90d_n_orders | 0.0% |
| F6 | recent_180d_n_orders | 0.0% |
| F6 | recent_365d_n_orders | 0.0% |
| F6 | pre_dormancy_90d_n_orders | 0.0% |
| F6 | pre_dormancy_90d_amount | 0.0% |

---

## 3. 各數值特徵分位數

| 特徵 | min | P25 | P50 | P75 | max |
|---|---|---|---|---|---|
| sleep_days | 30.00 | 74.00 | 142.00 | 249.00 | 668.00 |
| hist_n_orders | 1.00 | 2.00 | 3.00 | 7.00 | 648.00 |
| hist_total_amount | 1.00 | 625.00 | 1528.00 | 3626.00 | 798162.00 |
| hist_avg_ticket | 1.00 | 257.00 | 420.20 | 670.17 | 75609.83 |
| hist_max_ticket | 1.00 | 394.00 | 697.00 | 1200.00 | 450000.00 |
| hist_total_qty | 1.00 | 5.00 | 12.00 | 27.00 | 4163.00 |
| personal_ci | 1.00 | 10.00 | 17.00 | 28.00 | 205.50 |
| gap_std | 0.00 | 12.27 | 21.93 | 38.44 | 326.82 |
| gap_cv | 0.00 | 0.72 | 1.18 | 2.11 | 248.84 |
| hist_discount_ratio | 0.00 | 0.15 | 0.22 | 0.31 | 1.00 |
| hist_coupon_order_ratio | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| hist_return_ratio | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| member_tenure_days | 30.00 | 172.00 | 260.00 | 368.00 | 668.00 |
| was_sealed | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| prior_once_only | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| t0_amount | 1.00 | 190.00 | 360.00 | 653.00 | 219560.00 |
| t0_ts_count | 1.00 | 2.00 | 3.00 | 5.00 | 61.00 |
| t0_qty | 1.00 | 2.00 | 3.00 | 5.00 | 1006.00 |
| t0_used_coupon | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| t0_discount_ratio | 0.00 | 0.10 | 0.19 | 0.29 | 1.00 |
| t0_month | 1.00 | 3.00 | 8.00 | 10.00 | 12.00 |
| t0_weekday | 0.00 | 2.00 | 4.00 | 5.00 | 6.00 |
| t0_is_weekend | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| t0_is_big_promo | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| hist_distinct_salepages | 0.00 | 0.00 | 0.00 | 0.00 | 291.00 |
| hist_repeat_sku_ratio | 0.00 | 0.00 | 0.00 | 0.12 | 1.00 |
| t0_n_distinct_salepages | 0.00 | 0.00 | 0.00 | 0.00 | 50.00 |
| t0_buys_familiar_product | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| t0_avg_unit_price | 1.00 | 104.67 | 169.00 | 260.00 | 9498.00 |
| t0_min_unit_price | 1.00 | 49.00 | 99.00 | 180.00 | 9498.00 |
| t0_max_unit_price | 1.00 | 139.00 | 245.00 | 388.00 | 9900.00 |
| age_at_t0 | -420.02 | 35.52 | 43.90 | 53.37 | 2006.55 |
| days_since_register | -408.00 | 335.00 | 799.00 | 2027.00 | 14406.00 |
| member_card_level | 10.00 | 10.00 | 10.00 | 20.00 | 50.00 |
| is_app_installed | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| marketing_optin_count | 0.00 | 3.00 | 3.00 | 3.00 | 3.00 |
| t0_vs_avg_ticket_ratio | 0.00 | 0.46 | 0.90 | 1.60 | 636.00 |
| t0_vs_max_ticket_ratio | 0.00 | 0.26 | 0.54 | 1.02 | 636.00 |
| sleep_vs_ci_ratio | 0.25 | 3.71 | 5.03 | 8.00 | 605.00 |
| t0_discount_vs_hist_ratio | 0.00 | 0.45 | 0.86 | 1.40 | 85.43 |

---

## 4. F5 比率特徵的極端值

| 特徵 | P95 | P99 | max | 建議clip上限 |
|---|---|---|---|---|
| t0_vs_avg_ticket_ratio | 3.95 | 8.91 | 636.00 | 17.8 |
| t0_vs_max_ticket_ratio | 2.91 | 7.57 | 636.00 | 15.1 |
| sleep_vs_ci_ratio | 22.40 | 56.40 | 605.00 | 20.0 |
| t0_discount_vs_hist_ratio | 3.61 | 12.87 | 85.43 | 20.0 |

> 建議在建模前對上表「建議clip上限」欄位值做 winsorize，防止少數極端比率主導樹分裂。

---

## 5. 防洩漏自我驗證

隨機抽 5 筆樣本，確認「歷史聚合最大 OrderDateTime」嚴格 < t0_date：

| sample_id | t0_date | max_hist_OrderDateTime | leakage_ok |
|---|---|---|---|
| 64573 | 2022-09-19 00:00:00 | 2022-06-15 17:55:03 | True |
| 90682 | 2022-11-04 00:00:00 | 2022-06-17 18:30:42 | True |
| 100723 | 2022-11-19 00:00:00 | 2022-01-15 19:02:19 | True |
| 101594 | 2022-11-20 00:00:00 | 2022-08-22 15:02:48 | True |
| 121180 | 2022-12-19 00:00:00 | 2022-04-20 15:25:39 | True |

✅ 所有抽查樣本均通過防洩漏驗證。

---

*Generated by feature_engineering.py — 2026-06-03*
