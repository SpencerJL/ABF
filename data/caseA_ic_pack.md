# IC Pack (Auto-generated)

## 1. Executive Summary
- Deal / pool description: Mixed whole-loan forward-flow pool (800 loans) as of 2025-01-31, current balance $42,106,606.26, WAC 6.97%, balance-weighted remaining term 13.81 years.
- Purchase price / target return: Purchase price $41,264,474.13 (98.00% of current balance); target IRR 3.50%.
- Base case (IRR / NPV / WAL): 3.81% / -$3,975,096.83 / 2.76 years.
- Downside summary (stress returns and key breakevens): Combined downside IRR 1.26% (delta -255 bp), default breakeven 3.60% (mult 1.20x base), LGD breakeven 64.95%, recovery lag breakeven 59 months.
- Top 3 risks + mitigants:
  - Return cushion vs target is 31 bp; mitigant: tighten purchase price or reserve structure.
  - Product concentration risk: Mortgage at 63.09%; mitigant: concentration limits and segment stress overlays.
  - Geographic concentration risk: UK at 32.33%; mitigant: region-level intake caps and trigger monitoring.
- Recommendation / open questions: Conditional pass. Open question: Confirm legal treatment for data exceptions before sign-off.

## 2. Data Quality & Reconciliation
### Data received
- Loan tape fields + history depth: 14 tape fields; 12 months of aggregate performance history (2024-02-29 to 2025-01-31).
- Servicer / trustee aggregates: total loans, current balance, WAC, WAL estimate, and status/region/product distributions were provided.
- Definitions provided (default / charge-off / recovery): not explicitly defined in source files; this run treats `default_principal` as gross default flow and `recovery_cash` as recoveries.

### QC checks performed
- Schema & type checks: tape missing columns=0, perf missing columns=0, unparseable dates=0.
- Duplicate / key integrity: duplicate loan_id=0, duplicate rows=0, origination after as-of breaches=0.
- Missingness (by field, by vintage/segment): max field missing=0; dataset-level missing ratio=0.00%.
- Outliers (unit errors vs true risk): unit-error flags=0; current>orig exceptions=8 loans / $2,020,549.80.
- Roll-forward identity checks: max absolute monthly difference=$0.00.
- Reconciliation to servicer/trustee (if available): loan-count diff=0, current-balance diff=$0.00, WAC diff=0 bp.

### Exceptions log
| Issue | Count / exposure | Treatment | Estimated impact |
| --- | --- | --- | --- |
| Current balance above original balance | 8 loans / $2,020,549.80 | Confirm capitalization or fix tape | Low to medium |
| Performance/servicer cut-off mismatch | gap -$8,033,560.32 | Align cut-off date and population | High for calibration confidence |

## 3. Assumptions & Methodology
- Prepayment (SMM/CPR curve; segmentation): base CPR 10.00% (pool-level); trailing realized CPR 6.70%.
- Default (monthly default / CDR curve; segmentation): base CDR 3.00% (pool-level); trailing realized CDR 2.12%.
- LGD / recovery rate + recovery lag: base LGD 55.00% with recovery lag 3 months; trailing implied LGD 62.15%, recovery rate 37.85%.
- Fees (servicing, trustees, other): servicing fee 1.00% annualized; trustee/other fees not separately modeled in v1.
- Discounting / funding curve and cost of funds: flat annual discount rate 8.00%.
- Notes on calibration and rationale: v1 uses a deterministic pool-level engine; next iteration should segment CPR/CDR/LGD by product, region, and delinquency state.

## 4. Results
### Base cashflow profile
- Cashflow summary chart/table (first 12 months):
| Month | Begin bal | Interest | Total principal | Default | Recovery | Net cash | End bal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 42.11m | 0.24m | 1.06m | 0.10m | 0.00m | 1.27m | 40.94m |
| 2 | 40.94m | 0.24m | 1.00m | 0.10m | 0.00m | 1.20m | 39.85m |
| 3 | 39.85m | 0.23m | 0.98m | 0.10m | 0.00m | 1.17m | 38.77m |
| 4 | 38.77m | 0.22m | 0.97m | 0.10m | 0.05m | 1.20m | 37.70m |
| 5 | 37.70m | 0.21m | 0.95m | 0.09m | 0.05m | 1.18m | 36.66m |
| 6 | 36.66m | 0.20m | 0.94m | 0.09m | 0.04m | 1.16m | 35.63m |
| 7 | 35.63m | 0.20m | 0.92m | 0.09m | 0.04m | 1.13m | 34.62m |
| 8 | 34.62m | 0.19m | 0.91m | 0.09m | 0.04m | 1.11m | 33.62m |
| 9 | 33.62m | 0.18m | 0.89m | 0.08m | 0.04m | 1.09m | 32.65m |
| 10 | 32.65m | 0.17m | 0.88m | 0.08m | 0.04m | 1.06m | 31.69m |
| 11 | 31.69m | 0.17m | 0.86m | 0.08m | 0.04m | 1.04m | 30.75m |
| 12 | 30.75m | 0.16m | 0.84m | 0.08m | 0.04m | 1.01m | 29.83m |

### Stress & sensitivities
| Scenario | Default | LGD | Recovery lag | Prepay | IRR | WAL | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base | 3.00% | 55.00% | 3m | 10.00% | 3.81% | 2.76y | Reference |
| CDR x2 | 6.00% | 55.00% | 3m | 10.00% | 2.21% | 2.50y | Higher defaults |
| LGD +10pp | 3.00% | 65.00% | 3m | 10.00% | 3.50% | 2.76y | Higher severity |
| Recovery lag +6m | 3.00% | 55.00% | 9m | 10.00% | 3.78% | 2.76y | Slower recoveries |
| CPR down 40% | 3.00% | 55.00% | 3m | 6.00% | 3.10% | 3.20y | Slower prepay |
| Combined downside | 6.00% | 65.00% | 9m | 6.00% | 1.26% | 2.93y | All downside levers |

### Breakevens ("what kills the deal")
- Default breakeven (IRR falls below target): 3.60% (mult 1.20x base).
- LGD breakeven: 64.95%.
- Recovery lag breakeven: 59 months.
- Key driver attribution: IRR deltas vs base are CDR x2 = -160 bp, LGD +10pp = -31 bp, CPR down 40% = -71 bp, recovery lag +6m = -3 bp.

## 5. Monitoring Plan
- Weekly pack: delinq buckets (30+=8.38%, 60+=4.63%), vintage mix (2024 (29.88%), 2023 (27.35%), 2022 (26.45%)), prepay curve (avg CPR 6.66%, latest 9.23%), cumulative net loss 1.19%, trailing recovery rate 37.77%, concentration by region (UK (32.33%), DE (18.19%), IT (16.72%)) and product (Mortgage (63.09%), SME (13.83%), Consumer_Unsec (12.35%)).
- Early warning triggers: 30+ delinquency >10%, 60+ delinquency >6%, 3-month avg CPR <4% or >12%, 3-month avg CDR >4%, recovery rate <30%, and any concentration bucket >35%.
- Reporting cadence and governance: weekly dashboard + monthly IC refresh; store run manifest with data/config/code versions and QC exceptions.