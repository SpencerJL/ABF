# IC Pack Template (ABF / Whole Loan Pool / Forward Flow)

## 1. Executive Summary (half page)
- Deal / pool description:
- Purchase price / target return:
- Base case (IRR / NPV / WAL):
- Downside summary (stress returns and key breakevens):
- Top 3 risks + mitigants:
- Recommendation / open questions:

## 2. Data Quality & Reconciliation (1 page)
### Data received
- Loan tape fields + history depth:
- Servicer / trustee aggregates:
- Definitions provided (default / charge-off / recovery):

### QC checks performed
- Schema & type checks:
- Duplicate / key integrity:
- Missingness (by field, by vintage/segment):
- Outliers (unit errors vs true risk):
- Roll-forward identity checks:
- Reconciliation to servicer/trustee (if available):

### Exceptions log
| Issue | Count / exposure | Treatment | Estimated impact |
|---|---:|---|---|

## 3. Assumptions & Methodology (1 page)
- Prepayment (SMM/CPR curve; segmentation):
- Default (monthly default / CDR curve; segmentation):
- LGD / recovery rate + recovery lag:
- Fees (servicing, trustees, other):
- Discounting / funding curve and cost of funds:
- Notes on calibration and rationale:

## 4. Results (1 page)
### Base cashflow profile
- Cashflow summary chart/table:

### Stress & sensitivities
| Scenario | Default | LGD | Recovery lag | Prepay | IRR | WAL | Notes |
|---|---:|---:|---:|---:|---:|---:|---|

### Breakevens (“what kills the deal”)
- Default breakeven (IRR falls below target):
- LGD breakeven:
- Recovery lag breakeven:
- Key driver attribution:

## 5. Monitoring Plan (half page)
- Weekly pack: delinq buckets, roll rates, vintage curves, prepay curve, CNL, recoveries, concentration
- Early warning triggers:
- Reporting cadence and governance:
