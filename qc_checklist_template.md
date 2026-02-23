# Loan Tape QC & Reconciliation Checklist (Template)

## A. Schema & Types
- [ ] Required fields present (loan_id, balance, rate, dates, term, status, etc.)
- [ ] Data types parse correctly (dates, numerics)
- [ ] Units validated (rates in decimals vs %, balances in currency units)
- [ ] Allowed values for categorical fields (status, product_type, region)

## B. Key Integrity
- [ ] loan_id unique
- [ ] No duplicate rows (or duplicates explained)
- [ ] Stable fields consistent (origination_date <= as_of_date)

## C. Missingness
- [ ] Missingness report by field
- [ ] Missingness by segment (vintage, product_type, region)
- [ ] Identify systematic gaps; agree treatment (exclude vs proxy vs conservative)

## D. Outliers & Economic Sanity
- [ ] Negative / zero balances flagged
- [ ] Rates outside bounds flagged
- [ ] Term / remaining term sanity
- [ ] Attribute outliers (LTV/DTI/FICO) flagged
- [ ] Distinguish unit/encoding errors vs true risk tails

## E. Accounting / Roll-Forward
- [ ] If performance history exists: begin balance - flows = end balance (tolerance)
- [ ] Cashflow identity checks (principal conservation / monotonicity in stresses)

## F. Reconciliation to Servicer/Trustee
- [ ] Agree cut-off dates and population
- [ ] Reconcile: begin bal, sched prin, prepay, default/charge-off, recoveries, end bal
- [ ] Bucket recon differences (timing, definitions, population changes, data issues)
- [ ] Exceptions log created and signed off

## G. Audit Trail
- [ ] Data dictionary stored and versioned
- [ ] QC results stored with run manifest (data hash, config hash, code commit)
