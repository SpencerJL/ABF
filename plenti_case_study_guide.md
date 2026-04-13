# Plenti Auto ABS 2025-2 Case Study Guide

## 1. Case Introduction

This case models the **Plenti Auto ABS Trust 2025-2** transaction as a pool-level cashflow engine suitable for interview and early-stage buy-side analysis.

The objective is to convert presale-style transaction facts into a transparent analytical workflow:

- project collateral cashflows (scheduled amortization, defaults, recoveries, prepayments),
- apply structural waterfall rules (income vs principal, A-X treatment, loss allocation),
- test scenario sensitivities (timing, recovery, prepay, step-down behavior),
- produce tranche-level outputs (losses, WAL, cash profiles, simple IRR).

The model is designed as a **foundation** for investment screening and scenario framing, not a legal-document-precision trustee engine.

---

## 2. Code Map

Main files:

- `src/plenti_abs_analysis.py`: core modelling framework.
- `src/run_plenti_analysis.py`: command-line runner for full scenario output.
- `tests/test_plenti_abs_analysis.py`: regression checks on accounting identity and scenario behavior.

Core objects in `plenti_abs_analysis.py`:

- `DealAssumptions`: base deal inputs (pool size, coupon, loss/recovery assumptions, trigger settings, call month).
- `Scenario`: stress knobs (default multiplier, recovery multiplier, CPR multiplier, timing shape, force-sequential).
- `Tranche`: tranche state and cashflow history.
- `CollateralEngine`: projects monthly collateral state and swap placeholder metrics.
- `WaterfallEngine`: allocates income/principal/losses to liabilities under trigger logic.
- `run_plenti_scenario` / `run_case_study`: orchestration and scenario matrix generation.

---

## 3. How the Engine Works

### 3.1 Collateral projection

For each month:

1. Start from beginning balance.
2. Calculate scheduled principal using a level-payment profile.
3. Apply gross defaults from a timing curve (base/front/back-loaded).
4. Apply voluntary prepayments via CPR-to-monthly conversion.
5. Queue recoveries with lag (default 9 months).
6. Track ending balance, cumulative defaults/recoveries, and swap mismatch placeholder fields.

### 3.2 Waterfall allocation

For each month:

1. Build available income from asset interest plus swap net placeholder, less servicing fee.
2. Pay note interest by priority (A, A-X, then mezz/junior classes).
3. Apply A-X scheduled principal from income before mezz/junior interest layers.
4. Send residual excess spread to G.
5. Allocate realized net losses bottom-up (G first, then up the stack).
6. Repay liquidity draws from principal collections.
7. Allocate principal using sequential or pro-rata mode depending on step-down tests.

Step-down behavior:

- Earliest step-down month lockout is month 25.
- Returns to sequential under call/tail conditions.
- Can be forced sequential via scenario (`No Stepdown`).

---

## 4. Scenario Set Included

`default_scenarios()` currently runs:

- Base
- Front Loaded Defaults
- Recovery Stress
- High Prepay / Swap Mismatch
- No Stepdown
- AAA Proxy

Each scenario produces:

- collateral monthly path,
- waterfall monthly path,
- tranche summary (interest/principal/loss/WAL/simple IRR),
- one row in the cross-scenario matrix.

---

## 5. Running the Analysis

Run all scenarios:

```bash
python src/run_plenti_analysis.py
```

Run a single scenario by exact name:

```bash
python src/run_plenti_analysis.py --scenario "Recovery Stress"
```

Outputs are written to:

- `data/plenti_analysis/scenario_matrix.csv`
- `data/plenti_analysis/*_collateral.csv`
- `data/plenti_analysis/*_waterfall.csv`
- `data/plenti_analysis/*_tranche_summary.csv`

---

## 6. Validation Tests

Run:

```bash
python -m pytest -q
```

Tests cover:

- collateral roll-forward identity,
- forced-sequential scenario behavior,
- scenario matrix structure.

---

## 7. Practical Interpretation

Use the outputs to answer interview/investment questions:

- Which assumptions hurt equity (`G`) most?
- How much timing risk comes from front-loaded defaults?
- Does pro-rata step-down materially change WAL/cash timing?
- How sensitive is residual cash to recovery and prepayment assumptions?

---

## 8. Current Simplifications

Important simplifications (intentional in this version):

- pool-level model (not loan-level heterogeneity),
- stylized swap placeholder (not full market-curve swap cashflows),
- simplified liquidity/ledger mechanics,
- no tax/accounting/legal entity cash-trap detail.

Recommended next upgrades:

- loan tape segmentation and calibration by cohort,
- explicit swap leg and basis/day-count handling,
- document-precision principal draw/reimbursement ledgers,
- richer partner/IC reporting layer on top of `scenario_matrix.csv`.

