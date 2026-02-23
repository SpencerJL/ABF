# Case C: ABF Quant Platform Architecture (One-pager + detail)

## Objective
Build a scalable, auditable quant stack that supports the full ABF investment lifecycle:
- Ingest messy originator/servicer datasets
- Produce loan/pool cashflows + pricing + stress/breakevens for IC
- Monitor portfolios weekly with early warning indicators
- Maintain governance: reproducibility, versioning, testing, audit trail

## Core principles
1) Canonical data model + adapters per source
2) QA gates first (fail-fast) + exceptions log
3) Deterministic, reproducible model runs (data/config/code versioned)
4) Investor/IC-ready outputs (explainable drivers + breakevens)

## Canonical data model
- loans_current (one row per loan as-of)
- loans_history (loan-month performance)
- qc_exceptions (issue log per ingestion)
- recon_results (servicer/trustee reconciliation)
- cashflows_pool (pool cashflows per run_id)
- scenarios (stress definitions per run_id)
- monitoring_metrics (vintages, roll-rates, delinq, concentrations)

## Pipeline
1) Raw ingest (object store) + metadata
2) Source adapter -> canonical schema
3) QA checks (types, missingness, duplicates, outliers, roll-forward)
4) Reconciliation pack (aggregate flows vs servicer/trustee)
5) Model run (cashflow engine + scenarios) -> results tagged with run_id
6) Reporting layer: IC pack outputs + weekly monitoring dashboards

## Engineering choices (example)
- Python (pandas/numpy) for ingestion + modelling
- Postgres for canonical warehouse (or cloud DW later)
- Object store for raw + artifacts
- Orchestration: Airflow/Prefect
- CI/CD: unit tests + regression snapshots
- Containerised runs (Docker) when scaling

## Governance
- Run manifest: data hash, config hash, code commit, timestamp
- Assumption memos + change log for material changes
- Unit tests: accounting identities + monotonicity checks
- Regression tests: known tapes reproduce known outputs
