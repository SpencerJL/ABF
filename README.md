# ABF Homework Kit (Mock)

This kit contains three mock homework/case formats commonly used for ABF / structured credit quant roles:

- Case A: Loan tape + history -> produce an IC-style memo pack
- Case B: Build a pool cashflow + stress engine (Python v1 engine included)
- Case C: Design a scalable, governed quant platform architecture

## Contents
- data/
  - sample_loan_tape.csv
  - sample_monthly_performance.csv
  - sample_servicer_summary.json
- ic_pack_template.md
- qc_checklist_template.md
- src/
  - abf_cashflow_engine.py
  - run_case_A.py
  - run_case_B.py
- tests/
  - test_cashflow_identities.py
- architecture_case_C.md

## Quick start
From the extracted folder:

    python src/run_case_A.py
    python src/run_case_B.py
