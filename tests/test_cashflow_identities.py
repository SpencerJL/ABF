import pandas as pd
from pathlib import Path
from src.abf_cashflow_engine import Assumptions, Stress, simulate_pool_cashflows

def test_roll_forward_identity():
    tape = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "sample_loan_tape.csv")
    cf = simulate_pool_cashflows(tape, Assumptions(), horizon_m=24, stress=Stress(name="Base"))

    tol = 1e-6
    rhs = cf["begin_balance"] - cf["scheduled_principal"] - cf["prepayment"] - cf["default_principal"]
    diff = (cf["end_balance"] - rhs).abs().max()
    assert diff < 1e-2, f"Roll-forward identity failed: max diff={diff}"
