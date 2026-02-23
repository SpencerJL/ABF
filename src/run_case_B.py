import pandas as pd
from pathlib import Path
from abf_cashflow_engine import Assumptions, Stress, simulate_pool_cashflows, summarize_pool

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def main():
    tape = pd.read_csv(DATA_DIR / "sample_loan_tape.csv")
    pool_bal = float(tape["current_balance"].sum())
    purchase_price = 0.985 * pool_bal

    ass = Assumptions(base_cpr=0.12, base_cdr=0.035, lgd=0.60, recovery_lag_m=4, servicing_fee_ann=0.012, discount_rate_ann=0.09)
    cf = simulate_pool_cashflows(tape, ass, horizon_m=120, stress=Stress(name="Base"))
    summ = summarize_pool(cf, purchase_price=purchase_price, assumptions=ass)
    print("Summary:", summ)

if __name__ == "__main__":
    main()
