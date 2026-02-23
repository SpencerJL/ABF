import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

def smm_from_cpr(cpr: float) -> float:
    return 1.0 - (1.0 - cpr) ** (1.0 / 12.0)

@dataclass(frozen=True)
class Assumptions:
    base_cpr: float = 0.10
    base_cdr: float = 0.03
    lgd: float = 0.55
    recovery_lag_m: int = 3
    servicing_fee_ann: float = 0.010
    discount_rate_ann: float = 0.08

@dataclass(frozen=True)
class Stress:
    name: str
    cpr_mult: float = 1.0
    cdr_mult: float = 1.0
    lgd_add: float = 0.0
    recovery_lag_add_m: int = 0

def _level_payment(balance: float, annual_rate: float, n_months: int) -> float:
    r = annual_rate / 12.0
    if n_months <= 0:
        return 0.0
    if abs(r) < 1e-12:
        return balance / n_months
    return balance * (r * (1 + r) ** n_months) / ((1 + r) ** n_months - 1)

def simulate_pool_cashflows(
    tape: pd.DataFrame,
    assumptions: Assumptions,
    horizon_m: int = 120,
    stress: Optional[Stress] = None,
) -> pd.DataFrame:
    s = stress or Stress(name="Base")
    cpr = max(0.0, assumptions.base_cpr * s.cpr_mult)
    cdr = max(0.0, assumptions.base_cdr * s.cdr_mult)
    lgd = float(np.clip(assumptions.lgd + s.lgd_add, 0.0, 1.0))
    rec_lag = max(0, assumptions.recovery_lag_m + s.recovery_lag_add_m)

    smm = smm_from_cpr(cpr)
    mdr = smm_from_cpr(cdr)  # monthly default rate proxy

    bal = tape["current_balance"].astype(float).to_numpy().copy()
    rate = tape["coupon_rate"].astype(float).to_numpy()
    rem = tape["remaining_term_m"].astype(int).to_numpy()

    if "monthly_payment" in tape.columns:
        pmt = tape["monthly_payment"].astype(float).to_numpy()
        mask = (pmt <= 0) | ~np.isfinite(pmt)
        if mask.any():
            idx = np.where(mask)[0]
            for i in idx:
                pmt[i] = _level_payment(bal[i], rate[i], int(rem[i]))
    else:
        pmt = np.array([_level_payment(bal[i], rate[i], int(rem[i])) for i in range(len(bal))])

    rec_queue: List[Tuple[int, float]] = []
    rows = []

    for t in range(1, horizon_m + 1):
        begin_bal = float(bal.sum())
        interest = float((bal * (rate / 12.0)).sum())
        serv_fee = float(begin_bal * (assumptions.servicing_fee_ann / 12.0))

        sched_prin = np.minimum(pmt - (bal * (rate / 12.0)), bal)
        sched_prin = np.maximum(sched_prin, 0.0)
        sched_prin_amt = float(sched_prin.sum())

        bal_after_sched = bal - sched_prin

        prepay = smm * bal_after_sched
        prepay_amt = float(prepay.sum())
        bal_after_prepay = bal_after_sched - prepay

        default_prin = mdr * bal_after_prepay
        default_amt = float(default_prin.sum())

        recovery_cash = default_amt * (1.0 - lgd)
        loss_amt = default_amt * lgd
        if recovery_cash > 0:
            rec_queue.append((t + rec_lag, recovery_cash))

        bal = np.maximum(bal_after_prepay - default_prin, 0.0)

        rec_paid = 0.0
        if rec_queue:
            still = []
            for pay_month, amt in rec_queue:
                if pay_month == t:
                    rec_paid += amt
                else:
                    still.append((pay_month, amt))
            rec_queue = still

        end_bal = float(bal.sum())
        total_principal = sched_prin_amt + prepay_amt
        net_cash = (interest - serv_fee) + total_principal + rec_paid

        rows.append({
            "month": t,
            "begin_balance": begin_bal,
            "interest": interest,
            "servicing_fee": serv_fee,
            "scheduled_principal": sched_prin_amt,
            "prepayment": prepay_amt,
            "default_principal": default_amt,
            "loss": loss_amt,
            "recovery_paid": rec_paid,
            "end_balance": end_bal,
            "net_cash": net_cash,
        })

        rem = np.maximum(rem - 1, 0)
        if end_bal < 1e-6:
            break

    return pd.DataFrame(rows)

def _irr_newton(cash: np.ndarray, guess: float = 0.01) -> float:
    r = guess
    for _ in range(100):
        npv = 0.0
        d = 0.0
        for i, cf in enumerate(cash):
            npv += cf / ((1 + r) ** i)
            if i > 0:
                d -= i * cf / ((1 + r) ** (i + 1))
        if abs(npv) < 1e-8:
            return float(r)
        if abs(d) < 1e-12:
            break
        r -= npv / d
    return float(r)

def summarize_pool(cf_df: pd.DataFrame, purchase_price: float, assumptions: Assumptions) -> Dict[str, float]:
    cash = np.concatenate(([-purchase_price], cf_df["net_cash"].to_numpy()))
    irr_m = _irr_newton(cash)
    irr_a = (1 + irr_m) ** 12 - 1
    disc_m = assumptions.discount_rate_ann / 12.0
    npv = float(sum(cf / ((1 + disc_m) ** i) for i, cf in enumerate(cash)))
    prin = cf_df["scheduled_principal"].to_numpy() + cf_df["prepayment"].to_numpy()
    t = cf_df["month"].to_numpy()
    total = prin.sum()
    wal_y = float(((t * prin).sum() / total) / 12.0) if total > 0 else 0.0
    return {"irr_ann": float(irr_a), "npv": float(npv), "wal_years": wal_y}
