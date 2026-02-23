import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from abf_cashflow_engine import Assumptions, Stress, simulate_pool_cashflows, summarize_pool

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TAPE_PATH = DATA_DIR / "sample_loan_tape.csv"
PERF_PATH = DATA_DIR / "sample_monthly_performance.csv"
SERV_PATH = DATA_DIR / "sample_servicer_summary.json"


def pct(x: float, d: int = 2) -> str:
    return f"{100.0 * x:.{d}f}%"


def bp(x: float) -> str:
    return f"{x * 10000:.0f} bp"


def money(x: float) -> str:
    s = "-" if x < 0 else ""
    return f"{s}${abs(x):,.2f}"


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False, name=None):
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


def history_stats(perf: pd.DataFrame) -> dict:
    denom_smm = (perf["begin_balance"] - perf["scheduled_principal"]).replace(0.0, np.nan)
    smm = (perf["prepayment"] / denom_smm).clip(lower=0.0).fillna(0.0)
    denom_mdr = (perf["begin_balance"] - perf["scheduled_principal"] - perf["prepayment"]).replace(0.0, np.nan)
    mdr = (perf["default_principal"] / denom_mdr).clip(lower=0.0).fillna(0.0)
    rec_rate = (perf["recovery_cash"] / perf["default_principal"].replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.0)
    return {
        "months": int(len(perf)),
        "avg_cpr": float(1.0 - (1.0 - smm.mean()) ** 12),
        "avg_cdr": float(1.0 - (1.0 - mdr.mean()) ** 12),
        "avg_rec": float(rec_rate.mean()),
        "avg_lgd": float((1.0 - rec_rate).mean()),
    }


def qc_and_recon(tape: pd.DataFrame, perf: pd.DataFrame, serv: dict) -> tuple[dict, pd.DataFrame]:
    req_tape = {
        "loan_id", "product_type", "region", "origination_date", "as_of_date", "orig_balance", "current_balance",
        "coupon_rate", "orig_term_m", "remaining_term_m", "monthly_payment", "fico", "ltv", "status",
    }
    req_perf = {
        "month_end", "begin_balance", "interest", "scheduled_principal", "prepayment",
        "default_principal", "recovery_cash", "end_balance",
    }

    issues = []
    missing_tape = sorted(req_tape - set(tape.columns))
    missing_perf = sorted(req_perf - set(perf.columns))
    if missing_tape or missing_perf:
        issues.append({
            "Issue": "Missing required columns",
            "Count / exposure": f"tape:{len(missing_tape)} perf:{len(missing_perf)}",
            "Treatment": "Block run and fix schema mapping",
            "Estimated impact": "High",
        })

    dup_ids = int(tape.duplicated("loan_id").sum())
    dup_rows = int(tape.duplicated().sum())
    if dup_ids > 0 or dup_rows > 0:
        issues.append({
            "Issue": "Duplicate key rows",
            "Count / exposure": f"loan_id dup:{dup_ids}, row dup:{dup_rows}",
            "Treatment": "Deduplicate before valuation",
            "Estimated impact": "Medium",
        })

    t = tape.copy()
    t["origination_date"] = pd.to_datetime(t["origination_date"], errors="coerce")
    t["as_of_date"] = pd.to_datetime(t["as_of_date"], errors="coerce")
    bad_date = int(t["origination_date"].isna().sum() + t["as_of_date"].isna().sum())
    date_breach = int((t["origination_date"] > t["as_of_date"]).sum())

    unit_err = int(((tape["coupon_rate"] < 0) | (tape["coupon_rate"] > 1)).sum())
    unit_err += int(((tape["ltv"] < 0) | (tape["ltv"] > 2)).sum())
    unit_err += int(((tape["fico"] < 300) | (tape["fico"] > 850)).sum())
    unit_err += int((tape["remaining_term_m"] > tape["orig_term_m"]).sum())
    unit_err += int((tape["current_balance"] <= 0).sum())

    curr_gt_orig_mask = tape["current_balance"] > tape["orig_balance"]
    curr_gt_orig_n = int(curr_gt_orig_mask.sum())
    curr_gt_orig_exp = float(tape.loc[curr_gt_orig_mask, "current_balance"].sum())
    if curr_gt_orig_n > 0:
        issues.append({
            "Issue": "Current balance above original balance",
            "Count / exposure": f"{curr_gt_orig_n} loans / {money(curr_gt_orig_exp)}",
            "Treatment": "Confirm capitalization or fix tape",
            "Estimated impact": "Low to medium",
        })

    rhs = perf["begin_balance"] - perf["scheduled_principal"] - perf["prepayment"] - perf["default_principal"]
    roll_diff = float((perf["end_balance"] - rhs).abs().max())
    if roll_diff > 1e-2:
        issues.append({
            "Issue": "Roll-forward identity break",
            "Count / exposure": money(roll_diff),
            "Treatment": "Fix history feed before model run",
            "Estimated impact": "High",
        })

    tape_bal = float(tape["current_balance"].sum())
    tape_n = int(len(tape))
    tape_wac = float((tape["coupon_rate"] * tape["current_balance"]).sum() / tape_bal)
    serv_bal = float(serv.get("current_balance", 0.0))
    serv_n = int(serv.get("total_loans", 0))
    serv_wac = float(serv.get("wac", 0.0))
    perf_end = float(perf["end_balance"].iloc[-1])
    perf_cutoff_gap = perf_end - serv_bal
    if abs(perf_cutoff_gap) > max(1.0, 0.001 * max(serv_bal, 1.0)):
        issues.append({
            "Issue": "Performance/servicer cut-off mismatch",
            "Count / exposure": f"gap {money(perf_cutoff_gap)}",
            "Treatment": "Align cut-off date and population",
            "Estimated impact": "High for calibration confidence",
        })

    summary = {
        "missing_tape": missing_tape,
        "missing_perf": missing_perf,
        "dup_ids": dup_ids,
        "dup_rows": dup_rows,
        "bad_date": bad_date,
        "date_breach": date_breach,
        "unit_err": unit_err,
        "curr_gt_orig_n": curr_gt_orig_n,
        "curr_gt_orig_exp": curr_gt_orig_exp,
        "roll_diff": roll_diff,
        "tape_n": tape_n,
        "serv_n": serv_n,
        "n_diff": tape_n - serv_n,
        "tape_bal": tape_bal,
        "serv_bal": serv_bal,
        "bal_diff": tape_bal - serv_bal,
        "tape_wac": tape_wac,
        "serv_wac": serv_wac,
        "wac_diff": tape_wac - serv_wac,
    }
    ex_df = pd.DataFrame(issues, columns=["Issue", "Count / exposure", "Treatment", "Estimated impact"])
    return summary, ex_df

def run_scenarios(tape: pd.DataFrame, ass: Assumptions, purchase_price: float, horizon_m: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    stresses = [
        Stress("Base"),
        Stress("CDR x2", cdr_mult=2.0),
        Stress("LGD +10pp", lgd_add=0.10),
        Stress("Recovery lag +6m", recovery_lag_add_m=6),
        Stress("CPR down 40%", cpr_mult=0.6),
        Stress("Combined downside", cdr_mult=2.0, lgd_add=0.10, recovery_lag_add_m=6, cpr_mult=0.6),
    ]
    rows = []
    base_cf = None
    for st in stresses:
        cf = simulate_pool_cashflows(tape, ass, horizon_m=horizon_m, stress=st)
        sm = summarize_pool(cf, purchase_price=purchase_price, assumptions=ass)
        rows.append({
            "scenario": st.name,
            "cpr": ass.base_cpr * st.cpr_mult,
            "cdr": ass.base_cdr * st.cdr_mult,
            "lgd": float(np.clip(ass.lgd + st.lgd_add, 0.0, 1.0)),
            "recovery_lag_m": max(0, ass.recovery_lag_m + st.recovery_lag_add_m),
            "irr_ann": sm["irr_ann"],
            "npv": sm["npv"],
            "wal_years": sm["wal_years"],
        })
        if st.name == "Base":
            base_cf = cf.copy()
    return pd.DataFrame(rows), base_cf


def solve_breakevens(
    tape: pd.DataFrame,
    ass: Assumptions,
    purchase_price: float,
    horizon_m: int,
    target_irr: float,
) -> dict:
    cache = {}

    def irr(cdr_mult: float = 1.0, lgd: float | None = None, lag_add: int = 0) -> float:
        if lgd is None:
            lgd = ass.lgd
        key = (cdr_mult, lgd, lag_add)
        if key in cache:
            return cache[key]
        st = Stress("tmp", cdr_mult=cdr_mult, lgd_add=lgd - ass.lgd, recovery_lag_add_m=lag_add)
        cf = simulate_pool_cashflows(tape, ass, horizon_m=horizon_m, stress=st)
        r = summarize_pool(cf, purchase_price=purchase_price, assumptions=ass)["irr_ann"]
        cache[key] = float(r)
        return cache[key]

    base = irr()

    if base <= target_irr:
        cdr_out = {"status": "already_below"}
        lgd_out = {"status": "already_below"}
        lag_out = {"status": "already_below"}
    else:
        lo, hi = 1.0, 1.0
        while hi < 50 and irr(cdr_mult=hi) > target_irr:
            hi *= 1.25
        if irr(cdr_mult=hi) > target_irr:
            cdr_out = {"status": "not_found"}
        else:
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if irr(cdr_mult=mid) > target_irr:
                    lo = mid
                else:
                    hi = mid
            cdr_out = {"status": "ok", "cdr": ass.base_cdr * hi, "mult": hi}

        if irr(lgd=1.0) > target_irr:
            lgd_out = {"status": "not_found"}
        else:
            lo_lgd, hi_lgd = ass.lgd, 1.0
            for _ in range(50):
                mid = 0.5 * (lo_lgd + hi_lgd)
                if irr(lgd=mid) > target_irr:
                    lo_lgd = mid
                else:
                    hi_lgd = mid
            lgd_out = {"status": "ok", "lgd": hi_lgd}

        lag_hit = None
        for lag in range(0, 121):
            if irr(lag_add=lag) <= target_irr:
                lag_hit = ass.recovery_lag_m + lag
                break
        lag_out = {"status": "ok", "lag": lag_hit} if lag_hit is not None else {"status": "not_found"}

    return {"base": base, "default": cdr_out, "lgd": lgd_out, "lag": lag_out}


def fmt_be(be: dict, kind: str) -> str:
    if be["status"] == "already_below":
        return "Already below target in base case"
    if be["status"] == "not_found":
        return "No breach found in tested range"
    if kind == "cdr":
        return f"{pct(be['cdr'])} (mult {be['mult']:.2f}x base)"
    if kind == "lgd":
        return pct(be["lgd"])
    return f"{int(be['lag'])} months"


def monitoring_summary(tape: pd.DataFrame, perf: pd.DataFrame) -> dict:
    bal = float(tape["current_balance"].sum())
    status_bal = tape.groupby("status")["current_balance"].sum().sort_values(ascending=False) / bal
    product_bal = tape.groupby("product_type")["current_balance"].sum().sort_values(ascending=False) / bal
    region_bal = tape.groupby("region")["current_balance"].sum().sort_values(ascending=False) / bal
    vintage_bal = (
        tape.assign(vintage=pd.to_datetime(tape["origination_date"]).dt.year)
        .groupby("vintage")["current_balance"].sum().sort_values(ascending=False) / bal
    )

    denom_smm = (perf["begin_balance"] - perf["scheduled_principal"]).replace(0.0, np.nan)
    smm = (perf["prepayment"] / denom_smm).clip(0.0).fillna(0.0)
    cpr = 1.0 - (1.0 - smm) ** 12

    denom_mdr = (perf["begin_balance"] - perf["scheduled_principal"] - perf["prepayment"]).replace(0.0, np.nan)
    mdr = (perf["default_principal"] / denom_mdr).clip(0.0).fillna(0.0)
    cdr = 1.0 - (1.0 - mdr) ** 12

    net_loss = perf["default_principal"] - perf["recovery_cash"]
    cnl = float(net_loss.cumsum().iloc[-1] / perf["begin_balance"].iloc[0])
    rec_rate = float(perf["recovery_cash"].sum() / perf["default_principal"].sum()) if perf["default_principal"].sum() > 0 else 0.0

    d30 = float(tape.loc[tape["status"].isin(["30dpd", "60dpd", "90dpd", "Default"]), "current_balance"].sum() / bal)
    d60 = float(tape.loc[tape["status"].isin(["60dpd", "90dpd", "Default"]), "current_balance"].sum() / bal)

    return {
        "status_bal": status_bal,
        "product_bal": product_bal,
        "region_bal": region_bal,
        "vintage_bal": vintage_bal,
        "avg_cpr": float(cpr.mean()),
        "latest_cpr": float(cpr.iloc[-1]),
        "avg_cdr": float(cdr.mean()),
        "latest_cdr": float(cdr.iloc[-1]),
        "cnl": cnl,
        "rec_rate": rec_rate,
        "d30": d30,
        "d60": d60,
    }

def render_report(
    tape: pd.DataFrame,
    perf: pd.DataFrame,
    ass: Assumptions,
    purchase_price: float,
    target_irr: float,
    hist: dict,
    qc: dict,
    exceptions_df: pd.DataFrame,
    scenarios: pd.DataFrame,
    base_cf: pd.DataFrame,
    be: dict,
    mon: dict,
) -> str:
    bal = float(tape["current_balance"].sum())
    as_of = str(pd.to_datetime(tape["as_of_date"].iloc[0]).date())
    wac = float((tape["coupon_rate"] * tape["current_balance"]).sum() / bal)
    rem_term_y = float((tape["remaining_term_m"] * tape["current_balance"]).sum() / bal / 12.0)

    base = scenarios.loc[scenarios["scenario"] == "Base"].iloc[0]
    downside = scenarios.loc[scenarios["scenario"] == "Combined downside"].iloc[0]

    irr_cdr = float(scenarios.loc[scenarios["scenario"] == "CDR x2", "irr_ann"].iloc[0])
    irr_lgd = float(scenarios.loc[scenarios["scenario"] == "LGD +10pp", "irr_ann"].iloc[0])
    irr_lag = float(scenarios.loc[scenarios["scenario"] == "Recovery lag +6m", "irr_ann"].iloc[0])
    irr_cpr = float(scenarios.loc[scenarios["scenario"] == "CPR down 40%", "irr_ann"].iloc[0])

    cushion = float(base["irr_ann"] - target_irr)
    rec = "Conditional pass" if cushion >= 0 else "Do not proceed at current price"
    open_q = (
        "Confirm legal treatment for data exceptions before sign-off."
        if cushion >= 0 else
        "Re-cut purchase price or add structural protection to restore cushion."
    )

    top_prod = mon["product_bal"].head(1)
    top_reg = mon["region_bal"].head(1)
    top_prod_txt = f"{top_prod.index[0]} at {pct(float(top_prod.iloc[0]))}" if len(top_prod) else "n/a"
    top_reg_txt = f"{top_reg.index[0]} at {pct(float(top_reg.iloc[0]))}" if len(top_reg) else "n/a"

    ex_tbl = exceptions_df if len(exceptions_df) else pd.DataFrame([
        {"Issue": "None", "Count / exposure": "0", "Treatment": "n/a", "Estimated impact": "n/a"}
    ])

    stress_tbl = scenarios.copy()
    stress_tbl["Default"] = stress_tbl["cdr"].map(pct)
    stress_tbl["LGD"] = stress_tbl["lgd"].map(pct)
    stress_tbl["Recovery lag"] = stress_tbl["recovery_lag_m"].map(lambda x: f"{int(x)}m")
    stress_tbl["Prepay"] = stress_tbl["cpr"].map(pct)
    stress_tbl["IRR"] = stress_tbl["irr_ann"].map(pct)
    stress_tbl["WAL"] = stress_tbl["wal_years"].map(lambda x: f"{x:.2f}y")
    stress_tbl["Notes"] = [
        "Reference", "Higher defaults", "Higher severity", "Slower recoveries", "Slower prepay", "All downside levers"
    ]
    stress_tbl = stress_tbl[["scenario", "Default", "LGD", "Recovery lag", "Prepay", "IRR", "WAL", "Notes"]]
    stress_tbl.columns = ["Scenario", "Default", "LGD", "Recovery lag", "Prepay", "IRR", "WAL", "Notes"]

    cf_tbl = base_cf.head(12).copy()
    cf_tbl["Month"] = cf_tbl["month"].astype(int)
    cf_tbl["Begin bal"] = cf_tbl["begin_balance"].map(lambda x: f"{x/1_000_000:.2f}m")
    cf_tbl["Interest"] = cf_tbl["interest"].map(lambda x: f"{x/1_000_000:.2f}m")
    cf_tbl["Total principal"] = (cf_tbl["scheduled_principal"] + cf_tbl["prepayment"]).map(lambda x: f"{x/1_000_000:.2f}m")
    cf_tbl["Default"] = cf_tbl["default_principal"].map(lambda x: f"{x/1_000_000:.2f}m")
    cf_tbl["Recovery"] = cf_tbl["recovery_paid"].map(lambda x: f"{x/1_000_000:.2f}m")
    cf_tbl["Net cash"] = cf_tbl["net_cash"].map(lambda x: f"{x/1_000_000:.2f}m")
    cf_tbl["End bal"] = cf_tbl["end_balance"].map(lambda x: f"{x/1_000_000:.2f}m")
    cf_tbl = cf_tbl[["Month", "Begin bal", "Interest", "Total principal", "Default", "Recovery", "Net cash", "End bal"]]

    top_vintages = ", ".join(f"{int(k)} ({pct(float(v))})" for k, v in mon["vintage_bal"].head(3).items())
    top_regions = ", ".join(f"{k} ({pct(float(v))})" for k, v in mon["region_bal"].head(3).items())
    top_products = ", ".join(f"{k} ({pct(float(v))})" for k, v in mon["product_bal"].head(3).items())

    lines = []
    lines.append("# IC Pack (Auto-generated)")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append(
        f"- Deal / pool description: Mixed whole-loan forward-flow pool ({len(tape)} loans) as of {as_of}, current balance {money(bal)}, WAC {pct(wac)}, balance-weighted remaining term {rem_term_y:.2f} years."
    )
    lines.append(
        f"- Purchase price / target return: Purchase price {money(purchase_price)} ({purchase_price / bal:.2%} of current balance); target IRR {pct(target_irr)}."
    )
    lines.append(
        f"- Base case (IRR / NPV / WAL): {pct(float(base['irr_ann']))} / {money(float(base['npv']))} / {float(base['wal_years']):.2f} years."
    )
    lines.append(
        f"- Downside summary (stress returns and key breakevens): Combined downside IRR {pct(float(downside['irr_ann']))} (delta {bp(float(downside['irr_ann']) - float(base['irr_ann']))}), default breakeven {fmt_be(be['default'], 'cdr')}, LGD breakeven {fmt_be(be['lgd'], 'lgd')}, recovery lag breakeven {fmt_be(be['lag'], 'lag')}."
    )
    lines.append("- Top 3 risks + mitigants:")
    lines.append(f"  - Return cushion vs target is {bp(cushion)}; mitigant: tighten purchase price or reserve structure.")
    lines.append(f"  - Product concentration risk: {top_prod_txt}; mitigant: concentration limits and segment stress overlays.")
    lines.append(f"  - Geographic concentration risk: {top_reg_txt}; mitigant: region-level intake caps and trigger monitoring.")
    lines.append(f"- Recommendation / open questions: {rec}. Open question: {open_q}")
    lines.append("")
    lines.append("## 2. Data Quality & Reconciliation")
    lines.append("### Data received")
    lines.append(
        f"- Loan tape fields + history depth: {len(tape.columns)} tape fields; {hist['months']} months of aggregate performance history ({perf['month_end'].min()} to {perf['month_end'].max()})."
    )
    lines.append("- Servicer / trustee aggregates: total loans, current balance, WAC, WAL estimate, and status/region/product distributions were provided.")
    lines.append("- Definitions provided (default / charge-off / recovery): not explicitly defined in source files; this run treats `default_principal` as gross default flow and `recovery_cash` as recoveries.")
    lines.append("")
    lines.append("### QC checks performed")
    lines.append(
        f"- Schema & type checks: tape missing columns={len(qc['missing_tape'])}, perf missing columns={len(qc['missing_perf'])}, unparseable dates={qc['bad_date']}."
    )
    lines.append(
        f"- Duplicate / key integrity: duplicate loan_id={qc['dup_ids']}, duplicate rows={qc['dup_rows']}, origination after as-of breaches={qc['date_breach']}."
    )
    lines.append(
        f"- Missingness (by field, by vintage/segment): max field missing={int(tape.isna().sum().max())}; dataset-level missing ratio={float(tape.isna().mean().mean()):.2%}."
    )
    lines.append(
        f"- Outliers (unit errors vs true risk): unit-error flags={qc['unit_err']}; current>orig exceptions={qc['curr_gt_orig_n']} loans / {money(qc['curr_gt_orig_exp'])}."
    )
    lines.append(f"- Roll-forward identity checks: max absolute monthly difference={money(qc['roll_diff'])}.")
    lines.append(
        f"- Reconciliation to servicer/trustee (if available): loan-count diff={qc['n_diff']}, current-balance diff={money(qc['bal_diff'])}, WAC diff={bp(qc['wac_diff'])}."
    )
    lines.append("")
    lines.append("### Exceptions log")
    lines.append(md_table(ex_tbl))
    lines.append("")
    lines.append("## 3. Assumptions & Methodology")
    lines.append(f"- Prepayment (SMM/CPR curve; segmentation): base CPR {pct(ass.base_cpr)} (pool-level); trailing realized CPR {pct(hist['avg_cpr'])}.")
    lines.append(f"- Default (monthly default / CDR curve; segmentation): base CDR {pct(ass.base_cdr)} (pool-level); trailing realized CDR {pct(hist['avg_cdr'])}.")
    lines.append(f"- LGD / recovery rate + recovery lag: base LGD {pct(ass.lgd)} with recovery lag {ass.recovery_lag_m} months; trailing implied LGD {pct(hist['avg_lgd'])}, recovery rate {pct(hist['avg_rec'])}.")
    lines.append(f"- Fees (servicing, trustees, other): servicing fee {pct(ass.servicing_fee_ann)} annualized; trustee/other fees not separately modeled in v1.")
    lines.append(f"- Discounting / funding curve and cost of funds: flat annual discount rate {pct(ass.discount_rate_ann)}.")
    lines.append("- Notes on calibration and rationale: v1 uses a deterministic pool-level engine; next iteration should segment CPR/CDR/LGD by product, region, and delinquency state.")
    lines.append("")
    lines.append("## 4. Results")
    lines.append("### Base cashflow profile")
    lines.append("- Cashflow summary chart/table (first 12 months):")
    lines.append(md_table(cf_tbl))
    lines.append("")
    lines.append("### Stress & sensitivities")
    lines.append(md_table(stress_tbl))
    lines.append("")
    lines.append("### Breakevens (\"what kills the deal\")")
    lines.append(f"- Default breakeven (IRR falls below target): {fmt_be(be['default'], 'cdr')}.")
    lines.append(f"- LGD breakeven: {fmt_be(be['lgd'], 'lgd')}.")
    lines.append(f"- Recovery lag breakeven: {fmt_be(be['lag'], 'lag')}.")
    lines.append(
        f"- Key driver attribution: IRR deltas vs base are CDR x2 = {bp(irr_cdr - float(base['irr_ann']))}, LGD +10pp = {bp(irr_lgd - float(base['irr_ann']))}, CPR down 40% = {bp(irr_cpr - float(base['irr_ann']))}, recovery lag +6m = {bp(irr_lag - float(base['irr_ann']))}."
    )
    lines.append("")
    lines.append("## 5. Monitoring Plan")
    lines.append(
        f"- Weekly pack: delinq buckets (30+={pct(mon['d30'])}, 60+={pct(mon['d60'])}), vintage mix ({top_vintages}), prepay curve (avg CPR {pct(mon['avg_cpr'])}, latest {pct(mon['latest_cpr'])}), cumulative net loss {pct(mon['cnl'])}, trailing recovery rate {pct(mon['rec_rate'])}, concentration by region ({top_regions}) and product ({top_products})."
    )
    lines.append("- Early warning triggers: 30+ delinquency >10%, 60+ delinquency >6%, 3-month avg CPR <4% or >12%, 3-month avg CDR >4%, recovery rate <30%, and any concentration bucket >35%.")
    lines.append("- Reporting cadence and governance: weekly dashboard + monthly IC refresh; store run manifest with data/config/code versions and QC exceptions.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Case A IC pack artifacts.")
    parser.add_argument("--purchase-price-pct", type=float, default=0.98)
    parser.add_argument("--target-irr-ann", type=float, default=0.035)
    parser.add_argument("--horizon-m", type=int, default=120)
    parser.add_argument("--out-md", type=Path, default=DATA_DIR / "caseA_ic_pack.md")
    parser.add_argument("--out-scenarios", type=Path, default=DATA_DIR / "caseA_results.csv")
    parser.add_argument("--out-base-cf", type=Path, default=DATA_DIR / "caseA_base_cashflows.csv")
    parser.add_argument("--out-ex", type=Path, default=DATA_DIR / "caseA_qc_exceptions.csv")
    args = parser.parse_args()

    tape = pd.read_csv(TAPE_PATH)
    perf = pd.read_csv(PERF_PATH)
    serv = json.loads(SERV_PATH.read_text())

    ass = Assumptions(base_cpr=0.10, base_cdr=0.03, lgd=0.55, recovery_lag_m=3, servicing_fee_ann=0.010, discount_rate_ann=0.08)
    purchase_price = args.purchase_price_pct * float(tape["current_balance"].sum())

    hist = history_stats(perf)
    qc, ex_df = qc_and_recon(tape, perf, serv)
    scenarios, base_cf = run_scenarios(tape, ass, purchase_price, args.horizon_m)
    be = solve_breakevens(tape, ass, purchase_price, args.horizon_m, args.target_irr_ann)
    mon = monitoring_summary(tape, perf)

    report = render_report(tape, perf, ass, purchase_price, args.target_irr_ann, hist, qc, ex_df, scenarios, base_cf, be, mon)
    args.out_md.write_text(report, encoding="utf-8")
    scenarios.to_csv(args.out_scenarios, index=False)
    base_cf.to_csv(args.out_base_cf, index=False)
    ex_df.to_csv(args.out_ex, index=False)

    print(scenarios[["scenario", "irr_ann", "npv", "wal_years"]].to_string(index=False))
    print(f"\nIC pack markdown written to: {args.out_md}")
    print(f"Scenario results written to: {args.out_scenarios}")
    print(f"Base cashflows written to: {args.out_base_cf}")
    print(f"QC exceptions written to: {args.out_ex}")


if __name__ == "__main__":
    main()
