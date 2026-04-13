from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def annual_to_monthly_hazard(rate: float) -> float:
    if rate <= 0.0:
        return 0.0
    if rate >= 1.0:
        return rate / 12.0
    return 1.0 - (1.0 - rate) ** (1.0 / 12.0)


def level_payment(balance: float, annual_rate: float, n_months: int) -> float:
    if n_months <= 0:
        return 0.0
    r = annual_rate / 12.0
    if abs(r) < 1e-12:
        return balance / n_months
    return balance * r / (1.0 - (1.0 + r) ** (-n_months))


@dataclass
class Tranche:
    name: str
    initial_balance: float
    margin: float
    rating: Optional[str] = None
    from_income: bool = False
    step_up_margin: float = 0.0
    balance: float = field(init=False)
    principal_cf: List[float] = field(default_factory=list)
    interest_due_cf: List[float] = field(default_factory=list)
    interest_cf: List[float] = field(default_factory=list)
    interest_shortfall_cf: List[float] = field(default_factory=list)
    loss_cf: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.balance = float(self.initial_balance)

    def clone(self) -> "Tranche":
        return Tranche(
            name=self.name,
            initial_balance=self.initial_balance,
            margin=self.margin,
            rating=self.rating,
            from_income=self.from_income,
            step_up_margin=self.step_up_margin,
        )


@dataclass(frozen=True)
class DealAssumptions:
    original_pool: float = 550_000_000.0
    wa_coupon: float = 0.0862
    remaining_term_months: int = 61
    servicing_fee: float = 0.0100
    recovery_rate: float = 0.355
    recovery_lag_months: int = 9
    cumulative_default: float = 0.029
    cpr_annual: float = 0.10
    bbsw_annual: float = 0.040
    swap_fixed_rate: float = 0.040
    liquidity_facility_pct: float = 0.015
    liquidity_facility_floor: float = 1_500_000.0
    stepdown_subordination_test: float = 0.12
    max_90dpd: float = 0.025
    stepdown_earliest_month: int = 25
    cum_default_stepdown_24m: float = 0.03
    cum_default_stepdown_after: float = 0.06
    call_month: int = 49
    tail_switch_balance_pct: float = 0.10
    ax_target_amort_months: int = 24


@dataclass(frozen=True)
class Scenario:
    name: str
    cumulative_default_mult: float = 1.0
    recovery_rate_mult: float = 1.0
    cpr_mult: float = 1.0
    bbsw_shift: float = 0.0
    default_timing: str = "base"  # base | front | back
    force_sequential: bool = False
    disable_call_tail_switch: bool = False
    arrears_90dpd: float = 0.0


@dataclass
class ScenarioResult:
    scenario: Scenario
    assumptions: DealAssumptions
    collateral: pd.DataFrame
    waterfall: pd.DataFrame
    tranche_summary: pd.DataFrame


def build_standard_tranches() -> List[Tranche]:
    return [
        Tranche("A-X", 9_000_000.0, 0.015, "AAA", from_income=True),
        Tranche("A", 473_000_000.0, 0.017, "AAA", step_up_margin=0.0025),
        Tranche("B1", 16_500_000.0, 0.028),
        Tranche("B2", 16_500_000.0, 0.028),
        Tranche("C1", 8_250_000.0, 0.040),
        Tranche("C2", 8_250_000.0, 0.040),
        Tranche("D", 5_500_000.0, 0.055),
        Tranche("E", 9_900_000.0, 0.075),
        Tranche("F", 8_800_000.0, 0.095),
        Tranche("G", 3_300_000.0, 0.000),
    ]


def default_scenarios() -> List[Scenario]:
    return [
        Scenario(name="Base"),
        Scenario(name="Front Loaded Defaults", default_timing="front"),
        Scenario(name="Recovery Stress", recovery_rate_mult=0.75),
        Scenario(name="High Prepay / Swap Mismatch", cpr_mult=2.0),
        Scenario(name="No Stepdown", force_sequential=True),
        Scenario(
            name="AAA Proxy",
            cumulative_default_mult=5.0,
            recovery_rate_mult=0.50,
            default_timing="front",
            cpr_mult=0.8,
        ),
    ]


def apply_scenario(base: DealAssumptions, scenario: Scenario) -> DealAssumptions:
    return replace(
        base,
        cumulative_default=max(0.0, base.cumulative_default * scenario.cumulative_default_mult),
        recovery_rate=float(np.clip(base.recovery_rate * scenario.recovery_rate_mult, 0.0, 1.0)),
        cpr_annual=max(0.0, base.cpr_annual * scenario.cpr_mult),
        bbsw_annual=max(0.0, base.bbsw_annual + scenario.bbsw_shift),
    )


class CollateralEngine:
    def __init__(self, assumptions: DealAssumptions, scenario: Scenario) -> None:
        self.a = assumptions
        self.scenario = scenario
        self.term = assumptions.remaining_term_months
        self.default_curve = self._default_curve(
            self.term,
            assumptions.cumulative_default,
            scenario.default_timing,
        )
        self.cpr_curve = np.full(self.term, annual_to_monthly_hazard(assumptions.cpr_annual))
        self.swap_notional_schedule = self._swap_notional_schedule(self.term, assumptions.original_pool)

    @staticmethod
    def _default_curve(term: int, cumulative_default: float, timing: str) -> np.ndarray:
        months = np.arange(1, term + 1, dtype=float)
        if timing == "front":
            weights = np.exp(-months / 10.0)
        elif timing == "back":
            weights = months ** 2
        else:
            weights = months * np.exp(-months / 18.0)
        weights /= weights.sum()
        return cumulative_default * weights

    @staticmethod
    def _swap_notional_schedule(term: int, initial_notional: float) -> np.ndarray:
        schedule = np.linspace(initial_notional, 0.0, term + 1)
        return schedule[:-1]

    def project(self) -> pd.DataFrame:
        a = self.a
        balance = a.original_pool
        scheduled_payment = level_payment(a.original_pool, a.wa_coupon, a.remaining_term_months)
        recoveries_queue = [0.0] * (a.recovery_lag_months + 1)
        rows: List[Dict[str, float]] = []
        cumulative_default_amount = 0.0
        cumulative_recoveries = 0.0

        for month in range(1, self.term + 1):
            bop = balance
            asset_interest = bop * a.wa_coupon / 12.0
            scheduled_interest = bop * a.wa_coupon / 12.0
            scheduled_principal = max(0.0, min(bop, scheduled_payment - scheduled_interest))

            gross_default = min(
                max(0.0, bop - scheduled_principal),
                a.original_pool * self.default_curve[month - 1],
            )
            post_default_balance = max(0.0, bop - scheduled_principal - gross_default)
            voluntary_prepay = post_default_balance * self.cpr_curve[month - 1]

            future_recovery = gross_default * a.recovery_rate
            recoveries_queue.append(future_recovery)
            recoveries = recoveries_queue.pop(0)

            eop = max(0.0, bop - scheduled_principal - voluntary_prepay - gross_default)
            balance = eop
            cumulative_default_amount += gross_default
            cumulative_recoveries += recoveries

            swap_notional = float(self.swap_notional_schedule[month - 1])
            swap_receive_float = swap_notional * a.bbsw_annual / 12.0
            swap_pay_fixed = swap_notional * a.swap_fixed_rate / 12.0
            swap_net = swap_receive_float - swap_pay_fixed

            rows.append(
                {
                    "month": float(month),
                    "bop_balance": bop,
                    "asset_interest": asset_interest,
                    "scheduled_principal": scheduled_principal,
                    "voluntary_prepay": voluntary_prepay,
                    "gross_default": gross_default,
                    "recoveries": recoveries,
                    "net_loss": gross_default - recoveries,
                    "principal_collections": scheduled_principal + voluntary_prepay + recoveries,
                    "eop_balance": eop,
                    "cum_default_ratio": cumulative_default_amount / a.original_pool,
                    "cum_recovery_ratio": cumulative_recoveries / a.original_pool,
                    "swap_notional": swap_notional,
                    "swap_mismatch_balance": swap_notional - bop,
                    "swap_net": swap_net,
                }
            )
        return pd.DataFrame(rows)


class WaterfallEngine:
    INTEREST_ORDER = ["A", "A-X", "B1", "B2", "C1", "C2", "D", "E", "F", "G"]
    LOSS_GROUPS = [["G"], ["F"], ["E"], ["D"], ["C1", "C2"], ["B1", "B2"], ["A"]]
    SEQ_GROUPS = [["A"], ["B1", "B2"], ["C1", "C2"], ["D"], ["E"], ["F"], ["G"]]
    PRO_RATA_CLASSES = ["A", "B1", "B2", "C1", "C2", "D", "E", "F"]
    LIQUIDITY_SUPPORTED = {"A-X", "A", "B1", "B2", "C1", "C2", "D", "E", "F"}
    LIQUIDITY_COVERED = ["A-X", "A", "B1", "B2", "C1", "C2", "D", "E", "F"]

    def __init__(
        self,
        assumptions: DealAssumptions,
        scenario: Scenario,
        collateral_df: pd.DataFrame,
        tranches: Iterable[Tranche],
    ) -> None:
        self.a = assumptions
        self.scenario = scenario
        self.col = collateral_df.copy()
        self.tranches: Dict[str, Tranche] = {t.name: t for t in tranches}
        self.records: List[Dict[str, float]] = []
        self.initial_note_balance_ex_ax = sum(t.initial_balance for t in tranches if t.name != "A-X")
        self.liquidity_outstanding = 0.0

    def total_notes_ex_ax(self) -> float:
        return sum(t.balance for name, t in self.tranches.items() if name != "A-X")

    def class_a_subordination_pct(self) -> float:
        denom = self.total_notes_ex_ax()
        if denom <= 0.0:
            return 0.0
        junior = sum(t.balance for name, t in self.tranches.items() if name not in {"A-X", "A"})
        return junior / denom

    def _interest_due(self, tranche: Tranche, month: int) -> float:
        margin = tranche.margin + (tranche.step_up_margin if month >= self.a.call_month else 0.0)
        return tranche.balance * (self.a.bbsw_annual + margin) / 12.0

    def _liquidity_limit(self) -> float:
        covered_balance = sum(
            self.tranches[name].balance for name in self.LIQUIDITY_COVERED if name in self.tranches
        )
        return max(self.a.liquidity_facility_floor, self.a.liquidity_facility_pct * covered_balance)

    def _split_pro_rata(self, names: Iterable[str], amount: float) -> Dict[str, float]:
        names = [n for n in names if n in self.tranches]
        alloc = {n: 0.0 for n in names}
        if amount <= 0.0:
            return alloc
        active = [n for n in names if self.tranches[n].balance > 1e-10]
        if not active:
            return alloc

        total_balance = sum(self.tranches[n].balance for n in active)
        if total_balance <= 0.0:
            return alloc

        allocated = 0.0
        for n in active:
            raw = amount * self.tranches[n].balance / total_balance
            pay = min(raw, self.tranches[n].balance)
            alloc[n] = pay
            allocated += pay

        residual = amount - allocated
        if residual > 1e-8:
            for n in active:
                room = self.tranches[n].balance - alloc[n]
                if room <= 0.0:
                    continue
                top_up = min(room, residual)
                alloc[n] += top_up
                residual -= top_up
                if residual <= 1e-8:
                    break
        return alloc

    def _apply_principal_group(
        self, names: Iterable[str], available_principal: float, principal_paid: Dict[str, float]
    ) -> float:
        names = [n for n in names if n in self.tranches]
        group_balance = sum(self.tranches[n].balance for n in names)
        if available_principal <= 0.0 or group_balance <= 0.0:
            return available_principal

        group_payment = min(available_principal, group_balance)
        alloc = self._split_pro_rata(names, group_payment)
        paid = 0.0
        for n, amt in alloc.items():
            if amt <= 0.0:
                continue
            self.tranches[n].balance -= amt
            principal_paid[n] += amt
            paid += amt
        return max(0.0, available_principal - paid)

    def _allocate_sequential(self, available_principal: float, principal_paid: Dict[str, float]) -> float:
        for group in self.SEQ_GROUPS[:-1]:
            available_principal = self._apply_principal_group(group, available_principal, principal_paid)
            if available_principal <= 1e-8:
                return 0.0
        return self._apply_principal_group(self.SEQ_GROUPS[-1], available_principal, principal_paid)

    def _allocate_pro_rata(self, available_principal: float, principal_paid: Dict[str, float]) -> float:
        active = [n for n in self.PRO_RATA_CLASSES if n in self.tranches and self.tranches[n].balance > 1e-10]
        if not active or available_principal <= 0.0:
            return self._apply_principal_group(["G"], available_principal, principal_paid)

        total_balance = sum(self.tranches[n].balance for n in active)
        allocated = 0.0
        for n in active:
            target = available_principal * self.tranches[n].balance / total_balance
            pay = min(target, self.tranches[n].balance)
            if pay <= 0.0:
                continue
            self.tranches[n].balance -= pay
            principal_paid[n] += pay
            allocated += pay

        residual = max(0.0, available_principal - allocated)
        if residual > 1e-8:
            for n in active:
                room = self.tranches[n].balance
                if room <= 0.0:
                    continue
                top_up = min(room, residual)
                self.tranches[n].balance -= top_up
                principal_paid[n] += top_up
                residual -= top_up
                if residual <= 1e-8:
                    break
        return self._apply_principal_group(["G"], max(0.0, residual), principal_paid)

    def _allocate_losses(self, net_loss: float) -> tuple[Dict[str, float], float]:
        losses = {n: 0.0 for n in self.tranches}
        remaining = max(0.0, net_loss)
        for group in self.LOSS_GROUPS:
            names = [n for n in group if n in self.tranches]
            group_balance = sum(self.tranches[n].balance for n in names)
            if remaining <= 0.0 or group_balance <= 0.0:
                continue
            hit = min(remaining, group_balance)
            alloc = self._split_pro_rata(names, hit)
            paid = 0.0
            for n, amt in alloc.items():
                if amt <= 0.0:
                    continue
                self.tranches[n].balance -= amt
                losses[n] += amt
                paid += amt
            remaining = max(0.0, remaining - paid)
        return losses, remaining

    def _stepdown_allowed(self, month: int, cum_default_ratio: float) -> tuple[bool, str]:
        if self.scenario.force_sequential:
            return False, "forced_sequential"
        if month < self.a.stepdown_earliest_month:
            return False, "stepdown_lockout"
        tail_trigger = self.total_notes_ex_ax() <= self.initial_note_balance_ex_ax * self.a.tail_switch_balance_pct
        if not self.scenario.disable_call_tail_switch and month >= self.a.call_month:
            return False, "call_period"
        if not self.scenario.disable_call_tail_switch and tail_trigger:
            return False, "tail_trigger"
        if self.class_a_subordination_pct() < self.a.stepdown_subordination_test:
            return False, "subordination_test"
        if self.scenario.arrears_90dpd > self.a.max_90dpd:
            return False, "arrears_test"
        if month <= 24 and cum_default_ratio > self.a.cum_default_stepdown_24m:
            return False, "cum_default_24m"
        if month > 24 and cum_default_ratio > self.a.cum_default_stepdown_after:
            return False, "cum_default_after_24m"
        return True, "ok"

    def run(self) -> pd.DataFrame:
        for _, row in self.col.iterrows():
            month = int(row["month"])
            available_income = float(row["asset_interest"] + row["swap_net"])
            available_principal = float(row["principal_collections"])
            net_loss = float(max(0.0, row["net_loss"]))
            servicing_fee = float(row["bop_balance"] * self.a.servicing_fee / 12.0)
            available_income = max(0.0, available_income - servicing_fee)

            interest_due = {name: 0.0 for name in self.tranches}
            interest_paid = {name: 0.0 for name in self.tranches}
            interest_shortfall = {name: 0.0 for name in self.tranches}
            principal_paid = {name: 0.0 for name in self.tranches}
            liquidity_draw = 0.0

            def pay_interest(name: str) -> None:
                nonlocal available_income, liquidity_draw
                if name not in self.tranches:
                    return
                tranche = self.tranches[name]
                due = self._interest_due(tranche, month)
                interest_due[name] = due
                paid_cash = min(available_income, due)
                available_income -= paid_cash
                shortfall = due - paid_cash

                if shortfall > 0.0 and name in self.LIQUIDITY_SUPPORTED:
                    room = max(0.0, self._liquidity_limit() - self.liquidity_outstanding)
                    draw = min(shortfall, room)
                    self.liquidity_outstanding += draw
                    liquidity_draw += draw
                    paid_cash += draw
                    shortfall -= draw

                interest_paid[name] = paid_cash
                interest_shortfall[name] = shortfall

            pay_interest("A")
            pay_interest("A-X")

            ax_principal_from_income = 0.0
            if "A-X" in self.tranches and self.tranches["A-X"].balance > 0.0:
                ax = self.tranches["A-X"]
                target = ax.initial_balance / max(1, self.a.ax_target_amort_months)
                ax_principal_from_income = min(available_income, ax.balance, target)
                ax.balance -= ax_principal_from_income
                principal_paid["A-X"] += ax_principal_from_income
                available_income -= ax_principal_from_income

            for name in ["B1", "B2", "C1", "C2", "D", "E", "F", "G"]:
                pay_interest(name)

            excess_to_g = 0.0
            if "G" in self.tranches and available_income > 0.0:
                excess_to_g = available_income
                interest_paid["G"] += excess_to_g
                available_income = 0.0

            loss_paid, unallocated_loss = self._allocate_losses(net_loss)

            liquidity_repay = min(available_principal, self.liquidity_outstanding)
            self.liquidity_outstanding -= liquidity_repay
            available_principal -= liquidity_repay

            step_ok, step_reason = self._stepdown_allowed(month, float(row["cum_default_ratio"]))
            principal_mode = "pro_rata" if step_ok else "sequential"
            if self.scenario.force_sequential:
                principal_mode = "sequential"

            if principal_mode == "sequential":
                available_principal = self._allocate_sequential(available_principal, principal_paid)
            else:
                available_principal = self._allocate_pro_rata(available_principal, principal_paid)

            for name, tranche in self.tranches.items():
                tranche.interest_due_cf.append(interest_due[name])
                tranche.interest_cf.append(interest_paid[name])
                tranche.interest_shortfall_cf.append(interest_shortfall[name])
                tranche.principal_cf.append(principal_paid[name])
                tranche.loss_cf.append(loss_paid[name])

            rec: Dict[str, float] = {
                "month": float(month),
                "principal_mode": principal_mode,
                "stepdown_reason": step_reason,
                "servicing_fee": servicing_fee,
                "ax_principal_from_income": ax_principal_from_income,
                "excess_to_G": excess_to_g,
                "liquidity_limit": self._liquidity_limit(),
                "liquidity_draw": liquidity_draw,
                "liquidity_repay": liquidity_repay,
                "liquidity_outstanding": self.liquidity_outstanding,
                "class_a_subordination_pct": self.class_a_subordination_pct(),
                "total_notes_ex_ax": self.total_notes_ex_ax(),
                "unallocated_loss": unallocated_loss,
                "unallocated_principal": available_principal,
                "swap_mismatch_balance": float(row["swap_mismatch_balance"]),
            }
            for name in self.tranches:
                rec[f"int_due_{name}"] = interest_due[name]
                rec[f"int_paid_{name}"] = interest_paid[name]
                rec[f"int_short_{name}"] = interest_shortfall[name]
                rec[f"prin_{name}"] = principal_paid[name]
                rec[f"loss_{name}"] = loss_paid[name]
                rec[f"bal_{name}"] = self.tranches[name].balance
            self.records.append(rec)

        return pd.DataFrame(self.records)


def weighted_average_life(principal_cfs: List[float], initial_balance: float) -> float:
    if initial_balance <= 0.0:
        return 0.0
    principal = np.asarray(principal_cfs, dtype=float)
    if principal.size == 0:
        return 0.0
    months = np.arange(1, principal.size + 1, dtype=float)
    return float((months * principal).sum() / initial_balance / 12.0)


def _irr_newton(cashflows: np.ndarray, guess: float = 0.01) -> float:
    r = guess
    for _ in range(100):
        npv = 0.0
        deriv = 0.0
        for i, cf in enumerate(cashflows):
            disc = (1.0 + r) ** i
            npv += cf / disc
            if i > 0:
                deriv -= i * cf / ((1.0 + r) ** (i + 1))
        if abs(npv) < 1e-8:
            return float(r)
        if abs(deriv) < 1e-12:
            break
        next_r = r - npv / deriv
        if next_r <= -0.9999:
            break
        r = next_r
    return float("nan")


def tranche_summary_dataframe(tranches: Iterable[Tranche]) -> pd.DataFrame:
    rows = []
    for tranche in tranches:
        cashflows = np.array(
            [-tranche.initial_balance]
            + [i + p for i, p in zip(tranche.interest_cf, tranche.principal_cf)],
            dtype=float,
        )
        irr_m = _irr_newton(cashflows)
        irr_a = float((1.0 + irr_m) ** 12 - 1.0) if np.isfinite(irr_m) else float("nan")
        rows.append(
            {
                "tranche": tranche.name,
                "initial_balance": tranche.initial_balance,
                "ending_balance": tranche.balance,
                "total_interest": float(np.sum(tranche.interest_cf)),
                "total_principal": float(np.sum(tranche.principal_cf)),
                "total_losses": float(np.sum(tranche.loss_cf)),
                "interest_shortfall": float(np.sum(tranche.interest_shortfall_cf)),
                "wal_years": weighted_average_life(tranche.principal_cf, tranche.initial_balance),
                "simple_irr_annual": irr_a,
            }
        )
    return pd.DataFrame(rows)


def run_plenti_scenario(
    base_assumptions: DealAssumptions,
    scenario: Scenario,
    tranche_template: Optional[List[Tranche]] = None,
) -> ScenarioResult:
    assumptions = apply_scenario(base_assumptions, scenario)
    tranches = [t.clone() for t in (tranche_template or build_standard_tranches())]
    collateral = CollateralEngine(assumptions, scenario).project()
    waterfall_engine = WaterfallEngine(assumptions, scenario, collateral, tranches)
    waterfall = waterfall_engine.run()
    summary = tranche_summary_dataframe(waterfall_engine.tranches.values())
    return ScenarioResult(
        scenario=scenario,
        assumptions=assumptions,
        collateral=collateral,
        waterfall=waterfall,
        tranche_summary=summary,
    )


def run_case_study(
    base_assumptions: Optional[DealAssumptions] = None,
    scenarios: Optional[List[Scenario]] = None,
    tranche_template: Optional[List[Tranche]] = None,
) -> tuple[Dict[str, ScenarioResult], pd.DataFrame]:
    assumptions = base_assumptions or DealAssumptions()
    scenario_list = scenarios or default_scenarios()
    template = tranche_template or build_standard_tranches()
    results: Dict[str, ScenarioResult] = {}
    matrix_rows: List[Dict[str, float]] = []

    for scenario in scenario_list:
        result = run_plenti_scenario(assumptions, scenario, template)
        results[scenario.name] = result
        collateral = result.collateral
        waterfall = result.waterfall
        summary = result.tranche_summary.set_index("tranche")

        stepdown_month = float("nan")
        if (waterfall["principal_mode"] == "pro_rata").any():
            stepdown_month = float(waterfall.loc[waterfall["principal_mode"] == "pro_rata", "month"].iloc[0])

        gross_default_pct = float(collateral["gross_default"].sum() / assumptions.original_pool)
        net_loss_pct = float((collateral["gross_default"].sum() - collateral["recoveries"].sum()) / assumptions.original_pool)
        matrix_rows.append(
            {
                "scenario": scenario.name,
                "gross_default_pct": gross_default_pct,
                "net_loss_pct": net_loss_pct,
                "stepdown_month": stepdown_month,
                "class_A_loss": float(summary.loc["A", "total_losses"]),
                "class_G_loss": float(summary.loc["G", "total_losses"]),
                "class_A_wal_years": float(summary.loc["A", "wal_years"]),
                "class_G_irr_annual": float(summary.loc["G", "simple_irr_annual"]),
                "residual_cash_to_G": float(summary.loc["G", "total_interest"]),
            }
        )
    return results, pd.DataFrame(matrix_rows)
