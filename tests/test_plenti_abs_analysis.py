from src.plenti_abs_analysis import DealAssumptions, Scenario, run_case_study, run_plenti_scenario


def test_collateral_roll_forward_identity() -> None:
    result = run_plenti_scenario(DealAssumptions(), Scenario(name="Base"))
    cf = result.collateral
    rhs = cf["bop_balance"] - cf["scheduled_principal"] - cf["voluntary_prepay"] - cf["gross_default"]
    diff = (cf["eop_balance"] - rhs).abs().max()
    assert diff < 1e-2, f"Collateral roll-forward failed: max diff={diff}"


def test_no_stepdown_scenario_stays_sequential() -> None:
    result = run_plenti_scenario(DealAssumptions(), Scenario(name="No Stepdown", force_sequential=True))
    assert (result.waterfall["principal_mode"] == "sequential").all()


def test_case_study_returns_scenario_matrix() -> None:
    results, matrix = run_case_study()
    assert "Base" in results
    assert len(results) >= 5
    assert not matrix.empty
    expected_cols = {
        "scenario",
        "gross_default_pct",
        "net_loss_pct",
        "stepdown_month",
        "class_A_loss",
        "class_G_loss",
    }
    assert expected_cols.issubset(set(matrix.columns))

