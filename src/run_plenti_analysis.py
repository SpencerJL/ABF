import argparse
from pathlib import Path

from plenti_abs_analysis import default_scenarios, run_case_study


def slugify(value: str) -> str:
    parts = []
    for ch in value.lower():
        if ch.isalnum():
            parts.append(ch)
        else:
            parts.append("_")
    slug = "".join(parts).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Plenti Auto ABS 2025-2 case-study analysis.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "plenti_analysis",
        help="Directory for scenario outputs.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Scenario name to run (exact, case-insensitive) or 'all'.",
    )
    args = parser.parse_args()

    scenarios = default_scenarios()
    if args.scenario.lower() != "all":
        wanted = args.scenario.lower()
        scenarios = [s for s in scenarios if s.name.lower() == wanted]
        if not scenarios:
            raise ValueError(f"Scenario '{args.scenario}' not found.")

    results, matrix = run_case_study(scenarios=scenarios)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.out_dir / "scenario_matrix.csv", index=False)

    for scenario_name, result in results.items():
        tag = slugify(scenario_name)
        result.collateral.to_csv(args.out_dir / f"{tag}_collateral.csv", index=False)
        result.waterfall.to_csv(args.out_dir / f"{tag}_waterfall.csv", index=False)
        result.tranche_summary.to_csv(args.out_dir / f"{tag}_tranche_summary.csv", index=False)

    print(matrix.to_string(index=False))
    print(f"\nOutputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()

