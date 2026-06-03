"""
visualize_results.py
Generate result plots and portfolio statistics for all trained datasets.

Usage:
    python visualize_results.py
    python visualize_results.py --dataset nasdaq100
    python visualize_results.py --model_label "Stockformer v2"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.log_parser import load_all_datasets
from src.portfolio  import compute_all_portfolios
from src import plots

ROOT         = Path(__file__).parent
RESULTS_ROOT = ROOT / "data" / "results2"
DATASETS     = ["nasdaq100", "wig60", "nasdaq100_extended"]


def _print_summary(all_logs: dict) -> None:
    for ds, logs in all_logs.items():
        tests = [d["test"] for d in logs.values() if d.get("test")]
        if not tests:
            continue
        print(f"\n  {ds}  ({len(tests)} windows with test results)")
        for key, label in [("cls_acc", "Cls Acc "),
                            ("reg_mae", "Reg MAE "),
                            ("reg_rmse", "Reg RMSE")]:
            vals = [t[key] for t in tests]
            print(f"    {label}  mean={np.mean(vals):.4f}  "
                  f"std={np.std(vals):.4f}  "
                  f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default=None, choices=DATASETS,
                        help="Process only one dataset (default: all).")
    parser.add_argument("--model_label", default="Stockformer",
                        help="Label used in plot titles (default: Stockformer).")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else DATASETS
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # ── Load all logs ─────────────────────────────────────────────────────────
    print("Loading training logs...")
    all_logs = load_all_datasets(datasets)
    for ds, logs in all_logs.items():
        n = sum(1 for d in logs.values() if d.get("test"))
        print(f"  {ds}: {len(logs)} windows, {n} with test results")

    _print_summary(all_logs)

    # ── Main: dataset-level comparison ───────────────────────────────────────
    print("\nGenerating main comparison figures...")
    plots.plot_metrics_overview(all_logs, RESULTS_ROOT, args.model_label)
    plots.plot_temporal_progression(all_logs, RESULTS_ROOT)
    plots.plot_training_summary(all_logs, RESULTS_ROOT)

    # ── Portfolio: aggregate across windows ───────────────────────────────────
    print("\nLoading prediction CSVs for portfolio analysis...")
    all_portfolio = {}
    for ds, logs in all_logs.items():
        windows   = sorted(logs.keys())
        port_data = compute_all_portfolios(ds, windows)
        if port_data:
            print(f"  {ds}: {len(port_data)} windows with prediction CSVs")
            all_portfolio[ds] = port_data
        else:
            print(f"  {ds}: no prediction CSVs — skipping portfolio")

    if all_portfolio:
        plots.plot_portfolio_summary(all_portfolio, RESULTS_ROOT)
        plots.plot_portfolio_wealth(all_portfolio, RESULTS_ROOT)

    print(f"\nDone.  Results saved under:  data/results/")
    print("  metrics_overview.png      — main dataset comparison")
    print("  temporal_progression.png  — metrics over time")
    print("  training_summary.png      — aggregated training curves")
    print("  portfolio_summary.png     — portfolio stats per dataset")


if __name__ == "__main__":
    main()
