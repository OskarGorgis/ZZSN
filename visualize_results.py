"""
visualize_results.py
Reads training logs for all datasets and generates result plots.

Usage:
    python visualize_results.py
    python visualize_results.py --dataset nasdaq100

Output: data/results/{dataset}/
"""

import argparse
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "data" / "processed"
RESULTS_ROOT = ROOT / "data" / "results"

DATASETS = ["nasdaq100", "wig60", "nasdaq100_extended"]

METRIC_STYLE = {
    "acc":  ("Accuracy",  "steelblue"),
    "mae":  ("MAE",       "coral"),
    "rmse": ("RMSE",      "mediumseagreen"),
    "mape": ("MAPE",      "mediumpurple"),
}


# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse_metrics(line):
    """Parse 'average, acc: X, mae: X, rmse: X, mape: X' → dict or None."""
    m = re.match(
        r"average, acc: ([^\s,]+), mae: ([^\s,]+), rmse: ([^\s,]+), mape: ([^\s,]+)",
        line,
    )
    if not m:
        return None
    return {k: float(v) for k, v in zip(("acc", "mae", "rmse", "mape"), m.groups())}


def parse_log(log_path):
    """Return training/val curves and final test metrics from one log file."""
    epochs, train_loss = [], []
    val_acc, val_mae, val_rmse = [], [], []
    test_metrics = None
    in_test = False

    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if "testing begin" in line:
                in_test = True
                continue

            m = re.match(r"epoch (\d+).*loss ([\d.]+)", line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                in_test = False
                continue

            metrics = _parse_metrics(line)
            if metrics:
                if in_test:
                    test_metrics = metrics
                else:
                    val_acc.append(metrics["acc"])
                    val_mae.append(metrics["mae"])
                    val_rmse.append(metrics["rmse"])

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_acc": val_acc,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "test_metrics": test_metrics,
    }


def load_dataset(dataset):
    log_dir = DATA_ROOT / dataset / "log"
    if not log_dir.exists():
        return {}
    return {
        f.name.replace("log_", ""): parse_log(f)
        for f in sorted(log_dir.iterdir())
        if f.is_file()
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def _bar_value_labels(ax, bars, values, fmt=".3f"):
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ax.get_ylim()[1] * 0.005,
            f"{val:{fmt}}",
            ha="center", va="bottom", fontsize=7,
        )


def plot_test_metrics(dataset, results, out_dir):
    """Bar chart of test acc / mae / rmse / mape across all windows."""
    valid = [(w, r["test_metrics"]) for w, r in results.items() if r["test_metrics"]]
    if not valid:
        print(f"  [skip] no test metrics found for {dataset}")
        return

    windows, test_data = zip(*valid)
    labels = [w[:7] for w in windows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{dataset} — Test metrics across rolling windows", fontsize=14, fontweight="bold")

    for ax, (key, (title, color)) in zip(axes.flat, METRIC_STYLE.items()):
        values = [d[key] for d in test_data]
        mean = np.mean(values)
        bars = ax.bar(x, values, color=color, alpha=0.8, width=0.65)
        ax.axhline(mean, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean:.4f}")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        _bar_value_labels(ax, bars, values)

    plt.tight_layout()
    path = out_dir / "test_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.relative_to(ROOT)}")


def plot_training_curves(dataset, results, out_dir):
    """Training loss + val MAE + val accuracy curves, all windows overlaid."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f"{dataset} — Training curves (all windows)", fontsize=14, fontweight="bold")

    cmap = plt.cm.tab20
    windows = list(results.keys())
    n = max(len(windows) - 1, 1)

    for i, (window, data) in enumerate(results.items()):
        color = cmap(i / n)
        label = window[:7]
        epochs = data["epochs"]
        if not epochs:
            continue
        axes[0].plot(epochs, data["train_loss"], color=color, alpha=0.75, linewidth=1.3, label=label)
        if data["val_mae"]:
            axes[1].plot(range(1, len(data["val_mae"]) + 1), data["val_mae"],
                         color=color, alpha=0.75, linewidth=1.3, label=label)
        if data["val_acc"]:
            axes[2].plot(range(1, len(data["val_acc"]) + 1), data["val_acc"],
                         color=color, alpha=0.75, linewidth=1.3, label=label)

    titles = ["Training Loss", "Validation MAE", "Validation Accuracy"]
    ylabels = ["Loss", "MAE", "Accuracy"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2, loc="upper right")

    axes[2].axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Random (0.5)")

    plt.tight_layout()
    path = out_dir / "training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.relative_to(ROOT)}")


def plot_best_val_vs_test(dataset, results, out_dir):
    """Scatter: best validation MAE vs test MAE per window."""
    windows, best_val, test_vals = [], [], []
    for w, data in results.items():
        if data["val_mae"] and data["test_metrics"]:
            windows.append(w[:7])
            best_val.append(min(data["val_mae"]))
            test_vals.append(data["test_metrics"]["mae"])

    if not windows:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(best_val, test_vals, c=range(len(windows)), cmap="tab20", s=80, zorder=3)
    for label, x, y in zip(windows, best_val, test_vals):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7)

    lo = min(min(best_val), min(test_vals)) * 0.95
    hi = max(max(best_val), max(test_vals)) * 1.05
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, alpha=0.6, label="y = x")
    ax.set_xlabel("Best Validation MAE")
    ax.set_ylabel("Test MAE")
    ax.set_title(f"{dataset} — Best val MAE vs test MAE", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "val_vs_test_mae.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.relative_to(ROOT)}")


def plot_dataset_comparison(all_results, out_dir):
    """Bar chart comparing average test metrics across datasets."""
    datasets = list(all_results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(datasets)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Cross-dataset comparison — Average test metrics (± std)", fontsize=14, fontweight="bold")

    for ax, key in zip(axes, ["acc", "mae", "rmse"]):
        title, _ = METRIC_STYLE[key]
        means, stds = [], []
        for ds in datasets:
            vals = [r["test_metrics"][key] for r in all_results[ds].values() if r["test_metrics"]]
            means.append(np.mean(vals) if vals else 0.0)
            stds.append(np.std(vals) if vals else 0.0)

        bars = ax.bar(datasets, means, yerr=stds, capsize=6, color=colors, alpha=0.85)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(title)
        ax.set_xticklabels([d.replace("_", "\n") for d in datasets], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + ax.get_ylim()[1] * 0.01,
                    f"{mean:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = out_dir / "dataset_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.relative_to(ROOT)}")


def print_summary(dataset, results):
    valid = [r["test_metrics"] for r in results.values() if r["test_metrics"]]
    if not valid:
        return
    for key, (title, _) in METRIC_STYLE.items():
        vals = [d[key] for d in valid]
        print(f"    {title:<12} mean={np.mean(vals):.4f}  std={np.std(vals):.4f}"
              f"  min={np.min(vals):.4f}  max={np.max(vals):.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, choices=DATASETS,
                        help="Process only one dataset (default: all).")
    args = parser.parse_args()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    datasets_to_run = [args.dataset] if args.dataset else DATASETS
    all_results = {}

    for dataset in datasets_to_run:
        print(f"\n{'='*55}")
        print(f"  {dataset}")
        print(f"{'='*55}")

        results = load_dataset(dataset)
        if not results:
            print("  No log files found — skipping.")
            continue

        n_complete = sum(1 for r in results.values() if r["test_metrics"])
        print(f"  Windows: {len(results)} total, {n_complete} with test results")
        print_summary(dataset, results)

        out_dir = RESULTS_ROOT / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_test_metrics(dataset, results, out_dir)
        plot_training_curves(dataset, results, out_dir)
        plot_best_val_vs_test(dataset, results, out_dir)

        all_results[dataset] = results

    if len(all_results) > 1:
        print(f"\n{'='*55}")
        print("  Cross-dataset comparison")
        print(f"{'='*55}")
        plot_dataset_comparison(all_results, RESULTS_ROOT)

    print(f"\nAll plots saved under: data/results/")


if __name__ == "__main__":
    main()
