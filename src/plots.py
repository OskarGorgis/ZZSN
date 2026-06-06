"""All plotting functions for Multitask-Stockformer results.

Design principles:
  - Rolling windows are treated as the TIME AXIS, not independent divisions.
  - Primary comparisons are at the DATASET level (aggregated across windows).
  - Functions accept a model_label parameter so a second model can be overlaid
    later simply by calling the same function with different data and label.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.dates as mdates

ROOT = Path(__file__).parent.parent

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.85,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
})

DS_COLORS = {
    "nasdaq100":          "#1565C0",
    "wig60":              "#BF360C",
    "nasdaq100_extended": "#1B5E20",
    "csi300":             "#E65100",
}
DS_SHORT = {
    "nasdaq100":          "NASDAQ100",
    "wig60":              "WIG60",
    "nasdaq100_extended": "NASDAQ100+",
    "csi300":             "CSI300", 
}
METRIC_INFO = {
    "cls_acc":  ("Classification Accuracy", "higher = better"),
    "reg_mae":  ("Regression MAE",          "lower = better"),
    "reg_rmse": ("Regression RMSE",         "lower = better"),
}
WINDOW_CMAP = plt.cm.tab20


def _save(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    try:
        print(f"  Saved: {path.relative_to(ROOT)}")
    except ValueError:
        print(f"  Saved: {path}")


def _ax_style(ax: plt.Axes, title: str, ylabel: str,
              xlabel: str = "") -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def _agg(values: list) -> tuple[float, float]:
    """mean, std — ignoring NaN."""
    arr = np.array([v for v in values if v is not None and not np.isnan(float(v))],
                   dtype=float)
    if len(arr) == 0:
        return float("nan"), 0.0
    return float(np.mean(arr)), float(np.std(arr))


def plot_metrics_overview(all_logs: dict, out_dir: Path,
                          model_label: str = "Stockformer") -> None:
    """
    PRIMARY figure: mean ± std of each metric, grouped by dataset.

    Designed so a second model can be added later as a second call with
    different color — just aggregate the bars side-by-side.

    all_logs : {dataset: {window: parsed_log}}
    """
    datasets = list(all_logs.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Model: {model_label} — Average test performance by dataset "
                 "(mean ± std across rolling windows)", fontsize=12, y=1.02)

    x = np.arange(len(datasets))
    bar_colors = [DS_COLORS.get(ds, "#546E7A") for ds in datasets]

    for ax, (key, (label, note)) in zip(axes, METRIC_INFO.items()):
        means, stds = [], []
        for ds in datasets:
            vals = [d["test"][key] for d in all_logs[ds].values()
                    if d.get("test")]
            m, s = _agg(vals)
            means.append(m)
            stds.append(s)

        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=bar_colors, alpha=0.85, width=0.55, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([DS_SHORT.get(d, d) for d in datasets],
                           rotation=20, ha="right")

        yhi = ax.get_ylim()[1]
        for bar, m, s in zip(bars, means, stds):
            if not np.isnan(m):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + s + yhi * 0.02,
                        f"{m:.4f}", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold")

        if key == "cls_acc":
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.2,
                       label="random (0.50)", zorder=2)
            ax.legend(fontsize=8)

        _ax_style(ax, f"{label}\n({note})", label)

    fig.tight_layout()
    _save(fig, out_dir / "metrics_overview.png")


def plot_temporal_progression(all_logs: dict, out_dir: Path) -> None:
    """
    Line chart: how each metric changes over time (windows = time axis).
    One line per dataset, X = window start year.
    Shows temporal stability and trend.
    """
    datasets = list(all_logs.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Metric progression over time — rolling windows as temporal axis",
                 fontsize=12, y=1.02)

    for ax, (key, (label, note)) in zip(axes, METRIC_INFO.items()):
        for ds in datasets:
            windows = sorted(all_logs[ds].keys())
            vals = [all_logs[ds][w]["test"][key]
                    if all_logs[ds][w].get("test") else float("nan")
                    for w in windows]
            x_labels = [w[:4] + "/" + w[5:7] for w in windows]  # "2021/01"
            x = range(len(windows))

            ax.plot(x, vals,
                    color=DS_COLORS.get(ds, "#546E7A"),
                    linewidth=2.0, marker="o", markersize=5,
                    label=DS_SHORT.get(ds, ds), zorder=3)

        if key == "cls_acc":
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0,
                       alpha=0.7, zorder=2)

        _ax_style(ax, f"{label} over time\n({note})", label,
                  xlabel="Window index")
        ax.legend(fontsize=8.5)

    fig.tight_layout()
    _save(fig, out_dir / "temporal_progression.png")


def plot_training_summary(all_logs: dict, out_dir: Path) -> None:
    """
    Aggregated training curves per dataset: mean ± shaded std across windows.
    3 panels: train loss, val classification acc, val regression MAE.
    """
    datasets = list(all_logs.keys())
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Training curves — mean ± std across rolling windows",
                 fontsize=12, y=1.02)

    panels = [
        ("train_loss",  "Training Loss",          "Loss"),
        ("val_cls_acc", "Val Classification Acc",  "Accuracy"),
        ("val_reg_mae", "Val Regression MAE",      "MAE"),
    ]

    for ax, (key, title, ylabel) in zip(axes, panels):
        for ds in datasets:
            curves = [d.get(key, []) for d in all_logs[ds].values()
                      if d.get(key)]
            if not curves:
                continue
            max_ep = max(len(c) for c in curves)
            mat = np.full((len(curves), max_ep), np.nan)
            for i, c in enumerate(curves):
                mat[i, :len(c)] = c

            mean = np.nanmean(mat, axis=0)
            std  = np.nanstd(mat,  axis=0)
            xs   = np.arange(1, max_ep + 1)
            color = DS_COLORS.get(ds, "#546E7A")

            ax.plot(xs, mean, color=color, linewidth=2.0,
                    label=DS_SHORT.get(ds, ds))
            ax.fill_between(xs, mean - std, mean + std,
                            color=color, alpha=0.15)

        if key == "val_cls_acc":
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0,
                       alpha=0.7)

        _ax_style(ax, title, ylabel, xlabel="Epoch")
        ax.legend(fontsize=8.5)

    fig.tight_layout()
    _save(fig, out_dir / "training_summary.png")


def plot_portfolio_summary(all_portfolio: dict, out_dir: Path) -> None:
    """
    Portfolio statistics aggregated across windows per dataset.
    Shows mean Sharpe, IC, ICIR, max drawdown.
    """
    datasets = [ds for ds in all_portfolio if all_portfolio[ds]]
    if not datasets:
        return

    stats_reg = ["sharpe", "ic", "icir", "max_drawdown", "ann_return", "hit_rate"]
    stat_labels = {
        "sharpe":       "Sharpe Ratio",
        "ic":           "IC (Spearman)",
        "icir":         "ICIR",
        "max_drawdown": "Max Drawdown (%)",
        "ann_return":   "Ann. Return (%)",
        "hit_rate":     "Hit Rate",
    }
    scale = {"max_drawdown": 100, "ann_return": 100}

    n_stats = len(stats_reg)
    cols, rows = 3, (n_stats + 2) // 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4))
    axes = np.array(axes).flatten()
    fig.suptitle("Portfolio statistics — regression-based long-short (top/bottom 20%)\n"
                 "mean ± std across rolling windows", fontsize=12, y=1.02)

    x = np.arange(len(datasets))
    colors = [DS_COLORS.get(ds, "#546E7A") for ds in datasets]

    for idx, stat in enumerate(stats_reg):
        ax = axes[idx]
        means, stds = [], []
        for ds in datasets:
            raw = [all_portfolio[ds][w]["regression"].get(stat, 0.0)
                   for w in all_portfolio[ds]
                   if all_portfolio[ds][w].get("regression")]
            raw = [v * scale.get(stat, 1) for v in raw]
            m, s = _agg(raw)
            means.append(m); stds.append(s)

        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=colors, alpha=0.85, width=0.5, zorder=3)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([DS_SHORT.get(d, d) for d in datasets],
                           rotation=20, ha="right")
        _ax_style(ax, stat_labels[stat], stat_labels[stat])

        yhi = ax.get_ylim()[1]
        for bar, m, s in zip(bars, means, stds):
            if not np.isnan(m):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + s + abs(yhi) * 0.02,
                        f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    for idx in range(n_stats, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    _save(fig, out_dir / "portfolio_summary.png")


def _test_dates_from_window(window_label: str, test_days: int = 40) -> pd.DatetimeIndex:
    """Return business-day dates for the test period of a rolling window.

    The window label is 'START_END' (e.g. '2021-01-05_2022-02-11').
    The test period occupies the last *test_days* trading days of the window.
    """
    end_str = window_label.split("_")[1]   # "2022-02-11"
    end_dt  = datetime.strptime(end_str, "%Y-%m-%d")
    return pd.bdate_range(end=end_dt, periods=test_days)


def plot_portfolio_wealth(all_portfolio: dict, out_dir: Path,
                          test_days: int = 40) -> None:
    """
    Equity-curve charts: continuous cumulative portfolio WEALTH over calendar time.

    Merges the daily returns from all rolling windows sequentially, dropping 
    overlapping duplicate dates to plot one single continuous portfolio growth line.
    Generates two separate files per dataset:
      portfolio_wealth_regression.png
      portfolio_wealth_classification.png
    """
    datasets = [ds for ds in all_portfolio if all_portfolio[ds]]
    if not datasets:
        return

    PORT_PANELS = [
        ("regression",
         "Regression-based  (long top 20% / short bottom 20% by predicted return)",
         "portfolio_wealth_regression.png"),
        ("classification",
         "Classification-based  (long predicted ↑ / short predicted ↓)",
         "portfolio_wealth_classification.png"),
    ]

    for ds in datasets:
        port_by_window = all_portfolio[ds]
        windows = sorted(port_by_window.keys())
        n = len(windows)
        if n == 0:
            continue

        for port_key, subtitle, fname in PORT_PANELS:
            all_dates = []
            all_returns = []

            for window in windows:
                if port_by_window[window].get(port_key) is None:
                    continue
                
                port    = port_by_window[window][port_key]
                daily_r = port["daily_returns"]
                n_days  = len(daily_r)
                dates   = _test_dates_from_window(window, test_days=n_days)
                
                all_dates.extend(dates)
                all_returns.extend(daily_r)
            
            if not all_dates:
                continue

            df_returns = pd.DataFrame({"date": all_dates, "return": all_returns})
            df_returns["date"] = pd.to_datetime(df_returns["date"])
            
            # If windows overlap, keep the first prediction or mean return for that calendar day
            df_returns = df_returns.groupby("date")["return"].mean().reset_index()
            df_returns = df_returns.sort_values("date").reset_index(drop=True)

            # starting wealth of 1.0 anchored one business day before the first return
            initial_date = df_returns["date"].iloc[0] - pd.offsets.BDay(1)
            
            dates_plot = [initial_date] + df_returns["date"].tolist()
            wealth_plot = np.concatenate([[1.0], np.cumprod(1 + df_returns["return"].values)])

            fig, ax = plt.subplots(figsize=(13, 6))

            ax.plot(dates_plot, wealth_plot,
                    color="#512DA8", alpha=0.9, linewidth=2.0, zorder=2)
            
            ax.fill_between(dates_plot, wealth_plot, 1.0, color="#512DA8", alpha=0.08, zorder=1)

            ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--",
                       alpha=0.55, zorder=3)

            final_total_wealth = wealth_plot[-1]
            total_return_pct = (final_total_wealth - 1.0) * 100
            ax.text(0.02, 0.95,
                    f"Final Continuous Wealth: {final_total_wealth:.4f}\n"
                    f"Total Return: {total_return_pct:+.2f}%",
                    transform=ax.transAxes, fontsize=10, va="top",
                    bbox=dict(boxstyle="round,pad=0.5",
                              facecolor="white", alpha=0.9, edgecolor="gray"))

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)

            ax.set_title(
                f"{ds} — {subtitle}\n"
                "Continuous Cumulative Equity Curve (Chained Out-of-Sample Window Returns)",
                fontsize=11, fontweight="bold", pad=10,
            )
            ax.set_ylabel("Portfolio wealth  (1.0 = initial capital)", fontsize=10)
            ax.set_xlabel("Date", fontsize=10)
            ax.legend(fontsize=9, loc="upper left")
            ax.grid(alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            fig.tight_layout()
            
            ds_dir = out_dir / ds
            ds_dir.mkdir(parents=True, exist_ok=True)
            _save(fig, ds_dir / fname)
            plt.close(fig)


def plot_cross_eval_overview(cross_data: dict, indomain: dict,
                              out_dir: Path,
                              model_label: str = "Stockformer") -> None:
    """
    PRIMARY cross-eval figure.

    Groups by TARGET dataset. Within each group:
      - in-domain bar (target trained on target)
      - one bar per cross-domain source

    cross_data : {(source, target): [rows]}
    indomain   : {dataset: {window: {cls_acc, reg_mae, reg_rmse}}}
    """
    targets = sorted({t for _, t in cross_data})
    sources = sorted({s for s, _ in cross_data})
    all_ds  = sorted(set(sources) | set(targets))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(f"Model: {model_label} — Cross-domain transfer "
                 "(mean ± std across rolling windows)",
                 fontsize=12, y=1.02)

    n_bars_per_group = 1 + len(sources)    # in-domain + n cross-sources
    width = 0.75 / n_bars_per_group
    group_positions = np.arange(len(targets))

    for ax, (key, (label, note)) in zip(axes, METRIC_INFO.items()):
        bar_handles = []

        for src_idx, src in enumerate(["_indom_"] + sources):
            offsets = (src_idx - n_bars_per_group / 2 + 0.5) * width
            xs = group_positions + offsets

            means, stds = [], []
            for tgt in targets:
                if src == "_indom_":
                    vals = [v[key] for v in indomain.get(tgt, {}).values()
                            if v and key in v]
                    color = DS_COLORS.get(tgt, "#546E7A")
                    bar_label = "In-domain"
                    hatch = "//"
                    alpha = 0.6
                else:
                    rows = cross_data.get((src, tgt), [])
                    vals = [r[key] for r in rows]
                    color = DS_COLORS.get(src, "#546E7A")
                    bar_label = f"Source: {DS_SHORT.get(src, src)}"
                    hatch = None
                    alpha = 0.85

                m, s = _agg(vals)
                means.append(m); stds.append(s)

            bars = ax.bar(xs, means, width * 0.92, yerr=stds, capsize=3,
                          color=color, alpha=alpha, hatch=hatch,
                          zorder=3, label=bar_label if src_idx == 0
                          else (bar_label if src_idx <= len(sources) else None))
            if src_idx == 0:
                bar_handles.append(mpatches.Patch(
                    facecolor=color, alpha=alpha, hatch=hatch, label="In-domain baseline"))
            else:
                bar_handles.append(mpatches.Patch(
                    facecolor=DS_COLORS.get(src, "#546E7A"), alpha=0.85,
                    label=f"From {DS_SHORT.get(src, src)}"))

        ax.set_xticks(group_positions)
        ax.set_xticklabels([DS_SHORT.get(t, t) for t in targets],
                           fontsize=9)
        ax.set_xlabel("Target dataset")
        if key == "cls_acc":
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0,
                       alpha=0.7, zorder=2)

        _ax_style(ax, f"{label}\n({note})", label)

    # Shared legend
    handles = list({p.get_label(): p for p in bar_handles}.values())
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), bbox_to_anchor=(0.5, -0.08),
               fontsize=9, framealpha=0.9)

    fig.tight_layout()
    _save(fig, out_dir / "cross_eval_overview.png")


def plot_cross_eval_temporal(cross_data: dict, indomain: dict,
                              out_dir: Path) -> None:
    """
    Line chart: how transfer quality evolves over time (windows = time axis).
    One line per (source→target) pair + dashed in-domain baselines.
    """
    pairs  = sorted(cross_data.keys())
    n_wins = max(len(v) for v in cross_data.values()) if cross_data else 0
    if n_wins == 0:
        return

    pair_colors = [WINDOW_CMAP(i / max(len(pairs) - 1, 1))
                   for i in range(len(pairs))]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Cross-domain transfer quality over time\n"
                 "(windows as temporal axis — dashed = in-domain baseline)",
                 fontsize=12, y=1.02)

    for ax, (key, (label, note)) in zip(axes, METRIC_INFO.items()):
        for (src, tgt), color in zip(pairs, pair_colors):
            rows = cross_data[(src, tgt)]
            xs   = range(len(rows))
            vals = [r[key] for r in rows]
            pair_label = f"{DS_SHORT.get(src, src)}→{DS_SHORT.get(tgt, tgt)}"
            ax.plot(xs, vals, color=color, linewidth=1.8,
                    marker="o", markersize=4.5, label=pair_label, zorder=3)

        for ds in {t for _, t in pairs}:
            vals_id = [v[key] for v in indomain.get(ds, {}).values()
                       if v and key in v]
            if vals_id:
                m = float(np.nanmean(vals_id))
                ax.axhline(m, color=DS_COLORS.get(ds, "gray"),
                           linestyle="--", linewidth=1.3, alpha=0.7,
                           label=f"{DS_SHORT.get(ds, ds)} in-domain ({m:.4f})")

        if key == "cls_acc":
            ax.axhline(0.5, color="lightgray", linestyle=":", linewidth=1.0,
                       zorder=1)

        _ax_style(ax, f"{label} over time\n({note})", label,
                  xlabel="Window index (time →)")
        ax.legend(fontsize=7, ncol=1, loc="best")

    fig.tight_layout()
    _save(fig, out_dir / "cross_eval_temporal.png")


def plot_transfer_delta(cross_data: dict, indomain: dict,
                         out_dir: Path) -> None:
    """
    Δ = cross_mean − in-domain_mean for each (source→target) pair.
    Green = cross-domain model is competitive; Red = clear in-domain advantage.
    """
    pairs = sorted(cross_data.keys())
    labels = [f"{DS_SHORT.get(s, s)}\n→{DS_SHORT.get(t, t)}"
              for s, t in pairs]
    x = np.arange(len(pairs))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Transfer gap:  Δ = cross-domain − in-domain mean\n"
                 "(negative → in-domain model outperforms; positive → surprising gain)",
                 fontsize=12, y=1.02)

    for ax, (key, (label, note)) in zip(axes, METRIC_INFO.items()):
        higher_better = "higher" in note
        deltas, stds_d = [], []

        for src, tgt in pairs:
            cross_vals = [r[key] for r in cross_data[(src, tgt)]]
            indom_vals = [v[key] for v in indomain.get(tgt, {}).values()
                          if v and key in v]
            mc = float(np.nanmean(cross_vals)) if cross_vals else float("nan")
            mi = float(np.nanmean(indom_vals)) if indom_vals else float("nan")
            deltas.append(mc - mi)
            stds_d.append(float(np.nanstd(cross_vals)) if cross_vals else 0.0)

        colors = []
        for d in deltas:
            if np.isnan(d):
                colors.append("lightgray")
            elif (d > 0) == higher_better:
                colors.append("#43A047")   # better than in-domain
            else:
                colors.append("#E53935")   # worse

        bars = ax.bar(x, deltas, color=colors, alpha=0.85, width=0.55,
                      zorder=3)
        ax.axhline(0, color="black", linewidth=1.0, zorder=4)

        ylo, yhi = ax.get_ylim()
        span = max(abs(ylo), abs(yhi), 1e-6)
        for bar, d in zip(bars, deltas):
            if np.isnan(d):
                continue
            va = "bottom" if d >= 0 else "top"
            offset = span * 0.04 * (1 if d >= 0 else -1)
            ax.text(bar.get_x() + bar.get_width() / 2, d + offset,
                    f"{d:+.4f}", ha="center", va=va,
                    fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        legend_handles = [
            mpatches.Patch(color="#43A047", alpha=0.85,
                           label="Cross-domain competitive"),
            mpatches.Patch(color="#E53935", alpha=0.85,
                           label="In-domain advantage"),
        ]
        ax.legend(handles=legend_handles, fontsize=8.5)
        _ax_style(ax, f"Δ {label}\n({note})", f"Δ {label}")

    fig.tight_layout()
    _save(fig, out_dir / "transfer_delta.png")


def plot_heatmaps(all_rows: list, datasets: list, out_dir: Path) -> None:
    """Heatmap matrix (source x target) for each metric."""
    n      = len(datasets)
    ds_idx = {ds: i for i, ds in enumerate(datasets)}
    ticks  = [DS_SHORT.get(d, d) for d in datasets]

    for key, (label, _) in METRIC_INFO.items():
        cmap = "RdYlGn" if "higher" in _ else "RdYlGn_r"
        grid   = np.full((n, n), np.nan)
        counts = np.zeros((n, n), int)

        for row in all_rows:
            si = ds_idx.get(row.get("source_dataset"))
            ti = ds_idx.get(row.get("target_dataset"))
            v  = row.get(key)
            if si is None or ti is None or v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if np.isnan(grid[si, ti]):
                grid[si, ti] = v
            else:
                grid[si, ti] += v
            counts[si, ti] += 1

        for si in range(n):
            for ti in range(n):
                if counts[si, ti] > 1:
                    grid[si, ti] /= counts[si, ti]

        vmin, vmax = float(np.nanmin(grid)), float(np.nanmax(grid))
        fig, ax = plt.subplots(figsize=(7.5, 6))
        im = ax.imshow(grid, cmap=cmap,
                       vmin=vmin - abs(vmin) * 0.02,
                       vmax=vmax + abs(vmax) * 0.02)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)

        ax.set_xticks(range(n)); ax.set_xticklabels(ticks, fontsize=9)
        ax.set_yticks(range(n)); ax.set_yticklabels(ticks, fontsize=9)
        ax.set_xlabel("Target dataset (evaluated on)", fontsize=9)
        ax.set_ylabel("Source dataset (model trained on)", fontsize=9)
        ax.set_title(f"Mean {label}  (source model → target data)\n"
                     "diagonal = in-domain")

        for si in range(n):
            for ti in range(n):
                v = grid[si, ti]
                if not np.isnan(v):
                    bg  = im.cmap(im.norm(v))
                    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
                    ax.text(ti, si, f"{v:.4f}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="white" if lum < 0.5 else "black")

        fig.tight_layout()
        _save(fig, out_dir / f"heatmap_{key}.png")
