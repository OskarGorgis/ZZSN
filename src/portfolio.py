"""Portfolio construction and statistics from model prediction CSV files."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "processed"

_CLS_PAIR = re.compile(r"\[\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s*\]")


# ── CSV parsing ───────────────────────────────────────────────────────────────

def parse_reg_csv(path: Path) -> np.ndarray:
    """Parse regression prediction CSV → (T, N) float array."""
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            vals = [float(x) for x in line.split(",") if x.strip()]
            if vals:
                rows.append(vals)
    if not rows:
        return np.empty((0, 0))
    max_n = max(len(r) for r in rows)
    arr = np.zeros((len(rows), max_n))
    for i, r in enumerate(rows):
        arr[i, : len(r)] = r
    return arr


def parse_cls_csv(path: Path) -> np.ndarray:
    """Parse classification CSV (numpy-style strings) → (T, N, 2) logit array."""
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            matches = _CLS_PAIR.findall(line)
            if matches:
                rows.append([[float(a), float(b)] for a, b in matches])
    if not rows:
        return np.empty((0, 0, 2))
    max_n = max(len(r) for r in rows)
    arr = np.zeros((len(rows), max_n, 2))
    for i, row in enumerate(rows):
        arr[i, : len(row)] = row
    return arr


# ── Stats helpers ─────────────────────────────────────────────────────────────

def _stats(daily_r: np.ndarray) -> dict:
    mean_r = float(np.mean(daily_r))
    std_r  = float(np.std(daily_r)) + 1e-10
    sharpe = float(np.sqrt(252) * mean_r / std_r)
    wealth = np.cumprod(1 + daily_r)
    peak   = np.maximum.accumulate(wealth)
    mdd    = float(np.min((wealth - peak) / (peak + 1e-10)))
    return {
        "sharpe":       sharpe,
        "max_drawdown": mdd,
        "hit_rate":     float(np.mean(daily_r > 0)),
        "total_return": float(wealth[-1] - 1) if len(wealth) else 0.0,
        "ann_return":   float(mean_r * 252),
        "ann_vol":      float(std_r * np.sqrt(252)),
    }


# ── Portfolio strategies ──────────────────────────────────────────────────────

def regression_portfolio(pred_ret: np.ndarray, actual_ret: np.ndarray,
                         quantile: float = 0.2) -> dict:
    """
    Long-short portfolio ranked by predicted return.

    Each day: long top-Q% stocks, short bottom-Q% stocks.
    Also computes IC (Spearman corr between pred and actual ranks).

    pred_ret, actual_ret : (T, N)
    """
    T, N = pred_ret.shape
    k = max(1, int(N * quantile))
    daily_r = np.zeros(T)
    ic_vals = np.zeros(T)

    for t in range(T):
        idx = np.argsort(pred_ret[t])
        daily_r[t] = (np.mean(actual_ret[t, idx[-k:]])
                      - np.mean(actual_ret[t, idx[:k]]))
        ic, _ = spearmanr(pred_ret[t], actual_ret[t])
        ic_vals[t] = 0.0 if np.isnan(ic) else float(ic)

    cumulative = np.cumprod(1 + daily_r) - 1
    ic_mean = float(np.mean(ic_vals))
    ic_std  = float(np.std(ic_vals)) + 1e-10
    return {
        "daily_returns": daily_r,
        "cumulative":    cumulative,
        "ic":            ic_mean,
        "icir":          ic_mean / ic_std,
        **_stats(daily_r),
    }


def classification_portfolio(pred_logits: np.ndarray,
                              actual_ret: np.ndarray) -> dict:
    """
    Long stocks predicted UP (class 1), short stocks predicted DOWN (class 0).

    pred_logits : (T, N, 2)
    actual_ret  : (T, N)
    """
    T = min(pred_logits.shape[0], actual_ret.shape[0])
    N = min(pred_logits.shape[1], actual_ret.shape[1])
    pred_labels = np.argmax(pred_logits[:T, :N], axis=-1)   # (T, N)
    actual_ret  = actual_ret[:T, :N]

    daily_r = np.zeros(T)
    for t in range(T):
        long_mask  = pred_labels[t] == 1
        short_mask = pred_labels[t] == 0
        long_r  = float(np.mean(actual_ret[t, long_mask]))  if long_mask.any()  else 0.0
        short_r = float(np.mean(actual_ret[t, short_mask])) if short_mask.any() else 0.0
        daily_r[t] = long_r - short_r

    # Classification accuracy vs actual direction
    actual_dir = (actual_ret > 0).astype(int)
    cls_acc = float(np.mean(pred_labels == actual_dir))

    cumulative = np.cumprod(1 + daily_r) - 1
    return {
        "daily_returns": daily_r,
        "cumulative":    cumulative,
        "cls_accuracy":  cls_acc,
        **_stats(daily_r),
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_window_outputs(dataset: str, window: str):
    """Load CSV outputs for one window.

    Returns (pred_ret, actual_ret, pred_cls) or None if files missing.
    """
    base = DATA_ROOT / dataset / "output" / f"log_{window}"
    reg_pred  = base / "regression"     / "regression_pred_last_step.csv"
    reg_label = base / "regression"     / "regression_label_last_step.csv"
    cls_pred  = base / "classification" / "classification_pred_last_step.csv"

    if not reg_pred.exists() or not reg_label.exists():
        return None

    p_ret = parse_reg_csv(reg_pred)
    a_ret = parse_reg_csv(reg_label)
    T = min(p_ret.shape[0], a_ret.shape[0])
    N = min(p_ret.shape[1], a_ret.shape[1])
    p_ret, a_ret = p_ret[:T, :N], a_ret[:T, :N]

    p_cls = None
    if cls_pred.exists():
        p_cls = parse_cls_csv(cls_pred)
        Tc = min(T, p_cls.shape[0])
        Nc = min(N, p_cls.shape[1])
        p_cls = p_cls[:Tc, :Nc, :]

    return p_ret, a_ret, p_cls


def compute_all_portfolios(dataset: str, windows: list) -> dict:
    """Return {window: {regression: ..., classification: ...}} for all windows."""
    results = {}
    for window in windows:
        data = load_window_outputs(dataset, window)
        if data is None:
            continue
        p_ret, a_ret, p_cls = data
        if p_ret.shape[0] < 2 or p_ret.shape[1] < 2:
            continue

        reg_port = regression_portfolio(p_ret, a_ret)
        cls_port = classification_portfolio(p_cls, a_ret) if p_cls is not None else None
        results[window] = {"regression": reg_port, "classification": cls_port}

    return results
