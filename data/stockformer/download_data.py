"""
build_dataset.py
Downloads Yahoo Finance data and builds folder structure
compatible with Multitask-Stockformer using Qlib Alpha158 factors.

Usage:
    python data/download_data.py --dataset nasdaq100 --start 2021-01-01 --end 2023-01-01 --smoke_test

--smoke_test uses 10 stocks and 1 subdataset for quick testing.
"""

import argparse
import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

NASDAQ100 = [
    "NVDA","AAPL","MSFT","AMZN","GOOGL","GOOG","AVGO","TSLA","META","MU",
    "WMT","AMD","ASML","INTC","CSCO","COST","LRCX","PLTR","NFLX",
    "AMAT","TXN","QCOM","KLAC","LIN","PANW","APP","TMUS","ADI",
    "STX","PEP","CRWD","WDC","AMGN","MRVL","GILD","SHOP","HON","ISRG",
    "BKNG","PDD","VRTX","SBUX","ADBE","CEG","CDNS","FTNT","MAR","SNPS",
    "INTU","CMCSA","ADP","DDOG","MNST","MELI","CSX","NXPI","ABNB","MDLZ",
    "MPWR","ROST","ORLY","DASH","AEP","CTAS","WBD","REGN","BKR",
    "PCAR","MSTR","FANG","MCHP","FAST","EA","XEL","ADSK","ODFL",
    "EXC","IDXX","TTWO","KDP","ALNY","CCEP","PYPL","TRI","AXON","WDAY",
    "PAYX","ROP","CPRT","KHC","DXCM","GEHC","CTSH","VRSK","ZS","CHTR"
]

WIG20 = [
    "MBK.WA","BDX.WA","KTY.WA","KGH.WA","LPP.WA","CDR.WA","PEO.WA",
    "PKN.WA","PKO.WA","PGE.WA","PZU.WA","TPE.WA","KRU.WA","ALR.WA",
    "DNP.WA","ALE.WA","PCO.WA","ZAB.WA","SPL.WA","MDV.WA",
]

MWIG40 = [
    "ABE.WA","EAT.WA","ACP.WA","DOM.WA","EUR.WA","BHW.WA","ING.WA",
    "CAR.WA","DVL.WA","MIL.WA","NEU.WA","OPL.WA","RBW.WA","CPS.WA",
    "ASB.WA","AGT.WA","ENA.WA","MRB.WA","ASE.WA","MBR.WA","GPW.WA",
    "BFT.WA","JSW.WA","SNT.WA","NWG.WA","TXT.WA","WPL.WA","BNP.WA",
    "XTB.WA","APT.WA","CBF.WA","TSG.WA","VRC.WA","CRI.WA","GPP.WA",
    "DIG.WA","PXM.WA","SNK.WA","LBW.WA","PEP.WA",
]

WIG60 = WIG20 + MWIG40

CSI300 = [
    '000001.SZ', '000002.SZ', '000063.SZ', '000069.SZ', '000100.SZ', '000157.SZ', '000166.SZ', '000301.SZ',
    '000333.SZ', '000338.SZ', '000425.SZ', '000538.SZ', '000568.SZ', '000617.SZ', '000625.SZ', '000651.SZ',
    '000661.SZ', '000708.SZ', '000725.SZ', '000733.SZ', '000768.SZ', '000776.SZ', '000786.SZ', '000800.SZ',
    '000858.SZ', '000876.SZ', '000877.SZ', '000895.SZ', '000938.SZ', '000963.SZ', '000977.SZ', '000983.SZ',
    '000999.SZ', '001289.SZ', '001979.SZ', '002001.SZ', '002007.SZ', '002027.SZ', '002049.SZ', '002050.SZ',
    '002074.SZ', '002129.SZ', '002142.SZ', '002179.SZ', '002180.SZ', '002230.SZ', '002236.SZ', '002241.SZ',
    '002252.SZ', '002271.SZ', '002304.SZ', '002311.SZ', '002352.SZ', '002371.SZ', '002410.SZ', '002415.SZ',
    '002459.SZ', '002460.SZ', '002466.SZ', '002475.SZ', '002493.SZ', '002594.SZ', '002601.SZ', '002603.SZ',
    '002648.SZ', '002709.SZ', '002736.SZ', '002812.SZ', '002821.SZ', '002841.SZ', '002916.SZ', '002920.SZ',
    '002938.SZ', '003816.SZ', '300014.SZ', '300015.SZ', '300033.SZ', '300059.SZ', '300122.SZ', '300124.SZ',
    '300142.SZ', '300223.SZ', '300274.SZ', '300308.SZ', '300316.SZ', '300347.SZ', '300408.SZ', '300413.SZ',
    '300433.SZ', '300450.SZ', '300454.SZ', '300628.SZ', '300661.SZ', '300750.SZ', '300751.SZ', '300759.SZ',
    '300760.SZ', '300763.SZ', '300782.SZ', '300896.SZ', '300919.SZ', '300957.SZ', '300999.SZ', '600010.SS',
    '600011.SS', '600015.SS', '600016.SS', '600018.SS', '600019.SS', '600023.SS', '600025.SS', '600028.SS',
    '600029.SS', '600030.SS', '600031.SS', '600036.SS', '600039.SS', '600048.SS', '600050.SS', '600061.SS',
    '600111.SS', '600115.SS', '600132.SS', '600176.SS', '600183.SS', '600219.SS', '600233.SS', '600276.SS',
    '600332.SS', '600346.SS', '600372.SS', '600406.SS', '600426.SS', '600436.SS', '600438.SS', '600460.SS',
    '600489.SS', '600519.SS', '600547.SS', '600570.SS', '600584.SS', '600585.SS', '600600.SS', '600606.SS',
    '600660.SS', '600674.SS', '600690.SS', '600732.SS', '600745.SS', '600754.SS', '600760.SS', '600803.SS',
    '600809.SS', '600837.SS', '600845.SS', '600875.SS', '600886.SS', '600887.SS', '600893.SS', '600905.SS',
    '600918.SS', '600919.SS', '600926.SS', '600938.SS', '600941.SS', '600958.SS', '600989.SS', '600999.SS',
    '601006.SS', '601009.SS', '601021.SS', '601066.SS', '601088.SS', '601100.SS', '601111.SS', '601117.SS',
    '601138.SS', '601155.SS', '601166.SS', '601169.SS', '601186.SS', '601211.SS', '601225.SS', '601229.SS',
    '601236.SS', '601238.SS', '601288.SS', '601318.SS', '601319.SS', '601328.SS', '601336.SS', '601360.SS',
    '601377.SS', '601390.SS', '601398.SS', '601600.SS', '601601.SS', '601607.SS', '601615.SS', '601618.SS',
    '601628.SS', '601633.SS', '601658.SS', '601668.SS', '601669.SS', '601688.SS', '601689.SS', '601699.SS',
    '601728.SS', '601766.SS', '601788.SS', '601799.SS', '601800.SS', '601808.SS', '601816.SS', '601818.SS',
    '601838.SS', '601857.SS', '601865.SS', '601868.SS', '601872.SS', '601877.SS', '601878.SS', '601881.SS',
    '601888.SS', '601898.SS', '601899.SS', '601901.SS', '601916.SS', '601919.SS', '601939.SS', '601985.SS',
    '601988.SS', '601989.SS', '601995.SS', '601998.SS', '603019.SS', '603195.SS', '603259.SS', '603260.SS',
    '603288.SS', '603290.SS', '603369.SS', '603392.SS', '603486.SS', '603501.SS', '603659.SS', '603799.SS',
    '603806.SS', '603833.SS', '603899.SS', '603986.SS', '603993.SS', '605117.SS', '605499.SS',
]

DATASETS = {
    "nasdaq100":          NASDAQ100,
    "wig60":              WIG60,
    "nasdaq100_extended": NASDAQ100 + ["GLD","SLV","TLT","SHY"],
    "csi300":             CSI300,
    "smoke_test":         NASDAQ100[:10],
}

TRAIN_DAYS = 200
VAL_DAYS   = 40
TEST_DAYS  = 40
WINDOW     = TRAIN_DAYS + VAL_DAYS + TEST_DAYS

ALPHA_CATEGORIES = ["CLOSE", "OPEN", "HIGH", "LOW", "VWAP", "VOLUME"]


def download_raw(tickers, start, end):
    """Download raw OHLCV data from Yahoo Finance."""
    print(f"  Downloading {len(tickers)} tickers ({start} → {end})...")
    df = yf.download(
        tickers, start=start, end=end,
        interval="1d", auto_adjust=True,
        group_by="ticker", threads=True, progress=False
    )
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_product([[tickers[0]], df.columns])
    return df


def clean_tickers(df, tickers):
    """Drop tickers with >20% missing data, forward-fill the rest."""
    good = []
    for t in tickers:
        if t not in df.columns.get_level_values(0):
            continue
        close = df[t]["Close"]
        missing_pct = close.isna().mean()
        if missing_pct > 0.20:
            print(f"    Skipping {t}: {missing_pct:.1%} missing")
            continue
        good.append(t)
    df = df[good].copy()
    df = df.ffill().bfill()
    return df, good


def compute_returns(close_df):
    """Daily returns: (close_t - close_{t-1}) / close_{t-1}"""
    return close_df.pct_change().fillna(0)


def compute_trend(returns_df):
    """Trend: 1 if return > 0, else 0"""
    return (returns_df > 0).astype(float)


def build_alpha360(df, tickers):
    """
    Build 360 Alpha360 factors.
    Each factor: Ref(X, t) / close_t for t in 0..59.
    Returns dict: {"CLOSE0": DataFrame(T x N), ...}
    """
    factors = {}
    close = df.xs("Close", axis=1, level=1)[tickers]

    for cat in ALPHA_CATEGORIES:
        if cat == "VOLUME":
            base = df.xs("Volume", axis=1, level=1)[tickers]
            for lag in range(60):
                key = f"{cat}{lag}"
                shifted = base.shift(lag)
                factors[key] = shifted / (base + 1e-12)
        elif cat == "VWAP":
            # VWAP approximated as (High+Low+Close)/3 when no separate VWAP column is available
            try:
                base = df.xs("VWAP", axis=1, level=1)[tickers]
            except KeyError:
                high  = df.xs("High",  axis=1, level=1)[tickers]
                low   = df.xs("Low",   axis=1, level=1)[tickers]
                base  = (high + low + close) / 3
            for lag in range(60):
                key = f"{cat}{lag}"
                factors[key] = base.shift(lag) / close
        else:
            col_map = {"CLOSE": "Close", "OPEN": "Open",
                       "HIGH": "High",   "LOW":  "Low"}
            base = df.xs(col_map[cat], axis=1, level=1)[tickers]
            for lag in range(60):
                key = f"{cat}{lag}"
                factors[key] = base.shift(lag) / close

    for k in factors:
        factors[k] = factors[k].replace([np.inf, -np.inf], np.nan).fillna(0)

    return factors


def neutralize_factors(factors, returns):
    """
    Cross-sectional z-score per day.
    """
    neutralized = {}
    for k, df in factors.items():
        neu = df.sub(df.mean(axis=1), axis=0)
        std = df.std(axis=1).replace(0, 1)
        neu = neu.div(std, axis=0)
        neutralized[k] = neu.clip(-3, 3)
    return neutralized


def compute_corr_matrix(returns_df):
    """Pearson correlation matrix between stocks."""
    corr, _ = spearmanr(returns_df.values)
    if returns_df.shape[1] == 1:
        corr = np.array([[1.0]])
    corr = np.nan_to_num(corr, nan=0.0)
    return corr.astype(np.float32)


def simple_graph_embedding(corr_matrix, dim=128):
    """
    Graph embedding via SVD of correlation matrix.
    (Simplified replacement for Struc2Vec from the original paper.)
    """
    U, s, _ = np.linalg.svd(corr_matrix)
    k = min(dim, len(s))
    embedding = U[:, :k] * np.sqrt(s[:k])
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embedding = (embedding / norms).astype(np.float32)
    if k < dim:
        embedding = np.pad(embedding, ((0, 0), (0, dim - k)))
    return embedding


def generate_config(config_path, window_dir, label, out_root):
    """Generate a Stockformer .conf for one rolling window."""
    window_dir = os.path.abspath(window_dir).replace("\\", "/")
    alpha_dir  = f"{window_dir}/Alpha_360_{label}"
    cpt_dir    = os.path.abspath(os.path.join(out_root, "cpt")).replace("\\", "/")
    log_dir    = os.path.abspath(os.path.join(out_root, "log")).replace("\\", "/")
    os.makedirs(cpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_ratio = round(TRAIN_DAYS / WINDOW, 6)
    val_ratio   = round(VAL_DAYS   / WINDOW, 6)
    test_ratio  = round(TEST_DAYS  / WINDOW, 6)

    content = f"""[file]
traffic = {window_dir}/flow.npz
indicator = {window_dir}/trend_indicator.npz
adj = {window_dir}/corr_adj.npy
adjgat = {window_dir}/128_corr_struc2vec_adjgat.npy
model = {cpt_dir}/saved_model_{label}
log = {log_dir}/log_{label}
bonus = {alpha_dir}

[data]
dataset = STOCK
T1 = 20
T2 = 2
train_ratio = {train_ratio}
val_ratio = {val_ratio}
test_ratio = {test_ratio}

[train]
cuda = 0
max_epoch = 100
batch_size = 12
learning_rate = 0.001
seed = 1

[param]
layers = 2
heads = 1
dims = 128
samples = 1
wave = sym2
level = 1
"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        f.write(content)


def save_subdataset(out_dir, dates, returns, trend, factors, corr, emb):
    """Save one subdataset in Stockformer format."""
    os.makedirs(out_dir, exist_ok=True)

    ret_slice   = returns.loc[dates]
    trend_slice = trend.loc[dates]

    flow_data = ret_slice.values.astype(np.float32)
    np.savez(os.path.join(out_dir, "flow.npz"), data=flow_data)

    trend_data = trend_slice.values.astype(np.float32)
    np.savez(os.path.join(out_dir, "trend_indicator.npz"), data=trend_data)

    ret_slice.to_csv(os.path.join(out_dir, "label_processed.csv"))

    np.save(os.path.join(out_dir, "corr_adj.npy"), corr)
    np.save(os.path.join(out_dir, "128_corr_struc2vec_adjgat.npy"), emb)

    alpha_dir = os.path.join(out_dir, f"Alpha_360_{dates[0].strftime('%Y-%m-%d')}_{dates[-1].strftime('%Y-%m-%d')}")
    os.makedirs(alpha_dir, exist_ok=True)

    for factor_name, factor_df in factors.items():
        factor_slice = factor_df.loc[dates]
        factor_slice.to_csv(os.path.join(alpha_dir, f"{factor_name}.csv"))


def build_rolling_windows(dates, n_windows=14):
    """Generate rolling windows."""
    windows = []
    total = len(dates)

    if total < WINDOW:
        print(f"  ERROR: too few days ({total}) for window ({WINDOW})")
        return windows

    if n_windows == 1:
        steps = [0]
    else:
        step = (total - WINDOW) // (n_windows - 1)
        steps = [i * step for i in range(n_windows)]

    for start_idx in steps:
        end_idx = start_idx + WINDOW
        if end_idx > total:
            break
        w_dates = dates[start_idx:end_idx]
        windows.append({
            "train": w_dates[:TRAIN_DAYS],
            "val":   w_dates[TRAIN_DAYS:TRAIN_DAYS + VAL_DAYS],
            "test":  w_dates[TRAIN_DAYS + VAL_DAYS:],
            "label": f"{w_dates[0].strftime('%Y-%m-%d')}_{w_dates[-1].strftime('%Y-%m-%d')}"
        })
    return windows


def main(args):
    dataset_name = "smoke_test" if args.smoke_test else args.dataset
    tickers = DATASETS[dataset_name]
    out_root = os.path.join(args.out_dir, dataset_name)
    os.makedirs(out_root, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset:  {dataset_name} ({len(tickers)} tickers)")
    print(f"Period:   {args.start} → {args.end}")
    print(f"Output:   {out_root}")
    print(f"{'='*60}\n")


    raw = download_raw(tickers, args.start, args.end)
    raw, tickers = clean_tickers(raw, tickers)
    print(f"  {len(tickers)} tickers after cleaning\n")


    close = raw.xs("Close", axis=1, level=1)[tickers]
    returns = compute_returns(close).iloc[1:]
    trend   = compute_trend(returns)
    dates   = returns.index
    print(f"  Trading days: {len(dates)}")
    print(f"  Need minimum: {WINDOW} days ({TRAIN_DAYS}+{VAL_DAYS}+{TEST_DAYS})")

  
    print("\n  Computing Alpha360 factors (360 factors)...")
    factors_raw = build_alpha360(raw, tickers)
    factors     = neutralize_factors(factors_raw, returns)
    factors     = {k: v.loc[dates] for k, v in factors.items()}
    print(f"  Generated {len(factors)} factors")

    print("  Computing correlation matrix...")
    corr = compute_corr_matrix(returns)
    emb  = simple_graph_embedding(corr, dim=128)

    n_windows = 1 if args.smoke_test else 14
    windows = build_rolling_windows(dates, n_windows)
    print(f"\n  Generating {len(windows)} subdataset(s)...\n")

    for i, w in enumerate(tqdm(windows, desc="Subdatasets")):
        all_dates = w["train"].append(w["val"]).append(w["test"])
        folder_name = f"Stock_{dataset_name.upper()}_{w['label']}"
        out_dir = os.path.join(out_root, folder_name)

        corr_w = compute_corr_matrix(returns.loc[w["train"]])
        emb_w  = simple_graph_embedding(corr_w, dim=128)

        save_subdataset(
            out_dir   = out_dir,
            dates     = all_dates,
            returns   = returns,
            trend     = trend,
            factors   = factors,
            corr      = corr_w,
            emb       = emb_w,
        )

        config_path = os.path.join(args.config_dir, dataset_name, f"window_{w['label']}.conf")
        generate_config(config_path, out_dir, w["label"], out_root)

    print(f"\n✓ Done! Data saved in: {out_root}")
    print(f"  Stocks: {len(tickers)}")
    print(f"  Subdatasets: {len(windows)}")
    if windows:
        print(f"  Example folder: {os.path.join(out_root, windows[0]['label'])}")
    else:
        print("  ERROR: no windows generated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="smoke_test",
                        choices=list(DATASETS.keys()))
    parser.add_argument("--start",      default="2021-01-01")
    parser.add_argument("--end",        default="2026-01-31")
    parser.add_argument("--out_dir",    default="data/stockformer/processed")
    parser.add_argument("--config_dir", default="models/stockformer/config")
    parser.add_argument("--smoke_test", action="store_true",
                        help="10 stocks, 1 subdataset - quick test")
    args = parser.parse_args()

    if args.dataset == "smoke_test":
        args.smoke_test = True

    main(args)
