"""
Stock NeurIPS2018 Part 3. Backtest

This series is a reproduction of paper "Deep reinforcement learning for
automated stock trading: An ensemble strategy".

Introducing how to use the agents we trained to do backtest, and compare with baselines such as
Mean Variance Optimization and DJIA index.
"""

from __future__ import annotations

import argparse  # Added for argument handling
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from finrl.agents.stablebaselines3.models import DRLAgent
from configs.config import INDICATORS, TRAINED_MODEL_DIR, TRADE_START_DATE, TRADE_END_DATE, DATA_SAVE_DIR, RESULTS_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier


def load_data(train_path=f"{DATA_SAVE_DIR}/train_data.csv", trade_path=f"{DATA_SAVE_DIR}/trade_data.csv"):
    train = pd.read_csv(train_path)
    trade = pd.read_csv(trade_path)

    train = train.set_index(train.columns[0])
    train.index.names = [""]
    trade = trade.set_index(trade.columns[0])
    trade.index.names = [""]
    return train, trade


def backtest_agent(agent_name, trade):
    model_path = os.path.join(TRAINED_MODEL_DIR, agent_name)
    trained_sac = SAC.load(model_path)

    stock_dimension = len(trade.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }

    e_trade_gym = StockTradingEnv(
        df=trade, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs
    )

    df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction(model=trained_sac, environment=e_trade_gym)

    return df_account_value_sac, df_actions_sac


def process_df_for_mvo(df):
    return df.pivot(index="date", columns="tic", values="close")


def StockReturnsComputing(StockPrice, Rows, Columns):
    StockReturn = np.zeros([Rows - 1, Columns])
    for j in range(Columns):
        for i in range(Rows - 1):
            StockReturn[i, j] = (
                (StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j]
            ) * 100
    return StockReturn


def backtest_mvo(train, trade):
    StockData = process_df_for_mvo(train)
    TradeData = process_df_for_mvo(trade)

    arStockPrices = np.asarray(StockData)
    [Rows, Cols] = arStockPrices.shape
    arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

    meanReturns = np.mean(arReturns, axis=0)
    covReturns = np.cov(arReturns, rowvar=False)

    np.set_printoptions(precision=3, suppress=True)
    print("Mean returns of assets in portfolio\n", meanReturns)

    ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
    raw_weights_mean = ef_mean.max_sharpe()
    cleaned_weights_mean = ef_mean.clean_weights()
    mvo_weights = np.array(
        [1000000 * cleaned_weights_mean[i] for i in range(len(cleaned_weights_mean))]
    )

    LastPrice = np.array([1 / p for p in StockData.tail(1).to_numpy()[0]])
    Initial_Portfolio = np.multiply(mvo_weights, LastPrice)

    Portfolio_Assets = TradeData @ Initial_Portfolio
    MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])
    return MVO_result


def backtest_dji():
    df_dji = yf.download("^DJI", start=TRADE_START_DATE, end=TRADE_END_DATE)
    
    # Flatten MultiIndex columns if present (common in newer yfinance versions)
    if isinstance(df_dji.columns, pd.MultiIndex):
        df_dji.columns = df_dji.columns.get_level_values(0)
        
    df_dji = df_dji[["Close"]].reset_index()
    df_dji.columns = ["date", "close"]
    df_dji["date"] = df_dji["date"].astype(str)
    
    fst_day = df_dji["close"].iloc[0]
    dji = pd.merge(
        df_dji["date"],
        df_dji["close"].div(fst_day).mul(1000000),
        how="outer",
        left_index=True,
        right_index=True,
    ).set_index("date")

    return dji


def save_plot(result, path):
    plt.rcParams["figure.figsize"] = (15, 5)
    plt.figure()
    result.plot()
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    
    # Ensure target output folder exists before saving plot
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {path}\n")


def execute_pipeline(train_file, trade_file, agent_model, plot_name):
    print(f"\n=== Running Backtest Pipeline for {plot_name.upper()} ===")
    
    # 1. Load Data
    train_data, trade_data = load_data(
        train_path=f"{DATA_SAVE_DIR}/{train_file}", 
        trade_path=f"{DATA_SAVE_DIR}/{trade_file}"
    )

    # 2. Backtest Agent
    df_account_value_sac, _ = backtest_agent(agent_model, trade_data)
    df_result_sac = df_account_value_sac.set_index(df_account_value_sac.columns[0])

    # 3. Baselines (MVO & DJI)
    MVO_result = backtest_mvo(train_data, trade_data)
    dji = backtest_dji()
    

    result = pd.DataFrame(
        {
            "sac": df_result_sac["account_value"],
            "mvo": MVO_result["Mean Var"],
            "dji": dji["close"],
        }
    )

    print("\n=== Backtest Results Preview ===")
    print(result.head())

    # 5. Output Visualization
    save_plot(result, f"{RESULTS_DIR}/{plot_name}_backtest_result.png")


def run_backtest():
    dataset_configs = {
        "nasdaq100": ("NASDAQ100_train_data.csv", "NASDAQ100_trade_data.csv", "agent_sac_nasdaq100", "nasdaq100"),
        "nasdaq100_extended": ("NASDAQ100_EXT_train_data.csv", "NASDAQ100_EXT_trade_data.csv", "agent_sac_nasdaq100_ext", "nasdaq100_ext"),
        "wig60": ("WIG60_train_data.csv", "WIG60_trade_data.csv", "agent_sac_wig60", "wig60"),
        "csi300": ("CSI300_train_data.csv", "CSI300_trade_data.csv", "agent_sac_csi300", "csi300")
    }
    
    parser = argparse.ArgumentParser(description="Backtest trained FinRL agents against MVO and DJI baselines.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=list(dataset_configs.keys()) +["all"], 
        default="all",
        help="The dataset configuration to run backtesting on (default: all)"
    )
    args = parser.parse_args()

    
    if args.dataset == "all":
        for config in dataset_configs.values():
            execute_pipeline(*config)
    else:
        execute_pipeline(*dataset_configs[args.dataset])


if __name__ == "__main__":
    run_backtest()