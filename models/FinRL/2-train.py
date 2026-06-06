"""
Stock NeurIPS2018 Part 2. Train

This series is a reproduction of paper "Deep reinforcement learning for
automated stock trading: An ensemble strategy".

Introduce how to use FinRL to make data into the gym form environment, and train DRL agents on it.
"""

from __future__ import annotations

import argparse  
import os
import pandas as pd
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from configs.config import INDICATORS
from configs.config import RESULTS_DIR
from configs.config import TRAINED_MODEL_DIR
from configs.config import DATA_SAVE_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

# Ensure target directories exist
check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR])


def build_env(train_data_path=f"{DATA_SAVE_DIR}/NASDAQ100_train_data.csv"):
    print(f"\n=== Building Environment for: {os.path.basename(train_data_path)} ===")
    
    train = pd.read_csv(train_data_path)
    
    # Safely handle the index assignment
    if len(train.columns) > 0:
        train = train.set_index(train.columns[0])
    train.index.names = [""]

    stock_dimension = len(train.tic.unique())
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

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    return env_train


# --- SAC ---
def train_sac_agent(env_train, model_name="agent_sac"):
    print(f"=== Starting SAC Training for {model_name} ===")
    agent = DRLAgent(env=env_train)
    SAC_PARAMS = {
        "batch_size": 128,
        "buffer_size": 100000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    }
    model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)
    
    tmp_path = os.path.join(RESULTS_DIR, model_name)
    new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_sac.set_logger(new_logger_sac)

    # set total_timesteps to a small number for testing, and increase it for real training (250_000 is ok)
    trained_sac = agent.train_model(model=model_sac, tb_log_name=model_name, total_timesteps=250_000)

    save_path = os.path.join(TRAINED_MODEL_DIR, model_name)
    trained_sac.save(save_path)
    print(f"SAC agent trained and saved to {save_path}\n")


def run_training():
    # Setup argument parser matching the data script
    # Dictionary mapping choice names to their file paths and model naming targets
    dataset_configs = {
        "nasdaq100": (f"{DATA_SAVE_DIR}/NASDAQ100_train_data.csv", "agent_sac_nasdaq100"),
        "nasdaq100_extended": (f"{DATA_SAVE_DIR}/NASDAQ100_EXT_train_data.csv", "agent_sac_nasdaq100_ext"),
        "wig60": (f"{DATA_SAVE_DIR}/WIG60_train_data.csv", "agent_sac_wig60"),
        "csi300": (f"{DATA_SAVE_DIR}/CSI300_train_data.csv", "agent_sac_csi300")
    }

    parser = argparse.ArgumentParser(description="Train FinRL DRL agents on stock data.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices= list(dataset_configs.keys()) + ["all"], 
        default="all",
        help="The dataset environment to train the agent on (default: all)"
    )
    args = parser.parse_args()


    if args.dataset == "all":
        # Process all configurations sequentially
        for data_path, model_name in dataset_configs.values():
            env = build_env(train_data_path=data_path)
            train_sac_agent(env, model_name=model_name)
    else:
        # Process only the selected dataset environment
        data_path, model_name = dataset_configs[args.dataset]
        print(data_path)
        env = build_env(train_data_path=data_path)
        train_sac_agent(env, model_name=model_name)


if __name__ == "__main__":
    run_training()