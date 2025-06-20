"""Evaluate a trained DQN model on new data."""

import os, sys
sys.path.append(os.path.dirname(__file__))

import argparse
import pandas as pd
from stable_baselines3 import DQN

from trading_env import TradingEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="CSV file with columns Date, Open, High, Low, Close and Volume",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="dqn_trading_model",
        help="Path to the trained model (.zip)",
    )
    return parser.parse_args()


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=lambda c: c.strip().lower())
    if "close/last" in df.columns:
        df = df.rename(columns={"close/last": "close"})
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    return df[["open", "high", "low", "close", "volume"]]


def main():
    args = parse_args()
    df = load_data(args.data)
    env = TradingEnv(df)
    model = DQN.load(args.model_path)

    obs = env.reset()
    done = False
    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        action_name = ["Hold", "Buy", "Sell"][int(action)]
        print(
            f"Step {step}: action={action_name}, reward={reward:.2f}, balance={info['balance']:.2f}, position={info['position']}"
        )
        step += 1

    print(f"Final balance: {info['balance']:.2f}")


if __name__ == "__main__":
    main()
