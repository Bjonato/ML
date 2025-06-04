# Reinforcement Learning Trading Example

This repository contains a minimal example for setting up a trading environment and training a Deep Q-Network (DQN) agent.

## Structure

- `src/trading_env.py` – Custom OpenAI Gym environment for trading.
- `src/train_dqn.py` – Script to train a DQN agent using Stable Baselines3.
- `requirements.txt` – Dependencies.

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare a CSV file with the columns `Open`, `High`, `Low`, `Close` (or `Close/Last`) and `Volume`.
   The training script will normalize these features and ignore the `Date` column if present.

3. Train a DQN agent:

```bash
python -m src.train_dqn --data PATH_TO_CSV --timesteps 10000
```

Model weights will be saved to `dqn_trading_model.zip` by default.

All hyperparameters in `train_dqn.py` can be adjusted to tune training.

4. Evaluate a trained model on a test dataset (a plot of closing prices with buy
   and sell markers will be displayed):

```bash
python -m src.test_dqn --data PATH_TO_TEST_CSV --model-path dqn_trading_model.zip
```
