# RL Trading Agent

Reinforcement learning framework for stock trading with data download, feature engineering, training, backtesting, paper trading, and hyperparameter tuning.

## Features

- Multi-stock and single-stock trading modes
- PPO training pipeline with Stable-Baselines3
- Risk controls (max drawdown, stop-loss, position sizing)
- Backtesting with return, Sharpe ratio, and drawdown metrics
- Paper-trading loop using live market data from yfinance
- Optuna-based hyperparameter tuning

## Project Structure

- `config/default.yaml` - data sources, environment, risk, training, paper-trading, and tuning settings
- `scripts/` - runnable entry points (`download_data.py`, `train.py`, `backtest.py`, `paper_trade.py`, `tune.py`)
- `src/rl_trading_agent/data/` - data ingestion, feature engineering, and dataset pipeline
- `src/rl_trading_agent/env/` - Gymnasium trading environments
- `src/rl_trading_agent/training/` - model training and tuning logic
- `src/rl_trading_agent/evaluation/` - backtesting and metrics
- `src/rl_trading_agent/paper/` - simulated live/paper-trading logic
- `src/rl_trading_agent/dashboard/` - shared dashboard workflow contracts
- `ui_dashboard/` - integrated Streamlit UI dashboard

## Requirements

- Python 3.10+
- Windows PowerShell (commands below are shown for PowerShell)

## Installation

```powershell
cd "C:\Users\Dhimeji01\Projects\rl-trading-agent"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```powershell
# 1) Download and cache market data
.\.venv\Scripts\python.exe scripts\download_data.py

# 2) Train the agent
.\.venv\Scripts\python.exe scripts\train.py

# 3) Backtest the trained model
.\.venv\Scripts\python.exe scripts\backtest.py

# 4) Launch integrated UI dashboard
.\.venv\Scripts\python.exe scripts\dashboard.py
```

## One-Command Shortcuts (PowerShell)

Use the integrated runner:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 dashboard
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 download
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 train
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 backtest
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 tune -Trials 10
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 paper-once
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 paper-loop -MaxIterations 10
```

If you use VS Code, run predefined tasks from `Terminal -> Run Task`:
- `RL: Dashboard`
- `RL: Download Data`
- `RL: Train`
- `RL: Backtest`
- `RL: Fine-tune`
- `RL: Paper Trade Once`
- `RL: Paper Trade Loop (10)`

After training, the final model is saved to:

- `models/final_model.zip`

## Configuration

Main settings live in `config/default.yaml`.

### Multi-stock mode

Configure symbols:

```yaml
data:
  symbols:
    - AAPL
    - MSFT
    - GOOGL
    - AMZN
    - META
```

The agent uses a multi-discrete action format:

- `symbol_idx` -> which symbol to act on
- `action` -> hold / buy / sell

For single-stock mode, keep one symbol.

## Paper Trading

```powershell
# Run one decision step
.\.venv\Scripts\python.exe scripts\paper_trade.py --once

# Run continuously
.\.venv\Scripts\python.exe scripts\paper_trade.py

# Run a limited number of iterations
.\.venv\Scripts\python.exe scripts\paper_trade.py --max-iterations 10
```

Outputs:

- Portfolio state: `paper/portfolio_state.json`
- Trade log: `paper/trades.log`

## Hyperparameter Tuning

```powershell
.\.venv\Scripts\python.exe scripts\tune.py
.\.venv\Scripts\python.exe scripts\tune.py --trials 10
```

Best parameters are saved to:

- `tuning/best_params.json`

## Typical Outputs

- Cached market data: `data/cache/`
- Feature stats: `data/feature_stats.json`
- TensorBoard/logs: `runs/`
- Trained models: `models/`

## Disclaimer

This project is for research and education only, not financial advice. Validate thoroughly (including paper trading) before any live deployment.
