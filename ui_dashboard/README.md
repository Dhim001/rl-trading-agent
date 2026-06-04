# RL Trading UI Dashboard

Production-oriented Streamlit dashboard for RL-trading observability and operations.

## What it includes

- Page-based architecture (`overview`, `backtest`, `fine-tuning`, `paper trading`, `operations hub`, `artifacts`)
- Route-level lazy loading of heavy pages
- Data freshness badges and user-friendly empty/error states
- Operations Hub with state-aware workflow controls for:
  - Trading (`paper_trade.py`, `paper_trade.py --once`)
  - Training (`train.py`)
  - Fine-tuning (`tune.py`)
  - Backtest and data download
- Job progress tracking with status, stop controls, and log-tail viewer
- Two integration modes:
  - **Local process adapter** (default)
  - **Remote API adapter** via `RL_DASHBOARD_API_BASE`

## Run

```powershell
cd "C:\Users\Dhimeji01\Projects\rl-trading-agent"
.\.venv\Scripts\activate
pip install -r ui_dashboard\requirements.txt
python scripts\dashboard.py
```

Shortcut:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 dashboard
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 paper-loop -MaxIterations 10
```

## Optional API Integration

Set `RL_DASHBOARD_API_BASE` to connect Operations Hub to your backend jobs API:

```powershell
$env:RL_DASHBOARD_API_BASE = "http://localhost:8000"
python scripts\dashboard.py
```

Expected API routes:

- `POST /jobs` (launch workflow)
- `GET /jobs` (list jobs)
- `POST /jobs/{job_id}/stop` (stop job)
- `GET /jobs/{job_id}/logs?lines=80` (log tail)

## Local Runtime Storage

When API mode is not enabled, dashboard job state is persisted under:

- `ui_dashboard/.runtime/jobs.json`
- `ui_dashboard/.runtime/jobs/*.log`

## Data Sources

- `results/equity_curve.csv`
- `tuning/best_params.json`
- `tuning/trial_*/results/equity_curve.csv`
- `paper/portfolio_state.json`
- `paper/trades.log`
- `models/`, `runs/`, `tuning/`, `results/`, `paper/`

## Main-Project Integration

- Workflow definitions are shared from `src/rl_trading_agent/dashboard/workflows.py`
- Operations Hub and backend adapters use the same workflow IDs/labels/commands
- Dashboard launcher is integrated as `scripts/dashboard.py`
