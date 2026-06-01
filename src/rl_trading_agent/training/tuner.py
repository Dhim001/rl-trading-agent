from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import optuna
from optuna.pruners import MedianPruner

from rl_trading_agent.evaluation.backtest import run_backtest
from rl_trading_agent.training.trainer import train_agent


def run_hyperparameter_search(
    train_data: Any,
    eval_data: Any,
    test_data: Any,
    mode: str,
    env_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    tuning_cfg: dict[str, Any],
    root: Path,
) -> optuna.Study:
    study_dir = root / tuning_cfg.get("study_dir", "tuning")
    study_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        trial_env_cfg = deepcopy(env_cfg)
        trial_training_cfg = deepcopy(training_cfg)

        trial_training_cfg["learning_rate"] = trial.suggest_float(
            "learning_rate", *tuning_cfg["learning_rate_range"], log=True
        )
        trial_training_cfg["gamma"] = trial.suggest_float(
            "gamma", *tuning_cfg["gamma_range"]
        )
        trial_training_cfg["batch_size"] = trial.suggest_categorical(
            "batch_size", tuning_cfg["batch_size_options"]
        )
        trial_env_cfg["window_size"] = trial.suggest_categorical(
            "window_size", tuning_cfg["window_size_options"]
        )
        trial_training_cfg["total_timesteps"] = tuning_cfg["timesteps_per_trial"]
        trial_training_cfg["eval_freq"] = max(1000, tuning_cfg["timesteps_per_trial"] // 5)
        trial_training_cfg["save_freq"] = tuning_cfg["timesteps_per_trial"]
        trial_training_cfg["model_dir"] = str(study_dir / f"trial_{trial.number}" / "models")
        trial_training_cfg["log_dir"] = str(study_dir / f"trial_{trial.number}" / "runs")

        model_path = train_agent(
            train_data=train_data,
            eval_data=eval_data,
            mode=mode,
            env_cfg=trial_env_cfg,
            risk_cfg=risk_cfg,
            training_cfg=trial_training_cfg,
        )

        metrics = run_backtest(
            model_path=model_path,
            test_data=test_data,
            mode=mode,
            env_cfg=trial_env_cfg,
            risk_cfg=risk_cfg,
            output_dir=study_dir / f"trial_{trial.number}" / "results",
        )
        return metrics["sharpe_ratio"]

    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=1),
        study_name=tuning_cfg.get("study_name", "rl_trading_agent"),
    )
    study.optimize(objective, n_trials=tuning_cfg["n_trials"], show_progress_bar=True)

    best_path = study_dir / "best_params.json"
    import json

    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)

    return study
