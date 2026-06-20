param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("dashboard", "download", "train", "backtest", "tune", "paper", "paper-once", "paper-loop", "qa-dashboard")]
    [string]$Command,

    [int]$Port = 8501,
    [int]$Trials = 10,
    [int]$MaxIterations = 10
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $python = $venvPython
} else {
    $python = "python"
}

switch ($Command) {
    "dashboard" {
        & $python (Join-Path $root "scripts\dashboard.py") --port $Port
    }
    "download" {
        & $python (Join-Path $root "scripts\download_data.py")
    }
    "train" {
        & $python (Join-Path $root "scripts\train.py")
    }
    "backtest" {
        & $python (Join-Path $root "scripts\backtest.py")
    }
    "tune" {
        & $python (Join-Path $root "scripts\tune.py") --trials $Trials
    }
    "paper" {
        & $python (Join-Path $root "scripts\paper_trade.py")
    }
    "paper-once" {
        & $python (Join-Path $root "scripts\paper_trade.py") --once
    }
    "paper-loop" {
        & $python (Join-Path $root "scripts\paper_trade.py") --max-iterations $MaxIterations
    }
    "qa-dashboard" {
        & $python (Join-Path $root "scripts\qa_dashboard.py")
    }
}
