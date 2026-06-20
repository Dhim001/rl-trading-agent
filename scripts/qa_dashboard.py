#!/usr/bin/env python3
from __future__ import annotations

import importlib
import inspect
import sqlite3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def run_cmd(name: str, command: list[str]) -> bool:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    ok = result.returncode == 0
    status = "PASS" if ok else "FAIL"
    print(f"{status} | {name}")
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    return ok


def run_page_contracts_check() -> bool:
    try:
        from ui_dashboard.app import PAGES

        two_arg = {"Operations Hub", "Lineage", "Training Studio", "Strategy Studio"}
        bad: list[str] = []
        print("PASS | Page import and render contracts" if PAGES else "FAIL | Page import and render contracts")
        print(f"pages {len(PAGES)}")
        for name, modname in PAGES.items():
            mod = importlib.import_module(modname)
            render = getattr(mod, "render", None)
            ok = callable(render)
            params = list(inspect.signature(render).parameters.keys()) if ok else []
            exp = 2 if name in two_arg else 1
            row_ok = ok and len(params) == exp
            print(name, "OK" if row_ok else "FAIL", params)
            if not row_ok:
                bad.append(name)
        if bad:
            print(f"bad pages: {bad}")
            return False
        return True
    except Exception as exc:
        print(f"FAIL | Page import and render contracts")
        print(str(exc))
        return False


def run_service_smoke_check() -> bool:
    try:
        from ui_dashboard.core.config import load_config
        from ui_dashboard.services.cache_service import (
            get_backtest_curve_cached,
            get_paper_log_cached,
            get_paper_state_cached,
        )
        from ui_dashboard.services.job_service import JobService

        cfg = load_config()
        js = JobService(cfg)
        jobs = js.list_jobs()
        print("PASS | Service/data smoke checks")
        print("jobs", len(jobs))

        backtest_ok = get_backtest_curve_cached(cfg.project_root / "results" / "equity_curve.csv") is not None
        paper_state_ok = bool(get_paper_state_cached(cfg.project_root / "paper" / "portfolio_state.json"))
        paper_log_rows = len(get_paper_log_cached(cfg.project_root / "paper" / "trades.log"))
        print("paper_log_rows", paper_log_rows)
        if not backtest_ok:
            print("missing backtest cache")
            return False
        if not paper_state_ok:
            print("missing paper state")
            return False

        db = cfg.project_root / "artifacts" / "metadata.db"
        if not db.exists():
            print("metadata.db missing")
            return False
        con = sqlite3.connect(db)
        runs_rows = con.execute("select count(*) from runs").fetchone()[0]
        con.close()
        print("runs_rows", runs_rows)
        return True
    except Exception as exc:
        print("FAIL | Service/data smoke checks")
        print(str(exc))
        return False


def main() -> int:
    print("Dashboard QA Runner")
    print(f"Project root: {ROOT}")

    checks: list[bool] = []
    checks.append(run_cmd("Compile dashboard modules", [sys.executable, "-m", "compileall", "-q", "ui_dashboard"]))
    checks.append(run_page_contracts_check())
    checks.append(run_service_smoke_check())

    total = len(checks)
    passed = sum(1 for x in checks if x)
    print(f"\nSummary: {passed}/{total} checks passed.")

    checklist = ROOT / "ui_dashboard" / ".runtime" / "QUICK_SANITY_CHECKLIST.md"
    print(f"Manual signoff checklist: {checklist}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

