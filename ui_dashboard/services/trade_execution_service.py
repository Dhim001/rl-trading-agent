from __future__ import annotations

import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ui_dashboard.services.data_service import load_json, load_jsonl


def _runtime_orders_path(project_root: Path) -> Path:
    return project_root / "ui_dashboard" / ".runtime" / "pending_orders.json"


def load_pending_orders(project_root: Path) -> list[dict[str, Any]]:
    path = _runtime_orders_path(project_root)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []


def save_pending_orders(project_root: Path, orders: list[dict[str, Any]]) -> None:
    path = _runtime_orders_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(orders, f, indent=2)


def create_order(
    symbol: str,
    side: str,
    quantity: float,
    limit_price: float,
    stop_loss_pct: float,
    risk_note: str | None = None,
    source: str = "manual_override",
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": str(uuid.uuid4()),
        "symbol": symbol.upper().strip(),
        "side": side.lower().strip(),
        "quantity": float(quantity),
        "limit_price": float(limit_price),
        "stop_loss_pct": float(stop_loss_pct),
        "status": "pending",
        "source": source,
        "risk_note": risk_note or "",
        "created_at": now,
        "updated_at": now,
    }


def append_trade_log(project_root: Path, record: dict[str, Any]) -> None:
    path = project_root / "paper" / "trades.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def execute_order(project_root: Path, order: dict[str, Any], execution_price: float | None = None) -> dict[str, Any]:
    price = float(execution_price if execution_price is not None else order.get("limit_price", 0.0))
    qty = float(order.get("quantity", 0.0))
    side = str(order.get("side", "buy")).lower()
    notional = price * qty
    signed_notional = notional if side == "buy" else -notional
    now = datetime.now(timezone.utc).isoformat()

    order["status"] = "executed"
    order["updated_at"] = now
    order["executed_at"] = now
    order["executed_price"] = price

    append_trade_log(
        project_root,
        {
            "timestamp": now,
            "message": f"Manual override order executed: {side} {qty} {order.get('symbol')} @ {price}",
            "symbol": order.get("symbol"),
            "side": side,
            "quantity": qty,
            "price": price,
            "notional": signed_notional,
            "action": "manual_override",
            "outcome": "executed",
            "source": order.get("source", "manual_override"),
        },
    )
    return order


def cancel_order(order: dict[str, Any]) -> dict[str, Any]:
    order["status"] = "cancelled"
    order["updated_at"] = datetime.now(timezone.utc).isoformat()
    return order


def load_trade_history(project_root: Path) -> pd.DataFrame:
    rows = load_jsonl(project_root / "paper" / "trades.log")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if "symbol" not in df.columns:
        df["symbol"] = ""
    if "outcome" not in df.columns:
        df["outcome"] = "unknown"
    if "side" not in df.columns:
        df["side"] = df.get("action", "unknown")
    return df.sort_values("timestamp", ascending=False, na_position="last").reset_index(drop=True)


def get_available_symbols(project_root: Path) -> list[str]:
    cfg = load_json(project_root / "paper" / "portfolio_state.json") or {}
    shares = cfg.get("shares", {})
    symbols = sorted(str(s).upper() for s in shares.keys())
    if symbols:
        return symbols
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


def export_history_csv_bytes(df: pd.DataFrame) -> bytes:
    if df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8")


def export_history_pdf_bytes(df: pd.DataFrame) -> bytes:
    if df.empty:
        return b""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception:
        return b""

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 36
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(36, y, "Trade History Export")
    y -= 20
    pdf.setFont("Helvetica", 8)

    columns = [c for c in ["timestamp", "symbol", "side", "quantity", "price", "outcome", "message"] if c in df.columns]
    for _, row in df.head(500).iterrows():
        if y < 40:
            pdf.showPage()
            y = height - 36
            pdf.setFont("Helvetica", 8)
        parts = [f"{c}={row.get(c, '')}" for c in columns]
        line = " | ".join(parts)
        if len(line) > 160:
            line = f"{line[:157]}..."
        pdf.drawString(36, y, line)
        y -= 12
    pdf.save()
    return buffer.getvalue()


def run_paper_to_live_what_if(
    history_df: pd.DataFrame,
    slippage_bps: float,
    fee_bps: float,
) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame()
    df = history_df.copy()
    if "notional" in df.columns:
        notionals = pd.to_numeric(df["notional"], errors="coerce").fillna(0.0).abs()
    elif {"quantity", "price"}.issubset(df.columns):
        notionals = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0).abs() * pd.to_numeric(
            df["price"], errors="coerce"
        ).fillna(0.0).abs()
    else:
        notionals = pd.Series([0.0] * len(df))

    total_bps = float(slippage_bps) + float(fee_bps)
    df["estimated_live_cost"] = notionals * (total_bps / 10000.0)
    df["cumulative_live_cost"] = df["estimated_live_cost"].cumsum()
    return df
