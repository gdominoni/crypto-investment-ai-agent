"""Structured KPI reporting -- deliberately the ONLY numeric-reporting
path in this project that never touches an LLM. A command or button
press runs a real query against Freqtrade's trade database and renders
a template; there is no way for a KPI figure shown to the user to be
model-generated rather than computed.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def _signal_class(enter_tag: str | None) -> str:
    """'manual:shock_reactive:long' -> 'shock_reactive'; a battery
    candidate name (e.g. 'c2_long') -> 'battery'; missing -> 'unknown'."""
    if not enter_tag:
        return "unknown"
    if enter_tag.startswith("manual:"):
        parts = enter_tag.split(":")
        return parts[1] if len(parts) >= 3 else "manual"
    return "battery"


def _load_closed_trades(db_path: str | Path) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql(
            "SELECT pair, is_short, open_rate, close_rate, close_profit, "
            "open_date, close_date, enter_tag, exit_reason "
            "FROM trades WHERE is_open = 0", conn,
        )
    finally:
        conn.close()
    df["coin"] = df["pair"].str.split("/").str[0]
    df["signal"] = df["enter_tag"].fillna("unknown")
    df["signal_class"] = df["enter_tag"].apply(_signal_class)
    return df


def _sharpe(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return float("nan")
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def _sortino(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    downside = returns[returns < 0]
    if len(downside) < 2 or downside.std() == 0:
        return float("nan")
    return float(returns.mean() / downside.std() * np.sqrt(periods_per_year))


def _max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return float("nan")
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def kpi_table(db_path: str | Path, group_by: str = "coin") -> pd.DataFrame:
    """`group_by`: 'coin', 'signal', or None (overall)."""
    trades = _load_closed_trades(db_path)
    if len(trades) == 0:
        return pd.DataFrame(columns=["group", "n", "win_rate", "net_profit", "max_drawdown", "sharpe", "sortino"])

    groups = [None] if group_by is None else sorted(trades[group_by].unique())
    rows = []
    for g in groups:
        sub = trades if g is None else trades[trades[group_by] == g]
        returns = sub["close_profit"].to_numpy()
        wins = int((sub["close_profit"] > 0).sum())
        n = len(sub)
        rows.append({
            "group": g if g is not None else "ALL",
            "n": n,
            "win_rate": wins / n if n else np.nan,
            "net_profit": float(sub["close_profit"].sum()),
            "max_drawdown": _max_drawdown(sub["close_profit"]),
            "sharpe": _sharpe(returns),
            "sortino": _sortino(returns),
        })
    return pd.DataFrame(rows)


def format_kpi_table(df: pd.DataFrame, title: str) -> str:
    if len(df) == 0:
        return f"{title}\n\nNo closed trades yet."
    lines = [title, ""]
    lines.append(f"{'Group':<10} {'N':>4} {'WinRate':>8} {'NetProfit':>10} {'MaxDD':>8} {'Sharpe':>7} {'Sortino':>8}")
    for _, r in df.iterrows():
        lines.append(
            f"{str(r['group']):<10} {r['n']:>4} {r['win_rate']:>7.1%} {r['net_profit']:>+9.2%} "
            f"{r['max_drawdown']:>7.2%} {r['sharpe']:>7.2f} {r['sortino']:>8.2f}"
        )
    return "\n".join(lines)
