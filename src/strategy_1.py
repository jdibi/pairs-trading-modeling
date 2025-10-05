"""strategy_1.py

Pairs trading – Strategy 1 (rolling mean reversion with ±kσ bands).

Description
-----------
- Compute rolling mean (window=h) and rolling std of the spread.
- Enter SHORT when spread > mean + k*std; enter LONG when spread < mean - k*std.
- Hold one position at a time (unit size = 1 spread unit).
- Exit when spread crosses back to the rolling mean (or within a tolerance).
- Two reporting modes:
    1) All trades (including negative PnL)
    2) Profit-only view (keep only trades with positive realized PnL)

Notes
-----
- Transaction costs are charged once per round-trip via `tc_per_round`
  (applied at exit time). If you prefer per-side costs, split by 2.
- Slippage can be added as an extra absolute penalty per round-trip via `slippage`.
- The spread is treated as a directly tradable instrument for PnL accounting:
  PnL = position * (exit_price - entry_price) - (tc_per_round + slippage).
- If you need leg-level cashflows (A long / B short) supply them in a higher
  layer; this module focuses on spread-level signals and PnL.

Author: (c) 2025
License: MIT
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------
@dataclass
class Trade:
    direction: int  # +1 = long spread, -1 = short spread
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    pnl: float
    duration: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Strategy1Config:
    window: int = 60        # rolling window h
    k: float = 2.0          # band width multiplier
    exit_tol: float = 0.0   # tolerance around mean for exit
    tc_per_round: float = 0.0  # transaction cost per round-trip
    slippage: float = 0.0      # extra penalty per round-trip
    allow_long: bool = True
    allow_short: bool = True


# ----------------------------------------------------------------------------
# Signal generation
# ----------------------------------------------------------------------------
def generate_signals(spread: pd.Series, cfg: Strategy1Config) -> pd.DataFrame:
    """Compute rolling mean/std, bands and basic entry/exit triggers.

    Returns a DataFrame with columns:
    - spread, mean, std, upper, lower, z, mean_cross (bool)
    - long_entry (bool), short_entry (bool)
    """
    s = pd.Series(spread).astype(float)
    mean = s.rolling(cfg.window, min_periods=cfg.window).mean()
    std = s.rolling(cfg.window, min_periods=cfg.window).std(ddof=0)
    upper = mean + cfg.k * std
    lower = mean - cfg.k * std
    z = (s - mean) / std

    long_entry = (s < lower) if cfg.allow_long else pd.Series(False, index=s.index)
    short_entry = (s > upper) if cfg.allow_short else pd.Series(False, index=s.index)

    # Exit when spread returns near mean (within exit_tol)
    mean_cross = (s - mean).abs() <= cfg.exit_tol

    out = pd.DataFrame({
        "spread": s,
        "mean": mean,
        "std": std,
        "upper": upper,
        "lower": lower,
        "z": z,
        "long_entry": long_entry.fillna(False),
        "short_entry": short_entry.fillna(False),
        "mean_cross": mean_cross.fillna(False),
    })
    return out


# ----------------------------------------------------------------------------
# Backtest core (state machine, one position at a time)
# ----------------------------------------------------------------------------

def backtest_spread(
    spread: pd.Series,
    cfg: Strategy1Config,
) -> Dict[str, Any]:
    """Run Strategy 1 backtest on a spread series.

    Parameters
    ----------
    spread : pd.Series
        Time-indexed spread values (float). Index can be dates.
    cfg : Strategy1Config
        Strategy configuration.

    Returns
    -------
    dict with keys:
      - signals: DataFrame of indicators and triggers
      - trades_all: list[Trade] (all trades)
      - trades_profit_only: list[Trade] (PnL>0)
      - equity_all: pd.Series (equity from all trades, step-up at exits)
      - equity_profit_only: pd.Series (equity from positive trades only)
      - stats_all: dict of summary stats for all trades
      - stats_profit_only: dict of summary stats for positive trades only
    """
    signals = generate_signals(spread, cfg)
    s = signals["spread"].values
    mean = signals["mean"].values
    long_entry = signals["long_entry"].values
    short_entry = signals["short_entry"].values
    mean_cross = signals["mean_cross"].values

    n = len(signals)
    position = 0  # +1 long, -1 short, 0 flat
    entry_price = np.nan
    entry_idx = -1

    trades: List[Trade] = []
    equity_all = np.zeros(n, dtype=float)
    equity_profit_only = np.zeros(n, dtype=float)
    cum_pnl_all = 0.0
    cum_pnl_pos = 0.0

    for i in range(n):
        # skip until we have enough window
        if np.isnan(mean[i]):
            equity_all[i] = cum_pnl_all
            equity_profit_only[i] = cum_pnl_pos
            continue

        if position == 0:
            # Flat: check entries
            if cfg.allow_long and bool(long_entry[i]):
                position = +1
                entry_price = s[i]
                entry_idx = i
            elif cfg.allow_short and bool(short_entry[i]):
                position = -1
                entry_price = s[i]
                entry_idx = i
        else:
            # In a trade: check exit condition (return to mean band)
            if bool(mean_cross[i]):
                exit_price = s[i]
                pnl = position * (exit_price - entry_price) - (cfg.tc_per_round + cfg.slippage)
                trade = Trade(
                    direction=position,
                    entry_idx=entry_idx,
                    exit_idx=i,
                    entry_price=float(entry_price),
                    exit_price=float(exit_price),
                    pnl=float(pnl),
                    duration=i - entry_idx,
                )
                trades.append(trade)

                cum_pnl_all += pnl
                equity_all[i] = cum_pnl_all

                if pnl > 0:
                    cum_pnl_pos += pnl
                    equity_profit_only[i] = cum_pnl_pos
                else:
                    equity_profit_only[i] = cum_pnl_pos

                # Reset position
                position = 0
                entry_price = np.nan
                entry_idx = -1
            else:
                # carry equity forward when still in a trade (mark-to-market not applied; realized at exit)
                equity_all[i] = cum_pnl_all
                equity_profit_only[i] = cum_pnl_pos
                
        # carry forward equity when no event
        if i > 0 and equity_all[i] == 0.0 and cum_pnl_all != 0.0:
            equity_all[i] = cum_pnl_all
        if i > 0 and equity_profit_only[i] == 0.0 and cum_pnl_pos != 0.0:
            equity_profit_only[i] = cum_pnl_pos

    # Build trade DataFrames
    trades_all = trades
    trades_profit_only = [t for t in trades if t.pnl > 0]

    equity_all = pd.Series(equity_all, index=signals.index, name="equity_all")
    equity_profit_only = pd.Series(equity_profit_only, index=signals.index, name="equity_profit_only")

    stats_all = summarize_trades(trades_all)
    stats_profit_only = summarize_trades(trades_profit_only)

    return {
        "signals": signals,
        "trades_all": trades_all,
        "trades_profit_only": trades_profit_only,
        "equity_all": equity_all,
        "equity_profit_only": equity_profit_only,
        "stats_all": stats_all,
        "stats_profit_only": stats_profit_only,
    }


# ----------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------

def summarize_trades(trades: List[Trade]) -> Dict[str, Any]:
    if not trades:
        return {
            "n_trades": 0,
            "win_rate": np.nan,
            "avg_pnl": np.nan,
            "median_pnl": np.nan,
            "total_pnl": 0.0,
            "avg_duration": np.nan,
            "pnl_std": np.nan,
            "sharpe": np.nan,
        }
    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins = (pnls > 0).mean()
    durations = np.array([t.duration for t in trades], dtype=float)

    avg = float(np.mean(pnls))
    med = float(np.median(pnls))
    total = float(np.sum(pnls))
    dur = float(np.mean(durations))
    std = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0
    sharpe = float(avg / std) if std > 0 else np.nan

    return {
        "n_trades": int(len(trades)),
        "win_rate": float(wins),
        "avg_pnl": avg,
        "median_pnl": med,
        "total_pnl": total,
        "avg_duration": dur,
        "pnl_std": std,
        "sharpe": sharpe,
    }


# ----------------------------------------------------------------------------
# Convenience: convert trade list to DataFrame
# ----------------------------------------------------------------------------

def trades_to_frame(trades: List[Trade], index: Optional[pd.Index] = None) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=[
            "direction","entry_idx","exit_idx","entry_ts","exit_ts",
            "entry_price","exit_price","pnl","duration"
        ])
    df = pd.DataFrame([t.as_dict() for t in trades])
    if index is not None and len(index) > 0:
        df["entry_ts"] = index[df["entry_idx"].values]
        df["exit_ts"] = index[df["exit_idx"].values]
    else:
        df["entry_ts"] = pd.NaT
        df["exit_ts"] = pd.NaT
    cols = ["direction","entry_idx","exit_idx","entry_ts","exit_ts","entry_price","exit_price","pnl","duration"]
    return df[cols]


# ----------------------------------------------------------------------------
# Example usage (run as script)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # quick demo on a synthetic mean-reverting series
    rng = np.random.default_rng(7)
    n = 5000
    # OU-like discrete AR(1) for demo
    a = np.exp(-3.0 / 252.0)
    mu = 0.0
    var_eps = (0.5**2 / (2*3.0)) * (1 - a**2)
    x = np.zeros(n)
    x[0] = 0.0
    for t in range(1, n):
        eps = rng.normal(scale=np.sqrt(var_eps))
        x[t] = mu*(1 - a) + a*x[t-1] + eps

    s = pd.Series(x, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    cfg = Strategy1Config(window=60, k=2.0, exit_tol=0.0, tc_per_round=0.0)

    results = backtest_spread(s, cfg)
    print("Stats (all):", results["stats_all"]) 
    print("Stats (profit-only):", results["stats_profit_only"]) 

    # Trades head
    trades_df = trades_to_frame(results["trades_all"], index=s.index)
    print(trades_df.head())
