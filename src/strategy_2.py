"""strategy_2.py

Pairs trading – Strategy 2 (equilibrium-mean reversion using OU parameters).

Description
-----------
- Uses the OU *equilibrium mean* `mu_t` and stationary std `sigma_stat_t = sigma_t / sqrt(2*theta_t)`
  to form dynamic bands:  upper = mu_t + k * sigma_stat_t,  lower = mu_t - k * sigma_stat_t.
- Enter SHORT when spread > upper; enter LONG when spread < lower.
- Exit when spread returns to the equilibrium mean (within a tolerance `exit_tol`).
- Supports *time-varying/regime-switching* OU parameters in three ways:
    (A) Fixed OU params for all t
    (B) Two-regime params + regime path `s_t ∈ {0,1}` (e.g., from Kalman/HMM)
    (C) Fully time-varying arrays for (mu_t, theta_t, sigma_t)

Accounting
----------
- One position at a time (unit size = 1 spread unit).
- Round-trip costs via `tc_per_round` (+ optional `slippage`).
- Provides both *all-trades* and *profit-only* summaries like Strategy 1.

Author: (c) 2025
License: MIT
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

try:
    # Optional import – used in the __main__ demo
    from model_ou import OUParams
except Exception:  # pragma: no cover
    OUParams = None  # type: ignore


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------
@dataclass
class Trade:
    direction: int  # +1 long spread, -1 short spread
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    pnl: float
    duration: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Strategy2Config:
    k: float = 2.0              # band width multiplier around equilibrium mean
    exit_tol: float = 0.0       # tolerance around mu_t for exit
    tc_per_round: float = 0.0   # transaction cost per round-trip
    slippage: float = 0.0       # additional penalty per round-trip
    allow_long: bool = True
    allow_short: bool = True


# ----------------------------------------------------------------------------
# Parameter expansion utilities
# ----------------------------------------------------------------------------

def _expand_params(
    n: int,
    index: pd.Index,
    *,
    ou: Optional["OUParams"] = None,
    ou1: Optional["OUParams"] = None,
    ou2: Optional["OUParams"] = None,
    regime: Optional[pd.Series] = None,
    mu_t: Optional[pd.Series] = None,
    theta_t: Optional[pd.Series] = None,
    sigma_t: Optional[pd.Series] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return time series of (mu_t, theta_t, sigma_t) according to supplied inputs.

    Priority order:
        1) Explicit arrays mu_t/theta_t/sigma_t if all provided
        2) Two regimes (ou1, ou2) + regime path
        3) Single OU params (ou)
    """
    # Case 1: fully time-varying arrays
    if (mu_t is not None) and (theta_t is not None) and (sigma_t is not None):
        mu = pd.Series(mu_t, index=index).astype(float)
        th = pd.Series(theta_t, index=index).astype(float)
        sg = pd.Series(sigma_t, index=index).astype(float)
        return mu, th, sg

    # Case 2: two regimes
    if (ou1 is not None) and (ou2 is not None) and (regime is not None):
        r = pd.Series(regime, index=index).astype(int)
        if not set(r.unique()).issubset({0, 1}):
            raise ValueError("regime must contain only 0/1 values for two-regime case")
        mu = pd.Series(np.where(r == 0, ou1.mu, ou2.mu), index=index, dtype=float)
        th = pd.Series(np.where(r == 0, ou1.theta, ou2.theta), index=index, dtype=float)
        sg = pd.Series(np.where(r == 0, ou1.sigma, ou2.sigma), index=index, dtype=float)
        return mu, th, sg

    # Case 3: single OU params
    if ou is not None:
        mu = pd.Series(ou.mu, index=index, dtype=float)
        th = pd.Series(ou.theta, index=index, dtype=float)
        sg = pd.Series(ou.sigma, index=index, dtype=float)
        return mu, th, sg

    raise ValueError(
        "Provide either (mu_t, theta_t, sigma_t) arrays, or (ou1,ou2,regime), or a single 'ou'"
    )


# ----------------------------------------------------------------------------
# Signal generation
# ----------------------------------------------------------------------------

def generate_signals(
    spread: pd.Series,
    cfg: Strategy2Config,
    *,
    ou: Optional["OUParams"] = None,
    ou1: Optional["OUParams"] = None,
    ou2: Optional["OUParams"] = None,
    regime: Optional[pd.Series] = None,
    mu_t: Optional[pd.Series] = None,
    theta_t: Optional[pd.Series] = None,
    sigma_t: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Build dynamic equilibrium bands and entry/exit triggers.

    Returns DataFrame with columns:
      spread, mu, theta, sigma, sigma_stat, upper, lower, long_entry, short_entry, exit_flag
    """
    s = pd.Series(spread).astype(float)
    mu, th, sg = _expand_params(
        n=len(s), index=s.index,
        ou=ou, ou1=ou1, ou2=ou2, regime=regime,
        mu_t=mu_t, theta_t=theta_t, sigma_t=sigma_t,
    )

    # Stationary std per time t: sigma / sqrt(2*theta)
    sig_stat = sg / np.sqrt(2.0 * th)

    upper = mu + cfg.k * sig_stat
    lower = mu - cfg.k * sig_stat

    long_entry = (s < lower) if cfg.allow_long else pd.Series(False, index=s.index)
    short_entry = (s > upper) if cfg.allow_short else pd.Series(False, index=s.index)

    # Exit when close to equilibrium mean
    exit_flag = (s - mu).abs() <= cfg.exit_tol

    out = pd.DataFrame({
        "spread": s,
        "mu": mu,
        "theta": th,
        "sigma": sg,
        "sigma_stat": sig_stat,
        "upper": upper,
        "lower": lower,
        "long_entry": long_entry,
        "short_entry": short_entry,
        "exit_flag": exit_flag,
    })
    return out


# ----------------------------------------------------------------------------
# Backtest (one position at a time)
# ----------------------------------------------------------------------------

def backtest_spread(
    spread: pd.Series,
    cfg: Strategy2Config,
    *,
    ou: Optional["OUParams"] = None,
    ou1: Optional["OUParams"] = None,
    ou2: Optional["OUParams"] = None,
    regime: Optional[pd.Series] = None,
    mu_t: Optional[pd.Series] = None,
    theta_t: Optional[pd.Series] = None,
    sigma_t: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Run Strategy 2 backtest.

    Provide OU params via one of the supported modes (ou / regimes / per-t arrays).
    """
    signals = generate_signals(
        spread, cfg,
        ou=ou, ou1=ou1, ou2=ou2, regime=regime,
        mu_t=mu_t, theta_t=theta_t, sigma_t=sigma_t,
    )

    s = signals["spread"].values
    mu = signals["mu"].values
    long_entry = signals["long_entry"].values
    short_entry = signals["short_entry"].values
    exit_flag = signals["exit_flag"].values

    n = len(signals)
    position = 0
    entry_price = np.nan
    entry_idx = -1

    trades: List[Trade] = []
    equity_all = np.zeros(n, dtype=float)
    equity_profit_only = np.zeros(n, dtype=float)
    cum_pnl_all = 0.0
    cum_pnl_pos = 0.0

    for i in range(n):
        # No warm-up needed here (bands defined for all t), but ensure theta>0
        if not np.isfinite(mu[i]):
            equity_all[i] = cum_pnl_all
            equity_profit_only[i] = cum_pnl_pos
            continue

        if position == 0:
            if cfg.allow_long and bool(long_entry[i]):
                position = +1
                entry_price = s[i]
                entry_idx = i
            elif cfg.allow_short and bool(short_entry[i]):
                position = -1
                entry_price = s[i]
                entry_idx = i
        else:
            # Exit when spread is near current mu_t
            if bool(exit_flag[i]):
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

                # reset
                position = 0
                entry_price = np.nan
                entry_idx = -1
            else:
                equity_all[i] = cum_pnl_all
                equity_profit_only[i] = cum_pnl_pos

        # carry forward
        if i > 0 and equity_all[i] == 0.0 and cum_pnl_all != 0.0:
            equity_all[i] = cum_pnl_all
        if i > 0 and equity_profit_only[i] == 0.0 and cum_pnl_pos != 0.0:
            equity_profit_only[i] = cum_pnl_pos

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
    durations = np.array([t.duration for t in trades], dtype=float)
    win_rate = float((pnls > 0).mean())

    avg = float(np.mean(pnls))
    med = float(np.median(pnls))
    total = float(np.sum(pnls))
    dur = float(np.mean(durations))
    std = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0
    sharpe = float(avg / std) if std > 0 else np.nan

    return {
        "n_trades": int(len(trades)),
        "win_rate": win_rate,
        "avg_pnl": avg,
        "median_pnl": med,
        "total_pnl": total,
        "avg_duration": dur,
        "pnl_std": std,
        "sharpe": sharpe,
    }


# ----------------------------------------------------------------------------
# Convenience
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
    # Synthetic demo: switch between two OU regimes
    rng = np.random.default_rng(123)
    n = 5000

    # Define regimes and parameters
    mu1, th1, sg1 = 0.0, 5.0, 0.4
    mu2, th2, sg2 = 1.0, 1.0, 0.8
    p00, p11 = 0.98, 0.95
    # Simulate simple 2-state Markov chain
    s = np.zeros(n, dtype=int)
    for t in range(1, n):
        if s[t-1] == 0:
            s[t] = 0 if rng.random() < p00 else 1
        else:
            s[t] = 1 if rng.random() < p11 else 0

    # Discrete-time exact form per step a_t, c_t, var_eps_t using dt = 1/252
    dt = 1/252
    a1 = np.exp(-th1 * dt); c1 = mu1 * (1 - a1); v1 = (sg1**2 / (2*th1)) * (1 - a1**2)
    a2 = np.exp(-th2 * dt); c2 = mu2 * (1 - a2); v2 = (sg2**2 / (2*th2)) * (1 - a2**2)

    x = np.zeros(n)
    for t in range(1, n):
        if s[t] == 0:
            eps = rng.normal(scale=np.sqrt(v1))
            x[t] = c1 + a1 * x[t-1] + eps
        else:
            eps = rng.normal(scale=np.sqrt(v2))
            x[t] = c2 + a2 * x[t-1] + eps

    idx = pd.date_range("
