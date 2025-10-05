"""utils.py

General utilities for the pairs-trading OU project.

This module groups together:
- Data alignment & basic transforms
- Spread construction (static/rolling/kalman hedge ratio)
- Cointegration testing (Engle–Granger, Johansen when available)
- Performance metrics (drawdown, Sharpe, CAGR, etc.)
- OU parameter estimation helpers (wrappers)
- Simple pair ranking by cointegration p-value

Author: (c) 2025
License: MIT
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Optional dependencies
try:  # statsmodels is handy but not mandatory
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint as eg_coint
    HAVE_STATSMODELS = True
except Exception:  # pragma: no cover
    sm = None
    eg_coint = None
    HAVE_STATSMODELS = False

try:
    from kalman_filter import Kalman1D
except Exception:  # pragma: no cover
    Kalman1D = None  # type: ignore

try:
    from model_ou import OUParams, OUProcess
except Exception:  # pragma: no cover
    OUParams = None  # type: ignore
    OUProcess = None  # type: ignore

# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def align_series(a: pd.Series, b: pd.Series, dropna: bool = True) -> Tuple[pd.Series, pd.Series]:
    """Align two series on their common index (intersection)."""
    a2, b2 = a.align(b, join="inner")
    if dropna:
        mask = a2.notna() & b2.notna()
        a2, b2 = a2[mask], b2[mask]
    return a2.astype(float), b2.astype(float)


def infer_periods_per_year(index: pd.Index) -> int:
    """Infer trading periods per year from a DatetimeIndex (fallback 252)."""
    if not isinstance(index, pd.DatetimeIndex):
        return 252
    # Use pandas frequency if present
    if index.freqstr:
        f = index.freqstr.upper()
        if f.startswith("B") or f.startswith("C") or f.startswith("BD"):
            return 252
        if f.startswith("D"):
            return 365
        if "W" in f:
            return 52
        if "M" in f:
            return 12
    # Fallback: count business days between first/last and scale
    days = (index[-1] - index[0]).days
    if days <= 0:
        return 252
    obs_per_day = len(index) / days
    if obs_per_day > 0.9 and obs_per_day < 1.5:
        return 365
    # Assume business daily
    return 252


def zscore(x: pd.Series) -> pd.Series:
    """Standardize a series using its mean/std (population std)."""
    m = x.mean()
    s = x.std(ddof=0)
    return (x - m) / s if s > 0 else x * 0.0


# ---------------------------------------------------------------------------
# Spread construction
# ---------------------------------------------------------------------------

def hedge_ratio_static(y1: pd.Series, y2: pd.Series, add_const: bool = True) -> float:
    """OLS hedge ratio (y1 ~ const + beta*y2). Returns beta.

    If statsmodels is available, uses OLS; otherwise uses numpy lstsq.
    """
    a, b = align_series(y1, y2)
    if HAVE_STATSMODELS:
        X = b.values.reshape(-1, 1)
        if add_const:
            X = sm.add_constant(X)
        model = sm.OLS(a.values, X).fit()
        beta = float(model.params[-1])
        return beta
    # numpy fallback
    if add_const:
        X = np.vstack([np.ones(len(b)), b.values]).T
    else:
        X = b.values.reshape(-1, 1)
    beta, *_ = np.linalg.lstsq(X, a.values, rcond=None)
    beta = beta[-1]
    return float(beta)


def spread_from_beta(y1: pd.Series, y2: pd.Series, beta: float) -> pd.Series:
    a, b = align_series(y1, y2)
    return (a - beta * b).rename("spread")


def spread_static_ols(y1: pd.Series, y2: pd.Series, add_const: bool = True) -> Tuple[pd.Series, float]:
    beta = hedge_ratio_static(y1, y2, add_const=add_const)
    s = spread_from_beta(y1, y2, beta)
    return s, beta


def spread_rolling_ols(y1: pd.Series, y2: pd.Series, window: int = 60, min_periods: Optional[int] = None) -> Tuple[pd.Series, pd.Series]:
    """Rolling OLS hedge ratio and corresponding spread.

    Returns (spread, beta_series).
    """
    if min_periods is None:
        min_periods = window
    a, b = align_series(y1, y2)
    # Compute rolling covariance/variance for slope-only regression (no intercept)
    cov = a.rolling(window, min_periods=min_periods).cov(b)
    var = b.rolling(window, min_periods=min_periods).var(ddof=0)
    beta = cov / var
    spread = a - beta * b
    return spread.rename("spread"), beta.rename("beta")


def spread_kalman_beta(y1: pd.Series, y2: pd.Series, q: float = 1e-6, r: float = 1e-4, beta0: float = 1.0, P0: float = 1.0) -> Tuple[pd.Series, pd.Series]:
    """Time-varying hedge ratio via scalar Kalman filter (random-walk beta).

    Observation: y1_t = beta_t * y2_t + v_t,  v_t~N(0,r)
    State:       beta_t = beta_{t-1} + w_t,   w_t~N(0,q)
    """
    if Kalman1D is None:
        raise ImportError("Kalman1D not available. Ensure kalman_filter.py is in path.")
    a, b = align_series(y1, y2)
    y = a.values
    x = b.values

    # Build time-varying observation gain h_t = x_t (since y_t = h_t * beta_t + v_t)
    kf = Kalman1D(c=0.0, a=1.0, Q=q, h=x, R=r, x0=beta0, P0=P0)
    beta_filt, _, _ = kf.filter(y)
    beta_series = pd.Series(beta_filt, index=a.index, name="beta")
    spread = a - beta_series * b
    return spread.rename("spread"), beta_series


# ---------------------------------------------------------------------------
# Cointegration
# ---------------------------------------------------------------------------

def engle_granger(y1: pd.Series, y2: pd.Series, trend: str = "c") -> Dict[str, Any]:
    """Run Engle–Granger cointegration test. Returns dict of (stat, pvalue, crit).

    Parameters
    ----------
    trend: {'c','ct','nc'} matching statsmodels.coint
    """
    if not HAVE_STATSMODELS or eg_coint is None:
        raise ImportError("statsmodels is required for Engle–Granger test")
    a, b = align_series(y1, y2)
    stat, pval, crit = eg_coint(a.values, b.values, trend=trend)
    return {"stat": float(stat), "pvalue": float(pval), "crit": {"1%": crit[0], "5%": crit[1], "10%": crit[2]}}


def johansen(df: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> Any:
    """Johansen cointegration test (delegates to statsmodels if available)."""
    if not HAVE_STATSMODELS:
        raise ImportError("statsmodels is required for Johansen test")
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    return coint_johansen(df.dropna(), det_order, k_ar_diff)


def rank_pairs_by_coint(prices: pd.DataFrame, method: str = "eg", pvalue_thresh: float = 0.05, max_pairs: int = 20) -> pd.DataFrame:
    """Rank all column pairs by cointegration p-value (lower is better)."""
    cols = list(prices.columns)
    rows = []
    for c1, c2 in combinations(cols, 2):
        s1, s2 = align_series(prices[c1], prices[c2])
        try:
            if method == "eg":
                res = engle_granger(s1, s2)
                p = res["pvalue"]
            else:
                # fall back to EG if Johansen not suitable for 2 series
                res = engle_granger(s1, s2)
                p = res["pvalue"]
            rows.append({"asset1": c1, "asset2": c2, "pvalue": p})
        except Exception:
            continue
    out = pd.DataFrame(rows).sort_values("pvalue")
    if pvalue_thresh is not None:
        out = out[out["pvalue"] <= pvalue_thresh]
    return out.head(max_pairs).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def drawdown(equity: pd.Series) -> pd.DataFrame:
    e = equity.fillna(method="ffill").fillna(0.0)
    peak = e.cummax()
    dd = e - peak
    dd_pct = dd / peak.replace(0, np.nan)
    return pd.DataFrame({"equity": e, "peak": peak, "drawdown": dd, "drawdown_pct": dd_pct})


def performance_from_returns(returns: pd.Series, periods_per_year: Optional[int] = None) -> Dict[str, float]:
    r = returns.dropna()
    if periods_per_year is None:
        periods_per_year = infer_periods_per_year(r.index)
    mu = r.mean() * periods_per_year
    sig = r.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = mu / sig if sig > 0 else np.nan
    downside = r[r < 0].std(ddof=0) * np.sqrt(periods_per_year)
    sortino = mu / downside if downside > 0 else np.nan
    cum = (1 + r).prod() - 1
    return {
        "ann_return": float(mu),
        "ann_vol": float(sig),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "total_return": float(cum),
    }


def performance_from_equity(equity: pd.Series, periods_per_year: Optional[int] = None) -> Dict[str, float]:
    e = equity.fillna(method="ffill").fillna(0.0)
    rets = e.diff().fillna(0.0)  # equity modeled as cumulative PnL
    perf = performance_from_returns(rets, periods_per_year)
    dd_tbl = drawdown(e)
    mdd = dd_tbl["drawdown"].min()
    mdd_pct = dd_tbl["drawdown_pct"].min()
    perf.update({"max_drawdown": float(mdd), "max_drawdown_pct": float(mdd_pct)})
    return perf


# ---------------------------------------------------------------------------
# OU estimation helpers
# ---------------------------------------------------------------------------

def estimate_ou_params(x: pd.Series, dt: float = 1/252, method: str = "closed_form") -> Dict[str, float]:
    """Estimate OU parameters using model_ou or EM-on-AR1.

    method ∈ {"closed_form","numeric","em_ar1"}
    """
    xv = x.dropna().values
    if method == "em_ar1":
        from kalman_filter import estimate_ar1_kalman
        _, ou = estimate_ar1_kalman(xv, dt=dt)
        return ou
    if OUProcess is None or OUParams is None:
        raise ImportError("model_ou not available for OU estimation")
    if method == "closed_form":
        p = OUProcess.fit_mle_closed_form(xv, dt)
    elif method == "numeric":
        p = OUProcess.fit_mle_numeric(xv, dt)
    else:
        raise ValueError("Unknown method for OU estimation")
    return {"mu": p.mu, "theta": p.theta, "sigma": p.sigma, "dt": p.dt}


# ---------------------------------------------------------------------------
# Convenience: build equity series from trade list (step function)
# ---------------------------------------------------------------------------

def equity_from_trades(trades: List[Dict[str, Any]] | List[Any], index: pd.Index) -> pd.Series:
    """Create step cumulative equity from a list of Trade-like dicts.

    Each trade must expose keys/attrs: exit_idx, pnl.
    """
    eq = pd.Series(0.0, index=index)
    cum = 0.0
    for t in trades:
        exit_idx = int(getattr(t, "exit_idx", t["exit_idx"]))
        pnl = float(getattr(t, "pnl", t["pnl"]))
        cum += pnl
        if 0 <= exit_idx < len(index):
            eq.iloc[exit_idx] = cum
    return eq.replace(0, np.nan).ffill().fillna(0.0).rename("equity")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Small smoke test
    idx = pd.date_range("2022-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    p1 = pd.Series(100 + np.cumsum(rng.normal(scale=1.0, size=len(idx))), index=idx)
    p2 = pd.Series(80 + np.cumsum(rng.normal(scale=1.0, size=len(idx))), index=idx)

    s, beta = spread_static_ols(p1, p2)
    print("Static beta:", beta)

    if HAVE_STATSMODELS:
        print("EG pvalue:", engle_granger(p1, p2)["pvalue"])  

    eperf = performance_from_equity(s.cumsum())
    print("Perf from equity:", eperf)
