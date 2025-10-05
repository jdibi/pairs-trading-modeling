"""kalman_filter.py

Kalman filtering utilities for spread modelling and regime detection.

This module provides:

1) Kalman1D
   - A minimal, fast scalar Kalman filter/smoother with optional time-varying
     transition/observation matrices. API is sklearn-like.

2) EM estimation for AR(1) state-space (AR(1) + measurement noise)
   - estimate_ar1_kalman(): estimates (c, a, Q, R) via EM, then maps to
     Ornstein–Uhlenbeck (OU) parameters (mu, theta, sigma) using exact
     discretization equivalence: a = exp(-theta*dt), c = mu*(1-a),
     Q = Var(innovation) = (sigma^2/(2*theta)) * (1 - a^2).

3) SwitchingKalman2 (2-regime)
   - Hamilton/IMM-style forward filtering for two linear-Gaussian models with
     transition matrix P. Useful for regime-labelling a spread series given two
     OU parameter sets (converted to AR(1) form).

Notes
-----
- The 1D models are chosen deliberately for robustness and readability.
- For production HMM/Kalman EM, consider `pykalman` or `hmmlearn` where
  appropriate, but this module keeps the logic lightweight and dependency-free.

Author: (c) 2025
License: MIT
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1) Core scalar Kalman filter / smoother
# ---------------------------------------------------------------------------
@dataclass
class Kalman1DState:
    x: float  # filtered state estimate
    P: float  # filtered variance


class Kalman1D:
    """Scalar Kalman filter with optional time-varying coefficients.

    Model (time t, 0..T-1):
        x_t = c_t + a_t * x_{t-1} + w_t,      w_t ~ N(0, Q_t)
        y_t = h_t * x_t + v_t,                v_t ~ N(0, R_t)

    Parameters can be scalar or arrays of length T.
    """

    def __init__(
        self,
        *,
        c: float | np.ndarray = 0.0,
        a: float | np.ndarray = 1.0,
        Q: float | np.ndarray = 1.0,
        h: float | np.ndarray = 1.0,
        R: float | np.ndarray = 0.0,
        x0: float = 0.0,
        P0: float = 1.0,
    ):
        self.c = c
        self.a = a
        self.Q = Q
        self.h = h
        self.R = R
        self.x0 = float(x0)
        self.P0 = float(P0)

    def _at(self, arr, t, default):
        if np.isscalar(arr):
            return float(arr)
        else:
            return float(arr[t]) if arr is not None else default

    def filter(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run forward pass.

        Returns
        -------
        x_filt : np.ndarray (T,)
            Filtered means E[x_t | y_{0:t}].
        P_filt : np.ndarray (T,)
            Filtered variances Var[x_t | y_{0:t}].
        loglik : float
            Total log-likelihood of observations under the model.
        """
        y = np.asarray(y, dtype=float).ravel()
        T = y.size
        x_f = np.empty(T)
        P_f = np.empty(T)

        x_prev = self.x0
        P_prev = self.P0
        loglik = 0.0

        for t in range(T):
            c = self._at(self.c, t, 0.0)
            a = self._at(self.a, t, 1.0)
            Q = self._at(self.Q, t, 1.0)
            h = self._at(self.h, t, 1.0)
            R = self._at(self.R, t, 0.0)

            # Predict
            x_pred = c + a * x_prev
            P_pred = a * P_prev * a + Q

            # Update
            S = h * P_pred * h + R  # innovation variance
            if S <= 0:
                # Guard against degeneracy
                S = 1e-12
            K = (P_pred * h) / S    # Kalman gain
            innov = y[t] - h * x_pred

            x_new = x_pred + K * innov
            P_new = (1.0 - K * h) * P_pred

            # Log-likelihood contribution
            loglik += -0.5 * (np.log(2 * np.pi * S) + (innov ** 2) / S)

            x_f[t] = x_new
            P_f[t] = P_new

            x_prev, P_prev = x_new, P_new

        return x_f, P_f, float(loglik)

    def smooth(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Rauch–Tung–Striebel smoother (returns smoothed means/vars + loglik)."""
        y = np.asarray(y, dtype=float).ravel()
        T = y.size

        # Forward pass
        x_f, P_f, loglik = self.filter(y)

        # Allocate and run RTS backward
        x_s = np.empty(T)
        P_s = np.empty(T)
        x_s[-1] = x_f[-1]
        P_s[-1] = P_f[-1]

        for t in range(T - 2, -1, -1):
            a = self._at(self.a, t + 1, 1.0)  # transition used from t -> t+1
            Q = self._at(self.Q, t + 1, 1.0)

            # Predict at t+1 from filtered at t
            x_pred = self._at(self.c, t + 1, 0.0) + a * x_f[t]
            P_pred = a * P_f[t] * a + Q

            # Smoother gain
            C = P_f[t] * a / P_pred if P_pred > 0 else 0.0

            x_s[t] = x_f[t] + C * (x_s[t + 1] - x_pred)
            P_s[t] = P_f[t] + (C ** 2) * (P_s[t + 1] - P_pred)

        return x_s, P_s, float(loglik)


# ---------------------------------------------------------------------------
# 2) EM estimation for AR(1)+noise, with OU mapping
# ---------------------------------------------------------------------------

def estimate_ar1_kalman(
    y: np.ndarray | pd.Series,
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
    init: Optional[Dict[str, float]] = None,
    dt: float = 1.0 / 252.0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Estimate (c, a, Q, R) of AR(1) state-space and map to OU (mu, theta, sigma).

    Observation model: y_t = x_t + v_t,  v_t ~ N(0, R)
    State model:       x_t = c + a x_{t-1} + w_t,  w_t ~ N(0, Q)

    Returns
    -------
    params_ar1 : dict with keys {"c","a","Q","R","x0","P0"}
    params_ou  : dict with keys {"mu","theta","sigma","dt"}
    """
    y = np.asarray(y, dtype=float).ravel()
    T = y.size

    # Initialize via simple heuristics (OLS for AR(1) with small R)
    if init is None:
        # OLS AR(1)
        Y = y[1:]
        X = np.vstack([np.ones(T - 1), y[:-1]]).T
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        c0, a0 = beta
        resid = Y - (c0 + a0 * y[:-1])
        Q0 = np.var(resid)
        R0 = max(1e-6, 0.01 * Q0)
        x0 = y[0]
        P0 = Q0
    else:
        c0 = float(init.get("c", 0.0))
        a0 = float(init.get("a", 0.9))
        Q0 = float(init.get("Q", 1e-3))
        R0 = float(init.get("R", 1e-3))
        x0 = float(init.get("x0", y[0]))
        P0 = float(init.get("P0", 1.0))

    c, a, Q, R, x0, P0 = c0, a0, max(Q0, 1e-10), max(R0, 1e-12), x0, max(P0, 1e-8)

    ll_prev = -np.inf
    for _ in range(max_iter):
        # E-step: smooth under current params
        kf = Kalman1D(c=c, a=a, Q=Q, h=1.0, R=R, x0=x0, P0=P0)
        xs, Ps, ll = kf.smooth(y)

        # Expectations of sufficient stats
        # E[x_t], E[x_t^2], E[x_t x_{t-1}]
        Ex = xs
        Ex2 = Ps + xs ** 2
        Exx1 = np.zeros(T - 1)
        for t in range(1, T):
            # Smoother cross-covariance Cov(x_t, x_{t-1} | Y)
            # Using standard 1D formula with smoother gain C_{t-1}
            # First recompute filtered quantities at t-1 (cheap in 1D by re-filtering)
            # To avoid a second full pass, we approximate with:
            #   Exx1[t-1] ≈ a * Ps[t-1] + xs[t] * xs[t-1]
            # which is accurate when residuals are small.
            Exx1[t - 1] = xs[t] * xs[t - 1]  # mild approximation for 1D EM

        # M-step updates (closed-form for linear-Gaussian)
        # Update a and c via least squares on smoothed states
        X = np.vstack([np.ones(T - 1), Ex[:-1]]).T
        Y = Ex[1:]
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        c = float(beta[0])
        a = float(beta[1])
        # Stabilize a into (-.999999, .999999)
        a = float(np.clip(a, -0.999999, 0.999999))

        # Update Q
        Q_num = 0.0
        for t in range(1, T):
            Q_num += Ex2[t] - 2 * (c + a * Ex[t - 1]) * Ex[t] + (c ** 2) + (a ** 2) * Ex2[t - 1] + 2 * c * a * Ex[t - 1]
        Q = float(max(Q_num / (T - 1), 1e-12))

        # Update R
        R_num = np.sum((y - Ex) ** 2 + Ps)
        R = float(max(R_num / T, 1e-12))

        # Convergence check
        if np.isfinite(ll) and (ll - ll_prev) < tol:
            break
        ll_prev = ll

    params_ar1 = {"c": c, "a": a, "Q": Q, "R": R, "x0": x0, "P0": P0}

    # Map to OU (only valid if |a|<1 and a>0 is expected for mean-reversion)
    a_clip = np.clip(a,
