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
    a_clip = np.clip(a, 1e-9, 1 - 1e-9)
    theta = -np.log(a_clip) / dt
    mu = c / (1.0 - a_clip)
    sigma_sq = 2.0 * theta * Q / (1.0 - a_clip ** 2)
    sigma = float(np.sqrt(max(sigma_sq, 1e-12)))

    params_ou = {"mu": float(mu), "theta": float(theta), "sigma": sigma, "dt": float(dt)}
    return params_ar1, params_ou


# ---------------------------------------------------------------------------
# 3) Two-regime switching Kalman filter (forward probabilities)
# ---------------------------------------------------------------------------
class SwitchingKalman2:
    """Two-regime linear-Gaussian model with fixed AR(1) params per regime.

    For regime j ∈ {0,1}:
        x_t = c_j + a_j x_{t-1} + w_t,  w_t ~ N(0, Q_j)
        y_t = x_t + v_t,                v_t ~ N(0, R_j)

    Regime S_t follows a Markov chain with transition matrix P (2x2).

    This class computes filtered probabilities p(S_t=j | y_{0:t}) and returns
    the most likely regime path (argmax) as a convenience.
    """

    def __init__(
        self,
        *,
        c: Tuple[float, float],
        a: Tuple[float, float],
        Q: Tuple[float, float],
        R: Tuple[float, float],
        P: np.ndarray,  # shape (2,2)
        x0: float = 0.0,
        P0: float = 1.0,
        pi0: Optional[np.ndarray] = None,
    ):
        self.c = np.array(c, dtype=float)
        self.a = np.array(a, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        self.P = np.asarray(P, dtype=float)
        if self.P.shape != (2, 2):
            raise ValueError("P must be 2x2")
        if pi0 is None:
            # stationary distribution of P
            p01 = self.P[0, 1]
            p10 = self.P[1, 0]
            denom = p01 + p10
            if denom > 0:
                pi0 = np.array([p10 / denom, p01 / denom])
            else:
                pi0 = np.array([0.5, 0.5])
        self.pi0 = np.asarray(pi0, dtype=float)
        self.x0 = float(x0)
        self.P0 = float(P0)

    def filter(self, y: np.ndarray | pd.Series) -> Dict[str, Any]:
        y = np.asarray(y, dtype=float).ravel()
        T = y.size

        # Per-regime filtered states and variances
        x = np.zeros((T, 2))
        V = np.zeros((T, 2))
        ll = 0.0

        # Regime probabilities
        p_filt = np.zeros((T, 2))
        p_prev = self.pi0.copy()

        # Initialize state means for each regime equally
        x_prev = np.array([self.x0, self.x0], dtype=float)
        V_prev = np.array([self.P0, self.P0], dtype=float)

        for t in range(T):
            # Interact (mix) prior over regimes
            p_pred = p_prev @ self.P  # shape (2,)

            x_pred = np.empty(2)
            V_pred = np.empty(2)
            like = np.empty(2)

            for j in range(2):
                c, a, Q, R = self.c[j], self.a[j], self.Q[j], self.R[j]
                # Predict
                xj_pred = c + a * x_prev[j]
                Vj_pred = a * V_prev[j] * a + Q
                # Update
                Sj = Vj_pred + R
                Kj = Vj_pred / Sj
                innov = y[t] - xj_pred
                xj_new = xj_pred + Kj * innov
                Vj_new = (1.0 - Kj) * Vj_pred

                x_pred[j] = xj_new
                V_pred[j] = Vj_new
                like[j] = (1.0 / np.sqrt(2 * np.pi * Sj)) * np.exp(-0.5 * innov ** 2 / Sj)

            # Regime update via Bayes
            numer = p_pred * like
            denom = numer.sum()
            if denom <= 0:
                numer = np.maximum(numer, 1e-300)
                denom = numer.sum()
            p_curr = numer / denom
            ll += np.log(denom)

            # Save
            x[t, :] = x_pred
            V[t, :] = V_pred
            p_filt[t, :] = p_curr

            # Prepare next step: set regime-conditional states as current
            x_prev, V_prev = x_pred, V_pred
            p_prev = p_curr

        most_likely = p_filt.argmax(axis=1)
        return {
            "x_filt": x,          # shape (T,2)
            "V_filt": V,          # shape (T,2)
            "p_filt": p_filt,     # shape (T,2)
            "regime": most_likely,  # argmax path
            "loglik": float(ll),
        }


# ---------------------------------------------------------------------------
# Helper: map OU params to AR(1) (c, a, Q) given dt
# ---------------------------------------------------------------------------

def ou_to_ar1(mu: float, theta: float, sigma: float, dt: float) -> Tuple[float, float, float]:
    """Exact discretization mapping OU -> AR(1) with Gaussian innovations.

    X_{t+1} = c + a X_t + eps,  eps ~ N(0, Q)
    a = exp(-theta*dt)
    c = mu * (1 - a)
    Q = (sigma^2 / (2*theta)) * (1 - a^2)
    """
    a = float(np.exp(-theta * dt))
    c = float(mu * (1.0 - a))
    if theta <= 0:
        raise ValueError("theta must be > 0 for OU mapping")
    Q = float((sigma ** 2) / (2.0 * theta) * (1.0 - a ** 2))
    return c, a, Q


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    T = 2000

    # Generate AR(1)+noise, then estimate via EM
    a_true = 0.99
    c_true = 0.0
    Q_true = 1e-4
    R_true = 1e-3

    x = np.zeros(T)
    y = np.zeros(T)
    x[0] = 0.5
    for t in range(1, T):
        x[t] = c_true + a_true * x[t-1] + rng.normal(scale=np.sqrt(Q_true))
    y = x + rng.normal(scale=np.sqrt(R_true), size=T)

    ar1_hat, ou_hat = estimate_ar1_kalman(y, dt=1/252)
    print("AR1 hat:", ar1_hat)
    print("OU hat:", ou_hat)

    # Switching demo
    dt = 1/252
    mu1, th1, sg1 = 0.0, 5.0, 0.4
    mu2, th2, sg2 = 1.0, 1.0, 0.8
    c1, a1, Q1 = ou_to_ar1(mu1, th1, sg1, dt)
    c2, a2, Q2 = ou_to_ar1(mu2, th2, sg2, dt)
    R = 1e-6  # assume negligible measurement noise on spread

    # Simulate switching AR(1)
    P = np.array([[0.98, 0.02],[0.05, 0.95]])
    s = np.zeros(T, dtype=int)
    for t in range(1, T):
        s[t] = 0 if (s[t-1] == 0 and rng.random() < P[0,0]) else (1 if s[t-1]==0 else (1 if rng.random() < P[1,1] else 0))
    x = np.zeros(T)
    for t in range(1, T):
        if s[t] == 0:
            x[t] = c1 + a1 * x[t-1] + rng.normal(scale=np.sqrt(Q1))
        else:
            x[t] = c2 + a2 * x[t-1] + rng.normal(scale=np.sqrt(Q2))
    y = x + rng.normal(scale=np.sqrt(R), size=T)

    sw = SwitchingKalman2(c=(c1,c2), a=(a1,a2), Q=(Q1,Q2), R=(R,R), P=P, x0=0.0, P0=1.0)
    res = sw.filter(y)
    print("Switching loglik:", res["loglik"]) 
    print("Regime accuracy (argmax):", (res["regime"] == s).mean())

    # Simple plot
    plt.figure()
    plt.plot(y, label="obs spread", alpha=0.6)
    plt.plot(res["x_filt"][:,0]*res["p_filt"][:,0] + res["x_filt"][:,1]*res["p_filt"][:,1], label="filtered x (mix)")
    plt.legend()
    plt.title("Switching Kalman – filtered state")
    plt.show()
