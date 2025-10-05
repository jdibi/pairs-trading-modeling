"""model_ou.py

Ornstein–Uhlenbeck (OU) process utilities for pairs-trading spread modelling.

Features
--------
- Exact-discretization simulator for the OU process
- Fast closed-form MLE via AR(1) regression equivalence
- Optional numerical MLE (exact transition density)
- Convenience analytics (half-life, stationary variance, z-score)
- Two-regime OU simulator with Markov switching (for regime-change experiments)

Author: (c) 2025
License: MIT
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Union
import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize, stats


# ---------------------------------------------------------------------------
# Core OU process
# ---------------------------------------------------------------------------
@dataclass
class OUParams:
    """Parameters of a continuous-time OU process.

    dX_t = theta * (mu - X_t) dt + sigma dW_t

    Attributes
    ----------
    mu : float
        Long-run mean (equilibrium level).
    theta : float
        Mean-reversion speed (> 0).
    sigma : float
        Diffusion/volatility (> 0) of the Brownian shock.
    dt : float
        Time step for discretization (e.g., 1/252 for daily if time in years).
    """

    mu: float
    theta: float
    sigma: float
    dt: float = 1.0 / 252.0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    @property
    def a(self) -> float:
        """AR(1) equivalent coefficient a = exp(-theta * dt)."""
        return float(np.exp(-self.theta * self.dt))

    @property
    def c(self) -> float:
        """Intercept in discrete form: c = mu * (1 - a)."""
        return float(self.mu * (1.0 - self.a))

    @property
    def var_eps(self) -> float:
        """Innovation variance of the exact discretization.

        For X_{t+1} = c + a X_t + eps_t, eps_t ~ N(0, var_eps), we have
        var_eps = (sigma^2 / (2 theta)) * (1 - a^2).
        """
        if self.theta <= 0 or self.sigma <= 0:
            return np.nan
        return float((self.sigma ** 2) / (2.0 * self.theta) * (1.0 - self.a ** 2))

    @property
    def stationary_var(self) -> float:
        """Stationary variance: sigma^2 / (2 * theta)."""
        if self.theta <= 0:
            return np.nan
        return float((self.sigma ** 2) / (2.0 * self.theta))

    @property
    def stationary_std(self) -> float:
        v = self.stationary_var
        return float(np.sqrt(v)) if np.isfinite(v) else np.nan

    @property
    def half_life(self) -> float:
        """Mean-reversion half-life in *time units of dt*.

        Half-life h satisfies exp(-theta * dt * h) = 1/2 =>
        h = ln(2) / (theta * dt)
        """
        if self.theta <= 0:
            return np.inf
        return float(np.log(2.0) / (self.theta * self.dt))

    def zscore(self, x: ArrayLike) -> np.ndarray:
        """Z-score relative to stationary distribution."""
        return (np.asarray(x) - self.mu) / self.stationary_std


class OUProcess:
    """Ornstein–Uhlenbeck process with exact Gaussian transition.

    Provides simulation, likelihood, and parameter estimation.
    """

    def __init__(self, params: OUParams):
        self.params = params

    # --------------------------- Simulation ---------------------------------
    def simulate(
        self,
        n_steps: int,
        x0: Optional[float] = None,
        n_paths: int = 1,
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> np.ndarray:
        """Simulate OU paths using the *exact* discretization.

        Parameters
        ----------
        n_steps : int
            Number of time steps to simulate (length of each path).
        x0 : float, optional
            Initial value. If None, draws a stationary sample N(mu, stationary_var).
        n_paths : int
            Number of independent paths to generate.
        random_state : int | np.random.Generator, optional
            Seed or generator.

        Returns
        -------
        np.ndarray
            Array of shape (n_steps, n_paths).
        """
        rng = _to_rng(random_state)
        p = self.params
        a, c, var_eps = p.a, p.c, p.var_eps
        std_eps = np.sqrt(var_eps)

        x = np.empty((n_steps, n_paths), dtype=float)
        if x0 is None:
            x0 = rng.normal(loc=p.mu, scale=np.sqrt(p.stationary_var), size=n_paths)
        else:
            x0 = np.full(n_paths, float(x0))

        x[0] = x0
        for t in range(1, n_steps):
            eps = rng.normal(loc=0.0, scale=std_eps, size=n_paths)
            x[t] = c + a * x[t - 1] + eps
        return x

    # --------------------------- Likelihood ---------------------------------
    def loglik(self, x: ArrayLike) -> float:
        """Exact Gaussian log-likelihood of the OU under params for a 1D series.

        The discrete form is X_{t+1} | X_t ~ N(m_t, v), with
            m_t = c + a X_t,   v = var_eps.
        """
        x = np.asarray(x, dtype=float).ravel()
        p = self.params
        a, c, v = p.a, p.c, p.var_eps
        if not np.isfinite(v) or v <= 0:
            return -np.inf
        m = c + a * x[:-1]
        ll = stats.norm.logpdf(x[1:], loc=m, scale=np.sqrt(v)).sum()
        return float(ll)

    # --------------------------- Estimation ---------------------------------
    @staticmethod
    def fit_mle_closed_form(x: ArrayLike, dt: float) -> OUParams:
        """Closed-form MLE using AR(1) regression equivalence.

        We fit X_{t+1} = c + a X_t + eps by OLS. Then map to OU params:
            a = exp(-theta dt) ⇒ theta = -ln(a)/dt
            c = mu (1 - a)     ⇒ mu = c / (1 - a)
            var_eps = Var(eps)  ⇒ sigma^2 = 2 theta * var_eps / (1 - a^2)

        Notes
        -----
        - Requires at least 3 observations.
        - If a ≤ 0 or a ≥ 1, theta becomes invalid; we clip a to (1e-6, 1 - 1e-6).
        """
        x = np.asarray(x, dtype=float).ravel()
        if x.size < 3:
            raise ValueError("Need at least 3 observations for closed-form MLE.")

        X = x[:-1]
        Y = x[1:]
        # OLS for Y = alpha + beta X + eps
        X_mat = np.vstack([np.ones_like(X), X]).T
        beta_hat, *_ = np.linalg.lstsq(X_mat, Y, rcond=None)
        c_hat, a_hat = beta_hat

        # Residual variance (ML, not unbiased): mean of squared residuals
        resid = Y - (c_hat + a_hat * X)
        var_eps = float(np.mean(resid ** 2))

        # Stabilize a_hat
        a_hat = float(np.clip(a_hat, 1e-6, 1 - 1e-6))
        theta = -np.log(a_hat) / float(dt)
        mu = c_hat / (1.0 - a_hat)
        sigma_sq = 2.0 * theta * var_eps / (1.0 - a_hat ** 2)
        sigma = float(np.sqrt(max(sigma_sq, 1e-12)))

        return OUParams(mu=float(mu), theta=float(theta), sigma=float(sigma), dt=float(dt))

    @staticmethod
    def fit_mle_numeric(
        x: ArrayLike,
        dt: float,
        init: Optional[OUParams] = None,
        bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None,
        method: str = "L-BFGS-B",
    ) -> OUParams:
        """Numerical MLE maximizing the exact Gaussian likelihood.

        Parameters
        ----------
        x : array-like
            Observations (1D) of the spread.
        dt : float
            Time step.
        init : OUParams, optional
            Initial guess. If None, uses closed-form MLE as a warm start.
        bounds : tuple, optional
            ((mu_min, mu_max), (theta_min, theta_max), (sigma_min, sigma_max)).
        method : str
            SciPy optimizer (default L-BFGS-B supports bounds).
        """
        x = np.asarray(x, dtype=float).ravel()
        if x.size < 3:
            raise ValueError("Need at least 3 observations for MLE.")

        if init is None:
            init = OUProcess.fit_mle_closed_form(x, dt)

        if bounds is None:
            # Reasonable broad bounds
            mu_span = max(1.0, np.nanstd(x) * 10)
            bounds = ((init.mu - mu_span, init.mu + mu_span), (1e-6, 1000.0), (1e-8, 1e3))

        def nll(vec: np.ndarray) -> float:
            mu, theta, sigma = vec
            if theta <= 0 or sigma <= 0:
                return 1e50
            p = OUParams(mu=mu, theta=theta, sigma=sigma, dt=dt)
            ou = OUProcess(p)
            return -ou.loglik(x)

        x0 = np.array([init.mu, init.theta, init.sigma], dtype=float)
        res = optimize.minimize(nll, x0=x0, method=method, bounds=bounds)
        mu, theta, sigma = res.x
        # Clip to safe positive
        theta = max(theta, 1e-8)
        sigma = max(sigma, 1e-12)
        return OUParams(mu=float(mu), theta=float(theta), sigma=float(sigma), dt=float(dt))


# ---------------------------------------------------------------------------
# Two-regime OU with Markov switching (simulation only)
# ---------------------------------------------------------------------------
@dataclass
class TwoRegimeParams:
    """Parameters for a 2-state Markov-switching OU model (simulation).

    Attributes
    ----------
    ou1 : OUParams
        Parameters for regime 1.
    ou2 : OUParams
        Parameters for regime 2.
    P : np.ndarray, shape (2, 2)
        Transition matrix where P[i, j] = Pr(S_{t+1} = j | S_t = i).
    pi0 : np.ndarray, shape (2,)
        Initial regime distribution. If None, uses stationary of P.
    """

    ou1: OUParams
    ou2: OUParams
    P: np.ndarray
    pi0: Optional[np.ndarray] = None

    def __post_init__(self):
        self.P = np.asarray(self.P, dtype=float)
        if self.P.shape != (2, 2):
            raise ValueError("P must be 2x2.")
        if np.any(self.P < 0) or np.any(np.abs(self.P.sum(axis=1) - 1.0) > 1e-8):
            raise ValueError("Rows of P must be nonnegative and sum to 1.")
        if self.pi0 is None:
            self.pi0 = stationary_dist_2state(self.P)
        else:
            self.pi0 = np.asarray(self.pi0, dtype=float).ravel()
            if self.pi0.size != 2 or np.any(self.pi0 < 0) or not np.isclose(self.pi0.sum(), 1.0):
                raise ValueError("pi0 must be a valid probability vector of length 2.")
        # Enforce same dt across regimes
        if not np.isclose(self.ou1.dt, self.ou2.dt):
            raise ValueError("Both regimes must share the same dt.")


class TwoRegimeOU:
    """Simulator for a two-regime OU process with Markov switching.

    Estimation is intentionally left out (handled via Kalman/EM in a separate module).
    """

    def __init__(self, params: TwoRegimeParams):
        self.params = params

    def simulate(
        self,
        n_steps: int,
        x0: Optional[float] = None,
        s0: Optional[int] = None,
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate path and regime sequence.

        Parameters
        ----------
        n_steps : int
            Number of time steps.
        x0 : float, optional
            Initial X. If None, draws from regime-weighted stationary mix.
        s0 : int, optional
            Initial regime (0 or 1). If None, draws from pi0.
        random_state : int | Generator, optional
            RNG seed or generator.

        Returns
        -------
        (x, s) : (np.ndarray, np.ndarray)
            x shape (n_steps,), s shape (n_steps,) with values in {0,1}.
        """
        rng = _to_rng(random_state)
        prm = self.params
        ou_list = [prm.ou1, prm.ou2]

        # Draw initial regime
        if s0 is None:
            s = np.empty(n_steps, dtype=int)
            s[0] = rng.choice(2, p=prm.pi0)
        else:
            s = np.empty(n_steps, dtype=int)
            s[0] = int(s0)

        # Initial X
        if x0 is None:
            # Mixture of stationary distributions weighted by pi0
            mu_mix = prm.pi0[0] * ou_list[0].mu + prm.pi0[1] * ou_list[1].mu
            var_mix = (
                prm.pi0[0] * (ou_list[0].stationary_var + (ou_list[0].mu - mu_mix) ** 2)
                + prm.pi0[1] * (ou_list[1].stationary_var + (ou_list[1].mu - mu_mix) ** 2)
            )
            x_prev = rng.normal(loc=mu_mix, scale=np.sqrt(var_mix))
        else:
            x_prev = float(x0)

        x = np.empty(n_steps, dtype=float)
        x[0] = x_prev

        for t in range(1, n_steps):
            # Transition regime
            s[t] = rng.choice(2, p=prm.P[s[t - 1]])
            ou = ou_list[s[t]]
            a, c, var_eps = ou.a, ou.c, ou.var_eps
            eps = rng.normal(loc=0.0, scale=np.sqrt(var_eps))
            x[t] = c + a * x_prev + eps
            x_prev = x[t]

        return x, s


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def stationary_dist_2state(P: np.ndarray) -> np.ndarray:
    """Stationary distribution of a 2-state Markov chain.

    For P = [[p00, p01],[p10, p11]], stationary pi solves pi = pi P, sum(pi)=1.
    Closed-form: pi0 = p10 / (p10 + p01), pi1 = p01 / (p10 + p01)
    when chain is irreducible (p01 + p10 > 0).
    """
    P = np.asarray(P, dtype=float)
    p01 = P[0, 1]
    p10 = P[1, 0]
    denom = p01 + p10
    if denom <= 0:
        # Degenerate case: stick in state 0 or 1 depending on transitions
        if p01 == 0 and p10 > 0:
            return np.array([1.0, 0.0])
        if p10 == 0 and p01 > 0:
            return np.array([0.0, 1.0])
        # Fully absorbing both states (P=I) — default to uniform
        return np.array([0.5, 0.5])
    pi0 = p10 / denom
    pi1 = p01 / denom
    return np.array([pi0, pi1])


def _to_rng(random_state: Optional[Union[int, np.random.Generator]]) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(seed=random_state)


# ---------------------------------------------------------------------------
# Quick self-test / example usage (executed only if run as a script)
# ---------------------------------------------------------------------------
if __name
