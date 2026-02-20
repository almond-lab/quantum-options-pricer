"""
tests/conftest.py
Shared fixtures and environment setup for all test modules.
Forces BACKEND=cpu so tests run without a GPU.
"""

import os
import pytest
import numpy as np

# Force CPU backend before any settings are loaded
os.environ.setdefault("BACKEND", "cpu")
os.environ.setdefault("LOG_LEVEL", "DEBUG")
os.environ.setdefault("PRECISION", "double")


# ── Black-Scholes reference implementation ────────────────────────────────────

from scipy.stats import norm as _norm

def black_scholes(spot, strike, T, r, vol, option_type="call"):
    """Closed-form Black-Scholes for test reference values."""
    if T <= 0:
        payoff = max(spot - strike, 0) if option_type == "call" else max(strike - spot, 0)
        return payoff
    d1 = (np.log(spot / strike) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    if option_type == "call":
        return spot * _norm.cdf(d1) - strike * np.exp(-r * T) * _norm.cdf(d2)
    else:
        return strike * np.exp(-r * T) * _norm.cdf(-d2) - spot * _norm.cdf(-d1)


@pytest.fixture(scope="session")
def bs():
    """Black-Scholes reference pricer."""
    return black_scholes


@pytest.fixture(scope="session")
def atm_params():
    """ATM option params — keys match derive_lognormal_params signature + strike for circuit tests."""
    return dict(spot=100.0, strike=100.0, risk_free_rate=0.05, volatility=0.20, T=1.0)


@pytest.fixture(scope="session")
def itm_params():
    return dict(spot=110.0, strike=100.0, risk_free_rate=0.05, volatility=0.20, T=1.0)


@pytest.fixture(scope="session")
def otm_params():
    return dict(spot=90.0, strike=100.0, risk_free_rate=0.05, volatility=0.20, T=1.0)
