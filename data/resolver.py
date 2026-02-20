"""
data/resolver.py
Auto-resolution of optional pricing parameters from live market data.

  spot           → yfinance  fast_info["last_price"]
  risk_free_rate → yfinance  ^IRX (13-week T-bill annualised yield ÷ 100)
  volatility     → Databento OPRA.PILLAR NBBO mid + Black-Scholes inversion
                   (closest strike × nearest expiry to the requested position)

Ticker convention
-----------------
  Callers pass the equity root ticker  (e.g. "AAPL").
  Databento calls are handled inside data.market_data which appends ".OPT"
  automatically — callers never need to know about the OPRA parent format.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm as sp_norm

from data.market_data import get_options_chain

logger = logging.getLogger("pricer.resolver")

_TICKERS = {"T_BILL": "^IRX"}


# ── Black-Scholes helpers ─────────────────────────────────────────────────────

def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * sp_norm.cdf(d1) - K * np.exp(-r * T) * sp_norm.cdf(d2)


def _bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * sp_norm.cdf(-d2) - S * sp_norm.cdf(-d1)


def _implied_vol(
    S: float, K: float, T: float, r: float,
    market_price: float, option_type: str,
) -> Optional[float]:
    bs = _bs_call if option_type == "call" else _bs_put
    try:
        return brentq(
            lambda sigma: bs(S, K, T, r, sigma) - market_price,
            1e-4, 10.0, xtol=1e-6,
        )
    except Exception:
        return None


# ── Public resolvers ──────────────────────────────────────────────────────────

def fetch_spot(ticker: str) -> float:
    """Live last price from yfinance. Raises RuntimeError on failure."""
    try:
        price = yf.Ticker(ticker.upper()).fast_info["last_price"]
        if not price or price <= 0:
            raise ValueError(f"Invalid price returned: {price}")
        logger.info("Spot resolved | ticker=%s price=%.4f source=yfinance", ticker, price)
        return float(price)
    except Exception as exc:
        raise RuntimeError(f"Could not fetch spot for {ticker} from yfinance: {exc}") from exc


def fetch_risk_free_rate() -> float:
    """
    13-week US T-bill yield (^IRX) from yfinance, annualised, as a decimal.
    e.g. Yahoo returns 3.60 → we return 0.0360.

    This is the standard short-term risk-free proxy for equity options pricing.
    Raises RuntimeError if the feed is unavailable.
    """
    try:
        pct = yf.Ticker(_TICKERS["T_BILL"]).fast_info["last_price"]
        if not pct or pct <= 0:
            raise ValueError(f"Invalid yield returned: {pct}")
        rate = float(pct) / 100.0
        logger.info(
            "Risk-free rate resolved | ticker=%s yield=%.4f%% rate=%.6f source=yfinance",
            _TICKERS["T_BILL"], pct, rate,
        )
        return rate
    except Exception as exc:
        raise RuntimeError(f"Could not fetch risk-free rate from yfinance ^IRX: {exc}") from exc


def fetch_implied_vol(
    ticker: str,
    strike: float,
    days_to_expiry: int,
    option_type: str,
    spot: float,
    risk_free_rate: float,
) -> tuple[float, str]:
    """
    Back-calculate implied volatility from Databento OPRA NBBO mid.

    Strategy:
      1. Fetch the full options chain for `ticker` (AAPL → AAPL.OPT internally)
      2. Filter by option_type, find the row whose (strike, expiry) is closest
         to (requested_strike, today + days_to_expiry)
      3. Invert Black-Scholes on the NBBO mid price

    Returns
    -------
    (iv, source_description)
        iv  : annualised implied volatility as a decimal (e.g. 0.30)
        source : human-readable string describing which contract was used

    Raises RuntimeError if no suitable option is found or inversion fails.
    """
    # ticker passed as bare equity root ("AAPL") — market_data adds ".OPT"
    snapshot = get_options_chain(ticker)
    df = snapshot.calls if option_type == "call" else snapshot.puts

    if df.empty:
        raise RuntimeError(
            f"No {option_type} options found in Databento chain for {ticker}"
        )

    # Target expiry as a date
    target_expiry = date.today() + timedelta(days=days_to_expiry)

    # Find the row closest in (strike_distance + expiry_distance_days)
    df = df.copy()
    df["strike_dist"]  = (df["strike"] - strike).abs()
    df["expiry_dist"]  = (
        df["expiration"].dt.date.apply(lambda d: abs((d - target_expiry).days))
    )
    # Weighted score: normalise so 1-day expiry error ≈ 1-dollar strike error
    df["score"] = df["strike_dist"] + df["expiry_dist"] * (strike * 0.001)
    best = df.loc[df["score"].idxmin()]

    matched_strike = float(best["strike"])
    matched_expiry = best["expiration"].date()
    mid_price      = float(best["mid"])
    T              = float(best["time_to_expiry"])

    if mid_price <= 0:
        raise RuntimeError(
            f"NBBO mid is zero/negative for {ticker} {option_type} "
            f"K={matched_strike} exp={matched_expiry} — option may be illiquid"
        )

    iv = _implied_vol(spot, matched_strike, T, risk_free_rate, mid_price, option_type)

    if iv is None:
        raise RuntimeError(
            f"Black-Scholes IV inversion failed for {ticker} {option_type} "
            f"K={matched_strike} exp={matched_expiry} mid={mid_price:.4f}"
        )

    source = (
        f"databento_iv({ticker} {option_type[0].upper()}{matched_strike:.0f} "
        f"exp={matched_expiry} mid={mid_price:.4f} iv={iv*100:.2f}%)"
    )
    logger.info(
        "IV resolved | %s strike=%.2f→%.2f expiry=%s mid=%.4f iv=%.4f",
        ticker, strike, matched_strike, matched_expiry, mid_price, iv,
    )
    return float(iv), source
