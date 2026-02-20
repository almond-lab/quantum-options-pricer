"""
data/market_data.py
Databento OPRA.PILLAR data ingestion service.

Cost strategy (Pay-As-You-Go):
  - Definitions call  : full chain metadata for the trading day (static, cheap)
  - NBBO snapshot call: exactly 1-minute window at market close (fractions of a cent)
    → 15:59:00 → 16:00:00 ET captures final pre-close NBBO for the entire chain

Output: OptionsChainSnapshot — ready for Black-Scholes IV inversion
        and subsequent SABR Volatility Surface calibration.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional

import databento as db
import numpy as np
import pandas as pd

from config.settings import get_settings

logger = logging.getLogger("pricer.market_data")

# ── Constants ─────────────────────────────────────────────────────────────────

OPRA_DATASET       = "OPRA.PILLAR"
ET_ZONE            = ZoneInfo("America/New_York")
MARKET_CLOSE_HOUR  = 16        # 4:00 PM ET — standard US equity options close
SNAPSHOT_MINUTES   = 1         # 1-minute window: 15:59–16:00 ET

# Prices returned by .to_df(price_type='float') are already in USD.
# UNDEF_PRICE as float sentinel: ~9.22e9. Any real option price is far below this.
UNDEF_PRICE_FLOAT  = db.UNDEF_PRICE / db.FIXED_PRICE_SCALE   # ≈ 9_223_372_036.85

# instrument_class values for vanilla options
OPTION_CLASSES     = frozenset({"C", "P"})


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OptionsChainSnapshot:
    """
    Complete options chain for a single ticker at market close.

    .combined  — full chain DataFrame, sorted by (option_type, expiration, strike)
    .calls     — calls only
    .puts      — puts only

    Columns:
        instrument_id   int     Databento instrument identifier
        raw_symbol      str     OCC symbology string (e.g. AAPL  240119C00150000)
        option_type     str     'call' or 'put'
        strike          float   Strike price in USD
        expiration      dt      Expiration datetime
        days_to_expiry  int     Calendar days to expiration
        time_to_expiry  float   days_to_expiry / 365.0
        bid             float   NBBO best bid in USD
        ask             float   NBBO best ask in USD
        mid             float   (bid + ask) / 2
        spread          float   ask - bid
        bid_size        int     Best bid size (contracts)
        ask_size        int     Best ask size (contracts)
        ts_recv         dt      Timestamp of the NBBO update
    """
    ticker:         str
    snapshot_date:  date
    snapshot_time:  datetime    # start of 1-minute window in ET
    calls:          pd.DataFrame
    puts:           pd.DataFrame
    combined:       pd.DataFrame

    @property
    def unique_expirations(self) -> list[date]:
        return sorted(self.combined["expiration"].dt.date.unique())

    @property
    def unique_strikes(self) -> list[float]:
        return sorted(self.combined["strike"].unique())

    def filter_expiration(self, target: date) -> "OptionsChainSnapshot":
        """Return a snapshot containing only options expiring on target date."""
        mask = self.combined["expiration"].dt.date == target
        combined = self.combined[mask].reset_index(drop=True)
        calls    = combined[combined["option_type"] == "call"].reset_index(drop=True)
        puts     = combined[combined["option_type"] == "put"].reset_index(drop=True)
        return OptionsChainSnapshot(
            ticker=self.ticker,
            snapshot_date=self.snapshot_date,
            snapshot_time=self.snapshot_time,
            calls=calls,
            puts=puts,
            combined=combined,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _previous_trading_day(reference: Optional[date] = None) -> date:
    """
    Return the most recent trading day before reference (defaults to today).

    Skips Saturdays and Sundays. Does not account for market holidays —
    integrate exchange_calendars or pandas_market_calendars for full coverage.
    """
    ref    = reference or date.today()
    target = ref - timedelta(days=1)
    while target.weekday() >= 5:   # 5 = Sat, 6 = Sun
        target -= timedelta(days=1)
    logger.debug("Reference=%s → previous trading day=%s", ref, target)
    return target


def _close_window(trading_day: date) -> tuple[datetime, datetime]:
    """
    Build the 1-minute NBBO snapshot window in ET (DST-aware via zoneinfo).

    Captures 15:59:00 → 16:00:00 ET — the final trading minute for US equity
    options. Databento accepts timezone-aware datetimes directly.
    """
    start = datetime(
        trading_day.year, trading_day.month, trading_day.day,
        MARKET_CLOSE_HOUR - 1, 59, 0,
        tzinfo=ET_ZONE,
    )
    end = datetime(
        trading_day.year, trading_day.month, trading_day.day,
        MARKET_CLOSE_HOUR, 0, 0,
        tzinfo=ET_ZONE,
    )
    utc = ZoneInfo("UTC")
    logger.debug(
        "Close window | ET %s → %s | UTC %s → %s",
        start.strftime("%H:%M:%S"), end.strftime("%H:%M:%S"),
        start.astimezone(utc).strftime("%H:%M:%S"),
        end.astimezone(utc).strftime("%H:%M:%S"),
    )
    return start, end


def _is_undefined_price(series: pd.Series) -> pd.Series:
    """Return boolean mask: True where price is Databento's UNDEF sentinel or NaN."""
    return series.isna() | (series >= UNDEF_PRICE_FLOAT) | (series <= 0)


# ── Step 1A: Definitions ──────────────────────────────────────────────────────

def fetch_options_definitions(
    client: db.Historical,
    ticker: str,
    trading_day: date,
) -> pd.DataFrame:
    """
    Fetch all option definitions for the ticker on trading_day.

    Uses schema=DEFINITION + stype_in=PARENT to retrieve the full chain
    (every strike and expiration on OPRA for this underlying).

    Returns clean DataFrame with columns:
        instrument_id, raw_symbol, option_type, strike, expiration
    """
    day_str      = trading_day.strftime("%Y-%m-%d")
    next_day_str = (trading_day + timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(
        "Definitions fetch | ticker=%s dataset=%s date=%s",
        ticker, OPRA_DATASET, day_str,
    )

    store = client.timeseries.get_range(
        dataset  = OPRA_DATASET,
        schema   = db.Schema.DEFINITION,
        stype_in = db.SType.PARENT,
        symbols  = [ticker],
        start    = day_str,
        end      = next_day_str,
    )

    df_raw = store.to_df(price_type="float", pretty_ts=True)

    if df_raw.empty:
        logger.warning("No definitions returned | ticker=%s date=%s", ticker, day_str)
        return pd.DataFrame()

    logger.info("Definitions raw | rows=%d columns=%s", len(df_raw), df_raw.columns.tolist())

    # ── Filter: keep only vanilla Call and Put instruments ────────────────────
    df = df_raw[df_raw["instrument_class"].isin(OPTION_CLASSES)].copy()

    n_dropped = len(df_raw) - len(df)
    logger.info(
        "Definitions filtered | kept=%d dropped=%d (non C/P instrument classes)",
        len(df), n_dropped,
    )

    if df.empty:
        logger.warning("No C/P options found in definitions for %s on %s", ticker, day_str)
        return pd.DataFrame()

    # ── Build clean output schema ─────────────────────────────────────────────
    result = pd.DataFrame({
        "instrument_id": df["instrument_id"].astype(int),
        "raw_symbol":    df["raw_symbol"].astype(str),
        "option_type":   df["instrument_class"].map({"C": "call", "P": "put"}),
        "strike":        df["strike_price"].astype(float),   # USD, already scaled
        "expiration":    pd.to_datetime(df["expiration"]),
    })

    logger.info(
        "Definitions ready | calls=%d puts=%d unique_strikes=%d unique_expirations=%d",
        (result["option_type"] == "call").sum(),
        (result["option_type"] == "put").sum(),
        result["strike"].nunique(),
        result["expiration"].dt.date.nunique(),
    )

    return result


# ── Step 1B: NBBO Snapshot ────────────────────────────────────────────────────

def fetch_nbbo_snapshot(
    client: db.Historical,
    ticker: str,
    trading_day: date,
) -> pd.DataFrame:
    """
    Fetch the 1-minute NBBO snapshot at market close (cost-saving strategy).

    Uses schema=MBP_1 (Market by Price, Level 1 = National Best Bid and Offer).
    Requests exactly 1 minute: 15:59:00 → 16:00:00 ET.
    Takes the LAST update per instrument in the window (most recent pre-close quote).

    Returns clean DataFrame with columns:
        instrument_id, ts_recv, bid, ask, bid_size, ask_size, mid, spread
    """
    start_et, end_et = _close_window(trading_day)

    logger.info(
        "NBBO snapshot fetch | ticker=%s window=%s→%s ET (%d min)",
        ticker,
        start_et.strftime("%H:%M:%S"),
        end_et.strftime("%H:%M:%S"),
        SNAPSHOT_MINUTES,
    )

    store = client.timeseries.get_range(
        dataset  = OPRA_DATASET,
        schema   = db.Schema.MBP_1,
        stype_in = db.SType.PARENT,
        symbols  = [ticker],
        start    = start_et,
        end      = end_et,
    )

    df_raw = store.to_df(price_type="float", pretty_ts=True)

    if df_raw.empty:
        logger.warning(
            "No NBBO data returned | ticker=%s date=%s "
            "— possible holiday, early close, or market halt.",
            ticker, trading_day,
        )
        return pd.DataFrame()

    logger.info(
        "NBBO raw | rows=%d (all L1 updates in window) instruments=%d",
        len(df_raw), df_raw["instrument_id"].nunique(),
    )

    # ── Take the LAST update per instrument (final pre-close NBBO) ────────────
    df = (
        df_raw
        .sort_values("ts_recv")
        .groupby("instrument_id", as_index=False)
        .last()
    )
    logger.info("NBBO after last-per-instrument dedup | rows=%d", len(df))

    # ── Validate NBBO prices — drop undefined / stale quotes ──────────────────
    # MBP-1 level-0 columns after to_df() expansion: bid_px_00, ask_px_00, etc.
    bid_col = "bid_px_00"
    ask_col = "ask_px_00"

    valid_mask = (
        ~_is_undefined_price(df[bid_col]) &
        ~_is_undefined_price(df[ask_col]) &
        (df[bid_col] <= df[ask_col])   # bid must not exceed ask (crossed market guard)
    )

    n_stale = (~valid_mask).sum()
    if n_stale > 0:
        logger.warning(
            "Dropped %d instruments with stale/undefined/crossed NBBO "
            "(far OTM, not trading, or halted)", n_stale,
        )
    df = df[valid_mask].copy()

    if df.empty:
        logger.warning("All NBBO quotes were invalid for %s on %s", ticker, trading_day)
        return pd.DataFrame()

    # ── Build clean output schema ─────────────────────────────────────────────
    result = pd.DataFrame({
        "instrument_id": df["instrument_id"].astype(int),
        "ts_recv":       df["ts_recv"],
        "bid":           df[bid_col].astype(float),
        "ask":           df[ask_col].astype(float),
        "bid_size":      df["bid_sz_00"].astype(int),
        "ask_size":      df["ask_sz_00"].astype(int),
        "mid":           ((df[bid_col] + df[ask_col]) / 2).astype(float),
        "spread":        (df[ask_col] - df[bid_col]).astype(float),
    })

    logger.info(
        "NBBO snapshot ready | valid_quotes=%d avg_bid=%.4f avg_ask=%.4f avg_spread=%.4f",
        len(result),
        result["bid"].mean(),
        result["ask"].mean(),
        result["spread"].mean(),
    )

    return result


# ── Step 1C: Build Chain ──────────────────────────────────────────────────────

def build_options_chain(
    definitions: pd.DataFrame,
    nbbo: pd.DataFrame,
    trading_day: date,
    ticker: str,
) -> Optional[OptionsChainSnapshot]:
    """
    Inner join definitions with NBBO quotes and produce a clean OptionsChainSnapshot.

    Only options with BOTH a definition AND a valid NBBO quote are retained.
    Adds days_to_expiry and time_to_expiry for downstream IV / SABR calibration.
    Drops: expired options, zero-mid options, negative-spread data errors.
    """
    if definitions.empty or nbbo.empty:
        logger.error(
            "Cannot build chain | definitions_empty=%s nbbo_empty=%s",
            definitions.empty, nbbo.empty,
        )
        return None

    chain = definitions.merge(nbbo, on="instrument_id", how="inner")

    n_unmatched = len(definitions) - len(chain)
    logger.info(
        "Chain join | definitions=%d nbbo=%d matched=%d unmatched=%d",
        len(definitions), len(nbbo), len(chain), n_unmatched,
    )
    if n_unmatched > 0:
        logger.debug(
            "%d defined options had no NBBO in the close window "
            "(far OTM, inactive, or delisted).", n_unmatched,
        )

    # ── Time to expiry ────────────────────────────────────────────────────────
    chain["days_to_expiry"] = (
        chain["expiration"].dt.date.apply(lambda d: (d - trading_day).days)
    )
    chain["time_to_expiry"] = chain["days_to_expiry"] / 365.0

    # ── Filters ───────────────────────────────────────────────────────────────
    before = len(chain)
    chain = chain[
        (chain["days_to_expiry"] > 0) &   # drop expired / same-day
        (chain["mid"] > 0) &              # drop zero-mid (no real market)
        (chain["spread"] >= 0)            # drop crossed/data-error quotes
    ].copy()

    n_filtered = before - len(chain)
    if n_filtered:
        logger.debug("Post-join filters removed %d rows (expired/zero-mid/crossed)", n_filtered)

    # ── Final sort ────────────────────────────────────────────────────────────
    chain = chain.sort_values(
        ["option_type", "expiration", "strike"]
    ).reset_index(drop=True)

    calls = chain[chain["option_type"] == "call"].reset_index(drop=True)
    puts  = chain[chain["option_type"] == "put"].reset_index(drop=True)

    snapshot_time, _ = _close_window(trading_day)

    logger.info(
        "OptionsChainSnapshot ready | ticker=%s date=%s calls=%d puts=%d expirations=%d",
        ticker, trading_day, len(calls), len(puts),
        chain["expiration"].dt.date.nunique(),
    )

    return OptionsChainSnapshot(
        ticker        = ticker,
        snapshot_date = trading_day,
        snapshot_time = snapshot_time,
        calls         = calls,
        puts          = puts,
        combined      = chain,
    )


# ── Public entry point ────────────────────────────────────────────────────────

def get_options_chain(
    ticker: str,
    api_key: Optional[str] = None,
    reference_date: Optional[date] = None,
) -> OptionsChainSnapshot:
    """
    Fetch a complete institutional-grade options chain snapshot for ticker.

    Cost profile (Pay-As-You-Go):
      Call 1 — definitions : static metadata for the day  (negligible cost)
      Call 2 — mbp-1 NBBO  : exactly 1 minute of L1 data (fractions of a cent)

    Args:
        ticker:         Root symbol (e.g. "AAPL", "SPY", "TSLA")
        api_key:        Databento API key. Falls back to DATABENTO_API_KEY env var.
        reference_date: Pricing reference date. Defaults to today.
                        The snapshot uses the previous trading day from this date.

    Returns:
        OptionsChainSnapshot — .calls, .puts, .combined DataFrames ready for
        Black-Scholes IV inversion and SABR surface calibration.

    Raises:
        ValueError   : API key missing.
        RuntimeError : No data returned or chain build failed.
    """
    settings = get_settings()
    key = api_key or settings.databento_api_key

    if not key:
        raise ValueError(
            "Databento API key required. Set DATABENTO_API_KEY in .env "
            "or pass api_key= explicitly."
        )

    trading_day = _previous_trading_day(reference_date)
    ticker      = ticker.upper().strip()

    logger.info(
        "get_options_chain | ticker=%s trading_day=%s",
        ticker, trading_day,
    )

    client      = db.Historical(key=key)
    definitions = fetch_options_definitions(client, ticker, trading_day)
    nbbo        = fetch_nbbo_snapshot(client, ticker, trading_day)
    snapshot    = build_options_chain(definitions, nbbo, trading_day, ticker)

    if snapshot is None:
        raise RuntimeError(
            f"Failed to build options chain for {ticker} on {trading_day}. "
            "Check logs above for root cause."
        )

    return snapshot
