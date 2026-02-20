"""
tests/test_market_data.py
Unit tests for data/market_data.py

All tests run without a real Databento API key.
Live API calls are mocked using unittest.mock — no cost incurred.
"""

import os
from datetime import date, datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
import databento as db

from data.market_data import (
    OPRA_DATASET,
    UNDEF_PRICE_FLOAT,
    OptionsChainSnapshot,
    _close_window,
    _is_undefined_price,
    _previous_trading_day,
    build_options_chain,
    fetch_nbbo_snapshot,
    fetch_options_definitions,
    get_options_chain,
)

ET = ZoneInfo("America/New_York")


# ── _previous_trading_day ─────────────────────────────────────────────────────

class TestPreviousTradingDay:

    def test_monday_returns_friday(self):
        assert _previous_trading_day(date(2025, 1, 20)) == date(2025, 1, 17)

    def test_saturday_returns_friday(self):
        assert _previous_trading_day(date(2025, 1, 18)) == date(2025, 1, 17)

    def test_sunday_returns_friday(self):
        assert _previous_trading_day(date(2025, 1, 19)) == date(2025, 1, 17)

    def test_tuesday_returns_monday(self):
        assert _previous_trading_day(date(2025, 1, 21)) == date(2025, 1, 20)

    def test_returns_date_type(self):
        result = _previous_trading_day(date(2025, 6, 15))
        assert isinstance(result, date)

    def test_defaults_to_today_minus_one(self):
        """When no arg given, should return a date (not crash)."""
        result = _previous_trading_day()
        assert isinstance(result, date)


# ── _close_window ─────────────────────────────────────────────────────────────

class TestCloseWindow:

    def test_returns_two_datetimes(self):
        start, end = _close_window(date(2025, 1, 15))
        assert isinstance(start, datetime) and isinstance(end, datetime)

    def test_window_is_one_minute(self):
        start, end = _close_window(date(2025, 6, 15))
        delta = (end - start).total_seconds()
        assert delta == 60.0

    def test_end_is_4pm_et_winter(self):
        """EST (UTC-5): 4:00 PM ET = 21:00 UTC."""
        _, end = _close_window(date(2025, 1, 15))
        end_utc = end.astimezone(ZoneInfo("UTC"))
        assert end_utc.hour == 21 and end_utc.minute == 0

    def test_end_is_4pm_et_summer(self):
        """EDT (UTC-4): 4:00 PM ET = 20:00 UTC."""
        _, end = _close_window(date(2025, 6, 15))
        end_utc = end.astimezone(ZoneInfo("UTC"))
        assert end_utc.hour == 20 and end_utc.minute == 0

    def test_start_is_timezone_aware(self):
        start, _ = _close_window(date(2025, 1, 15))
        assert start.tzinfo is not None


# ── _is_undefined_price ───────────────────────────────────────────────────────

class TestIsUndefinedPrice:

    def test_nan_is_undefined(self):
        s = pd.Series([float("nan")])
        assert _is_undefined_price(s).iloc[0]

    def test_undef_sentinel_is_undefined(self):
        s = pd.Series([UNDEF_PRICE_FLOAT])
        assert _is_undefined_price(s).iloc[0]

    def test_zero_is_undefined(self):
        s = pd.Series([0.0])
        assert _is_undefined_price(s).iloc[0]

    def test_negative_is_undefined(self):
        s = pd.Series([-1.0])
        assert _is_undefined_price(s).iloc[0]

    def test_real_price_is_defined(self):
        s = pd.Series([1.50, 10.25, 0.05])
        assert not _is_undefined_price(s).any()


# ── fetch_options_definitions (mocked) ───────────────────────────────────────

class TestFetchOptionsDefinitions:

    def _mock_definitions_store(self):
        """Build a minimal mock DBNStore that returns a definitions DataFrame."""
        df = pd.DataFrame({
            "instrument_id":    [1001, 1002, 1003, 1004, 1005],
            "raw_symbol":       ["AAPL  250117C00150000", "AAPL  250117P00150000",
                                 "AAPL  250117C00160000", "AAPL  250117P00160000",
                                 "AAPL  250117F00000000"],  # future — should be dropped
            "instrument_class": ["C", "P", "C", "P", "F"],
            "strike_price":     [150.0, 150.0, 160.0, 160.0, 0.0],
            "expiration":       pd.to_datetime(["2025-01-17"] * 5),
        })
        store = MagicMock()
        store.to_df.return_value = df
        return store

    def test_returns_dataframe(self):
        client = MagicMock()
        client.timeseries.get_range.return_value = self._mock_definitions_store()
        result = fetch_options_definitions(client, "AAPL", date(2025, 1, 15))
        assert isinstance(result, pd.DataFrame)

    def test_filters_out_non_option_classes(self):
        client = MagicMock()
        client.timeseries.get_range.return_value = self._mock_definitions_store()
        result = fetch_options_definitions(client, "AAPL", date(2025, 1, 15))
        # 'F' (future) should be dropped — only 4 C/P rows remain
        assert len(result) == 4
        assert set(result["option_type"].unique()) == {"call", "put"}

    def test_maps_instrument_class_to_option_type(self):
        client = MagicMock()
        client.timeseries.get_range.return_value = self._mock_definitions_store()
        result = fetch_options_definitions(client, "AAPL", date(2025, 1, 15))
        assert "option_type" in result.columns
        assert not result["option_type"].isin(["C", "P"]).any()  # raw char gone
        assert result["option_type"].isin(["call", "put"]).all()

    def test_output_columns(self):
        client = MagicMock()
        client.timeseries.get_range.return_value = self._mock_definitions_store()
        result = fetch_options_definitions(client, "AAPL", date(2025, 1, 15))
        assert set(result.columns) >= {"instrument_id", "raw_symbol", "option_type", "strike", "expiration"}

    def test_empty_store_returns_empty_df(self):
        client = MagicMock()
        store  = MagicMock()
        store.to_df.return_value = pd.DataFrame()
        client.timeseries.get_range.return_value = store
        result = fetch_options_definitions(client, "AAPL", date(2025, 1, 15))
        assert result.empty

    def test_uses_correct_dataset_and_schema(self):
        client = MagicMock()
        client.timeseries.get_range.return_value = self._mock_definitions_store()
        fetch_options_definitions(client, "AAPL", date(2025, 1, 15))
        call_kwargs = client.timeseries.get_range.call_args.kwargs
        assert call_kwargs["dataset"] == OPRA_DATASET
        assert call_kwargs["schema"] == db.Schema.DEFINITION
        assert call_kwargs["stype_in"] == db.SType.PARENT
        assert call_kwargs["symbols"] == ["AAPL"]


# ── fetch_nbbo_snapshot (mocked) ─────────────────────────────────────────────

class TestFetchNbboSnapshot:

    def _mock_nbbo_store(self, include_stale: bool = False):
        rows = {
            "instrument_id": [1001, 1002, 1003, 1004],
            "ts_recv":       pd.to_datetime(["2025-01-15 20:59:30+00:00"] * 4),
            "bid_px_00":     [1.50, 2.00, 0.50, UNDEF_PRICE_FLOAT if include_stale else 3.00],
            "ask_px_00":     [1.55, 2.10, 0.55, UNDEF_PRICE_FLOAT if include_stale else 3.10],
            "bid_sz_00":     [10, 5, 20, 15],
            "ask_sz_00":     [10, 5, 20, 15],
        }
        df = pd.DataFrame(rows)
        store = MagicMock()
        store.to_df.return_value = df
        return store

    def test_returns_dataframe(self):
        client = MagicMock()
        client.timeseries.get_range.return_value = self._mock_nbbo_store()
        result = fetch_nbbo_snapshot(client, "AAPL", date(2025, 1, 15))
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self):
        client = MagicMock()
        client.timeseries.get_range.return_value = self._mock_nbbo_store()
        result = fetch_nbbo_snapshot(client, "AAPL", date(2025, 1, 15))
        assert set(result.columns) >= {"instrument_id", "bid", "ask", "mid", "spread"}

    def test_mid_computed_correctly(self):
        client = MagicMock()
        client.timeseries.get_range.return_value = self._mock_nbbo_store()
        result = fetch_nbbo_snapshot(client, "AAPL", date(2025, 1, 15))
        expected_mid = (result["bid"] + result["ask"]) / 2
        pd.testing.assert_series_equal(result["mid"], expected_mid, check_names=False)

    def test_spread_computed_correctly(self):
        client = MagicMock()
        client.timeseries.get_range.return_value = self._mock_nbbo_store()
        result = fetch_nbbo_snapshot(client, "AAPL", date(2025, 1, 15))
        assert (result["spread"] >= 0).all()

    def test_drops_undefined_prices(self):
        client = MagicMock()
        client.timeseries.get_range.return_value = self._mock_nbbo_store(include_stale=True)
        result = fetch_nbbo_snapshot(client, "AAPL", date(2025, 1, 15))
        # Row with UNDEF_PRICE should be dropped
        assert not (result["bid"] >= UNDEF_PRICE_FLOAT).any()

    def test_uses_correct_schema(self):
        client = MagicMock()
        client.timeseries.get_range.return_value = self._mock_nbbo_store()
        fetch_nbbo_snapshot(client, "AAPL", date(2025, 1, 15))
        call_kwargs = client.timeseries.get_range.call_args.kwargs
        assert call_kwargs["schema"] == db.Schema.MBP_1
        assert call_kwargs["stype_in"] == db.SType.PARENT

    def test_empty_store_returns_empty_df(self):
        client = MagicMock()
        store  = MagicMock()
        store.to_df.return_value = pd.DataFrame()
        client.timeseries.get_range.return_value = store
        result = fetch_nbbo_snapshot(client, "AAPL", date(2025, 1, 15))
        assert result.empty


# ── build_options_chain ───────────────────────────────────────────────────────

class TestBuildOptionsChain:

    @pytest.fixture
    def sample_definitions(self):
        return pd.DataFrame({
            "instrument_id": [1001, 1002, 1003, 1004],
            "raw_symbol":    ["AAPL  250117C00150000", "AAPL  250117P00150000",
                              "AAPL  250117C00160000", "AAPL  250117P00160000"],
            "option_type":   ["call", "put", "call", "put"],
            "strike":        [150.0, 150.0, 160.0, 160.0],
            "expiration":    pd.to_datetime(["2025-01-17"] * 4),
        })

    @pytest.fixture
    def sample_nbbo(self):
        return pd.DataFrame({
            "instrument_id": [1001, 1002, 1003, 1004],
            "ts_recv":       pd.to_datetime(["2025-01-15 20:59:30+00:00"] * 4),
            "bid":    [1.50, 2.00, 0.50, 3.00],
            "ask":    [1.55, 2.10, 0.55, 3.10],
            "bid_size": [10, 5, 20, 15],
            "ask_size": [10, 5, 20, 15],
            "mid":    [1.525, 2.05, 0.525, 3.05],
            "spread": [0.05, 0.10, 0.05, 0.10],
        })

    def test_returns_snapshot(self, sample_definitions, sample_nbbo):
        result = build_options_chain(sample_definitions, sample_nbbo, date(2025, 1, 15), "AAPL")
        assert isinstance(result, OptionsChainSnapshot)

    def test_calls_and_puts_split(self, sample_definitions, sample_nbbo):
        snap = build_options_chain(sample_definitions, sample_nbbo, date(2025, 1, 15), "AAPL")
        assert len(snap.calls) == 2
        assert len(snap.puts) == 2

    def test_combined_has_all_rows(self, sample_definitions, sample_nbbo):
        snap = build_options_chain(sample_definitions, sample_nbbo, date(2025, 1, 15), "AAPL")
        assert len(snap.combined) == 4

    def test_days_to_expiry_positive(self, sample_definitions, sample_nbbo):
        snap = build_options_chain(sample_definitions, sample_nbbo, date(2025, 1, 15), "AAPL")
        assert (snap.combined["days_to_expiry"] > 0).all()

    def test_time_to_expiry_is_days_over_365(self, sample_definitions, sample_nbbo):
        snap = build_options_chain(sample_definitions, sample_nbbo, date(2025, 1, 15), "AAPL")
        expected = snap.combined["days_to_expiry"] / 365.0
        pd.testing.assert_series_equal(snap.combined["time_to_expiry"], expected, check_names=False)

    def test_filter_expiration(self, sample_definitions, sample_nbbo):
        snap = build_options_chain(sample_definitions, sample_nbbo, date(2025, 1, 15), "AAPL")
        filtered = snap.filter_expiration(date(2025, 1, 17))
        assert len(filtered.combined) == 4  # all same expiration

    def test_unique_expirations_property(self, sample_definitions, sample_nbbo):
        snap = build_options_chain(sample_definitions, sample_nbbo, date(2025, 1, 15), "AAPL")
        assert snap.unique_expirations == [date(2025, 1, 17)]

    def test_unique_strikes_property(self, sample_definitions, sample_nbbo):
        snap = build_options_chain(sample_definitions, sample_nbbo, date(2025, 1, 15), "AAPL")
        assert snap.unique_strikes == [150.0, 160.0]

    def test_returns_none_on_empty_definitions(self, sample_nbbo):
        result = build_options_chain(pd.DataFrame(), sample_nbbo, date(2025, 1, 15), "AAPL")
        assert result is None

    def test_returns_none_on_empty_nbbo(self, sample_definitions):
        result = build_options_chain(sample_definitions, pd.DataFrame(), date(2025, 1, 15), "AAPL")
        assert result is None


# ── get_options_chain (integration mock) ─────────────────────────────────────

class TestGetOptionsChain:

    def test_raises_without_api_key(self):
        with patch.dict(os.environ, {"DATABENTO_API_KEY": ""}, clear=False):
            # Reset lru_cache so env change is picked up
            from config.settings import get_settings
            get_settings.cache_clear()
            with pytest.raises(ValueError, match="Databento API key required"):
                get_options_chain("AAPL", api_key=None)
        get_settings.cache_clear()

    def test_ticker_is_uppercased(self):
        """Ticker normalisation — lowercase input should be uppercased before API call."""
        with patch("data.market_data.fetch_options_definitions") as mock_def, \
             patch("data.market_data.fetch_nbbo_snapshot") as mock_nbbo, \
             patch("data.market_data.build_options_chain") as mock_build, \
             patch("data.market_data.db.Historical"):
            mock_def.return_value  = pd.DataFrame({"x": [1]})
            mock_nbbo.return_value = pd.DataFrame({"x": [1]})
            mock_build.return_value = MagicMock(spec=OptionsChainSnapshot)
            get_options_chain("aapl", api_key="fake-key")
            # ticker passed to fetch_options_definitions should be uppercase
            call_ticker = mock_def.call_args.args[1]
            assert call_ticker == "AAPL"
