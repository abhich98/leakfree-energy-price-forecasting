from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from ingestion.smard_client import (
    ENERGY_SOURCE,
    FORECAST_SIGNAL,
    MAX_TIME_OUT,
    REGION,
    RESOLUTION,
    SMARD_SIGNALS,
    Units,
    _get_index,
    _get_series,
    fetch_range,
    resolution_for_date,
    SMARD_QUARTER_HOUR_SWITCH_DATE,
)


@pytest.fixture(autouse=True)
def clear_smard_index_cache():
    _get_index.cache_clear()


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    if status_code >= 400:
        mock.raise_for_status.side_effect = requests.HTTPError(response=mock)
    else:
        mock.raise_for_status.return_value = None
    return mock


# ── _get_index ────────────────────────────────────────────────────────────────

class TestGetIndex:
    def test_returns_timestamps(self):
        expected = [1704067200000, 1704672000000]
        with patch("ingestion.smard_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response({"timestamps": expected})
            result = _get_index(ENERGY_SOURCE.WIND_ONSHORE, REGION.DE, RESOLUTION.HOUR)
        assert result == expected

    def test_missing_key_returns_empty_list(self):
        with patch("ingestion.smard_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response({})
            result = _get_index(ENERGY_SOURCE.WIND_ONSHORE, REGION.DE, RESOLUTION.HOUR)
        assert result == []

    def test_raises_on_http_error(self):
        with patch("ingestion.smard_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response({}, status_code=404)
            with pytest.raises(requests.HTTPError):
                _get_index(ENERGY_SOURCE.WIND_ONSHORE, REGION.DE, RESOLUTION.HOUR)

    def test_url_constructed_correctly(self):
        with patch("ingestion.smard_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response({"timestamps": []})
            _get_index(ENERGY_SOURCE.WIND_ONSHORE, REGION.DE, RESOLUTION.HOUR)
        mock_get.assert_called_once_with(
            f"https://www.smard.de/app/chart_data/{ENERGY_SOURCE.WIND_ONSHORE.value}/{REGION.DE.value}/index_{RESOLUTION.HOUR.value}.json",
            timeout=MAX_TIME_OUT
        )

    def test_quarter_hour_resolution_in_url(self):
        with patch("ingestion.smard_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response({"timestamps": []})
            _get_index(ENERGY_SOURCE.WIND_ONSHORE, REGION.DE_LU, RESOLUTION.QUARTER_HOUR)
        mock_get.assert_called_once_with(
            f"https://www.smard.de/app/chart_data/{ENERGY_SOURCE.WIND_ONSHORE.value}/{REGION.DE_LU.value}/index_{RESOLUTION.QUARTER_HOUR.value}.json",
            timeout=MAX_TIME_OUT
        )


# ── _get_series ───────────────────────────────────────────────────────────────

class TestGetSeries:
    CHUNK_TS = 1704067200000

    def test_returns_series_data(self):
        expected = [[1704067200000, 100.0], [1704070800000, 200.0]]
        with patch("ingestion.smard_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response({"series": expected})
            result = _get_series(ENERGY_SOURCE.WIND_ONSHORE, REGION.DE, RESOLUTION.HOUR, self.CHUNK_TS)
        assert result == expected

    def test_missing_key_returns_empty_list(self):
        with patch("ingestion.smard_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response({})
            result = _get_series(ENERGY_SOURCE.WIND_ONSHORE, REGION.DE, RESOLUTION.HOUR, self.CHUNK_TS)
        assert result == []

    def test_raises_on_http_error(self):
        with patch("ingestion.smard_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response({}, status_code=500)
            with pytest.raises(requests.HTTPError):
                _get_series(ENERGY_SOURCE.WIND_ONSHORE, REGION.DE, RESOLUTION.HOUR, self.CHUNK_TS)

    def test_url_constructed_correctly(self):
        with patch("ingestion.smard_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response({"series": []})
            _get_series(ENERGY_SOURCE.WIND_ONSHORE, REGION.DE, RESOLUTION.HOUR, self.CHUNK_TS)
        mock_get.assert_called_once_with(
            f"https://www.smard.de/app/chart_data/{ENERGY_SOURCE.WIND_ONSHORE.value}/{REGION.DE.value}/{ENERGY_SOURCE.WIND_ONSHORE.value}_{REGION.DE.value}_{RESOLUTION.HOUR.value}_{self.CHUNK_TS}.json",
            timeout=MAX_TIME_OUT
        )


# ── fetch_range ───────────────────────────────────────────────────────────────

START = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
END = datetime(2024, 1, 21, 23, 0, tzinfo=timezone.utc)

# Chunk that starts 1 hour before START — simulates a weekly chunk that overlaps the range
_CHUNK_TS = int(START.timestamp() * 1000) - 3_600_000

_SERIES_DATA = [
    [int(START.timestamp() * 1000), 500.0],
    [int(START.timestamp() * 1000) + 3_600_000, 600.0],
    [int(END.timestamp() * 1000), 700.0],
]


@pytest.fixture
def mock_index_and_series():
    with patch("ingestion.smard_client._get_index") as m_idx, \
         patch("ingestion.smard_client._get_series") as m_ser, \
         patch("ingestion.smard_client._get_series_payload_with_retry") as m_payload:
        m_idx.return_value = [_CHUNK_TS]
        m_ser.return_value = _SERIES_DATA
        m_payload.return_value = {
            "series": _SERIES_DATA,
            "meta_data": {"created": _CHUNK_TS},
        }
        yield m_idx, m_ser, m_payload


class TestFetchRange:
    def test_returns_expected_columns(self, mock_index_and_series):
        df = fetch_range(START, END)
        expected_columns = {"timestamp", "value", "signal", "unit", "resolution", "issue_timestamp", "fetched_at"}
        assert expected_columns.issubset(set(df.columns))

    def test_all_smard_signals_are_fetched(self, mock_index_and_series):
        m_idx, _, _ = mock_index_and_series
        fetch_range(START, END)
        assert m_idx.call_count == len(SMARD_SIGNALS)

    def test_result_contains_all_signal_names(self, mock_index_and_series):
        df = fetch_range(START, END)
        expected_names = {s.name for s in SMARD_SIGNALS}
        assert set(df["signal"].unique()) == expected_names

    def test_energy_source_signals_use_mw_unit(self, mock_index_and_series):
        df = fetch_range(START, END)
        mw_signals = {s.name for s, cfg in SMARD_SIGNALS.items() if cfg["unit"] == Units.MW}
        assert (df[df["signal"].isin(mw_signals)]["unit"] == "MW").all()

    def test_neighboring_region_signals_use_eur_mwh_unit(self, mock_index_and_series):
        df = fetch_range(START, END)
        eur_signals = {s.name for s, cfg in SMARD_SIGNALS.items() if cfg["unit"] == Units.EUR_MWH}
        assert (df[df["signal"].isin(eur_signals)]["unit"] == "EUR_MWH").all()

    def test_result_is_sorted_by_timestamp(self, mock_index_and_series):
        df = fetch_range(START, END)
        assert df["timestamp"].is_monotonic_increasing

    def test_timestamps_are_utc_aware(self, mock_index_and_series):
        df = fetch_range(START, END)
        assert str(df["timestamp"].dt.tz) == "UTC"

    def test_data_filtered_to_date_range(self):
        narrow_start = datetime(2024, 1, 16, 0, 0, tzinfo=timezone.utc)
        narrow_end = datetime(2024, 1, 17, 0, 0, tzinfo=timezone.utc)

        start_ms = int(narrow_start.timestamp() * 1000)
        end_ms = int(narrow_end.timestamp() * 1000)
        before_start_ms = start_ms - 3_600_000

        series = [
            [before_start_ms, 100.0],  # outside — excluded
            [start_ms, 200.0],         # at start — included
            [end_ms, 300.0],           # at end — included
        ]
        with patch("ingestion.smard_client._get_index") as m_idx, \
             patch("ingestion.smard_client._get_series") as m_ser, \
             patch("ingestion.smard_client._get_series_payload_with_retry") as m_payload:
            m_idx.return_value = [before_start_ms]
            m_ser.return_value = series
            m_payload.return_value = {"series": series, "meta_data": {"created": before_start_ms}}
            df = fetch_range(narrow_start, narrow_end)

        assert all(df["timestamp"] >= narrow_start)
        assert all(df["timestamp"] <= narrow_end)
        assert len(df) == 2 * len(SMARD_SIGNALS)

    def test_none_values_coerced_to_nan(self, mock_index_and_series):
        _, m_ser, m_payload = mock_index_and_series
        m_ser.return_value = [[int(START.timestamp() * 1000), None]]
        m_payload.return_value = {
            "series": [[int(START.timestamp() * 1000), None]],
            "meta_data": {"created": _CHUNK_TS},
        }
        df = fetch_range(START, END)
        assert df["value"].isna().sum() == len(SMARD_SIGNALS)

    def test_failed_signal_is_skipped_others_included(self):
        call_count = 0

        def failing_first(*_):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("simulated failure")
            return [_CHUNK_TS]

        with patch("ingestion.smard_client._get_index") as m_idx, \
             patch("ingestion.smard_client._get_series") as m_ser, \
             patch("ingestion.smard_client._get_series_payload_with_retry") as m_payload:
            m_idx.side_effect = failing_first
            m_ser.return_value = _SERIES_DATA
            m_payload.return_value = {"series": _SERIES_DATA, "meta_data": {"created": _CHUNK_TS}}
            df = fetch_range(START, END)

        assert len(df["signal"].unique()) == len(SMARD_SIGNALS) - 1

    def test_returns_empty_dataframe_when_all_signals_fail(self):
        with patch("ingestion.smard_client._get_index") as m_idx:
            m_idx.side_effect = ValueError("simulated failure")
            df = fetch_range(START, END)
        assert list(df.columns) == ["timestamp", "value", "signal", "unit", "resolution"]
        assert len(df) == 0

    def test_returns_empty_dataframe_when_all_series_empty(self):
        with patch("ingestion.smard_client._get_index") as m_idx, \
             patch("ingestion.smard_client._get_series") as m_ser, \
             patch("ingestion.smard_client._get_series_payload_with_retry") as m_payload:
            m_idx.return_value = [_CHUNK_TS]
            m_ser.return_value = []
            m_payload.return_value = {"series": [], "meta_data": {"created": _CHUNK_TS}}
            df = fetch_range(START, END)
        assert list(df.columns) == ["timestamp", "value", "signal", "unit", "resolution"]
        assert len(df) == 0

    def test_multiple_chunks_per_signal_are_concatenated(self):
        chunk1 = _CHUNK_TS
        chunk2 = _CHUNK_TS + 7 * 24 * 3_600_000

        series_chunk1 = [[int(START.timestamp() * 1000), 100.0]]
        series_chunk2 = [[int(START.timestamp() * 1000) + 7 * 24 * 3_600_000, 200.0]]

        wide_end = datetime(2024, 1, 29, 23, 0, tzinfo=timezone.utc)
        with patch("ingestion.smard_client._get_index") as m_idx, \
             patch("ingestion.smard_client._get_series") as m_ser, \
             patch("ingestion.smard_client._get_series_payload_with_retry") as m_payload:
            m_idx.return_value = [chunk1, chunk2]
            m_ser.side_effect = [series_chunk1, series_chunk2] * len(SMARD_SIGNALS)
            m_payload.side_effect = [
                {"series": series_chunk1, "meta_data": {"created": chunk1}},
                {"series": series_chunk2, "meta_data": {"created": chunk2}},
            ] * len(SMARD_SIGNALS)
            df = fetch_range(START, wide_end)

        assert len(df) == 2 * len(SMARD_SIGNALS)

    def test_forecast_issue_timestamp_rule(self):
        start = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 15, 23, 0, tzinfo=timezone.utc)
        start_ms = int(start.timestamp() * 1000)

        with patch("ingestion.smard_client._get_index") as m_idx, \
             patch("ingestion.smard_client._get_series_payload_with_retry") as m_payload:
            m_idx.return_value = [start_ms]
            m_payload.return_value = {
                "series": [[start_ms, 100.0]],
                "meta_data": {"created": start_ms},
            }

            df_load = fetch_range(start, end)

        total_load_row = df_load[df_load["signal"] == FORECAST_SIGNAL.TOTAL_CONSUMPTION_FC.name].iloc[0]
        total_generation_row = df_load[df_load["signal"] == FORECAST_SIGNAL.TOTAL_GENERATION_FC.name].iloc[0]

        assert total_load_row["issue_timestamp"] == pd.Timestamp("2024-01-14T10:00:00Z")
        assert total_generation_row["issue_timestamp"] == pd.Timestamp("2024-01-14T18:00:00Z")


# ── resolution_for_date ───────────────────────────────────────────────────────

class TestResolutionForDate:
    def test_returns_hour_before_switch_date(self):
        date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert resolution_for_date(date) == RESOLUTION.HOUR

    def test_returns_quarter_hour_on_switch_date(self):
        date = SMARD_QUARTER_HOUR_SWITCH_DATE
        assert resolution_for_date(date) == RESOLUTION.QUARTER_HOUR

    def test_returns_quarter_hour_after_switch_date(self):
        date = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert resolution_for_date(date) == RESOLUTION.QUARTER_HOUR

    def test_handles_naive_datetime(self):
        date = datetime(2026, 1, 1)  # no tzinfo
        assert resolution_for_date(date) == RESOLUTION.QUARTER_HOUR

    def test_fetch_range_uses_hour_resolution_for_old_dates(self, mock_index_and_series):
        old_start = datetime(2024, 1, 15, tzinfo=timezone.utc)
        old_end = datetime(2024, 1, 21, tzinfo=timezone.utc)
        df = fetch_range(old_start, old_end)
        assert (df["resolution"] == "hour").all()

    def test_fetch_range_uses_quarter_hour_resolution_for_new_dates(self, mock_index_and_series):
        new_start = datetime(2026, 1, 15, tzinfo=timezone.utc)
        new_end = datetime(2026, 1, 21, tzinfo=timezone.utc)
        df = fetch_range(new_start, new_end)
        assert (df["resolution"] == "quarter-hour").all()
