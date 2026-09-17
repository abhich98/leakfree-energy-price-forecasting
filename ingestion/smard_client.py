"""
SMARD client for fetching time series data from the SMARD API.

This module provides utilities to retrieve power generation, consumption,
and price signals for German and neighboring regions using the SMARD
chart_data endpoints. It supports multiple resolutions and returns data
as pandas DataFrames.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import lru_cache
from typing import Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


BASE_URL = os.environ.get("ZEPHYRWERK_SMARD_BASE_URL")
MAX_TIME_OUT = 30  # s


class RESOLUTION(Enum):
    HOUR = "hour"
    QUARTER_HOUR = "quarterhour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class ENERGY_SOURCE(Enum):
    WIND_ONSHORE = 4067
    WIND_OFFSHORE = 1225
    SOLAR = 4068
    BIOMASS = 4066
    HYDROPOWER = 1226
    PUMPED_STORAGE = 4070
    NATURAL_GAS = 4071
    HARD_COAL = 4069
    BROWN_COAL = 1223
    NUCLEAR = 1224
    OTHER_CONVENTIONAL = 1227
    OTHER_RENEWABLE = 1228


class NEIGHBORING_REGION(Enum):
    DE_LU = 4169
    AUSTRIA = 4170
    FRANCE = 254
    NETHERLANDS = 256
    POLAND = 258  # was 257 — filter 257 has gap 2017-2019, 258 is continuous
    SWITZERLAND = 259
    CZECHIA = 261
    DENMARK_1 = 252
    DENMARK_2 = 253


class CONSUMPTION_TYPE(Enum):
    TOTAL_CONSUMPTION = 410
    RESIDUAL_LOAD = 4359


class FORECAST_SIGNAL(Enum):
    """SMARD forecasted signals — published one day before

    The forecasted consumption (grid load) signal is published on D-1, two hours before the auction (10:00 CET).
    The forecasted residual load and forecasted generation signals are made available before 18:00 CET (on D-1).
    For further details, refer to documentation on the SMARD website.
    """

    TOTAL_CONSUMPTION_FC = 411

    RESIDUAL_LOAD_FC = 4362

    TOTAL_GENERATION_FC = 122
    WIND_ONSHORE_FC = 123
    SOLAR_FC = 125
    WIND_OFFSHORE_FC = 3791
    WIND_PV_FC = 5097


class REGION(Enum):
    DE = "DE"
    DE_LU = "DE-LU"


class Units(Enum):
    MW = "MW"
    EUR_MWH = "EUR_MWH"


# SMARD moved from hourly to quarter-hourly resolution for most signals around
# this date. Dates on/after the switch fetch QUARTER_HOUR; before fetch HOUR.
# The exact switch date varies per signal — this is the earliest known switch
# (prices moved first). Verify per-signal via the SMARD index endpoint.
SMARD_QUARTER_HOUR_SWITCH_DATE = pd.to_datetime("2025-09-30 00:00:00+02:00").tz_convert(
    timezone.utc
)


def resolution_for_date(date: datetime) -> RESOLUTION:
    """Return the SMARD resolution to fetch for the given date.

    SMARD moved from hourly to quarter-hourly resolution starting late 2025.
    Dates on/after the switch date use QUARTER_HOUR; earlier dates use HOUR.
    This avoids clobbering existing hourly rows when 15-min data becomes
    available, and lets the backfill loop pick the right resolution per day.

    param: date: The date to check (timezone-aware preferred).
    return: RESOLUTION.QUARTER_HOUR if date >= switch date, else RESOLUTION.HOUR.
    """
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    if date >= SMARD_QUARTER_HOUR_SWITCH_DATE:
        return RESOLUTION.QUARTER_HOUR
    return RESOLUTION.HOUR


SMARD_SIGNALS = {
    # Generation (region: DE, unit: MW)
    ENERGY_SOURCE.WIND_ONSHORE: {"region": REGION.DE, "unit": Units.MW},
    ENERGY_SOURCE.WIND_OFFSHORE: {"region": REGION.DE, "unit": Units.MW},
    ENERGY_SOURCE.SOLAR: {"region": REGION.DE, "unit": Units.MW},
    ENERGY_SOURCE.BIOMASS: {"region": REGION.DE, "unit": Units.MW},
    ENERGY_SOURCE.HYDROPOWER: {"region": REGION.DE, "unit": Units.MW},
    ENERGY_SOURCE.PUMPED_STORAGE: {"region": REGION.DE, "unit": Units.MW},
    ENERGY_SOURCE.NATURAL_GAS: {"region": REGION.DE, "unit": Units.MW},
    ENERGY_SOURCE.HARD_COAL: {"region": REGION.DE, "unit": Units.MW},
    ENERGY_SOURCE.BROWN_COAL: {"region": REGION.DE, "unit": Units.MW},
    ENERGY_SOURCE.NUCLEAR: {"region": REGION.DE, "unit": Units.MW},
    ENERGY_SOURCE.OTHER_CONVENTIONAL: {"region": REGION.DE, "unit": Units.MW},
    ENERGY_SOURCE.OTHER_RENEWABLE: {"region": REGION.DE, "unit": Units.MW},
    # Consumption (region: DE, unit: MW)
    CONSUMPTION_TYPE.TOTAL_CONSUMPTION: {"region": REGION.DE, "unit": Units.MW},
    CONSUMPTION_TYPE.RESIDUAL_LOAD: {"region": REGION.DE, "unit": Units.MW},
    # DE and Neighbour prices (region: DE-LU, unit: EUR_MWH)
    NEIGHBORING_REGION.DE_LU: {"region": REGION.DE_LU, "unit": Units.EUR_MWH},
    NEIGHBORING_REGION.AUSTRIA: {"region": REGION.DE_LU, "unit": Units.EUR_MWH},
    NEIGHBORING_REGION.FRANCE: {"region": REGION.DE_LU, "unit": Units.EUR_MWH},
    NEIGHBORING_REGION.NETHERLANDS: {"region": REGION.DE_LU, "unit": Units.EUR_MWH},
    NEIGHBORING_REGION.POLAND: {"region": REGION.DE_LU, "unit": Units.EUR_MWH},
    NEIGHBORING_REGION.SWITZERLAND: {"region": REGION.DE_LU, "unit": Units.EUR_MWH},
    NEIGHBORING_REGION.CZECHIA: {"region": REGION.DE_LU, "unit": Units.EUR_MWH},
    NEIGHBORING_REGION.DENMARK_1: {"region": REGION.DE_LU, "unit": Units.EUR_MWH},
    NEIGHBORING_REGION.DENMARK_2: {"region": REGION.DE_LU, "unit": Units.EUR_MWH},
    # Consumption (region: DE, unit: MW) — forecasted, published before auction
    FORECAST_SIGNAL.TOTAL_CONSUMPTION_FC: {"region": REGION.DE, "unit": Units.MW},
    FORECAST_SIGNAL.RESIDUAL_LOAD_FC: {"region": REGION.DE, "unit": Units.MW},
    # Forecasted generation (region: DE, unit: MW) — published before auction
    FORECAST_SIGNAL.TOTAL_GENERATION_FC: {"region": REGION.DE, "unit": Units.MW},
    FORECAST_SIGNAL.WIND_ONSHORE_FC: {"region": REGION.DE, "unit": Units.MW},
    FORECAST_SIGNAL.SOLAR_FC: {"region": REGION.DE, "unit": Units.MW},
    FORECAST_SIGNAL.WIND_OFFSHORE_FC: {"region": REGION.DE, "unit": Units.MW},
    FORECAST_SIGNAL.WIND_PV_FC: {"region": REGION.DE, "unit": Units.MW},
}


@lru_cache(maxsize=None)
def _get_index(
    filter_id: Union[ENERGY_SOURCE, CONSUMPTION_TYPE, NEIGHBORING_REGION],
    region: REGION,
    resolution: RESOLUTION,
) -> list:
    """Fetches the index of available timestamps for a given filter_id, region, and resolution.
        Cached with lru_cache — the index rarely changes, so repeated calls within a single
        pipeline run (e.g. backfill) hit the cache instead of re-fetching from SMARD.

    param: filter_id: The SMARD filter ID corresponding to the signal we want to fetch
                    (e.g. 4067 for onshore wind generation).
    param: region: The region for which to fetch the data (e.g. REGION.DE).
    param: resolution: The desired data resolution (e.g. Resolution.HOUR).
    return: a list of timestamps (in milliseconds) that mark the start of each weekly chunk of data available
    for the specified filter_id and region.
    """
    url = f"{BASE_URL}/{filter_id.value}/{region.value}/index_{resolution.value}.json"
    response = requests.get(url, timeout=MAX_TIME_OUT)
    response.raise_for_status()
    return response.json().get("timestamps", [])


def _is_retryable_http_error(exc: BaseException) -> bool:
    if not isinstance(exc, requests.HTTPError):
        return False
    status_code = exc.response.status_code
    return status_code == 429 or status_code >= 500


@retry(
    retry=retry_if_exception(_is_retryable_http_error),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _get_series_payload(
    filter_id: Union[ENERGY_SOURCE, CONSUMPTION_TYPE, NEIGHBORING_REGION],
    region: REGION,
    resolution: RESOLUTION,
    timestamp: int,
) -> dict:
    """Fetches the payload for a given filter_id, region, resolution, and timestamp.

    param: filter_id: The SMARD filter ID corresponding to the signal we want to fetch
                    (e.g. 4067 for onshore wind generation).
    param: region: The region for which to fetch the data (e.g. REGION.DE).
    param: resolution: The desired data resolution (e.g. Resolution.HOUR).
    param: timestamp: The timestamp (in milliseconds) that marks the start of the weekly chunk of data to fetch.
                      This timestamp should be one of the values returned by the _get_index function
                      for the specified filter_id, region, and resolution.
    return: the full JSON payload for the specified filter_id, region, resolution, and timestamp.
    """

    file_name = f"{filter_id.value}_{region.value}_{resolution.value}_{timestamp}.json"
    url = f"{BASE_URL}/{filter_id.value}/{region.value}/{file_name}"
    logger.debug("Fetching data payload from URL: %s", url)
    response = requests.get(url, timeout=MAX_TIME_OUT)
    response.raise_for_status()
    return response.json()


def _build_issue_timestamps(signal_name: Enum, timestamps: pd.Series) -> pd.Series:
    """Build deterministic issue timestamps for SMARD forecast signals.

    For target day D:
    - TOTAL_CONSUMPTION_FC: issue_timestamp = 10:00 CET or 8:00 UTC on D-1
    - all remaining forecast signals: issue_timestamp = 18:00 or 16:00 on D-1
    """
    assert timestamps.dt.tz is timezone.utc, "timestamps must be timezone-aware in UTC"
    timestamps_cet = timestamps.dt.tz_convert("Europe/Berlin")

    issue_hour = 10 if signal_name == FORECAST_SIGNAL.TOTAL_CONSUMPTION_FC else 18
    day_start = timestamps_cet.dt.floor("D")
    issue_timestamps_cet = (
        day_start - pd.Timedelta(days=1) + pd.Timedelta(hours=issue_hour)
    )
    return issue_timestamps_cet.dt.tz_convert("UTC")


def _fetch_range_single_signal(
    signal_name: Union[ENERGY_SOURCE, CONSUMPTION_TYPE, NEIGHBORING_REGION],
    start_date: datetime,
    end_date: datetime,
    region: REGION = REGION.DE,
    unit: Units = Units.MW,
    resolution: RESOLUTION = RESOLUTION.HOUR,
    chunk_workers: int = 8,
) -> pd.DataFrame:
    """Fetches time series data for a given signal, date range, and resolution from the SMARD API.
        Weekly chunks are fetched in parallel for speed.
    param: signal_name: The name of the signal to fetch. This can be an instance of
            ENERGY_SOURCE, CONSUMPTION_TYPE, or NEIGHBORING_REGION.
    param: start_date: The start date of the desired date range (inclusive).
    param: end_date: The end date of the desired date range (inclusive).
    param: resolution: The desired data resolution (e.g. Resolution.HOUR). Default is Resolution.HOUR.
    param: chunk_workers: Number of parallel workers for fetching weekly chunks. Default is 8.
    return: A pandas DataFrame containing the time series data for the specified signal, date range, and resolution.
            The DataFrame has columns "timestamp" (as a timezone-aware datetime in UTC),
                                        "value" (as a numeric value),
                                        "signal" (the name of the signal), and
                                        "unit" (the unit of the values).
    """

    filter_id = signal_name if isinstance(signal_name, Enum) else None
    if filter_id is None:
        raise ValueError(
            f"Unsupported signal name: {signal_name}. Must be an instance of \
                         ENERGY_SOURCE, CONSUMPTION_TYPE, or NEIGHBORING_REGION."
        )

    # Get the index for the specified filter_id, region, and resolution
    valid_timestamps = _get_index(filter_id, region, resolution)
    if not valid_timestamps:
        raise ValueError(
            f"No timestamps found in index for signal '{signal_name}' with \
                         filter_id {filter_id} and region {region.value}."
        )

    # Filter the index to get the relevant timestamps for the specified date range.
    # Each ts marks the start of a weekly chunk, so we must include the chunk whose
    # start is before start_date but whose data covers start_date (last ts <= start_timestamp).
    start_timestamp = int(start_date.timestamp() * 1000)  # Convert to milliseconds
    end_timestamp = int(end_date.timestamp() * 1000)  # Convert to milliseconds
    before_start = [ts for ts in valid_timestamps if ts <= start_timestamp]
    first_ts = before_start[-1] if before_start else valid_timestamps[0]
    relevant_timestamps = [
        ts for ts in valid_timestamps if first_ts <= ts <= end_timestamp
    ]

    def _fetch_one_chunk(ts: int):
        payload = _get_series_payload(filter_id, region, resolution, ts)
        series = payload.get("series", [])

        if not series:
            return None

        df = pd.DataFrame(series, columns=["timestamp_ms", "value"])
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.drop(columns=["timestamp_ms"], inplace=True)

        return df[["timestamp", "value"]]

    # Fetch the series data for each relevant timestamp in parallel and aggregate into a DataFrame
    data_frames = []
    with ThreadPoolExecutor(max_workers=chunk_workers) as pool:
        for df in pool.map(_fetch_one_chunk, relevant_timestamps):
            if df is not None and not df.empty:
                data_frames.append(df)

    if not data_frames:
        return pd.DataFrame(columns=["timestamp", "value", "signal", "unit"])

    df = pd.concat(data_frames, ignore_index=True)
    df["signal"] = signal_name.name
    df["unit"] = unit.value

    # For forecasted signals, build issue timestamps and add fetched_at column
    is_forecast_signal = isinstance(signal_name, FORECAST_SIGNAL)

    if is_forecast_signal:
        df["issue_timestamp"] = _build_issue_timestamps(signal_name, df["timestamp"])
        df["fetched_at"] = pd.Timestamp.now(tz="UTC")

    # Filter the final DataFrame to ensure it only contains data within the specified date range
    start_mask = df["timestamp"] >= pd.Timestamp(start_date).tz_convert("UTC")
    end_mask = df["timestamp"] <= pd.Timestamp(end_date).tz_convert("UTC")
    filtered_df = df[start_mask & end_mask]

    return filtered_df


def fetch_range_threaded(
    start_date: datetime,
    end_date: datetime,
    max_workers: int = 10,
    resolution: RESOLUTION | None = None,
) -> pd.DataFrame:
    """Fetches time series data for all the smard signals inside the time range from the SMARD API.
    param: start_date: The start date of the desired date range (inclusive).
    param: end_date: The end date of the desired date range (inclusive).
    param: max_workers: Number of parallel workers for fetching signals.
    param: resolution: The SMARD resolution to fetch. If None, picks per-date via resolution_for_date().

    return: A pandas DataFrame containing the time series data for the signals, date range.
            The DataFrame has columns "timestamp" (as a timezone-aware datetime in UTC),
                                        "value" (as a numeric value),
                                        "signal" (the name of the signal),
                                        "unit" (the unit of the values),
                                        "resolution" ("hour" or "quarter-hour").
    """
    logger.info(f"Running threaded fetch_range with max_workers={max_workers}")
    if resolution is None:
        resolution = resolution_for_date(start_date)

    def _fetch_one(signal, signal_config):
        try:
            return _fetch_range_single_signal(
                signal,
                start_date,
                end_date,
                signal_config["region"],
                signal_config["unit"],
                resolution,
            )
        except Exception as e:
            logger.error(f"Failed to fetch signal {signal}: {e}")
            return None

    data_frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_one, sig, cfg): sig for sig, cfg in SMARD_SIGNALS.items()
        }
        for fut in as_completed(futures):
            df = fut.result()
            if df is not None and not df.empty:
                data_frames.append(df)

    if not data_frames:
        return pd.DataFrame(
            columns=["timestamp", "value", "signal", "unit", "resolution"]
        )

    df = pd.concat(data_frames, ignore_index=True)
    df["resolution"] = resolution.value
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_range(
    start_date: datetime, end_date: datetime, resolution: RESOLUTION | None = None
):
    """Fetches time series data for all the smard signals inside the time range from the SMARD API.
    param: start_date: The start date of the desired date range (inclusive).
    param: end_date: The end date of the desired date range (inclusive).
    param: resolution: The SMARD resolution to fetch. If None, picks per-date via resolution_for_date().

    return: A pandas DataFrame containing the time series data for the signals, date range. The DataFrame has columns
        "timestamp" (as a timezone-aware datetime in UTC),
        "value" (as a numeric value),
        "signal" (the name of the signal),
        "unit" (the unit of the values),
        "resolution" ("hour" or "quarter-hour").
    """
    if resolution is None:
        resolution = resolution_for_date(start_date)

    data_frames = []
    for signal, signal_config in SMARD_SIGNALS.items():
        try:
            region = signal_config["region"]
            unit = signal_config["unit"]
            df = _fetch_range_single_signal(
                signal, start_date, end_date, region, unit, resolution
            )
            data_frames.append(df)
        except Exception as e:
            logger.error(f"Failed to fetch signal {signal}: {e}")

    if not data_frames:
        return pd.DataFrame(
            columns=["timestamp", "value", "signal", "unit", "resolution"]
        )

    df = pd.concat(data_frames, ignore_index=True)
    df["resolution"] = resolution.value
    return df.sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    start_date = datetime.now(timezone.utc) - timedelta(days=8)
    end_date = datetime.now(timezone.utc) - timedelta(days=7)

    df = fetch_range(start_date, end_date)
    print(df.head())
    print(len(df))
    print(df["value"].isna().sum())
