"""
Open-Meteo API client for fetching weather data.

This module provides utilities to retrieve current weather, forecasts,
and historical weather data for specified locations using the Open-Meteo API.
It supports multiple endpoints and returns data as pandas DataFrames
for easy analysis and integration with other data sources.
The client handles API requests, response parsing, and error handling
to ensure reliable data retrieval for various weather-related applications.

Three data sources are supported:
  1. Historical Weather API (ERA5 actuals) — for backfilling observed weather.
  2. Forecast API — for the daily production run (latest forecast).
  3. Historical Forecast API + Single Runs API - alternative to the Forecast API — for ML training on leak-safe weather forecasts (the forecast that was actually available at auction time).
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, cast

import openmeteo_requests
import pandas as pd
import requests
import requests_cache
from retry_requests import retry as retry_requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry_requests(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=cast(Any, retry_session))

# Actual weather data
BASE_HISTORICAL_URL = os.getenv("ZEPHYRWERK_OPENMETEO_HISTORICAL_URL")

# Forecast weather data
BASE_FORECAST_URL = os.getenv("ZEPHYRWERK_OPENMETEO_FORECAST_URL")
BASE_HISTORICAL_FORECAST_URL = os.getenv("ZEPHYRWERK_OPENMETEO_HISTORICAL_FORECAST_URL")
BASE_SINGLE_RUNS_FORECAST_URL = os.getenv(
    "ZEPHYRWERK_OPENMETEO_SINGLE_RUNS_FORECAST_URL"
)

MAX_TIME_OUT = 30  # s

# ECMWF IFS Single Runs are available from March 2024. Before this date, fall
# back to the Historical Forecast API (stitched icon_seamless) as a proxy.
ECMWF_SINGLE_RUNS_START_DATE = datetime(2024, 3, 14, tzinfo=timezone.utc)

# The auction-time forecast run: 00:00 UTC on D-1 (issued ~02:00 CET, well
# before the 12:00 CET auction close). The 06:00 UTC run is fresher but ECMWF
# IFS takes 4-6h to compute, so 06:00 may not be ready by auction time.
# 00:00 UTC is the safe default; 06:00 UTC is the fallback if available.
AUCTION_TIME_RUN_UTC = "00:00"
AUCTION_TIME_RUN_FALLBACK_UTC = "06:00"


class SignalType(Enum):
    WIND_SPEED = "wind_speed_100m"
    WIND_DIRECTION = "wind_direction_100m"
    SHORTWAVE_RADIATION = "shortwave_radiation"
    CLOUD_COVER = "cloud_cover"
    TEMPERATURE = "temperature_2m"


class Region(Enum):
    BRANDENBURG = "wind_region_brandenburg"
    SCHLESWIG = "wind_region_schleswig"
    BAVARIA = "solar_region_bavaria"
    BADEN_WURTTEMBERG = "solar_region_bawue"


REGION_COORDINATES = {
    Region.BRANDENBURG: {"latitude": 52.41, "longitude": 12.53},
    Region.SCHLESWIG: {"latitude": 54.51, "longitude": 9.55},
    Region.BAVARIA: {"latitude": 48.13, "longitude": 11.58},
    Region.BADEN_WURTTEMBERG: {"latitude": 48.77, "longitude": 9.18},
}

SIGNAL_UNITS = {
    SignalType.WIND_SPEED: "km/h",
    SignalType.WIND_DIRECTION: "°",
    SignalType.SHORTWAVE_RADIATION: "W/m²",
    SignalType.CLOUD_COVER: "%",
    SignalType.TEMPERATURE: "°C",
}


def _is_retryable_http_error(exc: BaseException) -> bool:
    if isinstance(exc, requests.HTTPError):
        status_code = exc.response.status_code if exc.response is not None else None
        return status_code == 429 or (status_code is not None and status_code >= 500)
    return isinstance(exc, (requests.ConnectionError, requests.Timeout, RuntimeError))


def _extract_timed_dataframe(response, region: Region, resolution: str = "hourly", model: str | None = None) -> pd.DataFrame:
    if resolution == "hourly":
        timed_data = response.Hourly()
    elif resolution == "minutely_15":
        timed_data = response.Minutely15()
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")

    timestamps = pd.date_range(
        start=pd.to_datetime(timed_data.Time(), unit="s", utc=True),
        end=pd.to_datetime(timed_data.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=timed_data.Interval()),
        inclusive="left",
    )

    frames = []
    for index, signal_type in enumerate(SignalType):
        values = timed_data.Variables(index).ValuesAsNumpy()
        signal_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "region": region.value,
                "signal_type": signal_type.value,
                "value": values,
                "unit": SIGNAL_UNITS[signal_type],
            }
        )
        if model is not None:
            signal_df["model"] = model
        frames.append(signal_df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@retry(
    retry=retry_if_exception(_is_retryable_http_error),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _fetch_single_region_weather(
    region: Region, start_date: datetime, end_date: datetime, url: str
) -> pd.DataFrame:
    """
    Fetch weather data for a single region from the Open-Meteo API.

    Args:
        region (Region): The region for which to fetch weather data.
        start_date (datetime): The start date and time for the data retrieval.
        end_date (datetime): The end date and time for the data retrieval.
        url (str): The API endpoint URL to use for the request.
    Returns:
        pd.DataFrame: A DataFrame containing the requested weather data
                    for the specified region with timestamps as the index.
    """
    coordinates = REGION_COORDINATES[region]

    params = {
        "latitude": coordinates["latitude"],
        "longitude": coordinates["longitude"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": [signal_type.value for signal_type in SignalType],
        "timezone": "UTC",
        "utc_offset_seconds": 0,
    }

    responses = openmeteo.weather_api(url, params=params)
    if not responses:
        raise RuntimeError("No weather response returned by Open-Meteo")

    return _extract_timed_dataframe(responses[0], region)[
        ["timestamp", "region", "signal_type", "value", "unit"]
    ]


def fetch_historical_weather(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch weather data from the Open-Meteo API for the specified parameters.

    Args:
        start_date (datetime): The start date and time for the data retrieval.
        end_date (datetime): The end date and time for the data retrieval.
    Returns:
        pd.DataFrame: A DataFrame containing the requested weather data with timestamps as the index.
    """
    results = []
    for region in Region:
        df = _fetch_single_region_weather(
            region, start_date, end_date, BASE_HISTORICAL_URL
        )

        if not df.empty:
            results.append(df)

    return pd.concat(results) if results else pd.DataFrame()


def fetch_forecast_weather(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch weather data forecasts from the Open-Meteo API for the specified parameters.

    Args:
        start_date (datetime): The start date and time for the data retrieval.
        end_date (datetime): The end date and time for the data retrieval.
    Returns:
        pd.DataFrame: A DataFrame containing the requested weather data with timestamps as the index.
    """
    results = []
    for region in Region:
        df = _fetch_single_region_weather(
            region, start_date, end_date, BASE_FORECAST_URL
        )

        if not df.empty:
            fetched_at = datetime.now(timezone.utc)
            df["fetched_at"] = fetched_at

            # Since issue_timestamp is not known when fetched with this API, we approximate it as same-day 00:00 UTC for all rows. This is a limitation of the Forecast API.
            df["issue_timestamp"] = df["timestamp"].dt.floor("D")
            results.append(df)

    return pd.concat(results) if results else pd.DataFrame()


# ── Historical/current weather forecasts for ML training (leak-safe) ──────────────────
# This is an alternative to the above fetch_forecast_weather function. It fetches the weather forecast that was actually available at auction time (12:00 CET on D-1), with issue/reference timestamps and model names. This avoids the leakage in the existing fct_ml_features join.
# Two sources are used (hybrid approach):
#   - May 2023 – Feb 2024: Historical Forecast API (stitched icon_seamless).
#     Mild leak: gives near-actual weather (short-lead-time stitched values),
#     not a specific auction-time run. Documented approximation.
#   - Mar 2024 – present: ECMWF IFS Single Runs (run=YYYY-MM-DDT00:00).
#     Strict auction-time, no leak. Consistent model for train + inference.


def _fetch_forecast_stitched(
    region: Region, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """Fetch stitched historical forecast (icon_seamless) from the Historical Forecast API.

    This is a fallback for the ECMWF IFS Single Runs API, which is only available from March 2024. This is a continuous hourly timeseries built by stitching the first hours of
    each successive model run. It closely tracks actual conditions but is NOT a
    specific auction-time run — use as a proxy when Single Runs are unavailable.

    Returns a DataFrame with columns: issue_timestamp, timestamp, region, signal_type, value, unit, model.
    """
    logger.info(
        f"Fetching stitched historical forecast for {region.value} from {start_date.date()} to {end_date.date()}"
    )

    coordinates = REGION_COORDINATES[region]
    params = {
        "latitude": coordinates["latitude"],
        "longitude": coordinates["longitude"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "minutely_15": ",".join([s.value for s in SignalType]),
        "timezone": "UTC",
        "utc_offset_seconds": 0,
        "models": "icon_seamless",
    }
    responses = openmeteo.weather_api(BASE_HISTORICAL_FORECAST_URL, params=params)
    if not responses:
        raise RuntimeError("No stitched forecast response returned by Open-Meteo")

    df = _extract_timed_dataframe(responses[0], region, resolution="minutely_15", model="icon_seamless")
    # issue_timestamp is approximate for stitched data — use the target hour minus a
    # nominal lead time. This column is informational; the stitched model doesn't
    # have a single issue_timestamp per target hour.
    df["issue_timestamp"] = df["timestamp"] - pd.Timedelta(hours=18)
    return df[
        [
            "issue_timestamp",
            "timestamp",
            "region",
            "signal_type",
            "value",
            "unit",
            "model",
        ]
    ]


@retry(
    retry=retry_if_exception(_is_retryable_http_error),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _fetch_single_run(
    region: Region, run_time: datetime, forecast_days: int = 2
) -> pd.DataFrame:
    """Fetch a single ECMWF IFS model run from the Single Runs API.

    Args:
        region: The region to fetch for.
        run_time: The model initialisation time (UTC), e.g. 2024-03-01T00:00.
        forecast_days: Number of forecast days to retrieve (default 2 = covers day D).

    Returns a DataFrame with columns: issue_timestamp, timestamp, region, signal_type, value, unit, model.
    """
    assert run_time.tzinfo is not None and run_time.tzinfo.utcoffset(
        run_time
    ) == timedelta(0), "run_time must be UTC"

    # print(f"Fetching ECMWF IFS Single Run for {region.value} at {run_time.isoformat()} for {forecast_days} days")

    coordinates = REGION_COORDINATES[region]
    run_str = run_time.strftime("%Y-%m-%dT%H:%M")
    params = {
        "latitude": coordinates["latitude"],
        "longitude": coordinates["longitude"],
        "run": run_str,
        "forecast_days": forecast_days,
        "minutely_15": ",".join([s.value for s in SignalType]),
        "timezone": "GMT",
        "models": "ecmwf_ifs",
    }
    responses = openmeteo.weather_api(BASE_SINGLE_RUNS_FORECAST_URL, params=params)
    if not responses:
        raise RuntimeError("No single-runs forecast response returned by Open-Meteo")

    df = _extract_timed_dataframe(responses[0], region, resolution="minutely_15", model="ecmwf_ifs")
    df["issue_timestamp"] = pd.Timestamp(run_time)
    return df[
        [
            "issue_timestamp",
            "timestamp",
            "region",
            "signal_type",
            "value",
            "unit",
            "model",
        ]
    ]


def fetch_forecast_for_day(
    target_date: datetime, run_utc_hour: int = 0
) -> pd.DataFrame:
    """Fetch the auction-time weather forecast for a single target day D.

    Uses the hybrid approach:
      - If target_date >= ECMWF_SINGLE_RUNS_START_DATE (Mar 2024): use ECMWF IFS
        Single Runs with run = D-1 at 00:00 UTC (strict auction-time, no leak).
      - Otherwise: use Historical Forecast API (stitched icon_seamless) as a proxy.

    Args:
        target_date: The day D to fetch the forecast for (the day being predicted).
        run_utc_hour: The model run hour on D-1 (0 or 6). Default 0 (00:00 UTC run).

    Returns a DataFrame with columns: issue_timestamp, timestamp, region, signal_type, value, unit, model.
    """
    if target_date.tzinfo is None:
        target_date = target_date.replace(tzinfo=timezone.utc)

    # The forecast is issued on D-1; we want the forecast covering all of D.
    issue_date = target_date - timedelta(days=1)

    if issue_date >= ECMWF_SINGLE_RUNS_START_DATE:
        # Strict auction-time: ECMWF IFS Single Run at 00:00 UTC on D-1.
        run_time = issue_date.replace(
            hour=run_utc_hour, minute=0, second=0, microsecond=0
        )
        results = []
        for region in Region:
            df = _fetch_single_run(region, run_time, forecast_days=2)
            if not df.empty:
                # Filter to only the target day D
                df = df[df["timestamp"].dt.date == target_date.date()]
                df["fetched_at"] = datetime.now(timezone.utc)
                results.append(df)
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame(
                columns=[
                    "issue_timestamp",
                    "timestamp",
                    "region",
                    "signal_type",
                    "value",
                    "unit",
                    "model",
                    "fetched_at",
                ]
            )
    else:
        # Proxy: stitched icon_seamless. Fetch a 2-day window and filter to D.
        results = []
        for region in Region:
            df = _fetch_forecast_stitched(region, issue_date, target_date)
            if not df.empty:
                df = df[df["timestamp"].dt.date == target_date.date()]
                df["fetched_at"] = datetime.now(timezone.utc)
                results.append(df)
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame(
                columns=[
                    "issue_timestamp",
                    "timestamp",
                    "region",
                    "signal_type",
                    "value",
                    "unit",
                    "model",
                    "fetched_at",
                ]
            )


def fetch_forecast_weather_2(
    start_date: datetime, end_date: datetime, run_utc_hour: int = 0
) -> pd.DataFrame:
    """Fetch auction-time weather forecasts for a range of target (historical or current) days.

    Fetches each day independently in parallel, while keeping the total number of
    concurrent workers low enough to avoid overloading the Open-Meteo API.

    Args:
        start_date: The first target day D (inclusive).
        end_date: The last target day D (inclusive).
        run_utc_hour: The model run hour on D-1 (0 or 6). Default 0.

    Returns a concatenated DataFrame with columns:
        issue_timestamp, timestamp, region, signal_type, value, unit, model.
    """
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    days = []
    current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    while current <= end_date:
        days.append(current)
        current += timedelta(days=1)

    if not days:
        return pd.DataFrame(
            columns=[
                "issue_timestamp",
                "timestamp",
                "region",
                "signal_type",
                "value",
                "unit",
                "model",
            ]
        )

    results: list[pd.DataFrame] = []
    max_workers = min(4, len(days))

    def _fetch_day(day: datetime) -> pd.DataFrame:
        try:
            df = fetch_forecast_for_day(day, run_utc_hour=run_utc_hour)
            if not df.empty:
                logger.info(
                    f"Fetched weather forecast for {day.date()} ({len(df)} rows)"
                )
            return df
        except Exception as e:
            logger.error(f"Failed to fetch weather forecast for {day.date()}: {e}")
            return pd.DataFrame(
                columns=[
                    "issue_timestamp",
                    "timestamp",
                    "region",
                    "signal_type",
                    "value",
                    "unit",
                    "model",
                    "fetched_at",
                ]
            )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_day, day): day for day in days}
        for future in as_completed(futures):
            df = future.result()
            if not df.empty:
                results.append(df)

    if not results:
        return pd.DataFrame(
            columns=[
                "issue_timestamp",
                "timestamp",
                "region",
                "signal_type",
                "value",
                "unit",
                "model",
            ]
        )
    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    start = datetime.now(timezone.utc) - timedelta(days=3)
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc) - timedelta(days=1)
    end = datetime(2023, 1, 15, tzinfo=timezone.utc)

    # df = fetch_forecast_weather(start, end)
    # print(df.head(10))
    # print(f"Total rows: {len(df)}")
    # print(f"Regions: {df['region'].unique()}")
    # print(f"Signals: {df['signal_type'].unique()}")
    # print(df["timestamp"].min())
    # print(df["timestamp"].max())
    # print(df["value"].isna().sum())

    df = fetch_forecast_weather_2(start, end)
    print(df.head(10))
    print(f"Total rows: {len(df)}")
    print(f"Regions: {df['region'].unique()}")
    print(f"Signals: {df['signal_type'].unique()}")
    print(df["timestamp"].min())
    print(df["timestamp"].max())
    print(df["value"].isna().sum())

    df = fetch_historical_weather(start, end)
    print(df.head(10))
    print(f"Total rows: {len(df)}")
    print(f"Regions: {df['region'].unique()}")
    print(f"Signals: {df['signal_type'].unique()}")
    print(df["timestamp"].min())
    print(df["timestamp"].max())
    print(df["value"].isna().sum())
