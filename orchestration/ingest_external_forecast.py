"""Ingest Electricity Maps day-ahead prices into published forecast storage."""

import argparse
import logging
import os
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import text

from db.database import engine
from ingestion.smard_client import RESOLUTION
from orchestration.run_audit import finish_pipeline_run, start_pipeline_run
from utils.config_hash import config_hash

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BERLIN = ZoneInfo("Europe/Berlin")
EXTERNAL_MODEL_NAME = "electricity_maps_day_ahead"
ZONE = "DE"
GRANULARITY = "15_minutes"


def target_window(target_date: date) -> tuple[datetime, datetime]:
    """Return the requested local delivery-day window."""
    start = datetime.combine(target_date, time.min, tzinfo=BERLIN)
    return start, start + timedelta(days=1)


def fetch_electricity_maps_forecast(session: requests.Session | None = None) -> dict[str, Any]:
    """Fetch the next available German day-ahead 15-minute price forecast."""
    endpoint = os.environ["ELECTRICITY_MAPS_API_URL"]
    token = os.environ["ELECTRICITY_MAPS_API_AUTH_TOKEN"]
    client = session or requests.Session()
    response = client.get(
        endpoint,
        params={"zone": ZONE, "horizonHours": 48, "temporalGranularity": GRANULARITY},
        headers={"auth-token": token},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Electricity Maps response does not contain a data array.")
    return payload


def normalize_forecast(payload: dict[str, Any], start: datetime, end: datetime) -> pd.DataFrame:
    """Select and validate one complete local delivery day from the API payload."""
    records = pd.DataFrame(payload["data"])
    required_columns = {"datetime", "value"}
    missing_columns = required_columns.difference(records.columns)
    if missing_columns:
        raise ValueError(f"Electricity Maps response is missing columns: {sorted(missing_columns)}")

    records["target_timestamp"] = pd.to_datetime(records["datetime"], utc=True, errors="coerce")
    records["predicted_value"] = pd.to_numeric(records["value"], errors="coerce")
    if records[["target_timestamp", "predicted_value"]].isna().any().any():
        raise ValueError("Electricity Maps response contains an invalid timestamp or price value.")

    window_start = pd.Timestamp(start).tz_convert("UTC")
    window_end = pd.Timestamp(end).tz_convert("UTC")
    selected = records.loc[
        (records["target_timestamp"] >= window_start) & (records["target_timestamp"] < window_end),
        ["target_timestamp", "predicted_value"],
    ].copy()
    selected["resolution"] = RESOLUTION.QUARTER_HOUR.value

    expected_timestamps = pd.date_range(window_start, window_end, freq="15min", inclusive="left")
    actual_timestamps = pd.DatetimeIndex(selected["target_timestamp"])
    missing_timestamps = expected_timestamps.difference(actual_timestamps)
    unexpected_timestamps = actual_timestamps.difference(expected_timestamps)
    is_incomplete = (
        selected["target_timestamp"].duplicated().any()
        or not missing_timestamps.empty
        or not unexpected_timestamps.empty
    )
    if is_incomplete:
        raise ValueError(
            "Electricity Maps has no complete forecast for the requested delivery day "
            f"{start.date().isoformat()}. Use a date within the endpoint's next-24-hour horizon."
        )
    return selected.sort_values("target_timestamp").reset_index(drop=True)


def _data_version_id(payload: dict[str, Any]) -> str:
    created_at = [str(record["createdAt"]) for record in payload["data"] if record.get("createdAt")]
    return max(created_at, default="electricity_maps_day_ahead")


def persist_external_forecasts(
    run_id: UUID,
    forecasts: pd.DataFrame,
    endpoint: str,
    data_version_id: str,
) -> int:
    """Persist external point forecasts under the audit run that retrieved them."""
    generated_at = datetime.now(timezone.utc)
    records = [
        {
            "forecast_id": uuid4(),
            "run_id": run_id,
            "model_name": EXTERNAL_MODEL_NAME,
            "model_version_uri": endpoint,
            "data_version_id": data_version_id,
            "generated_at": generated_at,
            "target_timestamp": row.target_timestamp,
            "resolution": row.resolution,
            "predicted_value": row.predicted_value,
        }
        for row in forecasts.itertuples(index=False)
    ]
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO analytics.forecast_results (
                    forecast_id, run_id, model_name, model_version_uri, data_version_id,
                    generated_at, target_timestamp, resolution, predicted_value, status, published_at
                ) VALUES (
                    :forecast_id, :run_id, :model_name, :model_version_uri, :data_version_id,
                    :generated_at, :target_timestamp, :resolution, :predicted_value, 'published', :generated_at
                )
                """
            ),
            records,
        )
    return len(records)


def ingest_external_forecast(target_date: date) -> UUID:
    """Fetch, validate, and persist a complete external forecast delivery day."""
    start, end = target_window(target_date)
    endpoint = os.environ["ELECTRICITY_MAPS_API_URL"]
    run_id = start_pipeline_run(
        "external_price_forecast",
        start,
        end,
        config_hash({"provider": "electricity_maps", "zone": ZONE, "granularity": GRANULARITY}),
    )
    try:
        payload = fetch_electricity_maps_forecast()
        forecasts = normalize_forecast(payload, start, end)
        persisted_rows = persist_external_forecasts(run_id, forecasts, endpoint, _data_version_id(payload))
        logger.info("Persisted %d Electricity Maps forecasts for %s", persisted_rows, target_date.isoformat())
    except Exception as exc:
        finish_pipeline_run(run_id, "failed", error_message=str(exc))
        raise
    finish_pipeline_run(run_id, "succeeded")
    return run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Electricity Maps day-ahead German price forecasts.")
    parser.add_argument("--date", help="Europe/Berlin delivery date in YYYY-MM-DD format; defaults to tomorrow.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    load_dotenv()
    delivery_date = (
        date.fromisoformat(arguments.date)
        if arguments.date
        else datetime.now(BERLIN).date() + timedelta(days=1)
    )
    ingest_external_forecast(delivery_date)
