"""
This is the orchestration entry point. It wires the three modules together to fetch SMARD data, 
fetch weather data, and upload both to S3 for a given date range.
The main function is `run_pipeline`, which takes a start date and end date as input, 
fetches the relevant data, and uploads it to S3.

It needs to support two modes of operation:
1. A "full backfill" mode, where the user can specify a start date and end date in the past, 
and the pipeline will fetch all relevant data for that date range and upload it to S3.
2. A "daily update" mode, where the user can specify a start date of yesterday and an end date of today, 
and the pipeline will fetch only the data for the last 24 hours and upload it to S3.

Incremental: fetches yesterday's data only.
"""

import argparse
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, TypeVar
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from ingestion.loader import load_range
from ingestion.raw_data_inventory import update_raw_data_inventory
from ingestion.s3_uploader import DATA_NAMES, create_bucket_if_not_exists, is_already_uploaded, upload_to_s3
from ingestion.smard_client import RESOLUTION, SMARD_QUARTER_HOUR_SWITCH_DATE, fetch_range
from ingestion.weather_client import fetch_forecast_weather_2, fetch_historical_weather
from orchestration.run_audit import (
    finish_pipeline_run,
    finish_stage_run,
    start_pipeline_run,
    start_stage_run,
)

load_dotenv()  # Load environment variables from .env file

logger = logging.getLogger(__name__)
StageResult = TypeVar("StageResult")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),                    # still prints to terminal
        logging.FileHandler("pipeline.log"),        # also writes to file
    ]
)

def parser():
    arg_parser = argparse.ArgumentParser(
        description=(
            "Run the data pipeline to fetch SMARD and weather data and upload to S3. "
            "The script receives dates in UTC timezone and processes them in UTC."
        )
    )
    arg_parser.add_argument("--start_date", 
                            type=str, 
                            help="The start date in YYYY-MM-DD format. Required for full_backfill."
                        )
    arg_parser.add_argument("--end_date", 
                            type=str, 
                            help="The end date in YYYY-MM-DD format. Required for full_backfill."
                        )
    arg_parser.add_argument("--force_upload",
                            action="store_true",
                            help="Force upload of data even if it already exists in S3."
                        )
    return arg_parser.parse_args()


def _split_and_upload_by_day(
    df: pd.DataFrame, data_name: DATA_NAMES, force_upload: bool = False
) -> list[dict]:
    """Split a DataFrame by calendar day and upload each day's data to S3,
    skipping days that are already uploaded. For SMARD data, also groups by
    resolution so hourly and quarter-hourly files are written separately."""
    if df is None or df.empty:
        logger.info(f"No {data_name.value} data to upload")
        return []

    uploaded_objects: list[dict] = []

    # For SMARD data, group by resolution as well as date
    if data_name == DATA_NAMES.SMARD and "resolution" in df.columns:
        grouped = df.groupby([
            df["timestamp"].dt.year,
            df["timestamp"].dt.month,
            df["timestamp"].dt.day,
            df["resolution"],
        ])
        for (year, month, day, resolution), group in grouped:
            resolution_str = str(resolution)
            y, m, d = int(str(year)), int(str(month)), int(str(day))
            if not force_upload and is_already_uploaded(data_name, y, m, d, resolution=resolution_str):
                logger.info(f"Skipping {y}-{m:02d}-{d:02d} {data_name.value} ({resolution_str}) — already uploaded")
                continue
            uploaded_objects.append(upload_to_s3(group, data_name, resolution=resolution_str))
    else:
        grouped = df.groupby([
            df["timestamp"].dt.year,
            df["timestamp"].dt.month,
            df["timestamp"].dt.day,
        ])
        for (year, month, day), group in grouped:
            y, m, d = int(str(year)), int(str(month)), int(str(day))
            if not force_upload and is_already_uploaded(data_name, y, m, d):
                logger.info(f"Skipping {y}-{m:02d}-{d:02d} {data_name.value} — already uploaded")
                continue
            uploaded_objects.append(upload_to_s3(group, data_name))

    return uploaded_objects


def _fetch_weather_chunked(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch weather data, chunking by year to stay within Open-Meteo's practical range limits."""
    frames = []
    chunk_start = start_date
    end_date_local = min(end_date, datetime.now(timezone.utc))

    while chunk_start <= end_date_local:
        # chunk boundary = end of chunk_start's year, or end_date, whichever is sooner
        chunk_end = min(
            datetime(chunk_start.year, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
            end_date_local,
        )
        logger.info(f"Fetching weather for {chunk_start.date()} → {chunk_end.date()}")
        df = fetch_historical_weather(chunk_start, chunk_end)
        if not df.empty:
            frames.append(df)
        chunk_start = datetime(chunk_start.year + 1, 1, 1, tzinfo=timezone.utc)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def ingest_and_upload(
    start_date: datetime, end_date: datetime, force_upload: bool = False
) -> list[dict[str, Any]]:
    """Fetch source data for a range and upload its raw partitions to S3."""
    create_bucket_if_not_exists()
    uploaded_objects: list[dict[str, Any]] = []

    logger.info(f"Fetching SMARD data for {start_date.date()} (UTC) → {end_date.date()} (UTC)")
    smard_data = fetch_range(start_date=start_date, end_date=end_date, resolution=RESOLUTION.HOUR)
    logger.info(f"SMARD fetch done: {len(smard_data)} rows")

    uploaded_objects.extend(_split_and_upload_by_day(
        smard_data, DATA_NAMES.SMARD, force_upload=force_upload
    ))

    # QUARTER-HOURLY DATA (only for dates after germany the date switched to quarter-hourly resolution)
    start_date_for_quarter_hour = max(start_date, SMARD_QUARTER_HOUR_SWITCH_DATE)
    if start_date_for_quarter_hour < end_date:
        logger.info(
            "Fetching SMARD quarter-hourly data for %s (UTC) -> %s (UTC)",
            start_date_for_quarter_hour.date(),
            end_date.date(),
        )
        smard_qh_data = fetch_range(
            start_date=start_date_for_quarter_hour,
            end_date=end_date,
            resolution=RESOLUTION.QUARTER_HOUR,
        )
        logger.info(f"SMARD quarter-hourly fetch done: {len(smard_qh_data)} rows")

        uploaded_objects.extend(
            _split_and_upload_by_day(
                smard_qh_data, DATA_NAMES.SMARD, force_upload=force_upload
            )
        )

    # ── Fetch and upload historical weather data ─────────────
    logger.info(f"Fetching weather data for {start_date.date()} → {end_date.date()}")
    weather_data = _fetch_weather_chunked(start_date, end_date)
    logger.info(f"Weather fetch done: {len(weather_data)} rows")

    uploaded_objects.extend(
        _split_and_upload_by_day(weather_data, DATA_NAMES.WEATHER, force_upload=force_upload)
    )

    # ── Fetch and upload historical/current weather forecasts (leak-safe) ───────
    logger.info(f"Fetching weather forecasts for {start_date.date()} → {end_date.date()}")
    weather_forecast_data = fetch_forecast_weather_2(start_date, end_date, run_utc_hour=0)
    logger.info(f"Weather forecast fetch done: {len(weather_forecast_data)} rows")

    uploaded_objects.extend(
        _split_and_upload_by_day(
            weather_forecast_data, DATA_NAMES.WEATHER_FORECAST, force_upload=force_upload
        )
    )

    return uploaded_objects


def record_raw_inventory(uploaded_objects: list[dict[str, Any]]) -> dict[str, Any]:
    """Persist an immutable inventory that records the uploaded S3 object versions."""
    inventory = update_raw_data_inventory(uploaded_objects)
    logger.info("Raw data inventory updated: %s", inventory["s3_uri"])
    return inventory


def load_postgres(start_date: datetime, end_date: datetime) -> None:
    """Load raw S3 partitions for a range into PostgreSQL."""
    load_range(start_date, end_date)


def _run_stage(
    run_id,
    stage_name: str,
    operation: Callable[[], StageResult],
    details: Callable[[StageResult], dict[str, Any]],
) -> StageResult:
    """Execute one stage and ensure its audit record is finalized on every outcome."""
    stage_run_id = start_stage_run(run_id, stage_name)
    try:
        result = operation()
    except Exception as exc:
        finish_stage_run(stage_run_id, "failed", error_message=str(exc))
        raise
    finish_stage_run(stage_run_id, "succeeded", details=details(result))
    return result


def run_pipeline(
    start_date: datetime, end_date: datetime, force_upload: bool = False
) -> None:
    """Run the auditable ingestion, inventory, and PostgreSQL-load stages."""
    start_time = datetime.now(timezone.utc)
    run_id = start_pipeline_run("ingestion", start_date, end_date)
    inventory: dict[str, Any] | None = None

    try:
        uploaded_objects = _run_stage(
            run_id,
            "ingest_and_upload",
            lambda: ingest_and_upload(start_date, end_date, force_upload),
            lambda objects: {"uploaded_object_count": len(objects)},
        )
        inventory = _run_stage(
            run_id,
            "record_raw_inventory",
            lambda: record_raw_inventory(uploaded_objects),
            lambda result: {
                "inventory_id": result["inventory_id"],
                "s3_uri": result["s3_uri"],
            },
        )
        _run_stage(
            run_id,
            "load_postgres",
            lambda: load_postgres(start_date, end_date),
            lambda _: {
                "requested_start": start_date.isoformat(),
                "requested_end": end_date.isoformat(),
            },
        )
    except Exception as exc:
        finish_pipeline_run(
            run_id,
            "failed",
            raw_inventory_uri=inventory["s3_uri"] if inventory else None,
            error_message=str(exc),
        )
        raise

    finish_pipeline_run(run_id, "succeeded", raw_inventory_uri=inventory["s3_uri"])
    logger.info("Pipeline run %s completed in %s", run_id, datetime.now(timezone.utc) - start_time)


if __name__ == "__main__":
    # The scripte receives dates in UTC timezome.
    args = parser()
    
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=ZoneInfo("UTC"))

        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=ZoneInfo("UTC"))
        end_date = end_date.replace(hour=23, minute=59, second=59)

    elif args.start_date or args.end_date:
        raise ValueError("Provide both --start_date and --end_date or neither.")
    else:
        start_date = (
            datetime.now(tz=ZoneInfo("UTC"))
            .replace(hour=0, minute=0, second=0, microsecond=0)
            - timedelta(days=1)
        )
        end_date = datetime.now(tz=ZoneInfo("UTC"))

    run_pipeline(
        start_date=start_date, 
        end_date=end_date,
        force_upload=args.force_upload
        )