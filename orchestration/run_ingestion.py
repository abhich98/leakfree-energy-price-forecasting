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
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from ingestion.loader import load_range
from ingestion.raw_data_inventory import update_raw_data_inventory
from ingestion.s3_uploader import DATA_NAMES, is_already_uploaded, upload_to_s3, create_bucket_if_not_exists
from ingestion.smard_client import fetch_range, RESOLUTION, SMARD_QUARTER_HOUR_SWITCH_DATE
from ingestion.weather_client import fetch_historical_weather, fetch_forecast_weather_2


load_dotenv()  # Load environment variables from .env file

logger = logging.getLogger(__name__)
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
        description="Run the data pipeline to fetch SMARD and weather data and upload to S3. The script receives dates in UTC timezone and processes them in UTC."
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


def run_pipeline(start_date: datetime, end_date: datetime, force_upload: bool):

    create_bucket_if_not_exists()  # Ensure the S3 bucket exists before uploading

    start_time = datetime.now()

    # ── Fetch the ENTIRE range in one pass ──────────────────────────────
    # SMARD serves weekly chunks, so fetching 1 day costs the same API calls
    # as fetching 7 days. Fetching the whole range at once eliminates ~7×
    # redundant weekly chunk downloads and ~2500× redundant index calls.

    # HOURLY DATA
    logger.info(f"Fetching SMARD data for {start_date.date()} (UTC) → {end_date.date()} (UTC)")
    smard_data = fetch_range(start_date=start_date, end_date=end_date, resolution=RESOLUTION.HOUR)
    logger.info(f"SMARD fetch done: {len(smard_data)} rows")

    # ── Split by day and upload (skipping days already in S3) ───────────
    uploaded_objects = _split_and_upload_by_day(
        smard_data, DATA_NAMES.SMARD, force_upload=force_upload
    )

    # QUARTER-HOURLY DATA (only for dates after the switch date)
    start_date_for_quarter_hour = max(start_date, SMARD_QUARTER_HOUR_SWITCH_DATE)
    if start_date_for_quarter_hour < end_date:
        logger.info(f"Fetching SMARD quarter-hourly data for {start_date_for_quarter_hour.date()} (UTC) → {end_date.date()} (UTC)")
        smard_qh_data = fetch_range(start_date=start_date_for_quarter_hour, end_date=end_date, resolution=RESOLUTION.QUARTER_HOUR)
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
    # These are the forecasts that were actually available at auction time,
    # NOT ERA5 actuals. Used for ML training to avoid the leakage in the
    # existing fct_ml_features weather join.
    logger.info(f"Fetching weather forecasts for {start_date.date()} → {end_date.date()}")
    weather_forecast_data = fetch_forecast_weather_2(start_date, end_date, run_utc_hour=0)
    logger.info(f"Weather forecast fetch done: {len(weather_forecast_data)} rows")

    uploaded_objects.extend(
        _split_and_upload_by_day(
            weather_forecast_data, DATA_NAMES.WEATHER_FORECAST, force_upload=force_upload
        )
    )

    inventory = update_raw_data_inventory(uploaded_objects)
    logger.info("Raw data inventory updated: %s", inventory["s3_uri"])

    # Load raw data from S3 into PostgreSQL
    load_range(start_date, end_date)
    end_time = datetime.now()
    logger.info(f"Pipeline completed in {end_time - start_time}")


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
        start_date = datetime.now(tz=ZoneInfo("UTC")).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        end_date = datetime.now(tz=ZoneInfo("UTC"))

    run_pipeline(
        start_date=start_date, 
        end_date=end_date,
        force_upload=args.force_upload
        )