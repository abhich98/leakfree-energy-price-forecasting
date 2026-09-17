from io import BytesIO
import logging
import os
from enum import Enum

import boto3
import pandas as pd
from botocore.exceptions import NoCredentialsError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DATA_NAMES(Enum):
    SMARD = "smard"
    WEATHER = "weather"
    WEATHER_FORECAST = "weather_forecast"  # historical and current weather forecast


def is_already_uploaded(
    data_name: DATA_NAMES,
    year: int,
    month: int,
    day: int,
    resolution: str | None = None,
) -> bool:
    """Check if a Parquet file already exists in S3 for the given date."""
    BUCKET_NAME = os.environ.get("ZEPHYRWERK_AWS_BUCKET_NAME")
    AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL") or None

    key = get_file_name(data_name, year, month, day, resolution)
    s3 = boto3.client("s3", endpoint_url=AWS_ENDPOINT_URL)

    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=key)
        return True
    except Exception:
        return False


def create_bucket_if_not_exists(
    bucket_name: str | None = None, endpoint_url: str | None = None
):
    """Creates an S3 bucket if it does not already exist."""
    if bucket_name is None:
        bucket_name = os.environ.get("ZEPHYRWERK_AWS_BUCKET_NAME")
    if endpoint_url is None:
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL") or None

    s3 = boto3.client("s3", endpoint_url=endpoint_url)
    try:
        s3.head_bucket(Bucket=bucket_name)
        logger.info(f"Bucket '{bucket_name}' already exists.")
    except Exception:
        logger.info(f"Bucket '{bucket_name}' does not exist. Creating it now.")
        s3.create_bucket(Bucket=bucket_name)


def get_file_name(
    data_name: DATA_NAMES,
    year: int,
    month: int,
    day: int,
    resolution: str | None = None,
) -> str:
    """Generates a file name for the Parquet file based on the given year, month, and day.
    The file name follows the format: "{data_name}_{year}_{month}_{day}.parquet" and is stored
    in a directory structure organized by year and month. For SMARD data, a resolution partition is added so hourly and quarter-hourly files for the same day don't collide.

    param: data_name: The name of the data (e.g. "smard").
    param: year: The year of the data (e.g. 2024).
    param: month: The month of the data (e.g. 6 for June).
    param: day: The day of the data (e.g. 15).
    param: resolution: The SMARD resolution ("hour" or "quarter-hour"). Only used for SMARD data.
    return: A string representing the file name and path where the Parquet file should be stored in the S3 bucket.
    """
    file_name = f"{data_name.value}_{year}_{month:02d}_{day:02d}.parquet"
    if data_name == DATA_NAMES.SMARD and resolution is not None:
        return f"raw/{data_name.value}/year={year}/month={month:02d}/resolution={resolution}/{file_name}"
    return f"raw/{data_name.value}/year={year}/month={month:02d}/{file_name}"


def upload_to_s3(
    dataframe: pd.DataFrame, data_name: DATA_NAMES, resolution: str | None = None
):
    """Uploads a pandas DataFrame to an S3 bucket as a Parquet file.
    The file name is generated based on the timestamp of the first row in the DataFrame.

    param: dataframe: A pandas DataFrame containing the data to be uploaded.
                    The DataFrame must have a "timestamp" column with timezone-aware datetime values in UTC.
    param: data_name: The name of the data (e.g. "smard").
    param: resolution: The SMARD resolution ("hour" or "quarter-hour"). Only used for SMARD data;
                    if None for SMARD data, derived from the dataframe's "resolution" column.
    return: None. The function uploads the DataFrame to the specified S3 bucket and does not return any value.
    """
    BUCKET_NAME = os.environ.get("ZEPHYRWERK_AWS_BUCKET_NAME")
    if BUCKET_NAME is None:
        raise ValueError("ZEPHYRWERK_AWS_BUCKET_NAME environment variable is not set.")

    AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL") or None

    if dataframe.empty:
        raise ValueError(
            f"There is not any data to write s3 with data name {data_name.value}."
        )

    date = dataframe["timestamp"].iloc[0]
    year, month, day = date.year, date.month, date.day

    # For SMARD data, resolve the resolution from the dataframe if not given explicitly
    if (
        data_name == DATA_NAMES.SMARD
        and resolution is None
        and "resolution" in dataframe.columns
    ):
        resolution = dataframe["resolution"].iloc[0]

    file_name = get_file_name(data_name, year, month, day, resolution)

    # Serialize to an in-memory buffer instead of writing a local file, then upload it to S3 via boto3
    buffer = BytesIO()
    dataframe.to_parquet(buffer, index=False)
    buffer.seek(0)

    s3 = boto3.client("s3", endpoint_url=AWS_ENDPOINT_URL)
    bucket_name = BUCKET_NAME
    try:
        s3.put_object(Bucket=bucket_name, Key=file_name, Body=buffer)
        logger.info(f"File uploaded successfully to {file_name}")
    except NoCredentialsError:
        logger.error(
            "AWS credentials not found. \
                     Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
        )
        raise
    except Exception as e:
        logger.error(f"Error uploading file to S3: {e}")
        raise
