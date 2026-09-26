"""Inference helpers for the production two-stage price forecasting model."""

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml.data_access import (
    load_hourly_price_model_features,
    load_quarter_hourly_price_model_features,
)
from ml.training_utils import ModelType
from ml.two_stage_price import TwoStagePriceModel, prepare_price_features
from utils.s3 import get_bucket_name, get_s3_client

DEFAULT_PRODUCTION_POINTER_URI = "s3://{bucket}/models/price/production.json"
logger = logging.getLogger(__name__)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://") or "/" not in uri[5:]:
        raise ValueError(f"Expected an S3 URI, got {uri!r}.")
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def load_production_model_bundle(pointer_uri: str | None = None) -> dict[str, Any]:
    """Load a production-pointer JSON document with immutable model artifact URIs."""
    if pointer_uri is None:
        bucket = get_bucket_name()
        pointer_uri = DEFAULT_PRODUCTION_POINTER_URI.format(bucket=bucket)
    bucket, key = _parse_s3_uri(pointer_uri)
    response = get_s3_client().get_object(Bucket=bucket, Key=key)
    bundle = json.loads(response["Body"].read())
    required = {
        "model_version",
        "data_version_id",
        "hourly_model_uri",
        "quarter_hourly_model_uri",
    }
    missing = required - bundle.keys()
    if missing:
        raise ValueError(f"Production model pointer is missing fields: {sorted(missing)}.")
    bundle["pointer_uri"] = pointer_uri
    return bundle


def save_production_model_bundle(bundle: dict[str, Any]) -> str:
    """Archive and publish an approved immutable two-stage production bundle."""
    required = {
        "model_version",
        "data_version_id",
        "hourly_model_uri",
        "quarter_hourly_model_uri",
    }
    missing = required - bundle.keys()
    if missing:
        raise ValueError(f"Production model bundle is missing fields: {sorted(missing)}.")

    bucket = get_bucket_name()
    s3 = get_s3_client()
    pointer_key = "models/price/production.json"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    archive_key = f"models/price/production/archive/{timestamp}.json"
    payload = {**bundle, "published_at": datetime.now(timezone.utc).isoformat()}
    body = json.dumps(payload, indent=2, default=str).encode("utf-8")

    s3.put_object(Bucket=bucket, Key=archive_key, Body=body)
    s3.copy_object(
        Bucket=bucket,
        Key=pointer_key,
        CopySource={"Bucket": bucket, "Key": archive_key},
    )
    pointer_uri = f"s3://{bucket}/{pointer_key}"
    logger.info("Published production model bundle to %s", pointer_uri)
    return pointer_uri


def load_pipeline_from_uri(uri: str):
    """Download and deserialize one immutable pipeline artifact."""
    bucket, key = _parse_s3_uri(uri)
    with tempfile.TemporaryDirectory() as directory:
        local_path = Path(directory) / "model.joblib"
        get_s3_client().download_file(bucket, key, str(local_path))
        return joblib.load(local_path)


def _select_requested_window(
    dataframe: pd.DataFrame, start_date: str, end_date_exclusive: str
) -> pd.DataFrame:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date_exclusive)
    return dataframe.loc[(dataframe.index >= start) & (dataframe.index < end)].copy()


def predict_two_stage_prices(
    start_date: str, end_date_exclusive: str, bundle: dict[str, Any]
) -> pd.DataFrame:
    """Predict point prices for a local delivery window using production model artifacts."""
    warmup_start = (pd.Timestamp(start_date) - pd.Timedelta(days=1)).date().isoformat()
    hourly = load_hourly_price_model_features(warmup_start, end_date_exclusive)
    quarter_hourly = load_quarter_hourly_price_model_features(
        warmup_start, end_date_exclusive
    )
    hourly = _select_requested_window(
        prepare_price_features(hourly, ModelType.PRICE_HOURLY, require_target=False),
        start_date,
        end_date_exclusive,
    )
    quarter_hourly = _select_requested_window(
        prepare_price_features(
            quarter_hourly, ModelType.PRICE_QUARTER_HOURLY, require_target=False
        ),
        start_date,
        end_date_exclusive,
    )
    model = TwoStagePriceModel(
        load_pipeline_from_uri(bundle["hourly_model_uri"]),
        load_pipeline_from_uri(bundle["quarter_hourly_model_uri"]),
    )
    predictions = model.predict(hourly, quarter_hourly)

    def as_rows(features: pd.DataFrame, predictions: pd.Series, resolution: str) -> pd.DataFrame:
        timestamps = features["timestamp"] if "timestamp" in features else features.index
        return pd.DataFrame(
            {
                "target_timestamp": pd.to_datetime(timestamps, utc=True),
                "resolution": resolution,
                "predicted_value": predictions.to_numpy(),
            }
        )

    return pd.concat(
        [
            as_rows(hourly, predictions.hourly, "hour"),
            as_rows(quarter_hourly, predictions.quarter_hourly, "quarter_hour"),
        ],
        ignore_index=True,
    )