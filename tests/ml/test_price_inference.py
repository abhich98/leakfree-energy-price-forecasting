import json

import boto3
import numpy as np
import pandas as pd
import pytest
from moto import mock_aws

import ml.price_inference as price_inference
from ml.price_inference import (
    load_production_model_bundle,
    save_production_model_bundle,
)

BUCKET = "zephyrwerk-test-price-models"
REGION = "eu-central-1"


@pytest.fixture
def fake_s3(monkeypatch):
    monkeypatch.setenv("ZEPHYRWERK_AWS_BUCKET_NAME", BUCKET)
    monkeypatch.setenv("AWS_ENDPOINT_URL", "")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)

    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": REGION},
        )
        yield client


def test_save_production_bundle_archives_and_publishes_pointer(fake_s3):
    bundle = {
        "model_version": "candidate-20260920",
        "data_version_id": "data-version-1",
        "hourly_model_uri": "s3://models/hourly.joblib",
        "quarter_hourly_model_uri": "s3://models/quarter-hourly.joblib",
    }

    pointer_uri = save_production_model_bundle(bundle)

    assert pointer_uri == f"s3://{BUCKET}/models/price/production.json"
    loaded = load_production_model_bundle()
    assert loaded["model_version"] == bundle["model_version"]
    assert loaded["pointer_uri"] == pointer_uri
    assert "published_at" in loaded

    keys = {
        item["Key"]
        for item in fake_s3.list_objects_v2(Bucket=BUCKET, Prefix="models/price/")[
            "Contents"
        ]
    }
    assert "models/price/production.json" in keys
    assert any(key.startswith("models/price/production/archive/") for key in keys)


def test_save_production_bundle_requires_model_artifacts(fake_s3):
    with pytest.raises(ValueError, match="missing fields"):
        save_production_model_bundle({"model_version": "candidate-20260920"})


def test_load_production_bundle_rejects_missing_required_fields(fake_s3):
    fake_s3.put_object(
        Bucket=BUCKET,
        Key="models/price/production.json",
        Body=json.dumps({"model_version": "candidate-20260920"}),
    )

    with pytest.raises(ValueError, match="missing fields"):
        load_production_model_bundle()


class _ConstantPipeline:
    def __init__(self, value: float):
        self.value = value

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.value)


def test_prediction_omits_rows_with_missing_inference_features(monkeypatch):
    hourly_index = pd.date_range("2026-09-19 23:00", periods=3, freq="h")
    qh_index = pd.date_range("2026-09-19 23:45", periods=9, freq="15min")
    hourly = pd.DataFrame(
        {"feature": [1.0, None, 3.0], "price_eur_mwh": [None] * 3},
        index=hourly_index,
    )
    quarter_hourly = pd.DataFrame(
        {"feature": [1.0, None] + [3.0] * 7, "price_eur_mwh": [None] * 9},
        index=qh_index,
    )
    requested_starts: list[str] = []

    def load_hourly(start_date: str, end_date_exclusive: str) -> pd.DataFrame:
        requested_starts.append(start_date)
        return hourly

    def load_quarter_hourly(start_date: str, end_date_exclusive: str) -> pd.DataFrame:
        requested_starts.append(start_date)
        return quarter_hourly

    pipelines = iter([_ConstantPipeline(100.0), _ConstantPipeline(5.0)])
    monkeypatch.setattr(price_inference, "load_hourly_price_model_features", load_hourly)
    monkeypatch.setattr(
        price_inference, "load_quarter_hourly_price_model_features", load_quarter_hourly
    )
    monkeypatch.setattr(price_inference, "load_pipeline_from_uri", lambda _: next(pipelines))

    forecasts = price_inference.predict_two_stage_prices(
        "2026-09-20", "2026-09-20 02:00", {"hourly_model_uri": "hourly", "quarter_hourly_model_uri": "qh"}
    )

    assert requested_starts == ["2026-09-19", "2026-09-19"]
    assert len(forecasts.loc[forecasts["resolution"] == "hour"]) == 1
    assert len(forecasts.loc[forecasts["resolution"] == "quarter_hour"]) == 4
    assert forecasts["predicted_value"].tolist() == [100.0] + [105.0] * 4