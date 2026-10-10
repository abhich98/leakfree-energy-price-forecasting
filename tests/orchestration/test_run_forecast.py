from datetime import datetime
from unittest.mock import patch
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from orchestration import run_forecast

BERLIN = ZoneInfo("Europe/Berlin")
START = datetime(2026, 9, 19, tzinfo=BERLIN)
END = datetime(2026, 9, 20, tzinfo=BERLIN)


def _complete_forecasts() -> pd.DataFrame:
    start_utc = START.astimezone(ZoneInfo("UTC"))
    hourly = pd.date_range(start_utc, periods=24, freq="h")
    quarter_hourly = pd.date_range(start_utc, periods=96, freq="15min")
    return pd.DataFrame(
        {
            "target_timestamp": hourly.append(quarter_hourly),
            "resolution": ["hour"] * 24 + ["quarter_hour"] * 96,
            "predicted_value": [50.0] * 120,
        }
    )


def test_validate_forecasts_accepts_complete_point_forecasts():
    run_forecast.validate_forecasts(_complete_forecasts(), START, END)


def test_validate_forecasts_rejects_missing_or_incomplete_predictions():
    forecasts = _complete_forecasts().iloc[:-1].copy()
    forecasts.loc[0, "predicted_value"] = None

    with pytest.raises(ValueError, match="missing predictions"):
        run_forecast.validate_forecasts(forecasts, START, END)


def test_validate_forecasts_allows_omitted_timestamps_for_unavailable_features():
    forecasts = _complete_forecasts().drop(index=[3, 27])

    run_forecast.validate_forecasts(forecasts, START, END)


def test_validate_forecasts_rejects_quantile_crossings():
    forecasts = _complete_forecasts()
    forecasts = pd.concat([forecasts, forecasts], ignore_index=True)
    forecasts["quantile"] = [0.1] * 120 + [0.9] * 120
    forecasts.loc[120:, "predicted_value"] = 40.0

    with pytest.raises(ValueError, match="quantile crossings"):
        run_forecast.validate_forecasts(forecasts, START, END)


def test_run_forecast_requires_a_successful_latest_elt_run():
    with (
        patch(
            "orchestration.run_forecast.get_latest_elt_run",
            side_effect=ValueError("Latest ELT run failed"),
        ),
        patch("orchestration.run_forecast.start_pipeline_run") as start_run,
    ):
        with pytest.raises(ValueError, match="Latest ELT run failed"):
            run_forecast.run_forecast(START, END)

    start_run.assert_not_called()


def test_run_forecast_persists_validated_predictions():
    elt_run_id = uuid4()
    forecast_run_id = uuid4()
    bundle = {
        "pointer_uri": "s3://bucket/models/price/production.json",
        "model_version": "20260918-090000",
        "data_version_id": "data-version-1",
    }
    forecasts = _complete_forecasts()

    with (
        patch(
            "orchestration.run_forecast.get_latest_elt_run",
            return_value={"run_id": elt_run_id, "status": "succeeded"},
        ),
        patch("orchestration.run_forecast.start_pipeline_run", return_value=forecast_run_id),
        patch("orchestration.run_forecast.start_stage_run", side_effect=[uuid4() for _ in range(4)]),
        patch("orchestration.run_forecast.finish_stage_run"),
        patch("orchestration.run_forecast.finish_pipeline_run") as finish_run,
        patch("orchestration.run_forecast.load_production_model_bundle", return_value=bundle),
        patch("orchestration.run_forecast.predict_two_stage_prices", return_value=forecasts),
        patch("orchestration.run_forecast.persist_forecasts", return_value=120) as persist,
    ):
        result = run_forecast.run_forecast(START, END)

    assert result == forecast_run_id
    persist.assert_called_once_with(forecast_run_id, bundle, forecasts.to_dict("records"))
    finish_run.assert_called_once_with(forecast_run_id, "succeeded")