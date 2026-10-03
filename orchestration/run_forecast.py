"""Run validated, daily two-stage price forecasting independent of ELT."""

import argparse
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from ml.price_inference import load_production_model_bundle, predict_two_stage_prices
from orchestration.forecast_repository import persist_forecasts
from orchestration.run_audit import (
    finish_pipeline_run,
    finish_stage_run,
    get_pipeline_run,
    start_pipeline_run,
    start_stage_run,
)
from utils.config_hash import config_hash

load_dotenv()


def _default_window() -> tuple[datetime, datetime]:
    berlin = ZoneInfo("Europe/Berlin")
    tomorrow = datetime.now(berlin).date() + timedelta(days=1)
    start = datetime.combine(tomorrow, datetime.min.time(), tzinfo=berlin)
    return start, start + timedelta(days=1)


def validate_forecasts(forecasts: pd.DataFrame, start: datetime, end: datetime) -> None:
    """Reject missing or incomplete point/quantile forecasts before persistence."""
    expected = {
        "hour": len(pd.date_range(start, end, freq="h", inclusive="left")),
        "quarter_hour": len(pd.date_range(start, end, freq="15min", inclusive="left")),
    }
    if forecasts.empty or forecasts["predicted_value"].isna().any():
        raise ValueError("Forecast contains missing predictions.")
    for resolution, expected_count in expected.items():
        actual_count = forecasts.loc[
            forecasts["resolution"] == resolution, "target_timestamp"
        ].nunique()
        if actual_count != expected_count:
            raise ValueError(f"Expected {expected_count} {resolution} forecasts, found {actual_count}.")
    if "quantile" in forecasts and forecasts["quantile"].notna().any():
        ordered = forecasts.dropna(subset=["quantile"]).sort_values("quantile")
        crossings = ordered.groupby(["target_timestamp", "resolution"])["predicted_value"].diff().lt(0)
        if crossings.any():
            raise ValueError("Forecast contains quantile crossings.")


def _run_stage(run_id: UUID, name: str, stage_config_hash: str, operation) -> Any:
    stage_run_id = start_stage_run(run_id, name, stage_config_hash)
    try:
        result = operation()
    except Exception as exc:
        finish_stage_run(stage_run_id, "failed", error_message=str(exc))
        raise
    details = result if isinstance(result, dict) else {"row_count": len(result)}
    finish_stage_run(stage_run_id, "succeeded", details=details)
    return result


def run_forecast(
    source_elt_run_id: UUID, start: datetime, end: datetime, model_uri: str | None = None
) -> UUID:
    """Generate and persist a validated price forecast from a successful ELT run."""
    elt_run = get_pipeline_run(source_elt_run_id)
    if elt_run["pipeline_name"] != "elt" or elt_run["status"] != "succeeded":
        raise ValueError("Forecasting requires a successful ELT run.")
    run_id = start_pipeline_run(
        "forecast",
        start,
        end,
        config_hash({"pipeline": "forecast", "model_uri": model_uri, "source_elt_run_id": str(source_elt_run_id)}),
        source_elt_run_id=source_elt_run_id,
    )
    def stage_config(stage_name: str) -> str:
        return config_hash(
            {
                "pipeline": "forecast",
                "stage": stage_name,
                "source_elt_run_id": str(source_elt_run_id),
                "model_uri": model_uri,
                "requested_start": start.isoformat(),
                "requested_end": end.isoformat(),
            }
        )
    try:
        bundle = _run_stage(
            run_id, "load_model", stage_config("load_model"), lambda: load_production_model_bundle(model_uri)
        )
        forecasts = _run_stage(
            run_id,
            "predict",
            stage_config("predict"),
            lambda: predict_two_stage_prices(start.date().isoformat(), end.date().isoformat(), bundle),
        )
        _run_stage(
            run_id,
            "validate",
            stage_config("validate"),
            lambda: validate_forecasts(forecasts, start, end) or {"validated_rows": len(forecasts)},
        )
        _run_stage(
            run_id,
            "persist",
            stage_config("persist"),
            lambda: {"persisted_rows": persist_forecasts(run_id, bundle, forecasts.to_dict("records"))},
        )
    except Exception as exc:
        finish_pipeline_run(run_id, "failed", error_message=str(exc))
        raise
    finish_pipeline_run(run_id, "succeeded")
    return run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a two-stage price forecast.")
    parser.add_argument("--elt-run-id", required=True, help="Successful ELT run UUID to consume.")
    parser.add_argument("--start-date", help="Local Europe/Berlin date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="Exclusive local Europe/Berlin date in YYYY-MM-DD format.")
    parser.add_argument("--model-uri", help="S3 URI of a model-bundle JSON; overrides the production pointer.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    default_start, default_end = _default_window()
    if bool(args.start_date) != bool(args.end_date):
        raise ValueError("Provide both --start-date and --end-date, or neither.")
    berlin = ZoneInfo("Europe/Berlin")
    start = (
        datetime.fromisoformat(args.start_date).replace(tzinfo=berlin)
        if args.start_date
        else default_start
    )
    end = (
        datetime.fromisoformat(args.end_date).replace(tzinfo=berlin)
        if args.end_date
        else default_end
    )
    if end <= start:
        raise ValueError("--end-date must be after --start-date.")
    run_forecast(UUID(args.elt_run_id), start, end, args.model_uri)