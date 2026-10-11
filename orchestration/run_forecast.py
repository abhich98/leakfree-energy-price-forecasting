"""Run validated, daily two-stage price forecasting independent of ELT."""

import argparse
import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from ingestion.smard_client import RESOLUTION
from ml.price_inference import load_production_model_bundle, predict_two_stage_prices
from orchestration.forecast_repository import persist_forecasts
from orchestration.run_audit import (
    finish_pipeline_run,
    finish_stage_run,
    get_latest_elt_run,
    start_pipeline_run,
    start_stage_run,
)
from utils.config_hash import config_hash

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()


def _default_window() -> tuple[datetime, datetime]:
    berlin = ZoneInfo("Europe/Berlin")
    tomorrow = datetime.now(berlin).date() + timedelta(days=1)
    start = datetime.combine(tomorrow, datetime.min.time(), tzinfo=berlin)
    return start, start + timedelta(days=1)


def validate_forecasts(forecasts: pd.DataFrame, start: datetime, end: datetime) -> None:
    """Reject invalid predictions while allowing omitted timestamps with unavailable features."""
    expected_timestamps = {
        RESOLUTION.HOUR.value: pd.date_range(start, end, freq="h", inclusive="left"),
        RESOLUTION.QUARTER_HOUR.value: pd.date_range(start, end, freq="15min", inclusive="left"),
    }
    if forecasts.empty or forecasts["predicted_value"].isna().any():
        raise ValueError("Forecast contains missing predictions.")
    forecast_key = ["target_timestamp", "resolution"]
    if "quantile" in forecasts:
        forecast_key.append("quantile")
    if forecasts[forecast_key].duplicated().any():
        raise ValueError("Forecast contains duplicate point predictions.")
    for resolution, expected in expected_timestamps.items():
        actual = pd.DatetimeIndex(
            pd.to_datetime(
                forecasts.loc[forecasts["resolution"] == resolution, "target_timestamp"],
                utc=True,
            ).unique()
        )
        expected = pd.DatetimeIndex(pd.to_datetime(expected, utc=True))
        unexpected = actual.difference(expected)
        if not unexpected.empty:
            raise ValueError(f"Forecast contains {resolution} timestamps outside the requested window.")
        missing = expected.difference(actual)
        if not missing.empty:
            logger.warning(
                "Omitting %d %s forecast timestamps with unavailable features: %s",
                len(missing),
                resolution,
                ", ".join(timestamp.isoformat() for timestamp in missing),
            )
    if "quantile" in forecasts and forecasts["quantile"].notna().any():
        ordered = forecasts.dropna(subset=["quantile"]).sort_values("quantile")
        crossings = ordered.groupby(["target_timestamp", "resolution"])["predicted_value"].diff().lt(0)
        if crossings.any():
            raise ValueError("Forecast contains quantile crossings.")


def _run_stage(run_id: UUID, name: str, stage_config_hash: str, operation) -> Any:
    logger.info("Starting forecast stage %s for run %s", name, run_id)
    stage_run_id = start_stage_run(run_id, name, stage_config_hash)
    try:
        result = operation()
    except Exception as exc:
        finish_stage_run(stage_run_id, "failed", error_message=str(exc))
        logger.exception("Forecast stage %s failed for run %s", name, run_id)
        raise
    details = result if isinstance(result, dict) else {"row_count": len(result)}
    finish_stage_run(stage_run_id, "succeeded", details=details)
    logger.info("Completed forecast stage %s for run %s: %s", name, run_id, details)
    return result


def run_forecast(
    start: datetime,
    end: datetime,
    model_uri: str | None = None,
    ignore_elt_gate: bool = False,
) -> UUID:
    """Generate and persist a validated price forecast using the latest ELT run."""
    logger.info(
        "Starting forecast for %s to %s with model override=%s",
        start.isoformat(),
        end.isoformat(),
        model_uri is not None,
    )
    if ignore_elt_gate:
        logger.warning("Bypassing latest ELT success gate at operator request")
    elt_run = get_latest_elt_run(require_success=not ignore_elt_gate)
    source_elt_run_id = elt_run["run_id"]
    logger.info(
        "Using latest ELT run %s with status %s as forecast lineage",
        source_elt_run_id,
        elt_run["status"],
    )
    run_id = start_pipeline_run(
        "forecast",
        start,
        end,
        config_hash({"pipeline": "forecast", "model_uri": model_uri, "source_elt_run_id": str(source_elt_run_id)}),
        source_elt_run_id=source_elt_run_id,
    )
    logger.info("Created forecast audit run %s", run_id)

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
        logger.exception("Forecast run %s failed", run_id)
        raise
    finish_pipeline_run(run_id, "succeeded")
    logger.info("Forecast run %s completed successfully", run_id)
    return run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a two-stage price forecast.")
    parser.add_argument(
        "--ignore-elt-gate",
        action="store_true",
        help="Allow forecasting even if the latest ELT run did not succeed.",
    )
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
    run_forecast(start, end, args.model_uri, args.ignore_elt_gate)