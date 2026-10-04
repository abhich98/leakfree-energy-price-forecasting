"""Evaluate published price forecasts against realized SMARD market prices."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import matplotlib
import pandas as pd

from utils.forecast_evaluation import forecast_error_summary, metrics_by_resolution
from utils.forecast_evaluation import load_latest_forecasts as load_latest_published_forecasts
from utils.forecast_evaluation import update_actual_values as update_published_actual_values

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

MODEL_NAME = "two_stage_price"
EXTERNAL_MODEL_NAME = "electricity_maps_day_ahead"
RESULTS_DIR = Path("results/forecast_evaluations")
MARKET_TIMEZONE = ZoneInfo("Europe/Berlin")


def update_actual_values(start: datetime, end: datetime, model_name: str = MODEL_NAME) -> int:
    """Backfill realized prices for published point forecasts in the requested window."""
    return update_published_actual_values(start, end, (model_name,))


def load_latest_forecasts(start: datetime, end_exclusive: datetime, model_name: str = MODEL_NAME) -> pd.DataFrame:
    """Load the latest published point forecast for every delivery timestamp and resolution."""
    return load_latest_published_forecasts(start, end_exclusive, (model_name,)).drop(columns="model_name")


def build_evaluation_report(
    forecasts: pd.DataFrame, start: datetime, end_exclusive: datetime, updated_actuals: int
) -> dict[str, Any]:
    """Calculate per-resolution metrics for forecasts with available actual prices."""
    matched = forecasts.dropna(subset=["actual_value"]).copy()
    if matched.empty:
        raise ValueError("No realized prices are available for the requested forecast window.")

    return {
        "model_name": MODEL_NAME,
        "window": {
            "start": start.isoformat(),
            "end_exclusive": end_exclusive.isoformat(),
        },
        "generated_at": datetime.now(ZoneInfo("UTC")).isoformat(),
        "latest_forecast_rows": len(forecasts),
        "evaluated_rows": len(matched),
        "actual_values_updated": updated_actuals,
        "unavailable_actual_rows": len(forecasts) - len(matched),
        "resolutions": metrics_by_resolution(forecasts),
    }


def _forecast_difference_summary(paired: pd.DataFrame) -> dict[str, float | int]:
    """Calculate external-minus-internal point forecast differences."""
    differences = paired["predicted_value_external"] - paired["predicted_value_internal"]
    return {
        "row_count": len(paired),
        "mae": float(differences.abs().mean()),
        "rmse": float((differences.pow(2).mean()) ** 0.5),
        "bias_external_minus_internal": float(differences.mean()),
    }


def build_forecast_comparison_report(
    forecasts: pd.DataFrame,
    start: datetime,
    end_exclusive: datetime,
    updated_actuals: int,
    internal_model_name: str = MODEL_NAME,
    external_model_name: str = EXTERNAL_MODEL_NAME,
) -> dict[str, Any]:
    """Compare current internal and external forecasts, plus their realized-price errors."""
    internal = forecasts.loc[forecasts["model_name"] == internal_model_name].copy()
    external = forecasts.loc[forecasts["model_name"] == external_model_name].copy()
    if internal.empty:
        raise ValueError(f"No published {internal_model_name} forecasts are available for the requested window.")
    if external.empty:
        raise ValueError(f"No published {external_model_name} forecasts are available for the requested window.")

    paired = external.merge(
        internal[["target_timestamp", "resolution", "predicted_value"]],
        on=["target_timestamp", "resolution"],
        how="inner",
        suffixes=("_external", "_internal"),
    )
    if paired.empty:
        raise ValueError("No matching internal and external forecast timestamps are available.")

    return {
        "internal_model_name": internal_model_name,
        "external_model_name": external_model_name,
        "window": {"start": start.isoformat(), "end_exclusive": end_exclusive.isoformat()},
        "generated_at": datetime.now(ZoneInfo("UTC")).isoformat(),
        "actual_values_updated": updated_actuals,
        "forecast_difference": _forecast_difference_summary(paired),
        "models": {
            internal_model_name: forecast_error_summary(internal),
            external_model_name: forecast_error_summary(external),
        },
    }


def save_comparison_plot(
    forecasts: pd.DataFrame,
    output_path: Path,
    internal_model_name: str | None = None,
    external_model_name: str | None = None,
) -> None:
    """Save internal, external, and actual time series by forecast resolution."""
    if internal_model_name is None or external_model_name is None:
        forecasts = forecasts.copy()
        forecasts["model_name"] = MODEL_NAME
        labels = {MODEL_NAME: "forecast"}
    else:
        labels = {internal_model_name: "internal", external_model_name: "external"}

    forecasts["target_timestamp"] = pd.to_datetime(forecasts["target_timestamp"], utc=True).dt.tz_convert(
        MARKET_TIMEZONE
    )
    resolutions = sorted(forecasts["resolution"].unique())
    figure, axes = plt.subplots(len(resolutions), 1, figsize=(14, 4 * len(resolutions)), squeeze=False)

    for axis, resolution in zip(axes[:, 0], resolutions, strict=True):
        rows = forecasts.loc[forecasts["resolution"] == resolution]
        for model_name, label in labels.items():
            model_rows = rows.loc[rows["model_name"] == model_name].sort_values("target_timestamp")
            axis.plot(model_rows["target_timestamp"], model_rows["predicted_value"], label=label, alpha=0.8)
        actuals = rows.dropna(subset=["actual_value"]).drop_duplicates("target_timestamp")
        if not actuals.empty:
            axis.plot(actuals["target_timestamp"], actuals["actual_value"], label="actual", linewidth=2)
        axis.set_title(f"{resolution}: forecast comparison")
        axis.set_ylabel("EUR/MWh")
        axis.grid(alpha=0.25)
        axis.legend()

    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def evaluate_forecasts(
    start: datetime,
    end_exclusive: datetime,
    output_dir: Path = RESULTS_DIR,
    model_name: str = MODEL_NAME,
    external_model_name: str = EXTERNAL_MODEL_NAME,
) -> dict[str, Any]:
    """Compare internal and external forecasts, then evaluate both against actual prices."""
    logger.info(
        "Comparing %s and %s forecasts from %s to %s",
        model_name,
        external_model_name,
        start.isoformat(),
        end_exclusive.isoformat(),
    )
    model_names = (model_name, external_model_name)
    updated_actuals = update_published_actual_values(start, end_exclusive, model_names)
    forecasts = load_latest_published_forecasts(start, end_exclusive, model_names)
    logger.info("Loaded %d latest point forecasts and updated %d actual values", len(forecasts), updated_actuals)

    report = build_forecast_comparison_report(
        forecasts,
        start,
        end_exclusive,
        updated_actuals,
        model_name,
        external_model_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = f"forecast_comparison_{start.date().isoformat()}_{end_exclusive.date().isoformat()}"
    report_path = output_dir / f"{artifact_name}.json"
    plot_path = output_dir / f"{artifact_name}.png"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_comparison_plot(forecasts, plot_path, model_name, external_model_name)
    logger.info("Saved forecast evaluation report to %s and plot to %s", report_path, plot_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate published forecasts against realized SMARD prices.")
    parser.add_argument("--start-date", required=True, help="Europe/Berlin start date in YYYY-MM-DD format.")
    parser.add_argument(
        "--end-date-exclusive",
        required=True,
        help="Exclusive Europe/Berlin end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for the JSON report and PNG plot.",
    )
    parser.add_argument("--model-name", default=MODEL_NAME, help="Internal forecast model name.")
    parser.add_argument("--external-model-name", default=EXTERNAL_MODEL_NAME, help="External forecast model name.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    start_date = datetime.fromisoformat(arguments.start_date).replace(tzinfo=MARKET_TIMEZONE)
    end_date_exclusive = datetime.fromisoformat(arguments.end_date_exclusive).replace(tzinfo=MARKET_TIMEZONE)
    if end_date_exclusive <= start_date:
        raise ValueError("--end-date-exclusive must be after --start-date.")
    evaluate_forecasts(
        start_date,
        end_date_exclusive,
        arguments.output_dir,
        arguments.model_name,
        arguments.external_model_name,
    )