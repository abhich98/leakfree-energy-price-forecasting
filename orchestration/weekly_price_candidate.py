"""Weekly candidate training and promotion for the two-stage price model."""

import argparse
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from botocore.exceptions import ClientError

from ml import ML_REPORT_VERSION, QUARTER_HOURLY_START_DATE
from ml.data_access import (
    load_hourly_price_model_features,
    load_quarter_hourly_price_model_features,
)
from ml.data_versioning import build_data_manifest, save_local_data_manifest
from ml.features.feature_engineering import (
    BASELINE_PRED_COLUMNS,
    TARGET_COLUMNS,
    HourlyPriceModelFeatureEngineer,
    QuarterHourPriceModelFeatureEngineer,
    split_x_y,
)
from ml.price_inference import (
    load_pipeline_from_uri,
    load_production_model_bundle,
    save_production_model_bundle,
)
from ml.promotion import evaluate_promotion
from ml.reporting import StandardReport
from ml.promotion import PromotionDecision
from ml.s3_model_io import load_best_hyperparameters, save_data_manifest, save_pipeline
from ml.training_utils import (
    ModelType,
    broadcast_predictions_h2qh,
    create_price_pipeline,
    save_report,
)
from ml.two_stage_price import TwoStagePriceModel, prepare_price_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


EVALUATION_DAYS = 7
HOURLY_TRAINING_YEARS = 2
WALK_FORWARD_WINDOW_DAYS = 7


@dataclass(frozen=True)
class WeeklyTrainingWindows:
    hourly_train_start: pd.Timestamp
    quarter_hourly_train_start: pd.Timestamp
    evaluation_start: pd.Timestamp
    evaluation_end_exclusive: pd.Timestamp

    def as_dict(self) -> dict:
        return {
            key: value.date().isoformat() for key, value in asdict(self).items()
        }


def build_weekly_training_windows(as_of_date: str | pd.Timestamp) -> WeeklyTrainingWindows:
    """Build fixed training and untouched evaluation windows for a weekly run.

    ``as_of_date`` is the exclusive end of the evaluation period,
    rather than the timestamp at which the command executes.
    """
    evaluation_end_exclusive = pd.Timestamp(as_of_date).normalize()
    evaluation_start = evaluation_end_exclusive - pd.Timedelta(days=EVALUATION_DAYS)
    return WeeklyTrainingWindows(
        hourly_train_start=evaluation_start - pd.DateOffset(years=HOURLY_TRAINING_YEARS),
        quarter_hourly_train_start=pd.Timestamp(QUARTER_HOURLY_START_DATE),
        evaluation_start=evaluation_start,
        evaluation_end_exclusive=evaluation_end_exclusive,
    )


def _load_hyperparameters(model_name: str) -> dict | None:
    try:
        params = load_best_hyperparameters(model_name=model_name)["params"]
        logger.info("Loaded tuned hyperparameters for %s", model_name)
        return params
    except Exception:
        logger.warning("No tuned hyperparameters found for %s; using defaults", model_name)
        return None


def _prepare_features(dataframe: pd.DataFrame, model_type: ModelType) -> pd.DataFrame:
    prepared = prepare_price_features(dataframe, model_type, require_target=True)
    logger.info(
        "Prepared %s features: %d rows -> %d rows",
        model_type.value,
        len(dataframe),
        len(prepared),
    )
    return prepared


def _slice(dataframe: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return dataframe.loc[(dataframe.index >= start) & (dataframe.index < end)].copy()


def _predict_hourly(
    hourly_training: pd.DataFrame,
    hourly_window: pd.DataFrame,
    hourly_params: dict | None,
) -> tuple[Any, pd.Series]:
    logger.info(
        "Fitting Stage 1 hourly model on %d rows and predicting %d rows",
        len(hourly_training),
        len(hourly_window),
    )
    X_train, y_train = split_x_y(hourly_training, ModelType.PRICE_HOURLY)
    X_window, _ = split_x_y(hourly_window, ModelType.PRICE_HOURLY)
    pipeline = create_price_pipeline(HourlyPriceModelFeatureEngineer, hourly_params)
    pipeline.fit(X_train, y_train)
    predictions = pd.Series(
        pipeline.predict(X_window), index=X_window.index, name="stage1_hourly_prediction"
    )
    return pipeline, predictions


def build_stage2_training_frame(
    hourly_training: pd.DataFrame,
    quarter_hourly_training: pd.DataFrame,
    hourly_params: dict | None,
) -> pd.DataFrame:
    """Create leak-safe Stage 2 rows using walk-forward Stage 1 predictions."""
    logger.info(
        "Building Stage 2 walk-forward training frame from %d hourly and %d quarter-hour rows",
        len(hourly_training),
        len(quarter_hourly_training),
    )
    annotated_windows: list[pd.DataFrame] = []
    first_prediction_start = quarter_hourly_training.index.min().normalize()
    prediction_starts = pd.date_range(
        first_prediction_start,
        quarter_hourly_training.index.max().normalize() + pd.Timedelta(days=1),
        freq=f"{WALK_FORWARD_WINDOW_DAYS}D",
        inclusive="left",
    )

    for prediction_start in prediction_starts:
        prediction_end = min(
            prediction_start + pd.Timedelta(days=WALK_FORWARD_WINDOW_DAYS),
            quarter_hourly_training.index.max().normalize() + pd.Timedelta(days=1),
        )
        hourly_start = prediction_start - pd.DateOffset(years=HOURLY_TRAINING_YEARS)
        stage1_training = _slice(hourly_training, hourly_start, prediction_start)
        hourly_window = _slice(hourly_training, prediction_start, prediction_end)
        qh_window = _slice(quarter_hourly_training, prediction_start, prediction_end)
        if stage1_training.empty or hourly_window.empty or qh_window.empty:
            continue

        _, stage1_predictions = _predict_hourly(
            stage1_training, hourly_window, hourly_params
        )
        qh_window["stage1_hourly_prediction"] = broadcast_predictions_h2qh(
            pd.DatetimeIndex(qh_window.index), stage1_predictions
        )
        qh_window = qh_window.dropna(subset=["stage1_hourly_prediction"])
        if not qh_window.empty:
            annotated_windows.append(qh_window)

    if not annotated_windows:
        raise ValueError("No walk-forward Stage 2 training rows were available")
    training_frame = pd.concat(annotated_windows).sort_index()
    logger.info(
        "Built Stage 2 training frame with %d rows across %d walk-forward windows",
        len(training_frame),
        len(annotated_windows),
    )
    return training_frame


def _fit_candidate(
    hourly: pd.DataFrame,
    quarter_hourly: pd.DataFrame,
    windows: WeeklyTrainingWindows,
    hourly_params: dict | None,
    qh_params: dict | None,
) -> tuple[Any, Any, dict[str, pd.Series]]:
    hourly_training = _slice(hourly, windows.hourly_train_start, windows.evaluation_start)
    qh_training = _slice(
        quarter_hourly, windows.quarter_hourly_train_start, windows.evaluation_start
    )
    hourly_evaluation = _slice(
        hourly, windows.evaluation_start, windows.evaluation_end_exclusive
    )
    qh_evaluation = _slice(
        quarter_hourly, windows.evaluation_start, windows.evaluation_end_exclusive
    )
    if any(frame.empty for frame in (hourly_training, qh_training, hourly_evaluation, qh_evaluation)):
        raise ValueError("Training or evaluation window has no complete feature rows")

    logger.info(
        "Fitting candidate: hourly train=%d, quarter-hour train=%d, hourly evaluation=%d, "
        "quarter-hour evaluation=%d",
        len(hourly_training),
        len(qh_training),
        len(hourly_evaluation),
        len(qh_evaluation),
    )

    hourly_pipeline, hourly_predictions = _predict_hourly(
        hourly_training, hourly_evaluation, hourly_params
    )
    stage2_training = build_stage2_training_frame(hourly_training, qh_training, hourly_params)
    X_qh_train, y_qh_train = split_x_y(stage2_training, ModelType.PRICE_QUARTER_HOURLY)
    stage1_qh_train = X_qh_train.pop("stage1_hourly_prediction")
    qh_pipeline = create_price_pipeline(QuarterHourPriceModelFeatureEngineer, qh_params)
    qh_pipeline.fit(X_qh_train, y_qh_train - stage1_qh_train)

    candidate_model = TwoStagePriceModel(hourly_pipeline, qh_pipeline)
    qh_predictions = candidate_model.predict_quarter_hourly(
        qh_evaluation, hourly_predictions
    )
    return hourly_pipeline, qh_pipeline, {
        "hourly_predictions": hourly_predictions,
        "hourly_actuals": hourly_evaluation[TARGET_COLUMNS[ModelType.PRICE_HOURLY]],
        "hourly_baseline": hourly_evaluation[BASELINE_PRED_COLUMNS[ModelType.PRICE_HOURLY]],
        "qh_predictions": qh_predictions,
        "qh_actuals": qh_evaluation[TARGET_COLUMNS[ModelType.PRICE_QUARTER_HOURLY]],
        "qh_baseline": qh_evaluation[BASELINE_PRED_COLUMNS[ModelType.PRICE_QUARTER_HOURLY]],
    }


def _champion_predictions(
    bundle: dict[str, Any], hourly: pd.DataFrame, quarter_hourly: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    logger.info(
        "Generating champion comparison predictions for %d hourly and %d quarter-hour rows",
        len(hourly),
        len(quarter_hourly),
    )
    champion = TwoStagePriceModel(
        load_pipeline_from_uri(bundle["hourly_model_uri"]),
        load_pipeline_from_uri(bundle["quarter_hourly_model_uri"]),
    ).predict(hourly, quarter_hourly)
    return champion.hourly, champion.quarter_hourly


def _load_champion() -> dict[str, Any] | None:
    try:
        return load_production_model_bundle()
    except ClientError:
        logger.info("No production champion exists; evaluating initial promotion")
        return None


def _initialize_report(
        windows: WeeklyTrainingWindows,
        hourly_decision: PromotionDecision,
        qh_decision: PromotionDecision,
) -> StandardReport:
    return StandardReport(
        report_version=ML_REPORT_VERSION,
        run={
            "name": "weekly_price_candidate",
            "as_of_date": windows.evaluation_end_exclusive.date().isoformat(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
        data={
            "evaluation": {
                "start_date": windows.evaluation_start.date().isoformat(),
                "end_date_exclusive": windows.evaluation_end_exclusive.date().isoformat(),
            },
        },
        sections={
            "windows": windows.as_dict(),
            "promotion": {
                "hourly": hourly_decision.as_dict(),
                "quarter_hourly": qh_decision.as_dict(),
            },
        },
    )


def run_weekly_price_candidate(as_of_date: str) -> dict[str, Any]:
    """Train, evaluate, archive, and conditionally promote a price-model candidate."""
    logger.info("Starting weekly price candidate run for as-of date %s", as_of_date)
    windows = build_weekly_training_windows(as_of_date)
    hourly_raw = load_hourly_price_model_features(
        start_date=windows.hourly_train_start.isoformat(),
        end_date_exclusive=windows.evaluation_end_exclusive.isoformat(),
        filter_by_local_timestamp=True,
    )
    qh_raw = load_quarter_hourly_price_model_features(
        start_date=windows.quarter_hourly_train_start.isoformat(),
        end_date_exclusive=windows.evaluation_end_exclusive.isoformat(),
        filter_by_local_timestamp=True,
    )
    logger.info(
        "Loaded raw price features: hourly=%d rows, quarter-hour=%d rows",
        len(hourly_raw),
        len(qh_raw),
    )
    hourly = _prepare_features(hourly_raw, ModelType.PRICE_HOURLY)
    quarter_hourly = _prepare_features(qh_raw, ModelType.PRICE_QUARTER_HOURLY)
    hourly_params = _load_hyperparameters("stage1_price_hourly_forecast")
    qh_params = _load_hyperparameters("stage2_price_quarter_hourly_forecast")
    hourly_pipeline, qh_pipeline, evaluation = _fit_candidate(
        hourly, quarter_hourly, windows, hourly_params, qh_params
    )

    champion = _load_champion()
    champion_hourly = champion_qh = None
    if champion is not None:
        hourly_evaluation = _slice(hourly, windows.evaluation_start, windows.evaluation_end_exclusive)
        qh_evaluation = _slice(quarter_hourly, windows.evaluation_start, windows.evaluation_end_exclusive)
        champion_hourly, champion_qh = _champion_predictions(
            champion, hourly_evaluation, qh_evaluation
        )

    hourly_decision = evaluate_promotion(
        evaluation["hourly_predictions"],
        evaluation["hourly_actuals"],
        evaluation["hourly_baseline"],
        champion_hourly,
    )
    qh_decision = evaluate_promotion(
        evaluation["qh_predictions"],
        evaluation["qh_actuals"],
        evaluation["qh_baseline"],
        champion_qh,
    )
    logger.info(
        "Promotion evaluation: hourly approved=%s (candidate MAE=%.4f, champion MAE=%s), "
        "quarter-hour approved=%s (candidate MAE=%.4f, champion MAE=%s)",
        hourly_decision.approved,
        hourly_decision.candidate_mae,
        hourly_decision.champion_mae,
        qh_decision.approved,
        qh_decision.candidate_mae,
        qh_decision.champion_mae,
    )

    report = _initialize_report(
        windows=windows,
        hourly_decision=hourly_decision,
        qh_decision=qh_decision,
    )
    manifest = build_data_manifest({"hourly": hourly, "quarter_hourly": quarter_hourly}, report)
    data_manifest_local_path = save_local_data_manifest(manifest)
    data_manifest_s3_uri = save_data_manifest(manifest)
    report.attach_data_manifest(manifest, data_manifest_local_path, data_manifest_s3_uri)
    logger.info(
        "Saved data manifest: version=%s, local=%s, s3=%s",
        manifest["data_version_id"],
        data_manifest_local_path,
        data_manifest_s3_uri,
    )

    model_version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    hourly_uri = save_pipeline(
        hourly_pipeline,
        model_name="price_candidate_hourly",
        metadata=report.to_dict(),
    )
    qh_uri = save_pipeline(
        qh_pipeline,
        model_name="price_candidate_quarter_hourly",
        metadata=report.to_dict(),
    )
    report["artifacts"] = {
        "hourly_model_uri": hourly_uri,
        "quarter_hourly_model_uri": qh_uri,
    }
    logger.info(
        "Saved candidate model artifacts: hourly=%s, quarter-hour=%s",
        hourly_uri,
        qh_uri,
    )

    if hourly_decision.approved and qh_decision.approved:
        report["production_pointer_uri"] = save_production_model_bundle(
            {
                "model_version": model_version,
                "data_version_id": manifest["data_version_id"],
                "hourly_model_uri": hourly_uri,
                "quarter_hourly_model_uri": qh_uri,
                "promotion": report["promotion"],
            }
        )
        report["promotion"]["approved"] = True
        logger.info("Candidate approved and production pointer published")
    else:
        report["promotion"]["approved"] = False
        logger.info("Candidate rejected; production pointer was not changed")

    save_report(report, "weekly_price_candidate_report")
    logger.info("Finished weekly price candidate run for as-of date %s", as_of_date)
    return report.to_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a weekly price-model candidate")
    parser.add_argument(
        "--as-of-date",
        required=True,
        help="Exclusive end date (YYYY-MM-DD) of the preceding 28-day evaluation period",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_weekly_price_candidate(arguments.as_of_date)