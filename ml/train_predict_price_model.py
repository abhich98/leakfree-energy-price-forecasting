import argparse
import logging
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from ml.data_access import (
    load_hourly_price_model_features,
    load_quarter_hourly_price_model_features,
)
from ml.data_versioning import (
    build_data_manifest,
    save_local_manifest,
)
from ml.features.feature_engineering import (
    BASELINE_PRED_COLUMNS,
    TARGET_COLUMNS,
    HourlyPriceModelFeatureEngineer,
    QuarterHourPriceModelFeatureEngineer,
    drop_incomplete_days,
    split_x_y,
)
from ml.s3_model_io import load_best_hyperparameters, save_data_manifest, save_pipeline
from ml.training_utils import (
    ModelType,
    broadcast_predictions_h2qh,
    build_model_report,
    create_price_pipeline,
    draw_predictions,
    evaluate_holdout,
    fill_short_feature_gaps,
    save_report,
)
from ml.wandb_tracking import (
    log_wandb_model_results,
    start_wandb_run,
    update_wandb_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

HOURLY_START_DATE = "2023-05-01"
QUARTER_HOURLY_START_DATE = "2025-10-01"
HOLDOUT_START_DATE = "2026-09-07"
HOLDOUT_END_DATE_EXCLUSIVE = "2026-09-12"

WEEKLY_PREDICTION_DAYS = 7
WEEK_START_DAY_IDX = 0  # Monday
STAGE1_HOURLY_PARAMS_VERSION = os.environ.get("STAGE1_HOURLY_PARAMS_VERSION", "latest")
STAGE2_QH_PARAMS_VERSION = os.environ.get("STAGE2_QH_PARAMS_VERSION", "latest")


def _load_tuned_hyperparameters(
    model_name: str, version: str
) -> tuple[dict | None, dict]:
    """Load tuned hyperparameters, falling back to defaults if unavailable."""
    try:
        payload = load_best_hyperparameters(model_name=model_name, version=version)
        return payload["params"], payload
    except Exception:
        logger.warning(
            "No tuned hyperparameters found for %s (version=%s); falling back to defaults",
            model_name,
            version,
        )
        return None, {}


def _predict_hourly_for_window(
    hourly_df: pd.DataFrame,
    prediction_start: pd.Timestamp,
    prediction_end_exclusive: pd.Timestamp,
    hourly_params: dict | None = None,
) -> tuple[Pipeline, pd.Series]:
    """Fit the hourly model only on rows before a prediction window."""

    train_df = hourly_df.loc[hourly_df.index < prediction_start]
    X_train, y_train = split_x_y(train_df, model_type=ModelType.PRICE_HOURLY)
    pipeline = create_price_pipeline(HourlyPriceModelFeatureEngineer, hourly_params)
    pipeline.fit(X_train, y_train)

    window_df = hourly_df.loc[
        (hourly_df.index >= prediction_start)
        & (hourly_df.index < prediction_end_exclusive)
    ]
    X_window, _ = split_x_y(window_df, model_type=ModelType.PRICE_HOURLY)
    predictions = pd.Series(
        pipeline.predict(X_window), index=X_window.index, name="hourly_prediction"
    )
    return pipeline, predictions


def _build_pred_windows() -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Build prediction-windows (start and end_exclusives) from warmup and holdout periods. They are weekly for now, but could be a different length in the future."""
    stage2_start = pd.Timestamp(QUARTER_HOURLY_START_DATE)
    holdout_start = pd.Timestamp(HOLDOUT_START_DATE)
    holdout_end_ex = pd.Timestamp(HOLDOUT_END_DATE_EXCLUSIVE)

    training_week_starts = pd.DatetimeIndex([QUARTER_HOURLY_START_DATE]).append(
        pd.date_range(
            start=stage2_start
            + pd.Timedelta(days=WEEKLY_PREDICTION_DAYS - stage2_start.weekday()),
            end=holdout_start,
            freq=f"{WEEKLY_PREDICTION_DAYS}D",
            inclusive="left",
        )
    )

    holdout_week_starts = pd.DatetimeIndex([HOLDOUT_START_DATE]).append(
        pd.date_range(
            start=holdout_start
            + pd.Timedelta(days=WEEKLY_PREDICTION_DAYS - holdout_start.weekday()),
            end=holdout_end_ex,
            freq=f"{WEEKLY_PREDICTION_DAYS}D",
            inclusive="left",
        )
    )

    week_starts = pd.DatetimeIndex(
        training_week_starts.append(holdout_week_starts)
    ).sort_values()
    week_ends_ex = week_starts[1:].append(pd.DatetimeIndex([holdout_end_ex]))
    week_ends_ex = week_ends_ex.unique().sort_values()

    return week_starts, pd.DatetimeIndex(week_ends_ex)


def _initialize_predictions_store(
    *model_types: ModelType,
) -> dict[str, dict[str, list[pd.Series]]]:
    """Create per-stage containers for predictions, baselines, and actuals."""
    return {
        model_type.value: {
            "predictions": [],
            "baseline_predictions": [],
            "actuals": [],
        }
        for model_type in model_types
    }


def _initialize_report() -> dict[str, Any]:
    train_start_time = datetime.now(timezone.utc)
    return {
        "run": {
            "name": "two_stage_price_model_training_prediction",
            "started_at": train_start_time.isoformat(),
            "wandb_run_id": None,
        },
        "data": {
            "hourly": {"start_date": HOURLY_START_DATE},
            "quarter_hourly": {"start_date": QUARTER_HOURLY_START_DATE},
            "holdout": {
                "start_date": HOLDOUT_START_DATE,
                "end_date_exclusive": HOLDOUT_END_DATE_EXCLUSIVE,
            },
        },
        "backtest": {
            "strategy": "weekly_expanding_window",
            "prediction_window_days": WEEKLY_PREDICTION_DAYS,
        },
        "models": {},
    }


def run_two_stage_price_model_training_prediction(
    hourly_raw: pd.DataFrame,
    quarter_hourly_raw: pd.DataFrame,
    wandb_track: bool = True,
) -> None:
    """Backtest/Train both stages in weekly expanding windows and save the last pipelines.

    Each weekly window is predicted by fresh Stage 1 and Stage 2 pipelines fit only
    on prior delivery days. Stage 2 learns the deviation from Stage 1's historical
    out-of-sample point prediction, then reconstructs quarter-hour prices.
    """
    stage1_hourly_model_type = ModelType.PRICE_HOURLY
    stage2_qh_model_type = ModelType.PRICE_QUARTER_HOURLY
    train_start_time = datetime.now(timezone.utc)

    report = _initialize_report()
    tracking_run = None
    if wandb_track:
        tracking_run = start_wandb_run(
            run_name=f"price-two-stage-training-prediction-{train_start_time.strftime('%Y%m%d-%H%M%S')}",
            group="price_training_prediction",
        )
        report["run"]["wandb_run_id"] = tracking_run.id
        update_wandb_config(
            tracking_run,
            {
                "workflow": report["run"]["name"],
                "started_at": report["run"]["started_at"],
                "data": report["data"],
                "backtest": report["backtest"],
            },
        )

    # Preprocessing the raw data
    logger.info(
        "%sFilling short gaps and dropping incomplete days in the data%s",
        "*" * 10,
        "*" * 10,
    )
    hourly_df = drop_incomplete_days(
        fill_short_feature_gaps(
            hourly_raw, omit_columns=["price_eur_mwh"], verbose=True
        )
    )
    qh_df = drop_incomplete_days(
        fill_short_feature_gaps(
            quarter_hourly_raw, omit_columns=["price_eur_mwh"], verbose=True
        )
    )

    # Save the training data manifest locally and to S3
    data_manifest = build_data_manifest(
        {"hourly": hourly_df, "quarter_hourly": qh_df},
        report=report,
    )
    data_manifest_local_path = save_local_manifest(data_manifest)
    data_manifest_s3_uri = save_data_manifest(data_manifest)

    report["data"]["data_version_id"] = data_manifest["data_version_id"]
    report["data"]["manifest_local_path"] = data_manifest_local_path
    report["data"]["manifest_s3_uri"] = data_manifest_s3_uri
    if tracking_run is not None:
        update_wandb_config(
            tracking_run,
            {
                "data_version_id": data_manifest["data_version_id"],
                "data_manifest_s3_uri": data_manifest_s3_uri,
                "raw_inventory": data_manifest["raw_inventory"],
            },
        )

    # Load the tuned hyperparameters for both stage 1 and stage 2 models
    stage1_hourly_params, _ = _load_tuned_hyperparameters(
        f"stage1_{stage1_hourly_model_type.value}_forecast",
        STAGE1_HOURLY_PARAMS_VERSION,
    )
    stage2_qh_params, _ = _load_tuned_hyperparameters(
        f"stage2_{stage2_qh_model_type.value}_forecast", STAGE2_QH_PARAMS_VERSION
    )

    # Prediction windows for backtesting and prediction
    holdout_start = pd.Timestamp(HOLDOUT_START_DATE)
    pred_windows = _build_pred_windows()

    annotated_windows: list[pd.DataFrame] = []
    predictions_dict = _initialize_predictions_store(
        stage1_hourly_model_type, stage2_qh_model_type
    )
    weekly_reports: list[dict[str, object]] = []
    final_stage1_hourly_pipeline = None
    final_stage2_qh_pipeline = None

    for prediction_start, prediction_end_exc in zip(*pred_windows):
        if prediction_start <= hourly_df.index.min().normalize():
            continue
        if prediction_start <= qh_df.index.min().normalize():
            continue

        # Window selection for the current prediction period
        hourly_window = hourly_df.loc[
            (hourly_df.index >= prediction_start)
            & (hourly_df.index < prediction_end_exc)
        ].copy()
        qh_window = qh_df.loc[
            (qh_df.index >= prediction_start) & (qh_df.index < prediction_end_exc)
        ].copy()

        if hourly_window.empty:
            logger.warning(
                "Skipping %s because its hourly rows are incomplete",
                prediction_start.date(),
            )
            continue

        # Stage1-hourly prediction for the current window
        hourly_pipeline, hourly_predictions = _predict_hourly_for_window(
            hourly_df,
            prediction_start,
            prediction_end_exc,
            stage1_hourly_params,
        )
        qh_window["stage1_hourly_prediction"] = broadcast_predictions_h2qh(
            pd.DatetimeIndex(qh_window.index),
            hourly_predictions,
        )

        if qh_window.empty:
            logger.warning(
                "Skipping %s because its quarter-hourly rows are incomplete",
                prediction_start.date(),
            )
            continue

        annotated_windows.append(qh_window)
        if prediction_start < holdout_start:
            continue

        # Collect stage1-hourly forecast results
        predictions_dict[stage1_hourly_model_type.value]["predictions"].append(
            hourly_predictions
        )
        predictions_dict[stage1_hourly_model_type.value]["baseline_predictions"].append(
            hourly_window[BASELINE_PRED_COLUMNS[stage1_hourly_model_type]]
        )
        predictions_dict[stage1_hourly_model_type.value]["actuals"].append(
            hourly_window[TARGET_COLUMNS[stage1_hourly_model_type]]
        )

        # Stage 2 training uses all prior Stage 1 predictions and actuals to learn the deviation
        qh_train = pd.concat(annotated_windows).sort_index()
        qh_train = qh_train.loc[qh_train.index < prediction_start]
        if qh_train.empty:
            raise ValueError(
                f"No Stage 2 training rows available before {prediction_start.date()}"
            )

        X_qh_train, price_qh_train = split_x_y(
            qh_train,
            model_type=stage2_qh_model_type,
        )
        price_deviation_train = price_qh_train - X_qh_train.pop(
            "stage1_hourly_prediction"
        )

        X_qh_window, price_qh_actual = split_x_y(
            qh_window,
            model_type=stage2_qh_model_type,
        )
        hourly_window_prediction = X_qh_window.pop("stage1_hourly_prediction")

        qh_pipeline = create_price_pipeline(
            QuarterHourPriceModelFeatureEngineer, stage2_qh_params
        )
        qh_pipeline.fit(X_qh_train, price_deviation_train)
        reconstructed_price = pd.Series(
            hourly_window_prediction.to_numpy() + qh_pipeline.predict(X_qh_window),
            index=price_qh_actual.index,
            name="price_eur_mwh_prediction",
        )

        predictions_dict[stage2_qh_model_type.value]["predictions"].append(
            reconstructed_price
        )
        predictions_dict[stage2_qh_model_type.value]["baseline_predictions"].append(
            qh_window[BASELINE_PRED_COLUMNS[stage2_qh_model_type]]
        )
        predictions_dict[stage2_qh_model_type.value]["actuals"].append(price_qh_actual)

        weekly_report: dict[str, object] = {
            "prediction_start": prediction_start.isoformat(),
            "prediction_end_exclusive": prediction_end_exc.isoformat(),
        }
        for stg_val, stg_dict in predictions_dict.items():
            stage_window_report = evaluate_holdout(
                stg_dict["predictions"][-1],
                stg_dict["actuals"][-1],
                stg_dict["baseline_predictions"][-1],
            )[0]
            weekly_report[stg_val] = stage_window_report

            num_prediction_days = 0
            if stg_val == stage1_hourly_model_type.value:
                num_prediction_days = (
                    pd.DatetimeIndex(hourly_window.index).floor("D").nunique()
                )
            if stg_val == stage2_qh_model_type.value:
                num_prediction_days = (
                    pd.DatetimeIndex(qh_window.index).floor("D").nunique()
                )

            logger.info(
                "Stage %s - window %s to %s (exclusive) (%s/%s days): MAE %.3f",
                stg_val,
                prediction_start.date(),
                prediction_end_exc.date(),
                num_prediction_days,
                (prediction_end_exc.date() - prediction_start.date()).days,
                stage_window_report["mae"],
            )

        weekly_reports.append(weekly_report)

        final_stage1_hourly_pipeline = hourly_pipeline
        final_stage2_qh_pipeline = qh_pipeline

    # Consolidating for the entire holdout period
    if (
        not predictions_dict
        or final_stage1_hourly_pipeline is None
        or final_stage2_qh_pipeline is None
    ):
        raise ValueError(
            "No complete Stage 2 holdout windows were available for weekly training"
        )

    for stg_val, stg_dict in predictions_dict.items():
        holdout_predictions = pd.concat(stg_dict["predictions"]).sort_index()
        holdout_baseline_predictions = pd.concat(
            stg_dict["baseline_predictions"]
        ).sort_index()
        holdout_actuals = pd.concat(stg_dict["actuals"]).sort_index()

        holdout_report, baseline_report = evaluate_holdout(
            holdout_predictions,
            holdout_actuals,
            holdout_baseline_predictions,
        )

        # Build the stage-specific report dictionary
        stg_report = {}
        stg_report["holdout_metrics"] = holdout_report
        stg_report["baseline_metrics"] = baseline_report
        stg_report["n_holdout"] = len(holdout_actuals)

        if stg_val == stage1_hourly_model_type.value:
            stg_report["n_features"] = final_stage1_hourly_pipeline.named_steps[
                "model"
            ].n_features_in_
            stg_report["hyperparameters"] = final_stage1_hourly_pipeline.named_steps[
                "model"
            ].get_params()
        if stg_val == stage2_qh_model_type.value:
            stg_report["n_features"] = final_stage2_qh_pipeline.named_steps[
                "model"
            ].n_features_in_
            stg_report["hyperparameters"] = final_stage2_qh_pipeline.named_steps[
                "model"
            ].get_params()

        report["models"][stg_val] = build_model_report(**stg_report)

        # Draw the predictions for the current stage
        draw_predictions(
            holdout_predictions,
            holdout_actuals,
            key_word=stg_val + "_weekly_expanding_window",
        )

    # Save final model archives and record their durable references in the report.
    report["backtest"]["windows"] = weekly_reports
    stage1_hourly_model_s3_uri = save_pipeline(
        final_stage1_hourly_pipeline,
        model_name=f"stage1_{stage1_hourly_model_type.value}_forecast",
        metadata=report,
    )
    stage2_qh_model_s3_uri = save_pipeline(
        final_stage2_qh_pipeline,
        model_name=f"stage2_{stage2_qh_model_type.value}_forecast",
        metadata=report,
    )
    report["models"][stage1_hourly_model_type.value].setdefault("artifacts", {})[
        "pipeline"
    ] = stage1_hourly_model_s3_uri
    report["models"][stage2_qh_model_type.value].setdefault("artifacts", {})[
        "pipeline"
    ] = stage2_qh_model_s3_uri
    save_report(report, f"{report['run']['name']}_report")

    logger.info("Saved weekly Stage 1 pipeline to %s", stage1_hourly_model_s3_uri)
    logger.info("Saved weekly Stage 2 pipeline to %s", stage2_qh_model_s3_uri)

    ## WandB Logging

    if tracking_run is not None:
        for model_name, model_report in report["models"].items():
            log_wandb_model_results(tracking_run, model_name, model_report)

        tracking_run.summary["stage1_hourly_model_s3_uri"] = stage1_hourly_model_s3_uri
        tracking_run.summary["stage2_quarter_hourly_model_s3_uri"] = (
            stage2_qh_model_s3_uri
        )
        tracking_run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train two-stage price forecasting models"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B tracking",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    hourly_raw = load_hourly_price_model_features(
        start_date=HOURLY_START_DATE,
        end_date_exclusive=HOLDOUT_END_DATE_EXCLUSIVE,
        filter_by_local_timestamp=True,
    )
    quarter_hourly_raw = load_quarter_hourly_price_model_features(
        start_date=QUARTER_HOURLY_START_DATE,
        end_date_exclusive=HOLDOUT_END_DATE_EXCLUSIVE,
        filter_by_local_timestamp=True,
    )
    logger.info("Loaded hourly data with %d rows and %d columns", *hourly_raw.shape)
    logger.info(
        "Loaded quarter-hour data with %d rows and %d columns",
        *quarter_hourly_raw.shape,
    )

    run_two_stage_price_model_training_prediction(
        hourly_raw, quarter_hourly_raw, wandb_track=not args.no_wandb
    )
