import argparse
import logging
from datetime import datetime, timezone
from typing import Any

import optuna
import pandas as pd

from ml.data_access import (
    load_hourly_price_model_features,
    load_quarter_hourly_price_model_features,
)
from ml.features.feature_engineering import (
    BASELINE_PRED_COLUMNS,
    TARGET_COLUMNS,
    drop_incomplete_days,
    split_x_y,
    temporal_split,
)
from ml.data_versioning import (
    build_data_manifest,
    save_local_manifest,
)
from ml.s3_model_io import save_best_hyperparameters, save_data_manifest, save_pipeline
from ml.training_utils import (
    ModelType,
    build_model_report,
    fill_short_feature_gaps,
    draw_predictions,
    evaluate_holdout,
    save_report,
)
from ml.tuning_utils import (
    create_hourly_pipeline,
    create_qh_pipeline,
    objective_stage1,
    objective_stage2,
    predict_hourly_for_qh_index,
)
from ml.wandb_tracking import start_wandb_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

HOURLY_START_DATE = "2023-05-01"
QUARTER_HOURLY_START_DATE = "2025-10-01"
HOLDOUT_START_DATE = "2026-01-01"
HOLDOUT_END_DATE_EXCLUSIVE = "2026-04-01"


def tune_two_stage_price_models(
    n_trials: int = 40,
    hourly_cv_splits: int = 3,
    qh_cv_splits: int = 2,
    wandb_track: bool = True,
) -> None:

    stage1_hourly_model_type = ModelType.PRICE_HOURLY
    stage2_qh_model_type = ModelType.PRICE_QUARTER_HOURLY
    train_start_time = datetime.now(timezone.utc)

    report: dict[str, Any] = {
        "run": {
            "name": "two_stage_price_model_hyperparameter_tuning",
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
        "search": {
            "method": "optuna",
            "objective": "mae",
            "n_trials": n_trials,
            "cv_splits": {"hourly": hourly_cv_splits, "quarter_hourly": qh_cv_splits},
        },
        "models": {},
    }

    tracking_run = None
    if wandb_track:
        tracking_run = start_wandb_run(
            run_name=f"price-two-stage-tuning-{train_start_time.strftime('%Y%m%d-%H%M%S')}",
            group="price_hyperparameter_tuning",
        )
        report["run"]["wandb_run_id"] = tracking_run.id
        tracking_run.config.update(report)

    logger.info("Loading raw feature contracts")
    hourly_raw = load_hourly_price_model_features(
        start_date=HOURLY_START_DATE,
        end_date_exclusive=HOLDOUT_END_DATE_EXCLUSIVE,
        filter_by_local_timestamp=True,
    )
    qh_raw = load_quarter_hourly_price_model_features(
        start_date=QUARTER_HOURLY_START_DATE,
        end_date_exclusive=HOLDOUT_END_DATE_EXCLUSIVE,
        filter_by_local_timestamp=True,
    )

    hourly_df = drop_incomplete_days(
        fill_short_feature_gaps(hourly_raw, omit_columns=["price_eur_mwh"])
    )
    qh_df = drop_incomplete_days(
        fill_short_feature_gaps(qh_raw, omit_columns=["price_eur_mwh"])
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

    # Split the data into training/validation and holdout sets for both hourly and quarter-hourly models
    X_hourly, y_hourly = split_x_y(hourly_df, model_type=stage1_hourly_model_type)
    X_hourly_trainval, X_hourly_holdout, y_hourly_trainval, y_hourly_holdout = (
        temporal_split(
            X_hourly,
            y_hourly,
            holdout_start_date=HOLDOUT_START_DATE,
        )
    )

    X_qh_trainval, X_qh_holdout, _, _ = temporal_split(
        qh_df,
        qh_df[TARGET_COLUMNS[ModelType.PRICE_QUARTER_HOURLY]],
        holdout_start_date=HOLDOUT_START_DATE,
    )

    if X_hourly_trainval.empty or X_qh_trainval.empty:
        raise ValueError("Insufficient training rows for tuning after preprocessing")

    # Stage 1: Hourly model tuning
    logger.info(
        "Starting hourly (stage 1) tuning with %d trials and %d splits",
        n_trials,
        hourly_cv_splits,
    )
    price_hourly_study = optuna.create_study(
        direction="minimize", study_name="stage1_price_hourly"
    )
    price_hourly_study.optimize(
        lambda trial: objective_stage1(
            trial,
            hourly_cv_splits,
            X_hourly_trainval,
            y_hourly_trainval,
            create_hourly_pipeline,
        ),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_hourly_params = price_hourly_study.best_params
    logger.info("Hourly model best CV MAE: %.4f", price_hourly_study.best_value)

    wandb_run_id = tracking_run.id if tracking_run is not None else None
    hourly_params_s3_uri = save_best_hyperparameters(
        model_name=f"stage1_{stage1_hourly_model_type.value}_forecast",
        params=best_hourly_params,
        metadata={
            "cv_mae": float(price_hourly_study.best_value),
            "n_trials": n_trials,
            "cv_splits": hourly_cv_splits,
            "data_version_id": data_manifest["data_version_id"],
        },
        wandb_run_id=wandb_run_id,
    )
    logger.info("Saved hourly best hyperparameters to %s", hourly_params_s3_uri)

    # Stage 2: Quarter-hourly model tuning
    logger.info(
        "Starting quarter-hourly (stage 2) tuning with %d trials and %d splits",
        n_trials,
        qh_cv_splits,
    )
    price_qh_study = optuna.create_study(
        direction="minimize", study_name="stage2_price_quarter_hourly"
    )
    price_qh_study.optimize(
        lambda trial: objective_stage2(
            trial,
            qh_cv_splits,
            hourly_df.loc[X_hourly_trainval.index],
            X_qh_trainval,
            best_hourly_params,
            stage1_hourly_model_type,
            stage2_qh_model_type,
            create_hourly_pipeline,
            create_qh_pipeline,
            predict_hourly_for_qh_index,
        ),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_qh_params = price_qh_study.best_params
    logger.info("Quarter-hourly model best CV MAE: %.4f", price_qh_study.best_value)

    qh_params_s3_uri = save_best_hyperparameters(
        model_name=f"stage2_{stage2_qh_model_type.value}_forecast",
        params=best_qh_params,
        metadata={
            "cv_mae": float(price_qh_study.best_value),
            "n_trials": n_trials,
            "cv_splits": qh_cv_splits,
            "data_version_id": data_manifest["data_version_id"],
        },
        wandb_run_id=wandb_run_id,
    )
    logger.info("Saved quarter-hourly best hyperparameters to %s", qh_params_s3_uri)

    # Stage 3: Final model training, and evaluation on holdout set
    final_hourly_pipeline = create_hourly_pipeline(best_hourly_params)
    final_hourly_pipeline.fit(X_hourly_trainval, y_hourly_trainval)

    prediction_hourly_holdout = pd.Series(
        final_hourly_pipeline.predict(X_hourly_holdout),
        index=X_hourly_holdout.index,
        name="price_eur_mwh_prediction",
    )
    baseline_hourly_holdout = X_hourly_holdout[
        BASELINE_PRED_COLUMNS[ModelType.PRICE_HOURLY]
    ]
    hourly_holdout_report, hourly_baseline_report = evaluate_holdout(
        prediction_hourly_holdout,
        y_hourly_holdout,
        baseline_hourly_holdout,
    )

    qh_trainval_with_stage1 = X_qh_trainval.copy()
    qh_holdout_with_stage1 = X_qh_holdout.copy()
    qh_trainval_with_stage1["hourly_prediction"] = predict_hourly_for_qh_index(
        final_hourly_pipeline,
        hourly_df,
        pd.DatetimeIndex(qh_trainval_with_stage1.index),
    )
    qh_holdout_with_stage1["hourly_prediction"] = predict_hourly_for_qh_index(
        final_hourly_pipeline,
        hourly_df,
        pd.DatetimeIndex(qh_holdout_with_stage1.index),
    )

    qh_trainval_with_stage1 = qh_trainval_with_stage1.dropna(
        subset=["hourly_prediction"]
    )
    qh_holdout_with_stage1 = qh_holdout_with_stage1.dropna(subset=["hourly_prediction"])

    X_qh_train, y_qh_train_price = split_x_y(
        qh_trainval_with_stage1, model_type=stage2_qh_model_type
    )
    X_qh_holdout, y_qh_holdout_price = split_x_y(
        qh_holdout_with_stage1, model_type=stage2_qh_model_type
    )

    prediction_hourly_train = X_qh_train.pop("hourly_prediction")
    prediction_hourly_holdout = X_qh_holdout.pop("hourly_prediction")
    y_qh_train_deviation = y_qh_train_price - prediction_hourly_train

    final_qh_pipeline = create_qh_pipeline(best_qh_params)
    final_qh_pipeline.fit(X_qh_train, y_qh_train_deviation)

    prediction_qh_holdout = pd.Series(
        prediction_hourly_holdout.to_numpy() + final_qh_pipeline.predict(X_qh_holdout),
        index=y_qh_holdout_price.index,
        name="price_eur_mwh_prediction",
    )
    baseline_qh_holdout = qh_holdout_with_stage1[
        BASELINE_PRED_COLUMNS[ModelType.PRICE_QUARTER_HOURLY]
    ]
    qh_holdout_report, qh_baseline_report = evaluate_holdout(
        prediction_qh_holdout,
        y_qh_holdout_price,
        baseline_qh_holdout,
    )

    # Save predictions - visualizations, reports
    draw_predictions(
        prediction_hourly_holdout,
        y_hourly_holdout,
        key_word="price_hourly_tuned",
    )
    draw_predictions(
        prediction_qh_holdout,
        y_qh_holdout_price,
        key_word="price_quarter_hourly_tuned",
    )

    # Build report for both hourly and quarter-hourly models
    report["models"][stage1_hourly_model_type.value] = build_model_report(
        n_train=len(X_hourly_trainval),
        n_holdout=len(X_hourly_holdout),
        hyperparameters=best_hourly_params,
        holdout_metrics=hourly_holdout_report,
        baseline_metrics=hourly_baseline_report,
        n_features=final_hourly_pipeline.named_steps["model"].n_features_in_,
        cv_mae=float(price_hourly_study.best_value),
        artifacts={"hyperparameters": hourly_params_s3_uri},
    )
    report["models"][stage2_qh_model_type.value] = build_model_report(
        n_train=len(X_qh_train),
        n_holdout=len(X_qh_holdout),
        hyperparameters=best_qh_params,
        holdout_metrics=qh_holdout_report,
        baseline_metrics=qh_baseline_report,
        n_features=final_qh_pipeline.named_steps["model"].n_features_in_,
        cv_mae=float(price_qh_study.best_value),
        artifacts={"hyperparameters": qh_params_s3_uri},
    )

    # Save the trained pipelines to S3 and update the report with their URIs
    hourly_model_s3_uri = save_pipeline(
        final_hourly_pipeline,
        model_name=f"stage1_{stage1_hourly_model_type.value}_forecast",
        metadata=report,
    )
    qh_model_s3_uri = save_pipeline(
        final_qh_pipeline,
        model_name=f"stage2_{stage2_qh_model_type.value}_forecast",
        metadata=report,
    )
    report["models"][stage1_hourly_model_type.value]["artifacts"][
        "pipeline"
    ] = hourly_model_s3_uri
    report["models"][stage2_qh_model_type.value]["artifacts"][
        "pipeline"
    ] = qh_model_s3_uri

    # Save the final report to S3
    save_report(report, "price_forecast_hyperparameter_tuning_report")

    logger.info("Saved tuned hourly (stage 1) model to %s", hourly_model_s3_uri)
    logger.info("Saved tuned quarter-hourly (stage 2) model to %s", qh_model_s3_uri)

    if tracking_run is not None:
        tracking_run.log(
            {
                "hourly/best_cv_mae": float(price_hourly_study.best_value),
                "quarter_hourly/best_cv_mae": float(price_qh_study.best_value),
                "hourly/holdout_mae": float(hourly_holdout_report["mae"]),
                "quarter_hourly/holdout_mae": float(qh_holdout_report["mae"]),
                "hourly/holdout_rmse": float(hourly_holdout_report["rmse"]),
                "quarter_hourly/holdout_rmse": float(qh_holdout_report["rmse"]),
            }
        )
        tracking_run.summary["stage1_hourly_model_s3_uri"] = hourly_model_s3_uri
        tracking_run.summary["stage2_quarter_hourly_model_s3_uri"] = qh_model_s3_uri
        tracking_run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune two-stage price forecasting models with Optuna"
    )
    parser.add_argument(
        "--trials", type=int, default=25, help="Number of Optuna trials per stage"
    )
    parser.add_argument(
        "--cv-splits-hourly",
        type=int,
        default=4,
        help="TimeSeriesSplit folds for CV hourly model",
    )
    parser.add_argument(
        "--cv-splits-quarter-hourly",
        type=int,
        default=2,
        help="TimeSeriesSplit folds for CV quarter-hourly model",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B tracking",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tune_two_stage_price_models(
        n_trials=args.trials,
        hourly_cv_splits=args.cv_splits_hourly,
        qh_cv_splits=args.cv_splits_quarter_hourly,
        wandb_track=not args.no_wandb,
    )
