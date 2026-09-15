from pathlib import Path
from typing import Any, Callable

import optuna
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from ml.features.feature_engineering import (
    HourlyPriceModelFeatureEngineer,
    QuarterHourPriceModelFeatureEngineer,
    split_x_y,
)
from ml.training_utils import (
    ModelType,
    broadcast_predictions_h2qh,
    create_price_pipeline,
)

SearchSpace = dict[str, dict[str, Any]]


def load_search_spaces(config_path: str | Path) -> dict[str, SearchSpace]:
    """Load and validate the per-study Optuna search spaces from YAML."""
    with Path(config_path).open() as config_file:
        config = yaml.safe_load(config_file)

    required_studies = {"hourly", "quarter_hourly"}
    if not (isinstance(config, dict) and required_studies.issubset(config)):
        raise ValueError(
            "Search-space YAML must contain exactly 'hourly' and 'quarter_hourly' mappings"
        )

    search_spaces: dict[str, SearchSpace] = {}
    for study_name, space in config.items():
        if not isinstance(space, dict) or not space:
            raise ValueError(f"Search space '{study_name}' must be a non-empty mapping")

        validated_space: SearchSpace = {}
        for parameter_name, specification in space.items():
            if not isinstance(specification, dict):
                raise ValueError(f"Parameter '{parameter_name}' must be a mapping")

            if specification.get("type") not in {"int", "float"}:
                raise ValueError(
                    f"Parameter '{parameter_name}' must have type 'int' or 'float'"
                )
            if "low" not in specification or "high" not in specification:
                raise ValueError(
                    f"Parameter '{parameter_name}' must define both low and high"
                )
            if specification["low"] >= specification["high"]:
                raise ValueError(f"Parameter '{parameter_name}' must have low < high")
            validated_space[parameter_name] = specification
        search_spaces[study_name] = validated_space

    return search_spaces


def suggest_xgb_params(
    trial: optuna.Trial, search_space: SearchSpace
) -> dict[str, Any]:
    """Suggest one XGBoost configuration from a validated YAML search space."""
    params: dict[str, Any] = {}
    for parameter_name, specification in search_space.items():
        suggest_kwargs: dict[str, Any] = {
            "log": bool(specification.get("log", False)),
        }
        if "step" in specification:
            suggest_kwargs["step"] = specification["step"]

        if specification["type"] == "int":
            params[parameter_name] = trial.suggest_int(
                parameter_name,
                int(specification["low"]),
                int(specification["high"]),
                **suggest_kwargs,
            )
        else:
            params[parameter_name] = trial.suggest_float(
                parameter_name,
                float(specification["low"]),
                float(specification["high"]),
                **suggest_kwargs,
            )
    return params


def create_hourly_pipeline(model_params: dict[str, Any]) -> Pipeline:
    return create_price_pipeline(HourlyPriceModelFeatureEngineer, model_params)


def create_qh_pipeline(model_params: dict[str, Any]) -> Pipeline:
    return create_price_pipeline(QuarterHourPriceModelFeatureEngineer, model_params)


def predict_hourly_for_qh_index(
    hourly_pipeline: Pipeline,
    hourly_df: pd.DataFrame,
    qh_index: pd.DatetimeIndex,
) -> pd.Series:
    """Predict the required hours and broadcast each value to quarter-hour rows."""
    needed_hours = pd.DatetimeIndex(qh_index).floor("h").unique().sort_values()
    hourly_window = hourly_df.loc[hourly_df.index.isin(needed_hours)]
    if hourly_window.empty:
        return pd.Series(index=qh_index, dtype=float, name="hourly_prediction")

    X_hourly, _ = split_x_y(hourly_window, model_type=ModelType.PRICE_HOURLY)
    hourly_prediction = pd.Series(
        hourly_pipeline.predict(X_hourly),
        index=X_hourly.index,
        name="hourly_prediction",
    )
    return broadcast_predictions_h2qh(pd.DatetimeIndex(qh_index), hourly_prediction)


# Stage 1 and Stage 2 tuning objectives for Optuna, agnostic to the specific ml model types and data.


# Do not modify, this function is used as the objective for Optuna stage 1 tuning
def objective_stage1(
    trial: optuna.Trial,
    search_space: SearchSpace,
    cv_splits: int,
    X_stage1_trainval: pd.DataFrame,
    y_stage1_trainval: pd.Series,
    create_stage1_pipeline: Callable[[dict[str, Any]], Pipeline],
) -> float:
    """Return mean time-series CV MAE for a candidate model."""
    params = suggest_xgb_params(trial, search_space)
    pipeline = create_stage1_pipeline(params)
    splitter = TimeSeriesSplit(n_splits=cv_splits, gap=24)

    fold_mae: list[float] = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(X_stage1_trainval), start=1
    ):
        X_train = X_stage1_trainval.iloc[train_idx]
        y_train = y_stage1_trainval.iloc[train_idx]
        X_test = X_stage1_trainval.iloc[test_idx]
        y_test = y_stage1_trainval.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        mae = float(mean_absolute_error(y_test, pipeline.predict(X_test)))
        fold_mae.append(mae)
        trial.set_user_attr(f"fold_{fold_idx}_mae", mae)

    score = float(sum(fold_mae) / len(fold_mae))
    trial.set_user_attr("cv_mae", score)
    return score


# Do not modify, this function is used as the objective for Optuna stage 2 tuning
def objective_stage2(
    trial: optuna.Trial,
    search_space: SearchSpace,
    cv_splits: int,
    stage1_trainval: pd.DataFrame,
    stage2_trainval: pd.DataFrame,
    best_stage1_params: dict[str, Any],
    stage1_model: ModelType,
    stage2_model: ModelType,
    create_stage1_pipeline: Callable[[dict[str, Any]], Pipeline],
    create_stage2_pipeline: Callable[[dict[str, Any]], Pipeline],
    predict_stage1_for_stage2_index: Callable[
        [Any, pd.DataFrame, pd.DatetimeIndex], pd.Series
    ],
) -> float:
    """Return reconstructed-price CV MAE for a candidate stage 2 model, which uses stage 1 predictions as features."""
    stage2_params = suggest_xgb_params(trial, search_space)
    splitter = TimeSeriesSplit(n_splits=cv_splits, gap=4)
    fold_mae: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(stage2_trainval), start=1
    ):
        stage2_train_fold = stage2_trainval.iloc[train_idx].copy()
        stage2_test_fold = stage2_trainval.iloc[test_idx].copy()
        stage1_train_end = stage2_train_fold.index.max().floor("h")
        stage1_train_fold = stage1_trainval.loc[
            stage1_trainval.index <= stage1_train_end
        ]
        if stage1_train_fold.empty:
            continue

        X_stage1_train, y_stage1_train = split_x_y(
            stage1_train_fold, model_type=stage1_model
        )
        stage1_pipeline = create_stage1_pipeline(best_stage1_params)
        stage1_pipeline.fit(X_stage1_train, y_stage1_train)

        for stage2_fold in (stage2_train_fold, stage2_test_fold):
            stage2_fold["stage1_prediction"] = predict_stage1_for_stage2_index(
                stage1_pipeline,
                stage1_trainval,
                pd.DatetimeIndex(stage2_fold.index),
            )

        stage2_train_fold = stage2_train_fold.dropna(subset=["stage1_prediction"])
        stage2_test_fold = stage2_test_fold.dropna(subset=["stage1_prediction"])
        if stage2_train_fold.empty or stage2_test_fold.empty:
            continue

        X_stage2_train, y_stage2_train = split_x_y(stage2_train_fold, stage2_model)
        X_stage2_test, y_stage2_test = split_x_y(stage2_test_fold, stage2_model)
        stage1_prediction_train = X_stage2_train.pop("stage1_prediction")
        stage1_prediction_test = X_stage2_test.pop("stage1_prediction")

        stage2_pipeline = create_stage2_pipeline(stage2_params)
        stage2_pipeline.fit(X_stage2_train, y_stage2_train - stage1_prediction_train)
        stage2_prediction = stage1_prediction_test.to_numpy() + stage2_pipeline.predict(
            X_stage2_test
        )

        mae = float(mean_absolute_error(y_stage2_test, stage2_prediction))
        fold_mae.append(mae)
        trial.set_user_attr(f"fold_{fold_idx}_mae", mae)

    if not fold_mae:
        return float("inf")

    score = float(sum(fold_mae) / len(fold_mae))
    trial.set_user_attr("cv_mae", score)
    return score
