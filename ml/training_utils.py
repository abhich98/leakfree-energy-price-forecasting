import json
from enum import Enum
from pathlib import Path
from typing import overload

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ml.evaluate import baseline_persistence, directional_accuracy, full_evaluation_report


class ModelType(Enum):
    PRICE_HOURLY = "price_hourly"
    PRICE_QUARTER_HOURLY = "price_quarter_hourly"
    PRICE = "price"
    WIND = "wind"
    SOLAR = "solar"


def filter_raw_data(X_raw, y_raw, mode: ModelType):
    # Imported here, not at module scope: feature_engineering imports ModelType
    # from this module, so a top-level import back would be circular.

    # Run the transformer ONCE to identify NaN rows, then drop from raw indices
    if mode == ModelType.PRICE:
        from ml.features.feature_engineering import PriceModelFeatureEngineer
        feature_engineer = PriceModelFeatureEngineer()
    elif mode == ModelType.WIND or mode == ModelType.SOLAR:
        from ml.features.feature_engineering import GenerationModelFeatureEngineer
        feature_engineer = GenerationModelFeatureEngineer()

    tmp = feature_engineer.transform(X_raw)
    valid_idx = tmp.dropna().index
    del tmp
    
    X_raw = X_raw.loc[valid_idx]
    y = y_raw.loc[valid_idx]

    return X_raw, y

def create_preprocessor():
    """
    Create a ColumnTransformer for preprocessing the features.

    Returns:
        ColumnTransformer: A configured ColumnTransformer object.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num", StandardScaler(), 
                make_column_selector(dtype_include="number")
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")   # keep as DataFrame for readability
    return preprocessor

def create_ml_model():
    """
    Create an XGBRegressor model with predefined hyperparameters.

    Returns:
        XGBRegressor: A configured XGBRegressor model.
    """
    xgb_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",   # fast histogram-based training
    )
    model = XGBRegressor(**xgb_params)
    return model


@overload
def save_report(report: dict, mode: ModelType) -> None: ...
@overload
def save_report(report: dict, file_name: str) -> None: ...

def save_report(report: dict, target: ModelType | str) -> None:
    name = f"{target.value}_model_report" if isinstance(target, ModelType) else target
    file_path = f"ml/artifacts/{name}.json"
    Path("ml/artifacts").mkdir(exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(report, f, indent=2)


def evaluate_holdout(y_pred, y_true, test_baseline_pred):
    """
    Return the holdout and baseline evaluation reports.

    Args:
        y_pred (pd.Series): Predicted values for the test set.
        y_true (pd.Series): Actual values for the test set.
        test_baseline_pred (pd.Series): Baseline predictions for the test set.

    Returns:
        tuple: A tuple containing the holdout report and baseline report.
    """
    
    holdout_report = full_evaluation_report(
        y_true, y_pred, reference=test_baseline_pred,
        include_directional=True, include_peak=False,
    )
    baseline_report = baseline_persistence(test_baseline_pred, y_true)

    return holdout_report, baseline_report

def draw_predictions(y_pred, y_test, key_word: str):
    """
    Draw predictions for one of the models, right after y_pred is computed.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15, 4))
    y_test.plot(ax=ax, label="actual", alpha=0.7)
    pd.Series(y_pred, index=y_test.index).plot(ax=ax, label="predicted", alpha=0.7)
    ax.legend()
    ax.set_title(f"{key_word} — holdout")
    plt.tight_layout()
    plt.savefig(f"ml/artifacts/{key_word}_holdout.png", dpi=100)
