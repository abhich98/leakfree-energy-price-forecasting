import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def regression_metrics(y_true, y_pred) -> dict:
    """Standard regression metrics: MAE, RMSE, R²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.abs(y_true - y_pred).mean()
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def directional_accuracy(y_true, y_pred) -> float:
    """Hour-to-hour direction agreement."""
    y_true = pd.Series(np.asarray(y_true, dtype=float))
    y_pred = pd.Series(np.asarray(y_pred, dtype=float))
    return float(((y_pred.diff() * y_true.diff()) > 0).mean())


def deviation_directional_accuracy(y_true, y_pred, reference) -> float:
    """Direction of departure from a reference series (e.g., yesterday same hour)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    reference = np.asarray(reference, dtype=float)
    model_up = y_pred > reference
    actual_up = y_true > reference
    return float((model_up == actual_up).mean())


def peak_mae(y_true, y_pred, top_pct: float = 0.9) -> float:
    """MAE restricted to the top decile of actuals."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    threshold = np.quantile(y_true, top_pct)
    mask = y_true > threshold
    return float(np.abs(y_true[mask] - y_pred[mask]).mean())


def baseline_persistence(reference_series, y_true) -> dict:
    """Compute metrics for the naive persistence baseline."""
    metrics = regression_metrics(y_true, reference_series)
    metrics["directional_accuracy"] = directional_accuracy(y_true, reference_series)
    return metrics


def full_evaluation_report(
    y_true,
    y_pred,
    reference=None,
    include_directional: bool = True,
    include_peak: bool = True,
) -> dict:
    """One-call assembly of all metrics into a dict ready for JSON."""
    report = regression_metrics(y_true, y_pred)
    if include_directional:
        if reference is None:
            raise ValueError("reference is required when include_directional=True")
        report["directional_accuracy"] = directional_accuracy(y_true, y_pred)
        report["deviation_directional_accuracy"] = deviation_directional_accuracy(
            y_true, y_pred, reference
        )
    if include_peak:
        report["peak_mae"] = peak_mae(y_true, y_pred)
    return report
