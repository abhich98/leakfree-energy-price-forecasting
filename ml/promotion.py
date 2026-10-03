"""Statistical gates for promoting a candidate forecast model."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

MINIMUM_MAE_IMPROVEMENT = 0.02
MINIMUM_BOOTSTRAP_CONFIDENCE = 0.90
DEFAULT_BOOTSTRAP_SAMPLES = 10_000


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    candidate_mae: float
    baseline_mae: float
    champion_mae: float | None
    improvement_over_champion: float | None
    bootstrap_confidence: float | None
    reasons: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def _validate_aligned_series(*series: pd.Series) -> None:
    reference_index = series[0].index
    if not isinstance(reference_index, pd.DatetimeIndex):
        raise TypeError("promotion inputs require a DatetimeIndex")
    if reference_index.has_duplicates:
        raise ValueError("promotion inputs must not contain duplicate timestamps")

    for values in series:
        if not values.index.equals(reference_index):
            raise ValueError("promotion inputs must have identical timestamp indexes")
        if values.isna().any():
            raise ValueError("promotion inputs must not contain missing values")


def _daily_mae(predictions: pd.Series, actuals: pd.Series) -> pd.Series:
    absolute_errors = (predictions - actuals).abs()
    delivery_days = pd.DatetimeIndex(absolute_errors.index).normalize()
    return absolute_errors.groupby(delivery_days).mean()


def _bootstrap_confidence(
    candidate_daily_mae: pd.Series,
    champion_daily_mae: pd.Series,
    samples: int,
    random_seed: int,
) -> float:
    differences = (champion_daily_mae - candidate_daily_mae).to_numpy()
    generator = np.random.default_rng(random_seed)
    resample_indices = generator.integers(0, len(differences), size=(samples, len(differences)))
    resampled_mean_differences = differences[resample_indices].mean(axis=1)
    return float((resampled_mean_differences > 0).mean())


def evaluate_promotion(
    candidate_predictions: pd.Series,
    actuals: pd.Series,
    baseline_predictions: pd.Series,
    champion_predictions: pd.Series | None = None,
    *,
    minimum_improvement: float = MINIMUM_MAE_IMPROVEMENT,
    minimum_confidence: float = MINIMUM_BOOTSTRAP_CONFIDENCE,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    random_seed: int = 42,
) -> PromotionDecision:
    """Apply daily-block MAE gates to one candidate model stage."""
    if not 0 <= minimum_improvement < 1:
        raise ValueError("minimum_improvement must be in [0, 1)")
    if not 0 < minimum_confidence <= 1:
        raise ValueError("minimum_confidence must be in (0, 1]")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    series = [candidate_predictions, actuals, baseline_predictions]
    if champion_predictions is not None:
        series.append(champion_predictions)
    _validate_aligned_series(*series)

    candidate_daily_mae = _daily_mae(candidate_predictions, actuals)
    baseline_daily_mae = _daily_mae(baseline_predictions, actuals)
    candidate_mae = float(candidate_daily_mae.mean())
    baseline_mae = float(baseline_daily_mae.mean())
    reasons: list[str] = []

    if candidate_mae >= baseline_mae:
        reasons.append("candidate_does_not_beat_baseline")

    if champion_predictions is None:
        return PromotionDecision(
            approved=not reasons,
            candidate_mae=candidate_mae,
            baseline_mae=baseline_mae,
            champion_mae=None,
            improvement_over_champion=None,
            bootstrap_confidence=None,
            reasons=tuple(reasons),
        )

    champion_daily_mae = _daily_mae(champion_predictions, actuals)
    champion_mae = float(champion_daily_mae.mean())
    improvement = (champion_mae - candidate_mae) / champion_mae
    confidence = _bootstrap_confidence(
        candidate_daily_mae,
        champion_daily_mae,
        bootstrap_samples,
        random_seed,
    )

    if improvement < minimum_improvement:
        reasons.append("candidate_does_not_meet_minimum_improvement")
    if confidence < minimum_confidence:
        reasons.append("candidate_does_not_meet_bootstrap_confidence")

    return PromotionDecision(
        approved=not reasons,
        candidate_mae=candidate_mae,
        baseline_mae=baseline_mae,
        champion_mae=champion_mae,
        improvement_over_champion=improvement,
        bootstrap_confidence=confidence,
        reasons=tuple(reasons),
    )