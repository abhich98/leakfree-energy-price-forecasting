import pandas as pd
import pytest

from ml.promotion import evaluate_candidate_rejection, evaluate_promotion


def _series(values: list[float]) -> pd.Series:
    return pd.Series(
        values,
        index=pd.date_range("2026-08-01", periods=len(values), freq="D"),
    )


def test_initial_champion_is_approved_when_candidate_beats_baseline():
    decision = evaluate_promotion(
        candidate_predictions=_series([10.0, 10.0, 10.0, 10.0]),
        actuals=_series([11.0, 11.0, 11.0, 11.0]),
        baseline_predictions=_series([14.0, 14.0, 14.0, 14.0]),
    )

    assert decision.approved
    assert decision.champion_mae is None
    assert decision.reasons == ()


def test_candidate_is_approved_when_it_beats_champion_by_two_percent():
    actuals = _series([100.0] * 28)
    decision = evaluate_promotion(
        candidate_predictions=_series([101.0] * 28),
        actuals=actuals,
        baseline_predictions=_series([105.0] * 28),
        champion_predictions=_series([103.0] * 28),
    )

    assert decision.approved
    assert decision.improvement_over_champion == pytest.approx(2 / 3)
    assert decision.bootstrap_confidence == 1.0


def test_candidate_is_rejected_when_it_misses_minimum_improvement():
    actuals = _series([100.0] * 28)
    decision = evaluate_promotion(
        candidate_predictions=_series([102.0] * 28),
        actuals=actuals,
        baseline_predictions=_series([105.0] * 28),
        champion_predictions=_series([102.01] * 28),
    )

    assert not decision.approved
    assert "candidate_does_not_meet_minimum_improvement" in decision.reasons


def test_candidate_rejection_accepts_small_regression_within_tolerance():
    actuals = _series([100.0] * 28)

    decision = evaluate_candidate_rejection(
        candidate_predictions=_series([102.03] * 28),
        actuals=actuals,
        baseline_predictions=_series([105.0] * 28),
        champion_predictions=_series([102.0] * 28),
    )

    assert decision.approved
    assert decision.reasons == ()
    assert decision.bootstrap_confidence == 0.0


def test_candidate_rejection_rejects_material_regression_with_bootstrap_confidence():
    actuals = _series([100.0] * 28)

    decision = evaluate_candidate_rejection(
        candidate_predictions=_series([103.0] * 28),
        actuals=actuals,
        baseline_predictions=_series([105.0] * 28),
        champion_predictions=_series([102.0] * 28),
    )

    assert not decision.approved
    assert decision.reasons == ("candidate_regression_exceeds_tolerance",)
    assert decision.bootstrap_confidence == 1.0


def test_candidate_rejection_rejects_candidate_worse_than_baseline():
    actuals = _series([100.0] * 28)

    decision = evaluate_candidate_rejection(
        candidate_predictions=_series([103.0] * 28),
        actuals=actuals,
        baseline_predictions=_series([101.0] * 28),
        champion_predictions=_series([104.0] * 28),
    )

    assert not decision.approved
    assert decision.reasons == ("candidate_worse_than_baseline",)


def test_candidate_rejection_rejects_initial_candidate_worse_than_baseline():
    decision = evaluate_candidate_rejection(
        candidate_predictions=_series([105.0] * 4),
        actuals=_series([100.0] * 4),
        baseline_predictions=_series([101.0] * 4),
    )

    assert not decision.approved
    assert decision.champion_mae is None
    assert decision.reasons == ("candidate_worse_than_baseline",)


def test_candidate_is_rejected_when_input_indexes_do_not_align():
    with pytest.raises(ValueError, match="identical timestamp indexes"):
        evaluate_promotion(
            candidate_predictions=_series([1.0, 2.0]),
            actuals=_series([1.0, 2.0]),
            baseline_predictions=pd.Series(
                [1.0, 2.0], index=pd.date_range("2026-09-01", periods=2, freq="D")
            ),
        )