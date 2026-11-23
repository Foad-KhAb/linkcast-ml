"""Scores for expected-count predictions, shared by every model.

- MAE: average absolute error in conversions per link (required, but it favors
  medians on skewed counts, so it is not the deciding score).
- Bias: mean prediction minus mean actual; near zero for good expected counts.
- Poisson deviance: quality of predicted averages for count data (lower is
  better); it rewards predicting the true expected count.
- Total error: how far predicted totals are from actual totals, overall and
  summed per group (e.g. per store or campaign), relative to actual total.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PREDICTION = 1e-9
TOTAL_GROUPS = ("store_id", "campaign_id")


def poisson_deviance(actual, predicted) -> float:
    y = np.asarray(actual, dtype=float)
    mu = np.maximum(np.asarray(predicted, dtype=float), MIN_PREDICTION)
    log_term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(np.mean(2 * (log_term - (y - mu))))


def total_error(actual: pd.Series, predicted: pd.Series, groups=None) -> float:
    """Sum of |predicted total - actual total| per group, over actual total."""
    if groups is None:
        gap = abs(predicted.sum() - actual.sum())
    else:
        sums = pd.DataFrame({"a": actual, "p": predicted}).groupby(groups).sum()
        gap = (sums.p - sums.a).abs().sum()
    return float(gap / actual.sum()) if actual.sum() else float("nan")


def score(actual: pd.Series, predicted: pd.Series, groups: pd.DataFrame) -> dict:
    """All scores for aligned actual/predicted series; groups has TOTAL_GROUPS."""
    actual = actual.astype(float)
    predicted = pd.Series(np.asarray(predicted, dtype=float), index=actual.index)
    error = predicted - actual
    result = {
        "links": int(len(actual)),
        "mae": round(float(error.abs().mean()), 4),
        "bias": round(float(error.mean()), 4),
        "poisson_deviance": round(poisson_deviance(actual, predicted), 4),
        "mean_prediction": round(float(predicted.mean()), 4),
        "mean_actual": round(float(actual.mean()), 4),
        "total_error": round(total_error(actual, predicted), 4),
    }
    for group in TOTAL_GROUPS:
        name = f"total_error_by_{group.removesuffix('_id')}"
        result[name] = round(total_error(actual, predicted, groups[group]), 4)
    return result


SCORE_COLUMNS = (
    ("mae", "MAE"),
    ("poisson_deviance", "deviance"),
    ("bias", "bias"),
    ("total_error", "total"),
    ("total_error_by_store", "by store"),
    ("total_error_by_campaign", "by camp."),
)


def score_header(label_width: int) -> str:
    cells = " ".join(f"{title:>9}" for _, title in SCORE_COLUMNS)
    return f"  {'':<{label_width}} {cells}"


def score_line(label: str, scores: dict, label_width: int) -> str:
    cells = []
    for key, _ in SCORE_COLUMNS:
        value = scores[key]
        if key == "bias":
            cells.append(f"{value:>+9.4f}")
        elif key.startswith("total"):
            cells.append(f"{value:>9.1%}")
        else:
            cells.append(f"{value:>9.4f}")
    return f"  {label:<{label_width}} " + " ".join(cells)
