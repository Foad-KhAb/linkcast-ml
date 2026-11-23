"""Historical-mean baseline with smoothed fallbacks.

For each link, start from the global training mean and refine it level by
level (platform, then store, then media). At each level the estimate blends the
group's own training mean with the estimate from the level above:

    estimate = (sum of group targets + k * parent estimate) / (group links + k)

A group with few links stays close to its parent (shrinkage); a group with many
links mostly uses its own mean. Groups absent from training keep the parent
estimate, which is how new media or stores fall back. ``k`` is the only
hyperparameter; it is chosen on validation, never on test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from linkcast.schema import TARGET, LinkRow
from linkcast.splits import SplitConfig, assign_splits

LEVELS = ("platform", "store_id", "media_id")  # general to specific
DEFAULT_K = 10.0
K_GRID = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0)


@dataclass
class HistoricalMeanBaseline:
    k: float = DEFAULT_K
    levels: tuple[str, ...] = LEVELS
    global_mean: float | None = field(default=None, init=False)
    stats: dict[str, pd.DataFrame] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("k must be nonnegative")

    def fit(self, train: pd.DataFrame) -> HistoricalMeanBaseline:
        target = train[TARGET].astype(float)
        if target.isna().any() or target.empty:
            raise ValueError("training rows need completed, non-null targets")
        self.global_mean = float(target.mean())
        self.stats = {
            level: target.groupby(train[level]).agg(["sum", "count"])
            for level in self.levels
        }
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Return the expected count and the most specific level seen in training."""
        if self.global_mean is None:
            raise ValueError("fit the baseline before predicting")
        estimate = pd.Series(self.global_mean, index=rows.index, dtype=float)
        level_used = pd.Series("global", index=rows.index, dtype=object)
        for level in self.levels:
            stats = self.stats[level]
            total = rows[level].map(stats["sum"])
            count = rows[level].map(stats["count"])
            seen = count.notna()
            denominator = count[seen] + self.k
            # k == 0 and count > 0 is a plain group mean; count is never 0 here.
            estimate[seen] = (total[seen] + self.k * estimate[seen]) / denominator
            level_used[seen] = level.removesuffix("_id")
        return pd.DataFrame({"prediction": estimate, "level_used": level_used})


def link_frame(rows: list[LinkRow], config: SplitConfig) -> pd.DataFrame:
    """Baseline inputs, targets, and split membership for every link."""
    splits = assign_splits(rows, config).set_index("link_id")
    frame = pd.DataFrame(
        {
            "link_id": [row.link_id for row in rows],
            "platform": [row.platform for row in rows],
            "store_id": [row.store_id for row in rows],
            "media_id": [row.media_id for row in rows],
            TARGET: [row.purchase_conversions_7d for row in rows],
        }
    ).set_index("link_id")
    return frame.join(splits[["split", "in_final_train"]])


def _scores(actual: pd.Series, predicted: pd.Series) -> dict:
    error = predicted - actual
    return {
        "links": int(len(actual)),
        "mae": round(float(error.abs().mean()), 4),
        "bias": round(float(error.mean()), 4),
        "mean_prediction": round(float(predicted.mean()), 4),
        "mean_actual": round(float(actual.mean()), 4),
    }


def evaluate(
    rows: list[LinkRow],
    config: SplitConfig | None = None,
    k: float = DEFAULT_K,
    grid: tuple[float, ...] = K_GRID,
) -> dict:
    """Fit on train and score on validation for the default k and a k grid.

    The test split is deliberately never read here.
    """
    config = config or SplitConfig()
    frame = link_frame(rows, config)
    train = frame[frame.split == "train"]
    validation = frame[frame.split == "validation"]
    if train.empty or validation.empty:
        raise ValueError("train and validation splits must both contain links")
    actual = validation[TARGET].astype(float)
    train_target = train[TARGET].astype(float)

    references = {
        "train_mean": {"value": round(float(train_target.mean()), 4)},
        "train_median": {"value": float(train_target.median())},
    }
    for name, reference in references.items():
        constant = pd.Series(reference["value"], index=validation.index)
        reference |= _scores(actual, constant)

    tuning = []
    for value in sorted(set(grid) | {k}):
        predicted = HistoricalMeanBaseline(value).fit(train).predict(validation)
        tuning.append({"k": value} | _scores(actual, predicted.prediction))
    best = min(tuning, key=lambda result: (result["mae"], result["k"]))
    default = next(result for result in tuning if result["k"] == k)

    chosen = HistoricalMeanBaseline(best["k"]).fit(train).predict(validation)
    by_level = {
        level: _scores(actual[chosen.level_used == level], part.prediction)
        for level, part in chosen.groupby("level_used")
    }
    return {
        "fallback_order": ["media", "store", "platform", "global"],
        "train_links": int(len(train)),
        "validation_links": int(len(validation)),
        "references": references,
        "default": default,
        "tuned": best,
        "tuning": tuning,
        "by_level_tuned": by_level,
        "final_train_links": int(frame.in_final_train.sum()),
    }


def format_report(report: dict) -> str:
    default, tuned = report["default"], report["tuned"]
    lines = [
        "Historical-mean baseline (fallback: media -> store -> platform -> global)",
        f"  fitted on train ({report['train_links']} links), "
        f"scored on validation ({report['validation_links']} links); test untouched",
        "",
        f"  {'validation':<38} {'MAE':>7} {'bias':>8}",
    ]
    for name, reference in report["references"].items():
        label = f"predict {name.replace('_', ' ')} ({reference['value']:g})"
        lines.append(
            f"  {label:<38} {reference['mae']:>7.4f} {reference['bias']:>+8.4f}"
        )
    lines.append(
        f"  {f'baseline, default k={default["k"]:g}':<38} "
        f"{default['mae']:>7.4f} {default['bias']:>+8.4f}"
    )
    lines.append(
        f"  {f'baseline, tuned k={tuned["k"]:g}':<38} "
        f"{tuned['mae']:>7.4f} {tuned['bias']:>+8.4f}"
    )
    lines += ["", "Tuning k on validation", f"  {'k':>6} {'MAE':>8} {'bias':>8}"]
    for result in report["tuning"]:
        marker = "  <- best" if result == tuned else ""
        lines.append(
            f"  {result['k']:>6g} {result['mae']:>8.4f} {result['bias']:>+8.4f}{marker}"
        )
    lines += [
        "",
        f"By fallback level used (tuned k={tuned['k']:g})",
        f"  {'level':<10} {'links':>6} {'MAE':>8} {'mean pred':>10} {'mean actual':>12}",
    ]
    for level in ("media", "store", "platform", "global"):
        if level in report["by_level_tuned"]:
            s = report["by_level_tuned"][level]
            lines.append(
                f"  {level:<10} {s['links']:>6} {s['mae']:>8.4f} "
                f"{s['mean_prediction']:>10.4f} {s['mean_actual']:>12.4f}"
            )
    lines += [
        "",
        f"For the test, refit with k={tuned['k']:g} on the final training set "
        f"({report['final_train_links']} links) and score once (step 8).",
    ]
    return "\n".join(lines)
