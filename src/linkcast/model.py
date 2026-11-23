"""CatBoost count model for expected seven-day conversions.

CatBoost fits many small trees in sequence (gradient boosting), each correcting
the previous ones. Settings used here:

- Poisson loss: the target is a count, and predictions are expected counts
  (positive, fractional averages), which is what budgeting needs.
- Categorical IDs (media, store, ...) are encoded by CatBoost with ordered
  target statistics, so a row's encoding never uses its own label.
- Missing values stay missing: numeric nulls are NaN, which CatBoost handles
  natively; categorical nulls become an explicit "missing" category.
- Early stopping on validation picks the number of trees; a small grid of tree
  depth and learning rate is compared by validation Poisson deviance.

Only ``train`` labels are fitted and only ``validation`` is scored. The test
split is never read in this module's evaluation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from linkcast.baseline import HistoricalMeanBaseline
from linkcast.metrics import score, score_header, score_line
from linkcast.schema import (
    CATEGORICAL_FEATURES,
    DEFAULT_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    LinkRow,
)
from linkcast.splits import SplitConfig, assign_splits

FEATURE_SETS = {
    "default": DEFAULT_FEATURES,
    "all": CATEGORICAL_FEATURES + NUMERIC_FEATURES,
}
MISSING_CATEGORY = "__missing__"
GRID = {"depth": (4, 6, 8), "learning_rate": (0.03, 0.1)}
MAX_ITERATIONS = 3000
EARLY_STOPPING_ROUNDS = 100
SEED = 0
# Scoring needs IDs for totals even when a feature set does not use them.
GROUP_COLUMNS = ("store_id", "campaign_id")


@dataclass(frozen=True)
class Params:
    depth: int = 6
    learning_rate: float = 0.1
    l2_leaf_reg: float = 3.0
    iterations: int = MAX_ITERATIONS


def feature_frame(rows: list[LinkRow], features: tuple[str, ...]) -> pd.DataFrame:
    """Model inputs only, via the schema allowlist; nulls stay explicit."""
    frame = pd.DataFrame(
        [row.model_inputs(features) for row in rows],
        index=pd.Index([row.link_id for row in rows], name="link_id"),
    )
    for column in features:
        if column in CATEGORICAL_FEATURES:
            frame[column] = frame[column].astype(object).fillna(MISSING_CATEGORY)
            frame[column] = frame[column].astype(str)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    return frame


def model_data(
    rows: list[LinkRow], config: SplitConfig, features: tuple[str, ...]
) -> pd.DataFrame:
    """Features plus target, split membership, and grouping IDs for scoring."""
    splits = assign_splits(rows, config).set_index("link_id")
    extra = splits[["split", "in_final_train", TARGET, *GROUP_COLUMNS]]
    extra = extra.rename(columns={column: f"_{column}" for column in GROUP_COLUMNS})
    return feature_frame(rows, features).join(extra)


def _pool(frame: pd.DataFrame, features: tuple[str, ...], labeled: bool = True) -> Pool:
    categorical = [f for f in features if f in CATEGORICAL_FEATURES]
    label = frame[TARGET].astype(float) if labeled else None
    return Pool(frame[list(features)], label=label, cat_features=categorical)


def fit(
    train: pd.DataFrame,
    features: tuple[str, ...],
    params: Params,
    validation: pd.DataFrame | None = None,
) -> CatBoostRegressor:
    """Fit on labeled rows; with validation, stop when it stops improving."""
    if train[TARGET].isna().any():
        raise ValueError("training rows need completed, non-null targets")
    model = CatBoostRegressor(
        loss_function="Poisson",
        eval_metric="Poisson",
        depth=params.depth,
        learning_rate=params.learning_rate,
        l2_leaf_reg=params.l2_leaf_reg,
        iterations=params.iterations,
        random_seed=SEED,
        thread_count=4,  # fixed for reproducible results across machines
        verbose=False,
        allow_writing_files=False,
    )
    if validation is None:
        model.fit(_pool(train, features))
    else:
        model.fit(
            _pool(train, features),
            eval_set=_pool(validation, features),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )
    return model


def predict(model: CatBoostRegressor, frame: pd.DataFrame, features) -> np.ndarray:
    """Expected conversion counts (never negative)."""
    pool = _pool(frame, features, labeled=False)
    return model.predict(pool, prediction_type="Exponent")


def _groups(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({column: frame[f"_{column}"] for column in GROUP_COLUMNS})


def tune(data: pd.DataFrame, features: tuple[str, ...], grid: dict = GRID) -> dict:
    """Fit each grid setting on train with early stopping; score validation."""
    train = data[data.split == "train"]
    validation = data[data.split == "validation"]
    if train.empty or validation.empty:
        raise ValueError("train and validation splits must both contain links")
    actual = validation[TARGET]
    results = []
    best_model = None
    for depth, learning_rate in itertools.product(grid["depth"], grid["learning_rate"]):
        params = Params(depth=depth, learning_rate=learning_rate)
        model = fit(train, features, params, validation)
        predicted = predict(model, validation, features)
        result = {
            "depth": depth,
            "learning_rate": learning_rate,
            "iterations": int(model.get_best_iteration() + 1),
        } | score(actual, predicted, _groups(validation))
        results.append(result)
        key = (result["poisson_deviance"], depth, learning_rate)
        if best_model is None or key < best_model[0]:
            best_model = (key, result, model)
    _, best, model = best_model
    importance = model.get_feature_importance(prettified=True)
    return {
        "features": list(features),
        "tuning": results,
        "best": best,
        "importance": {
            str(row["Feature Id"]): round(float(row["Importances"]), 2)
            for _, row in importance.iterrows()
        },
    }


def evaluate(
    rows: list[LinkRow],
    config: SplitConfig | None = None,
    feature_sets: dict[str, tuple[str, ...]] = FEATURE_SETS,
    grid: dict = GRID,
    baseline_k: float = 1.0,
) -> dict:
    """Compare CatBoost feature sets with the baseline on validation."""
    config = config or SplitConfig()
    report = {"models": {}, "baseline_k": baseline_k}
    for name, features in feature_sets.items():
        data = model_data(rows, config, features)
        report["models"][name] = tune(data, features, grid)

    # Baseline on the same validation rows, fitted on the same training rows.
    any_data = model_data(rows, config, ("platform",))
    ids = pd.DataFrame(
        {
            "platform": [row.platform for row in rows],
            "store_id": [row.store_id for row in rows],
            "media_id": [row.media_id for row in rows],
        },
        index=pd.Index([row.link_id for row in rows], name="link_id"),
    ).join(any_data[["split", TARGET]])
    train = ids[ids.split == "train"]
    validation = ids[ids.split == "validation"]
    baseline = HistoricalMeanBaseline(baseline_k).fit(train).predict(validation)
    report["baseline"] = score(
        validation[TARGET],
        baseline.prediction,
        _groups(any_data.loc[validation.index]),
    )
    report["train_links"] = int(len(train))
    report["validation_links"] = int(len(validation))
    report["final_train_links"] = int(any_data.in_final_train.sum())
    return report


def format_report(report: dict) -> str:
    width = 34
    lines = [
        "CatBoost count model (Poisson loss), compared on validation",
        f"  fitted on train ({report['train_links']} links), scored on validation "
        f"({report['validation_links']} links); test untouched",
        "  settings chosen by Poisson deviance; trees chosen by early stopping",
        "",
        score_header(width),
        score_line(f"baseline (k={report['baseline_k']:g})", report["baseline"], width),
    ]
    for name, model in report["models"].items():
        best = model["best"]
        label = f"catboost {name} ({len(model['features'])} features)"
        lines.append(score_line(label, best, width))

    for name, model in report["models"].items():
        lines += [
            "",
            f"Tuning: {name} features",
            f"  {'depth':>5} {'lr':>5} {'trees':>6}"
            + score_header(0)[2:].replace(" ", " ", 1),
        ]
        for result in model["tuning"]:
            marker = "  <- best" if result == model["best"] else ""
            prefix = (
                f"  {result['depth']:>5} {result['learning_rate']:>5g} "
                f"{result['iterations']:>6}"
            )
            lines.append(prefix + score_line("", result, 0)[2:] + marker)
        lines += ["", f"Feature importance: {name} (share of prediction changes, %)"]
        for feature, value in model["importance"].items():
            if value > 0:
                lines.append(f"  {feature:<42} {value:>6.2f}")
    lines += [
        "",
        "For the test (step 8), refit the chosen model on the final training set "
        f"({report['final_train_links']} links) with its chosen number of trees.",
    ]
    return "\n".join(lines)
