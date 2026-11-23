"""Final evaluation on the test split.

Every choice is made on validation first: the baseline's k, each CatBoost
feature set's depth/learning rate/number of trees, and which CatBoost feature
set is chosen. Then every model is refit on the final training set (labels
available by ``validation_end``) and scored once on ``test``. Test labels are
used only for scoring, never for any choice or fit.

Reports: overall scores, scores for links on entities new vs known to the final
training set, and predicted vs actual averages by prediction band.
"""

from __future__ import annotations

import pandas as pd

from linkcast import baseline as baseline_module
from linkcast.baseline import HistoricalMeanBaseline, link_frame
from linkcast.metrics import score, score_header, score_line
from linkcast.model import (
    FEATURE_SETS,
    GRID,
    Params,
    fit,
    model_data,
    predict,
    tune,
)
from linkcast.schema import TARGET, LinkRow
from linkcast.splits import SplitConfig, assign_splits

ENTITIES = ("store", "media", "campaign", "creative")
BANDS = 5


def predict_test(
    rows: list[LinkRow],
    config: SplitConfig,
    grid: dict = GRID,
    k_grid: tuple[float, ...] = baseline_module.K_GRID,
) -> tuple[dict, pd.DataFrame]:
    """Make every choice on validation, refit on final train, predict test.

    Returns the selection and one row per test link with its label, grouping
    IDs, new-entity flags, and each model's prediction.
    """
    # 1. Choices, on validation only.
    baseline_choice = baseline_module.evaluate(rows, config, grid=k_grid)["tuned"]
    selection = {
        name: tune(model_data(rows, config, features), features, grid)["best"]
        for name, features in FEATURE_SETS.items()
    }
    chosen = min(selection, key=lambda name: selection[name]["poisson_deviance"])

    # 2. Refit on the final training set; predict the test split.
    splits = assign_splits(rows, config).set_index("link_id")
    test_index = splits.index[splits.split == "test"]
    if test_index.empty:
        raise ValueError("test split is empty")
    predictions = splits.loc[
        test_index, [TARGET, "store_id", "campaign_id"] + [f"new_{e}" for e in ENTITIES]
    ].copy()

    ids = link_frame(rows, config)
    final_baseline = HistoricalMeanBaseline(baseline_choice["k"]).fit(
        ids[ids.in_final_train]
    )
    predictions["baseline"] = final_baseline.predict(ids.loc[test_index]).prediction

    for name, features in FEATURE_SETS.items():
        data = model_data(rows, config, features)
        best = selection[name]
        params = Params(
            depth=best["depth"],
            learning_rate=best["learning_rate"],
            iterations=best["iterations"],
        )
        model = fit(data[data.in_final_train], features, params)
        predictions[f"catboost_{name}"] = predict(model, data.loc[test_index], features)

    selection = {
        "baseline_k": baseline_choice["k"],
        "catboost": selection,
        "chosen": f"catboost_{chosen}",
        "rule": "lowest validation Poisson deviance",
        "final_train_links": int(splits.in_final_train.sum()),
    }
    return selection, predictions


def evaluate(
    rows: list[LinkRow],
    config: SplitConfig | None = None,
    grid: dict = GRID,
    k_grid: tuple[float, ...] = baseline_module.K_GRID,
) -> dict:
    """Score the test split once with every model chosen on validation."""
    config = config or SplitConfig()
    selection, predictions = predict_test(rows, config, grid, k_grid)
    chosen = selection["chosen"]

    # 3. Score test once.
    models = ["baseline"] + [f"catboost_{name}" for name in FEATURE_SETS]
    groups = predictions[["store_id", "campaign_id"]]
    actual = predictions[TARGET].astype(float)

    def scores(model: str, mask=None) -> dict:
        part = predictions if mask is None else predictions[mask]
        return score(actual.loc[part.index], part[model], groups.loc[part.index])

    report = {
        "selection": {k: v for k, v in selection.items() if k != "final_train_links"},
        "final_train_links": selection["final_train_links"],
        "test_links": int(len(predictions)),
        "test_period": [
            config.validation_end.isoformat(),
            config.test_end.isoformat(),
        ],
        "overall": {model: scores(model) for model in models},
        "segments": {},
        "bands": _bands(predictions, actual, chosen, models),
    }
    for entity in ENTITIES:
        flag = predictions[f"new_{entity}"]
        report["segments"][entity] = {
            state: {
                model: scores(model, flag.fillna(False) == value) for model in models
            }
            for state, value in (("known", False), ("new", True))
            if ((flag.fillna(False) == value) & flag.notna()).any()
        }
    return report


def _bands(predictions, actual, chosen: str, models: list[str]) -> list[dict]:
    """Group test links by the chosen model's prediction; compare averages."""
    band = pd.qcut(predictions[chosen].rank(method="first"), BANDS, labels=False)
    rows = []
    for value in range(BANDS):
        mask = band == value
        part = predictions[mask]
        rows.append(
            {
                "band": value + 1,
                "links": int(mask.sum()),
                "prediction_range": [
                    round(float(part[chosen].min()), 3),
                    round(float(part[chosen].max()), 3),
                ],
                "mean_actual": round(float(actual[mask].mean()), 4),
            }
            | {f"mean_{model}": round(float(part[model].mean()), 4) for model in models}
        )
    return rows


def format_report(report: dict) -> str:
    width = 22
    selection = report["selection"]
    lines = [
        "Final evaluation on the test split (used once)",
        f"  test: links created {report['test_period'][0][:10]} to "
        f"{report['test_period'][1][:10]} ({report['test_links']} links)",
        f"  models refit on the final training set ({report['final_train_links']} links)",
        "",
        "Choices made on validation before reading test labels",
        f"  baseline k = {selection['baseline_k']:g}",
    ]
    for name, best in selection["catboost"].items():
        lines.append(
            f"  catboost {name}: depth {best['depth']}, learning rate "
            f"{best['learning_rate']:g}, {best['iterations']} trees "
            f"(validation deviance {best['poisson_deviance']:.4f})"
        )
    lines += [
        f"  chosen model: {selection['chosen']} ({selection['rule']})",
        "",
        "Test scores",
        score_header(width),
    ]
    for model, scores in report["overall"].items():
        marker = " *" if model == selection["chosen"] else ""
        lines.append(score_line(model + marker, scores, width))

    lines += ["", "Test scores for links on entities new vs known to final training"]
    for entity, states in report["segments"].items():
        for state, models in states.items():
            links = next(iter(models.values()))["links"]
            lines.append(f"  {entity}: {state} ({links} links)")
            lines.append(score_header(width))
            for model, scores in models.items():
                lines.append(score_line(model, scores, width))

    lines += [
        "",
        f"Predicted vs actual averages by band of {selection['chosen']} predictions",
        f"  {'band':>4} {'links':>6} {'prediction range':>18} {'actual':>8} "
        + " ".join(f"{name:>18}" for name in report["overall"]),
    ]
    for band in report["bands"]:
        low, high = band["prediction_range"]
        cells = " ".join(f"{band[f'mean_{name}']:>18.3f}" for name in report["overall"])
        lines.append(
            f"  {band['band']:>4} {band['links']:>6} {f'{low:.2f}-{high:.2f}':>18} "
            f"{band['mean_actual']:>8.3f} {cells}"
        )
    lines += [
        "",
        "Synthetic data: these results show pipeline behavior, not business value.",
    ]
    return "\n".join(lines)
