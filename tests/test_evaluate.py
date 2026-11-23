import dataclasses
import datetime as dt
import json
import subprocess
import sys

import pandas as pd
import pytest

from linkcast.evaluate import evaluate, format_report, predict_test
from linkcast.generate import GeneratorConfig, generate, write
from linkcast.schema import TARGET
from linkcast.splits import SplitConfig, assign_splits

DATA = GeneratorConfig(days=70, n_stores=6, n_publishers=10, n_media=25)
UTC = dt.UTC
CONFIG = SplitConfig(
    train_cutoff=dt.datetime(2026, 2, 1, tzinfo=UTC),
    validation_end=dt.datetime(2026, 2, 15, tzinfo=UTC),
    test_end=dt.datetime(2026, 3, 1, tzinfo=UTC),
)
GRID = {"depth": (4,), "learning_rate": (0.1,)}
K_GRID = (1.0, 10.0)
MODELS = ["baseline", "catboost_default", "catboost_all"]


@pytest.fixture(scope="module")
def rows():
    return generate(DATA).rows


@pytest.fixture(scope="module")
def result(rows):
    return predict_test(rows, CONFIG, GRID, K_GRID)


@pytest.fixture(scope="module")
def report(rows):
    return evaluate(rows, CONFIG, GRID, K_GRID)


def test_predictions_cover_exactly_the_test_split(rows, result):
    _, predictions = result
    splits = assign_splits(rows, CONFIG)
    assert set(predictions.index) == set(splits[splits.split == "test"].link_id)
    for model in MODELS:
        assert (predictions[model] > 0).all()


def test_test_labels_never_influence_choices_or_predictions(rows, result):
    test_ids = set(predictions_index := result[1].index)
    corrupted = [
        dataclasses.replace(row, purchase_conversions_7d=999)
        if row.link_id in test_ids
        else row
        for row in rows
    ]
    selection, predictions = predict_test(corrupted, CONFIG, GRID, K_GRID)
    assert selection == result[0]
    pd.testing.assert_frame_equal(
        predictions[MODELS], result[1].loc[predictions_index, MODELS]
    )
    assert (predictions[TARGET] == 999).all()


def test_chosen_model_has_lowest_validation_deviance(report):
    selection = report["selection"]
    deviances = {
        f"catboost_{name}": best["poisson_deviance"]
        for name, best in selection["catboost"].items()
    }
    assert selection["chosen"] == min(deviances, key=deviances.get)
    assert selection["baseline_k"] in K_GRID


def test_segments_split_test_links_into_known_and_new(report):
    for entity, states in report["segments"].items():
        totals = sum(states[state]["baseline"]["links"] for state in states)
        expected = report["test_links"]
        if entity == "creative":  # links without a creative have no flag
            assert totals <= expected
        else:
            assert totals == expected


def test_bands_cover_all_test_links_and_increase(report):
    bands = report["bands"]
    assert sum(band["links"] for band in bands) == report["test_links"]
    chosen = f"mean_{report['selection']['chosen']}"
    means = [band[chosen] for band in bands]
    assert means == sorted(means)
    json.dumps(report)
    assert "used once" in format_report(report)


def test_evaluate_command_prints_report(tmp_path):
    out = write(generate(DATA), tmp_path / "data")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "linkcast.cli",
            "evaluate",
            str(out / "link_rows.jsonl"),
            "--train-cutoff",
            "2026-02-01",
            "--validation-end",
            "2026-02-15",
            "--test-end",
            "2026-03-01",
        ],  # fmt: skip
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Test scores" in result.stdout
    assert "chosen model" in result.stdout
