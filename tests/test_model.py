import datetime as dt
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from linkcast.generate import GeneratorConfig, generate, write
from linkcast.metrics import poisson_deviance, total_error
from linkcast.model import (
    MISSING_CATEGORY,
    Params,
    evaluate,
    feature_frame,
    fit,
    model_data,
    predict,
    tune,
)
from linkcast.schema import DEFAULT_FEATURES, TARGET
from linkcast.splits import SplitConfig

DATA = GeneratorConfig(days=70, n_stores=6, n_publishers=10, n_media=25)
UTC = dt.UTC
CONFIG = SplitConfig(
    train_cutoff=dt.datetime(2026, 2, 1, tzinfo=UTC),
    validation_end=dt.datetime(2026, 2, 15, tzinfo=UTC),
    test_end=dt.datetime(2026, 3, 1, tzinfo=UTC),
)
SMALL_GRID = {"depth": (4,), "learning_rate": (0.1,)}
FAST = Params(depth=4, learning_rate=0.1, iterations=100)


@pytest.fixture(scope="module")
def rows():
    return generate(DATA).rows


@pytest.fixture(scope="module")
def data(rows):
    return model_data(rows, CONFIG, DEFAULT_FEATURES)


def test_feature_frame_contains_only_allowed_inputs_with_explicit_missing(rows):
    frame = feature_frame(rows, DEFAULT_FEATURES)
    assert list(frame.columns) == list(DEFAULT_FEATURES)
    assert TARGET not in frame.columns
    assert frame.creative_id.eq(MISSING_CATEGORY).any()  # null creative kept
    assert frame.audience_size.isna().any()  # numeric unknown stays NaN, not 0
    assert not frame.store_id.isna().any()


def test_predictions_are_positive_expected_counts(data):
    train = data[data.split == "train"]
    model = fit(train, DEFAULT_FEATURES, FAST)
    predicted = predict(model, data[data.split == "validation"], DEFAULT_FEATURES)
    assert (predicted > 0).all()
    assert abs(predicted.mean() - train[TARGET].mean()) < 1.0


def test_training_is_deterministic(data):
    train = data[data.split == "train"]
    first = predict(fit(train, DEFAULT_FEATURES, FAST), train, DEFAULT_FEATURES)
    second = predict(fit(train, DEFAULT_FEATURES, FAST), train, DEFAULT_FEATURES)
    np.testing.assert_array_equal(first, second)


def test_fit_rejects_rows_without_completed_labels(data):
    with pytest.raises(ValueError):
        fit(data[data.split == "excluded"], DEFAULT_FEATURES, FAST)


def test_test_labels_never_affect_tuning(data):
    corrupted = data.copy()
    test = corrupted.split == "test"
    assert test.any()
    corrupted.loc[test, TARGET] = 999
    original = tune(data, DEFAULT_FEATURES, SMALL_GRID)
    changed = tune(corrupted, DEFAULT_FEATURES, SMALL_GRID)
    assert original["best"] == changed["best"]


def test_evaluate_compares_feature_sets_with_baseline(rows):
    report = evaluate(rows, CONFIG, grid=SMALL_GRID)
    assert set(report["models"]) == {"default", "all"}
    assert len(report["models"]["all"]["features"]) == 29
    for model in report["models"].values():
        deviances = [r["poisson_deviance"] for r in model["tuning"]]
        assert model["best"]["poisson_deviance"] == min(deviances)
        assert model["importance"]
    assert report["baseline"]["links"] == report["validation_links"]


def test_metrics_reward_true_means_and_measure_totals():
    actual = pd.Series([0.0, 0.0, 1.0, 2.0, 7.0])
    mean = pd.Series([2.0] * 5)
    median = pd.Series([1.0] * 5)
    assert total_error(actual, mean) == 0.0
    assert total_error(actual, median) == 0.5
    assert (median - actual).abs().mean() < (mean - actual).abs().mean()
    assert poisson_deviance(actual, mean) < poisson_deviance(actual, median)
    groups = pd.Series(["a", "a", "b", "b", "b"])
    assert total_error(actual, mean, groups) == pytest.approx((4 + 4) / 10)


def test_train_command_prints_comparison(tmp_path):
    out = write(generate(DATA), tmp_path / "data")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "linkcast.cli",
            "train",
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
    assert "catboost default" in result.stdout
    assert "catboost all" in result.stdout
    assert "test untouched" in result.stdout
