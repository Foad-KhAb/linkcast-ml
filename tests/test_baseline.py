import datetime as dt
import subprocess
import sys

import pandas as pd
import pytest

from linkcast.baseline import HistoricalMeanBaseline, evaluate, link_frame
from linkcast.generate import GeneratorConfig, generate, write
from linkcast.schema import TARGET
from linkcast.splits import SplitConfig

DATA = GeneratorConfig(days=70, n_stores=6, n_publishers=10, n_media=25)
UTC = dt.UTC
CONFIG = SplitConfig(
    train_cutoff=dt.datetime(2026, 2, 1, tzinfo=UTC),
    validation_end=dt.datetime(2026, 2, 15, tzinfo=UTC),
    test_end=dt.datetime(2026, 3, 1, tzinfo=UTC),
)


def toy(rows):
    return pd.DataFrame(rows, columns=["platform", "store_id", "media_id", TARGET])


TRAIN = toy(
    [
        ("telegram", "s1", "m1", 0),
        ("telegram", "s1", "m1", 4),
        ("telegram", "s1", "m2", 2),
        ("instagram", "s2", "m3", 6),
    ]
)


@pytest.fixture(scope="module")
def rows():
    return generate(DATA).rows


def test_k_zero_uses_plain_group_means():
    predicted = HistoricalMeanBaseline(0).fit(TRAIN).predict(TRAIN.iloc[[0, 2, 3]])
    assert predicted.prediction.tolist() == [2.0, 2.0, 6.0]
    assert set(predicted.level_used) == {"media"}


def test_unseen_entities_fall_back_level_by_level():
    new = toy(
        [
            ("telegram", "s1", "new_media", None),
            ("telegram", "new_store", "new_media", None),
            ("new_platform", "new_store", "new_media", None),
        ]
    )
    predicted = HistoricalMeanBaseline(0).fit(TRAIN).predict(new)
    assert predicted.level_used.tolist() == ["store", "platform", "global"]
    assert predicted.prediction.tolist() == [2.0, 2.0, 3.0]


def test_smoothing_pulls_small_groups_toward_their_parent():
    one_link = toy([("instagram", "s2", "m3", None)])
    raw = HistoricalMeanBaseline(0).fit(TRAIN).predict(one_link).prediction[0]
    smooth = HistoricalMeanBaseline(10).fit(TRAIN).predict(one_link).prediction[0]
    heavy = HistoricalMeanBaseline(1e9).fit(TRAIN).predict(one_link).prediction[0]
    assert raw == 6.0
    assert 3.0 < smooth < raw  # between the global mean (3) and the media mean
    assert heavy == pytest.approx(3.0)


def test_fit_rejects_missing_targets_and_predict_requires_fit():
    with pytest.raises(ValueError):
        HistoricalMeanBaseline().fit(toy([("telegram", "s1", "m1", None)]))
    with pytest.raises(ValueError):
        HistoricalMeanBaseline().predict(TRAIN)
    with pytest.raises(ValueError):
        HistoricalMeanBaseline(-1)


def test_baseline_never_learns_from_validation_or_test_labels(rows):
    frame = link_frame(rows, CONFIG)
    train = frame[frame.split == "train"]
    later = frame[frame.split.isin(["validation", "test"])]
    baseline = HistoricalMeanBaseline(0).fit(train)
    before = baseline.predict(later).prediction
    # Corrupting every non-training label must not change predictions.
    corrupted = frame.copy()
    corrupted.loc[later.index, TARGET] = 999
    refit = HistoricalMeanBaseline(0).fit(corrupted[corrupted.split == "train"])
    pd.testing.assert_series_equal(before, refit.predict(later).prediction)


def test_evaluate_reports_default_tuned_and_references(rows):
    report = evaluate(rows, CONFIG, k=10, grid=(0, 3, 30))
    assert [r["k"] for r in report["tuning"]] == [0, 3, 10, 30]
    assert report["default"]["k"] == 10
    assert report["tuned"]["mae"] == min(r["mae"] for r in report["tuning"])
    assert set(report["references"]) == {"train_mean", "train_median"}
    assert (
        sum(level["links"] for level in report["by_level_tuned"].values())
        == (report["validation_links"])
    )


def test_baseline_command_prints_report(tmp_path):
    out = write(generate(DATA), tmp_path / "data")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "linkcast.cli",
            "baseline",
            str(out / "link_rows.jsonl"),
            "--train-cutoff",
            "2026-02-01",
            "--validation-end",
            "2026-02-15",
            "--test-end",
            "2026-03-01",
            "--grid",
            "0,10",
        ],  # fmt: skip
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Tuning k on validation" in result.stdout
    assert "test untouched" in result.stdout
