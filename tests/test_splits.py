import datetime as dt
import json
import subprocess
import sys

import pandas as pd
import pytest

from linkcast.generate import GeneratorConfig, generate, write
from linkcast.splits import SplitConfig, assign_splits, summarize

# 70 days from 2026-01-01: labels are mostly complete through early March.
DATA = GeneratorConfig(days=70, n_stores=6, n_publishers=10, n_media=25)
UTC = dt.UTC
CONFIG = SplitConfig(
    train_cutoff=dt.datetime(2026, 2, 1, tzinfo=UTC),
    validation_end=dt.datetime(2026, 2, 15, tzinfo=UTC),
    test_end=dt.datetime(2026, 3, 1, tzinfo=UTC),
)


@pytest.fixture(scope="module")
def rows():
    return generate(DATA).rows


@pytest.fixture(scope="module")
def splits(rows):
    return assign_splits(rows, CONFIG)


@pytest.fixture(scope="module")
def by_id(rows):
    return {row.link_id: row for row in rows}


def test_every_link_gets_exactly_one_split(rows, splits):
    assert len(splits) == len(rows)
    assert splits.link_id.is_unique
    assert set(splits.split) <= {"train", "validation", "test", "excluded"}


def test_train_rows_have_labels_available_by_cutoff(splits, by_id):
    train = splits[splits.split == "train"]
    assert len(train)
    for link_id in train.link_id:
        row = by_id[link_id]
        assert row.label_status == "ready"
        assert row.label_available_at <= CONFIG.train_cutoff


def test_ready_links_just_before_cutoff_fall_in_the_gap(splits, by_id):
    gap = splits[splits.reason == "label_after_cutoff"]
    assert len(gap)  # links created days before the cutoff, labels known after it
    for link_id in gap.link_id:
        row = by_id[link_id]
        assert row.created_at < CONFIG.train_cutoff
        assert row.label_available_at > CONFIG.train_cutoff
    train_end = splits[splits.split == "train"].created_at.max()
    assert train_end < pd.Timestamp(CONFIG.train_cutoff - dt.timedelta(days=7))


def test_evaluation_rows_are_ready_and_inside_their_period(splits):
    validation = splits[splits.split == "validation"]
    test = splits[splits.split == "test"]
    assert (validation.label_status == "ready").all()
    assert (test.label_status == "ready").all()
    assert validation.created_at.between(
        CONFIG.train_cutoff, CONFIG.validation_end, inclusive="left"
    ).all()
    assert test.created_at.between(
        CONFIG.validation_end, CONFIG.test_end, inclusive="left"
    ).all()


def test_unknown_and_pending_labels_are_never_used(splits):
    usable = splits[splits.split != "excluded"]
    assert set(usable.label_status) == {"ready"}
    assert not splits[splits.label_status != "ready"].in_final_train.any()


def test_final_train_adds_validation_rows_but_never_test_rows(splits):
    final = splits[splits.in_final_train]
    assert set(splits[splits.split == "train"].link_id) <= set(final.link_id)
    assert (final.split == "validation").any()
    assert not (final.split == "test").any()


def test_new_entity_flags_compare_with_the_right_training_set(splits):
    train_media = set(splits[splits.split == "train"].media_id)
    final_media = set(splits[splits.in_final_train].media_id)
    validation = splits[splits.split == "validation"]
    test = splits[splits.split == "test"]
    assert (validation.new_media == ~validation.media_id.isin(train_media)).all()
    assert (test.new_media == ~test.media_id.isin(final_media)).all()
    assert splits[splits.split == "train"].new_media.isna().all()


def test_invalid_dates_are_rejected():
    with pytest.raises(ValueError):
        SplitConfig(
            train_cutoff=dt.datetime(2026, 3, 1, tzinfo=UTC),
            validation_end=dt.datetime(2026, 2, 1, tzinfo=UTC),
        )
    with pytest.raises(ValueError):
        SplitConfig(train_cutoff=dt.datetime(2026, 5, 1))


def test_summary_warns_when_test_period_is_still_pending(rows):
    late = SplitConfig(
        train_cutoff=dt.datetime(2026, 2, 15, tzinfo=UTC),
        validation_end=dt.datetime(2026, 3, 1, tzinfo=UTC),
        test_end=dt.datetime(2026, 3, 12, tzinfo=UTC),
    )
    summary = summarize(assign_splits(rows, late), late)
    assert any("pending" in warning for warning in summary["warnings"])
    assert not summarize(assign_splits(rows, CONFIG), CONFIG)["warnings"]


def test_split_command_writes_csv_and_summary(tmp_path):
    out = write(generate(DATA), tmp_path / "data")
    summary_path = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "linkcast.cli",
            "split",
            str(out / "link_rows.jsonl"),
            "--train-cutoff",
            "2026-02-01",
            "--validation-end",
            "2026-02-15",
            "--test-end",
            "2026-03-01",
            "--json",
            str(summary_path),
        ],  # fmt: skip
        capture_output=True,
        text=True,
        check=True,
    )
    written = pd.read_csv(out / "splits.csv")
    assert "purchase_conversions_7d" not in written.columns
    summary = json.loads(summary_path.read_text())
    assert summary["splits"]["train"]["links"] == (written.split == "train").sum()
