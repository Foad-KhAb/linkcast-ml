import datetime as dt
import subprocess
import sys

import pandas as pd
import pytest

from linkcast.generate import GeneratorConfig, generate, write
from linkcast.schema import read_dataset

SMALL = GeneratorConfig(
    days=70,
    n_stores=6,
    n_publishers=10,
    n_media=25,
    outage_rate=0.1,
    unverified_rate=0.05,
)


@pytest.fixture(scope="module")
def data():
    return generate(SMALL)


@pytest.fixture(scope="module")
def frames(data):
    rows = pd.DataFrame([row.to_dict() for row in data.rows])
    for column in [c for c in rows.columns if c.endswith("_at") or c == "window_end"]:
        rows[column] = pd.to_datetime(rows[column])
    return rows, {
        name: frame.apply(
            lambda col: pd.to_datetime(col) if col.name.endswith("_at") else col
        )
        for name, frame in data.sources.items()
    }


def test_same_seed_is_identical_and_other_seed_differs(data):
    again = generate(SMALL)
    assert again.rows == data.rows
    for name, frame in data.sources.items():
        pd.testing.assert_frame_equal(frame, again.sources[name])
    other = generate(GeneratorConfig(**{**SMALL.__dict__, "seed": SMALL.seed + 1}))
    assert other.rows != data.rows


def test_written_dataset_validates_and_has_every_label_state(data, tmp_path):
    out = write(data, tmp_path / "synthetic")
    rows = read_dataset(out / "link_rows.jsonl")
    assert len(rows) == len(data.rows)
    assert {row.label_status for row in rows} == {"ready", "pending", "unknown"}
    assert (out / "manifest.json").exists()
    assert (out / "truth" / "links.csv").exists()


def test_hidden_variables_stay_out_of_sources(data):
    for frame in data.sources.values():
        assert not [c for c in frame.columns if c.startswith(("hidden_", "_"))]
    assert "hidden_expected_conversions_7d" in data.truth["links"].columns


def test_ready_targets_match_approved_conversions_in_window(frames):
    rows, sources = frames
    conversions = sources["conversions"].merge(
        rows[["link_id", "created_at", "window_end"]], on="link_id"
    )
    in_window = conversions[
        (conversions.occurred_at >= conversions.created_at)
        & (conversions.occurred_at < conversions.window_end)
        & (conversions.status == "approved")
    ]
    expected = in_window.groupby("link_id").size()
    ready = rows[rows.label_status == "ready"].set_index("link_id")
    recomputed = expected.reindex(ready.index, fill_value=0)
    assert (ready.purchase_conversions_7d == recomputed).all()
    assert (ready.purchase_conversions_7d == 0).any()


def test_post_history_uses_only_posts_known_at_creation(frames):
    rows, sources = frames
    posts = sources["posts"]
    sample = rows[rows.previous_posts_7d.notna()].sample(50, random_state=0)
    for row in sample.itertuples():
        media_posts = posts[posts.media_id == row.media_id]
        known = media_posts[
            (media_posts.published_at >= row.created_at - dt.timedelta(hours=168))
            & (media_posts.published_at < row.created_at)
            & (media_posts.known_at <= row.created_at)
        ]
        assert row.previous_posts_7d == len(known)


def test_features_never_come_from_after_creation(frames):
    rows, _ = frames
    observed = rows[rows.audience_observed_at.notna()]
    assert (observed.audience_observed_at <= observed.created_at).all()
    first_week = rows.created_at < rows.created_at.min() + dt.timedelta(hours=168)
    # Before a full week of collected history exists, counts are unknown, not zero.
    assert rows[first_week].previous_posts_7d.isna().all()


def test_pending_rows_keep_partial_counts_without_targets(frames):
    rows, _ = frames
    pending = rows[rows.label_status == "pending"]
    assert pending.purchase_conversions_7d.isna().all()
    assert pending.observed_purchase_conversions.notna().all()
    unknown = rows[rows.label_status == "unknown"]
    assert set(unknown.tracking_coverage) <= {"incomplete", "unknown"}


def test_generate_command_writes_valid_dataset(tmp_path):
    out = tmp_path / "cli"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "linkcast.cli",
            "generate",
            "--out",
            str(out),
            "--days",
            "70",
            "--media",
            "15",
        ],  # fmt: skip
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Wrote" in result.stdout
    assert read_dataset(out / "link_rows.jsonl")
