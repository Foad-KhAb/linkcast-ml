import datetime as dt
import json
import math
from pathlib import Path

import pytest

from linkcast.schema import DEFAULT_FEATURES, LinkRow, SchemaError, read_dataset

EXAMPLE = Path(__file__).parents[1] / "examples" / "link_rows_v1.jsonl"


@pytest.fixture
def record():
    return json.loads(EXAMPLE.read_text().splitlines()[0])


def test_examples_keep_zero_pending_and_unknown_separate():
    ready, pending, unknown = read_dataset(EXAMPLE)
    assert ready.purchase_conversions_7d == 0
    assert pending.purchase_conversions_7d is None
    assert pending.observed_purchase_conversions == 2
    assert unknown.purchase_conversions_7d is None
    assert not unknown.eligible_for_training(dt.datetime(2026, 2, 1, tzinfo=dt.UTC))


def test_training_cutoff_requires_label_availability(record):
    row = LinkRow.from_dict(record)
    assert not row.eligible_for_training(row.window_end)
    assert not row.eligible_for_training(
        row.label_available_at - dt.timedelta(seconds=1)
    )
    assert row.eligible_for_training(row.label_available_at)


def test_roundtrip_preserves_nullable_fields_and_timestamps(record):
    row = LinkRow.from_dict(record)
    assert LinkRow.from_dict(json.loads(json.dumps(row.to_dict()))) == row


def test_model_inputs_are_allowlisted_and_preserve_missing_values():
    row = read_dataset(EXAMPLE)[1]
    inputs = row.model_inputs()
    assert tuple(inputs) == DEFAULT_FEATURES
    assert inputs["audience_size"] is None
    assert "link_id" not in inputs
    assert "purchase_conversions_7d" not in inputs
    assert "observed_purchase_conversions" not in inputs


@pytest.mark.parametrize(
    "name",
    [
        "purchase_conversions_7d",
        "observed_purchase_conversions",
        "asset_ref",
        "snapshot_at",
        "publisher_id",
        "hidden_reach",
    ],
)
def test_model_inputs_reject_audit_labels_and_simulator_fields(record, name):
    with pytest.raises(SchemaError, match="Not model features"):
        LinkRow.from_dict(record).model_inputs((name,))


def test_engagement_coverage_uses_only_measured_posts(record):
    row = LinkRow.from_dict(record)
    assert row.model_inputs(("previous_posts_view_coverage_7d",)) == {
        "previous_posts_view_coverage_7d": 0.4
    }


def test_no_posts_gives_unknown_view_coverage_not_zero(record):
    record.update(
        previous_posts_7d=0,
        previous_posts_with_views_7d=0,
        previous_posts_median_observed_views_7d=None,
    )
    assert (
        LinkRow.from_dict(record).model_inputs(("previous_posts_view_coverage_7d",))[
            "previous_posts_view_coverage_7d"
        ]
        is None
    )


def test_calendar_features_use_utc_for_offset_timestamps(record):
    record["created_at"] = "2026-01-01T12:30:00+03:30"
    assert LinkRow.from_dict(record).model_inputs()["creation_hour_utc"] == 9


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("schema_version", "2.0"),
        ("store_id", 12),
        ("store_id", " "),
        ("created_at", "2026-01-01T09:00:00"),
        ("created_at", "not-a-date"),
        ("snapshot_at", "2025-12-31T09:00:00Z"),
        ("features_available_at", "2026-01-01T10:00:00Z"),
        ("window_end", "2026-01-08T10:00:00Z"),
        ("label_available_at", "2026-01-08T09:00:00Z"),
        ("label_available_at", "2026-01-21T09:00:00Z"),
        ("label_available_at", None),
        ("purchase_conversions_7d", None),
        ("purchase_conversions_7d", -1),
        ("purchase_conversions_7d", 1.5),
        ("purchase_conversions_7d", True),
        ("label_status", "pending"),
        ("label_status", "unknown"),
        ("label_status", "invalid"),
        ("tracking_coverage", "unknown"),
        ("tracking_coverage", "incomplete"),
        ("reporting_delay_hours", -1),
        ("audience_size", -1),
        ("audience_size", None),
        ("audience_observed_at", "2026-01-01T10:00:00Z"),
        ("publisher_share_percent", 101),
        ("media_age_days", math.nan),
        ("media_age_days", math.inf),
        ("creative_width_px", 0),
        ("creative_sequence_length", 0),
        ("has_cta", 1),
        ("previous_posts_with_views_7d", 21),
        ("previous_posts_with_views_7d", 0),
        ("previous_posts_with_views_7d", None),
        ("previous_posts_median_observed_views_7d", None),
        ("observed_through", "2026-01-02T09:00:00Z"),
    ],
)
def test_rejects_invalid_or_leaky_rows(record, column, value):
    record[column] = value
    with pytest.raises(SchemaError):
        LinkRow.from_dict(record)


@pytest.mark.parametrize("through", ["2026-01-18T09:00:00Z", "2026-01-21T09:00:00Z"])
def test_partial_observation_must_fit_creation_and_snapshot(through):
    record = json.loads(EXAMPLE.read_text().splitlines()[1])
    record["observed_through"] = through
    with pytest.raises(SchemaError):
        LinkRow.from_dict(record)


def test_rejects_duplicate_link_ids(tmp_path, record):
    dataset = tmp_path / "duplicate.jsonl"
    dataset.write_text((json.dumps(record) + "\n") * 2)
    with pytest.raises(SchemaError, match="duplicate.jsonl:2:.*duplicate link_id"):
        read_dataset(dataset)


@pytest.mark.parametrize("content", ["", "[]\n", "not-json\n", '{"hidden_reach": 1}\n'])
def test_rejects_empty_malformed_and_unknown_columns(tmp_path, content):
    dataset = tmp_path / "invalid.jsonl"
    dataset.write_text(content)
    with pytest.raises(SchemaError):
        read_dataset(dataset)


def test_missing_required_field_is_explained(record):
    del record["store_id"]
    with pytest.raises(SchemaError, match="store_id"):
        LinkRow.from_dict(record)


def test_training_cutoff_must_be_timezone_aware(record):
    with pytest.raises(SchemaError, match="timezone-aware"):
        LinkRow.from_dict(record).eligible_for_training(dt.datetime(2026, 2, 1))
