"""Versioned, flat link records with explicit feature and label boundaries."""

from __future__ import annotations

import datetime as dt
import json
import math
import types
from dataclasses import asdict, dataclass, fields
from functools import cache
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

SCHEMA_VERSION = "1.0"
TARGET = "purchase_conversions_7d"
HORIZON = dt.timedelta(hours=168)

CATEGORICAL_FEATURES = (
    "store_id",
    "media_id",
    "platform",
    "campaign_id",
    "creative_id",
    "landing_id",
    "media_category",
    "creative_format",
    "offer_type",
    "offer_source",
)
NUMERIC_FEATURES = (
    "creation_hour_utc",
    "creation_weekday_utc",
    "audience_size",
    "media_age_days",
    "previous_posts_7d",
    "previous_posts_with_views_7d",
    "previous_posts_view_coverage_7d",
    "previous_posts_median_observed_views_7d",
    "previous_links_7d",
    "hours_since_previous_link",
    "creative_width_px",
    "creative_height_px",
    "caption_length",
    "creative_sequence_length",
    "has_cta",
    "lead_price_rials",
    "publisher_share_percent",
    "campaign_daily_budget_rials",
    "is_pushed",
)
DEFAULT_FEATURES = (
    "store_id",
    "media_id",
    "platform",
    "campaign_id",
    "creative_id",
    "landing_id",
    "creation_hour_utc",
    "creation_weekday_utc",
    "audience_size",
    "creative_format",
)


class SchemaError(ValueError):
    """A record violates the dataset contract."""


@dataclass(frozen=True, kw_only=True)
class LinkRow:
    schema_version: str
    link_id: str
    created_at: dt.datetime
    snapshot_at: dt.datetime
    features_available_at: dt.datetime
    window_end: dt.datetime
    reporting_delay_hours: int
    label_status: str
    tracking_coverage: str
    store_id: str
    media_id: str
    platform: str
    campaign_id: str
    label_available_at: dt.datetime | None = None
    purchase_conversions_7d: int | None = None
    creative_id: str | None = None
    landing_id: str | None = None
    publisher_id: str | None = None
    asset_ref: str | None = None
    audience_size: int | None = None
    audience_observed_at: dt.datetime | None = None
    media_category: str | None = None
    media_age_days: float | None = None
    previous_posts_7d: int | None = None
    previous_posts_with_views_7d: int | None = None
    previous_posts_median_observed_views_7d: float | None = None
    previous_links_7d: int | None = None
    hours_since_previous_link: float | None = None
    creative_format: str | None = None
    creative_width_px: int | None = None
    creative_height_px: int | None = None
    caption_length: int | None = None
    creative_sequence_length: int | None = None
    has_cta: bool | None = None
    lead_price_rials: int | None = None
    publisher_share_percent: float | None = None
    campaign_daily_budget_rials: int | None = None
    offer_type: str | None = None
    offer_source: str | None = None
    is_pushed: bool | None = None
    observed_through: dt.datetime | None = None
    observed_purchase_conversions: int | None = None

    def __post_init__(self) -> None:
        for name, annotation in _row_types().items():
            value = getattr(self, name)
            alternatives = (
                get_args(annotation)
                if get_origin(annotation) is types.UnionType
                else (annotation,)
            )
            if value is None and type(None) in alternatives:
                continue
            expected = next(item for item in alternatives if item is not type(None))
            if expected is float:
                valid = (
                    type(value) in (int, float) and math.isfinite(value) and value >= 0
                )
            elif expected is int:
                valid = type(value) is int and value >= 0
            elif expected is dt.datetime:
                valid = isinstance(value, dt.datetime) and value.utcoffset() is not None
            elif expected is str:
                valid = isinstance(value, str) and bool(value.strip())
            else:
                valid = type(value) is expected
            if not valid:
                raise SchemaError(
                    f"{name}: expected {annotation}, with a valid nonnegative value where numeric"
                )

        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(f"schema_version must be {SCHEMA_VERSION!r}")
        if self.window_end != self.created_at.astimezone(dt.UTC) + HORIZON:
            raise SchemaError("window_end must equal created_at + 168 hours")
        if self.snapshot_at < self.created_at:
            raise SchemaError("snapshot_at must not precede created_at")
        if self.features_available_at > self.created_at:
            raise SchemaError("features_available_at must not be after creation")
        if self.label_status not in {"pending", "ready", "unknown"}:
            raise SchemaError("label_status must be pending, ready, or unknown")
        if self.tracking_coverage not in {"complete", "incomplete", "unknown"}:
            raise SchemaError(
                "tracking_coverage must be complete, incomplete, or unknown"
            )
        if self.label_status == "ready":
            if self.purchase_conversions_7d is None or self.label_available_at is None:
                raise SchemaError(
                    "ready labels require a target and label_available_at"
                )
            earliest = self.window_end.astimezone(dt.UTC) + dt.timedelta(
                hours=self.reporting_delay_hours
            )
            if not earliest <= self.label_available_at <= self.snapshot_at:
                raise SchemaError(
                    "label_available_at must follow the window and delay, and not exceed snapshot_at"
                )
            if self.tracking_coverage != "complete":
                raise SchemaError("ready labels require complete tracking coverage")
        elif (
            self.purchase_conversions_7d is not None
            or self.label_available_at is not None
        ):
            raise SchemaError(
                "pending/unknown labels must have null target and label_available_at"
            )

        if (self.audience_size is None) != (self.audience_observed_at is None):
            raise SchemaError(
                "audience_size and audience_observed_at must be supplied together"
            )
        if (
            self.audience_observed_at is not None
            and self.audience_observed_at > self.features_available_at
        ):
            raise SchemaError(
                "audience observation must be available by features_available_at"
            )
        if (
            self.publisher_share_percent is not None
            and self.publisher_share_percent > 100
        ):
            raise SchemaError("publisher_share_percent must be between 0 and 100")
        for name in (
            "creative_width_px",
            "creative_height_px",
            "creative_sequence_length",
        ):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise SchemaError(f"{name} must be positive when supplied")

        measured = self.previous_posts_with_views_7d
        median = self.previous_posts_median_observed_views_7d
        if measured is not None:
            if self.previous_posts_7d is None or measured > self.previous_posts_7d:
                raise SchemaError(
                    "measured post count requires a total at least as large"
                )
            if (measured == 0 and median is not None) or (
                measured > 0 and median is None
            ):
                raise SchemaError(
                    "post-view median requires at least one measured post, and vice versa"
                )
        elif median is not None:
            raise SchemaError("post-view median requires a measured post count")

        if (self.observed_through is None) != (
            self.observed_purchase_conversions is None
        ):
            raise SchemaError(
                "partial observations require both count and observed_through"
            )
        if (
            self.observed_through is not None
            and not self.created_at
            <= self.observed_through
            <= min(self.window_end, self.snapshot_at)
        ):
            raise SchemaError(
                "observed_through must be within the target window and snapshot"
            )

    def model_inputs(self, names: tuple[str, ...] = DEFAULT_FEATURES) -> dict:
        """Return only explicitly allowed, creation-time features; retain nulls."""
        allowed = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES)
        if forbidden := set(names) - allowed:
            raise SchemaError(f"Not model features: {sorted(forbidden)}")
        created_utc = self.created_at.astimezone(dt.UTC)
        coverage = None
        if self.previous_posts_7d and self.previous_posts_with_views_7d is not None:
            coverage = self.previous_posts_with_views_7d / self.previous_posts_7d
        derived = {
            "creation_hour_utc": created_utc.hour,
            "creation_weekday_utc": created_utc.weekday(),
            "previous_posts_view_coverage_7d": coverage,
        }
        return {
            name: derived[name] if name in derived else getattr(self, name)
            for name in names
        }

    def eligible_for_training(self, cutoff: dt.datetime) -> bool:
        """Whether this snapshot declares a completed label available by cutoff."""
        if cutoff.utcoffset() is None:
            raise SchemaError("training cutoff must be timezone-aware")
        return self.label_status == "ready" and self.label_available_at <= cutoff

    def to_dict(self) -> dict:
        return {
            name: value.astimezone(dt.UTC).isoformat()
            if isinstance(value, dt.datetime)
            else value
            for name, value in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, record: dict) -> LinkRow:
        if not isinstance(record, dict):
            raise SchemaError("each row must be a JSON object")
        known = {field.name for field in fields(cls)}
        if unknown := set(record) - known:
            raise SchemaError(f"Unknown columns: {sorted(unknown)}")
        parsed = dict(record)
        for name, annotation in _row_types().items():
            alternatives = get_args(annotation) or (annotation,)
            if dt.datetime in alternatives and isinstance(parsed.get(name), str):
                try:
                    parsed[name] = dt.datetime.fromisoformat(parsed[name])
                except ValueError as error:
                    raise SchemaError(f"{name}: invalid ISO 8601 timestamp") from error
        try:
            return cls(**parsed)
        except TypeError as error:
            raise SchemaError(str(error)) from error


@cache
def _row_types() -> dict:
    return get_type_hints(LinkRow)


LABEL_FIELDS = (TARGET, "label_status", "label_available_at")
AUDIT_FIELDS = tuple(
    field.name
    for field in fields(LinkRow)
    if field.name not in CATEGORICAL_FEATURES + NUMERIC_FEATURES + LABEL_FIELDS
)


def read_dataset(path: str | Path) -> list[LinkRow]:
    """Read JSONL, rejecting malformed rows, unknown columns, and duplicate IDs."""
    rows = []
    seen = set()
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = LinkRow.from_dict(json.loads(line))
                if row.link_id in seen:
                    raise SchemaError(f"duplicate link_id: {row.link_id}")
            except (ValueError, TypeError) as error:
                raise SchemaError(f"{path}:{line_number}: {error}") from error
            seen.add(row.link_id)
            rows.append(row)
    if not rows:
        raise SchemaError("dataset must contain at least one row")
    return rows
