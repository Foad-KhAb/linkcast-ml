"""Chronological train/validation/test splits that respect label availability.

A cutoff represents "the day the model is trained". Only links whose completed
label was available by that cutoff may be used for training; a link created
shortly before the cutoff is excluded (the "gap") because its seven-day outcome
was not yet known. Evaluation rows need a ready label, judged at snapshot time.

Two training sets exist:

- ``train``: labels available by ``train_cutoff``; the model scored on
  ``validation`` (used for comparing and tuning models).
- ``in_final_train``: labels available by ``validation_end``; the refit that is
  scored once on ``test``, as a deployment on that date would be.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from linkcast.schema import TARGET, LinkRow

SPLITS = ("train", "validation", "test", "excluded")
ENTITIES = ("store_id", "media_id", "campaign_id", "creative_id")
PENDING_WARNING = 0.05


@dataclass(frozen=True)
class SplitConfig:
    train_cutoff: dt.datetime = dt.datetime(2026, 5, 1, tzinfo=dt.UTC)
    validation_end: dt.datetime = dt.datetime(2026, 6, 1, tzinfo=dt.UTC)
    test_end: dt.datetime = dt.datetime(2026, 6, 23, tzinfo=dt.UTC)

    def __post_init__(self) -> None:
        times = (self.train_cutoff, self.validation_end, self.test_end)
        if any(time.utcoffset() is None for time in times):
            raise ValueError("split dates must be timezone-aware")
        if not self.train_cutoff < self.validation_end < self.test_end:
            raise ValueError("require train_cutoff < validation_end < test_end")


def assign_splits(
    rows: list[LinkRow], config: SplitConfig | None = None
) -> pd.DataFrame:
    """One row per link: split, reason, final-train membership, new-entity flags."""
    config = config or SplitConfig()
    records = []
    for row in rows:
        split, reason = _assign(row, config)
        records.append(
            {
                "link_id": row.link_id,
                "created_at": row.created_at.astimezone(dt.UTC),
                "split": split,
                "reason": reason,
                "in_final_train": row.eligible_for_training(config.validation_end),
                "label_status": row.label_status,
                TARGET: row.purchase_conversions_7d,
                **{entity: getattr(row, entity) for entity in ENTITIES},
            }
        )
    splits = pd.DataFrame.from_records(records)
    splits[TARGET] = splits[TARGET].astype("Int64")
    _flag_new_entities(splits)
    return splits


def _assign(row: LinkRow, config: SplitConfig) -> tuple[str, str]:
    created = row.created_at
    if created >= config.test_end:
        return "excluded", "after_test_period"
    if created < config.train_cutoff:
        if row.eligible_for_training(config.train_cutoff):
            return "train", "label_ready_by_cutoff"
        if row.label_status == "ready":
            return "excluded", "label_after_cutoff"  # the gap before the cutoff
        return "excluded", f"label_{row.label_status}"
    if row.label_status != "ready":
        return "excluded", f"label_{row.label_status}"  # cannot be scored
    return ("validation" if created < config.validation_end else "test"), "scored"


def _flag_new_entities(splits: pd.DataFrame) -> None:
    """Mark evaluation rows whose entity never appears in their training set."""
    seen_by = {
        "validation": splits[splits.split == "train"],
        "test": splits[splits.in_final_train],
    }
    for entity in ENTITIES:
        flag = pd.Series(pd.NA, index=splits.index, dtype="boolean")
        for split, training in seen_by.items():
            rows = (splits.split == split) & splits[entity].notna()
            seen = set(training[entity].dropna())
            flag[rows] = ~splits.loc[rows, entity].isin(seen)
        splits[f"new_{entity.removesuffix('_id')}"] = flag


def summarize(splits: pd.DataFrame, config: SplitConfig) -> dict:
    def describe(part: pd.DataFrame) -> dict:
        target = part[TARGET].dropna().astype(float)
        return {
            "links": int(len(part)),
            "first_created_at": _iso(part.created_at.min()) if len(part) else None,
            "last_created_at": _iso(part.created_at.max()) if len(part) else None,
            "target_mean": round(float(target.mean()), 4) if len(target) else None,
            "zero_share": round(float((target == 0).mean()), 4)
            if len(target)
            else None,
        }

    summary = {
        "config": {key: _iso(value) for key, value in config.__dict__.items()},
        "splits": {split: describe(splits[splits.split == split]) for split in SPLITS},
        "final_train": describe(splits[splits.in_final_train]),
        "excluded_reasons": {
            str(reason): int(count)
            for reason, count in splits[splits.split == "excluded"]
            .reason.value_counts()
            .sort_index()
            .items()
        },
        "new_entities": {},
        "warnings": [],
    }
    for split in ("validation", "test"):
        part = splits[splits.split == split]
        summary["new_entities"][split] = {
            entity: {
                "links": int(part[f"new_{entity}"].sum()),
                "distinct": int(part.loc[part[f"new_{entity}"].fillna(False), f"{entity}_id"].nunique()),
            }
            for entity in ("store", "media", "campaign", "creative")
        }  # fmt: skip

    evaluated = splits[
        (splits.created_at >= config.train_cutoff)
        & (splits.created_at < config.test_end)
    ]
    pending = (
        float((evaluated.label_status == "pending").mean()) if len(evaluated) else 0
    )
    if pending > PENDING_WARNING:
        summary["warnings"].append(
            f"{pending:.1%} of validation/test-period links are still pending; "
            "test_end may be too close to the snapshot"
        )
    for split in ("train", "validation", "test"):
        if summary["splits"][split]["links"] == 0:
            summary["warnings"].append(f"{split} split is empty")
    return summary


def _iso(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_summary(summary: dict) -> str:
    config = summary["config"]
    lines = [
        "Split dates",
        f"  train:      labels available by {config['train_cutoff']}",
        f"  validation: created {config['train_cutoff']} to {config['validation_end']}",
        f"  test:       created {config['validation_end']} to {config['test_end']}",
        f"  final train (refit before test): labels available by {config['validation_end']}",
        "",
        f"  {'split':<12} {'links':>7} {'target mean':>12} {'zeros':>7}  created",
    ]
    parts = summary["splits"] | {"final train": summary["final_train"]}
    for name, part in parts.items():
        mean = "" if part["target_mean"] is None else f"{part['target_mean']:.3f}"
        zeros = "" if part["zero_share"] is None else f"{part['zero_share']:.3f}"
        created = (
            f"{part['first_created_at'][:10]} to {part['last_created_at'][:10]}"
            if part["links"]
            else ""
        )
        lines.append(
            f"  {name:<12} {part['links']:>7} {mean:>12} {zeros:>7}  {created}"
        )
    lines += ["", "Excluded rows by reason"]
    lines += [
        f"  {reason}: {count}" for reason, count in summary["excluded_reasons"].items()
    ]
    lines += ["", "Rows on entities unseen in their training set (links / distinct)"]
    for split, entities in summary["new_entities"].items():
        cells = ", ".join(
            f"{entity} {value['links']} / {value['distinct']}"
            for entity, value in entities.items()
        )
        lines.append(f"  {split}: {cells}")
    lines += ["", "Warnings"]
    lines += [f"  {warning}" for warning in summary["warnings"]] or ["  none"]
    return "\n".join(lines)


def write_splits(splits: pd.DataFrame, path) -> None:
    out = splits.drop(columns=["label_status", TARGET])
    out = out.assign(created_at=out.created_at.dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    out.to_csv(path, index=False)
