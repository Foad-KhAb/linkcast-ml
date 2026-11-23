"""Dataset inspection before modeling.

The report answers: which rows are trainable over time, what the target looks
like, how missingness changes, when new entities appear, and whether basic
quality/leakage checks pass. It reads only schema-v1 rows, never hidden truth.
"""

from __future__ import annotations

import pandas as pd

from linkcast.schema import (
    AUDIT_FIELDS,
    CATEGORICAL_FEATURES,
    DEFAULT_FEATURES,
    LABEL_FIELDS,
    NUMERIC_FEATURES,
    TARGET,
    LinkRow,
)

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
STATUSES = ("ready", "pending", "unknown")
ENTITIES = ("store_id", "media_id", "campaign_id", "creative_id")
READY_SHARE = 0.95  # label horizon: days where most resolvable labels are ready
STALE_PENDING_DAYS = 7  # pending this long after the reporting buffer is suspicious
UNKNOWN_SHARE_WARNING = 0.2
MISSINGNESS_SHIFT_WARNING = 0.2
MIN_MONTH_LINKS = 100  # ignore tiny partial months when comparing months


def inspect_rows(rows: list[LinkRow]) -> dict:
    """Return a JSON-serializable inspection report."""
    data = _frame(rows)
    month = data.created_at.dt.strftime("%Y-%m")
    return {
        "overview": _overview(data),
        "labels_by_month": _labels_by_month(data, month),
        "label_horizon": _label_horizon(data),
        "target": _target(data),
        "missing": _missing(data, month),
        "new_entities": _new_entities(data, month),
        "checks": _checks(data, month),
    }


def _frame(rows: list[LinkRow]) -> pd.DataFrame:
    records = [
        row.to_dict() | row.model_inputs(FEATURES) for row in rows
    ]  # model_inputs adds derived features such as creation hour
    data = pd.DataFrame.from_records(records)
    for column in (
        "created_at",
        "snapshot_at",
        "features_available_at",
        "window_end",
        "audience_observed_at",
    ):
        data[column] = pd.to_datetime(data[column], utc=True)
    return data


def _iso(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _overview(data: pd.DataFrame) -> dict:
    return {
        "links": len(data),
        "first_created_at": _iso(data.created_at.min()),
        "last_created_at": _iso(data.created_at.max()),
        "snapshot_at": sorted(_iso(value) for value in data.snapshot_at.unique()),
    }


def _labels_by_month(data: pd.DataFrame, month: pd.Series) -> dict:
    table = pd.crosstab(month, data.label_status).reindex(
        columns=STATUSES, fill_value=0
    )
    return {
        str(m): {status: int(table.loc[m, status]) for status in STATUSES}
        for m in table.index
    }


def _label_horizon(data: pd.DataFrame) -> dict:
    """First creation day whose resolvable labels are mostly not ready yet."""
    day = data.created_at.dt.floor("D")
    resolvable = data.label_status != "unknown"
    ready = (data.label_status == "ready")[resolvable].groupby(day[resolvable]).mean()
    late = ready[ready < READY_SHARE]
    first_incomplete = _iso(late.index.min()) if len(late) else None
    return {
        "first_incomplete_day": first_incomplete,
        "rule": f"ready / (ready + pending) < {READY_SHARE} for links created that day",
    }


def _target(data: pd.DataFrame) -> dict:
    target = data.loc[data.label_status == "ready", TARGET].astype(float)
    if target.empty:
        return {"ready_links": 0}
    mean = float(target.mean())
    variance = float(target.var(ddof=0))
    return {
        "ready_links": int(len(target)),
        "mean": round(mean, 4),
        "variance": round(variance, 4),
        "variance_to_mean": round(variance / mean, 4) if mean else None,
        "zero_share": round(float((target == 0).mean()), 4),
        "median": float(target.median()),
        "p90": float(target.quantile(0.9)),
        "p99": float(target.quantile(0.99)),
        "max": int(target.max()),
    }


def _missing(data: pd.DataFrame, month: pd.Series) -> dict:
    nulls = data[list(FEATURES)].isna()
    overall = nulls.mean()
    by_month = nulls.groupby(month).mean()
    with_nulls = [feature for feature in FEATURES if overall[feature] > 0]
    return {
        "overall": {feature: round(float(overall[feature]), 4) for feature in FEATURES},
        "by_month": {
            feature: {
                str(m): round(float(by_month.loc[m, feature]), 4)
                for m in by_month.index
            }
            for feature in with_nulls
        },
    }


def _new_entities(data: pd.DataFrame, month: pd.Series) -> dict:
    """Entities first seen each month, and the share of that month's links on them."""
    report = {}
    links_per_month = month.value_counts()
    for entity in ENTITIES:
        known = data[entity].notna()
        first_seen = data[known].groupby(entity).created_at.transform("min")
        is_first_month = first_seen.dt.strftime("%Y-%m") == month[known]
        new_counts = (
            data[known]
            .groupby(entity)
            .created_at.min()
            .dt.strftime("%Y-%m")
            .value_counts()
        )
        on_new = is_first_month.groupby(month[known]).sum()
        report[entity] = {
            str(m): {
                "new": int(new_counts.get(m, 0)),
                "links_share": round(float(on_new.get(m, 0) / links_per_month[m]), 4),
            }
            for m in sorted(links_per_month.index)
        }
    return report


def _check(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}


def _checks(data: pd.DataFrame, month: pd.Series) -> list[dict]:
    checks = []

    leaked = sorted(set(FEATURES) & set(LABEL_FIELDS + AUDIT_FIELDS))
    default_leaked = sorted(set(DEFAULT_FEATURES) - set(FEATURES))
    checks.append(
        _check(
            "model inputs exclude labels and audit fields",
            "fail" if leaked or default_leaked else "pass",
            f"overlap: {leaked + default_leaked}" if leaked or default_leaked else
            f"{len(FEATURES)} allowed features, {len(DEFAULT_FEATURES)} default",
        )
    )  # fmt: skip

    late_audience = data.audience_observed_at > data.created_at
    late_features = data.features_available_at > data.created_at
    late = int((late_audience | late_features).sum())
    checks.append(
        _check(
            "feature timestamps precede link creation",
            "fail" if late else "pass",
            f"{late} rows with feature availability after created_at",
        )
    )

    snapshots = data.snapshot_at.nunique()
    checks.append(
        _check(
            "single snapshot time",
            "pass" if snapshots == 1 else "warn",
            f"{snapshots} distinct snapshot_at values",
        )
    )

    grace_hours = data.reporting_delay_hours.astype("int64") + STALE_PENDING_DAYS * 24
    deadline = data.window_end + pd.to_timedelta(grace_hours, unit="h")
    stale = int(
        ((data.label_status == "pending") & (deadline < data.snapshot_at)).sum()
    )
    checks.append(
        _check(
            "no stale pending labels",
            "fail" if stale else "pass",
            f"{stale} pending rows more than {STALE_PENDING_DAYS} days past "
            "window end plus reporting delay",
        )
    )

    mature = data.label_status != "pending"
    unknown_share = (
        (data.label_status == "unknown")[mature].groupby(month[mature]).mean()
    )
    high = {
        m: round(float(s), 3)
        for m, s in unknown_share.items()
        if s > UNKNOWN_SHARE_WARNING
    }
    checks.append(
        _check(
            "unknown label share per month",
            "warn" if high else "pass",
            f"months above {UNKNOWN_SHARE_WARNING}: {high}" if high else
            f"max {unknown_share.max():.3f} of mature rows",
        )
    )  # fmt: skip

    constant = [f for f in FEATURES if data[f].nunique(dropna=True) <= 1]
    checks.append(
        _check(
            "no constant features",
            "warn" if constant else "pass",
            f"constant or empty: {constant}" if constant else "all features vary",
        )
    )

    sizes = month.value_counts()
    full_months = sizes[sizes >= MIN_MONTH_LINKS].index
    by_month = (
        data[list(FEATURES)].isna().groupby(month).mean().loc[sorted(full_months)]
    )
    spread = (by_month.max() - by_month.min()).round(3)
    shifted = {f: float(s) for f, s in spread.items() if s > MISSINGNESS_SHIFT_WARNING}
    checks.append(
        _check(
            "missingness stable across months",
            "warn" if shifted else "pass",
            f"null-share range above {MISSINGNESS_SHIFT_WARNING}: {shifted}" if shifted
            else f"max monthly null-share range {spread.max():.3f}",
        )
    )  # fmt: skip
    return checks


# --- Text report ------------------------------------------------------------


def format_report(report: dict) -> str:
    lines = []
    overview = report["overview"]
    lines += [
        "Overview",
        f"  links: {overview['links']}",
        f"  created: {overview['first_created_at']} to {overview['last_created_at']}",
        f"  snapshot: {', '.join(overview['snapshot_at'])}",
        "",
        "Labels by creation month",
        f"  {'month':<8} {'ready':>7} {'pending':>8} {'unknown':>8}",
    ]
    for m, counts in report["labels_by_month"].items():
        lines.append(
            f"  {m:<8} {counts['ready']:>7} {counts['pending']:>8} {counts['unknown']:>8}"
        )
    horizon = report["label_horizon"]
    lines += [
        f"  labels mostly incomplete from: {horizon['first_incomplete_day'] or 'never'}",
        f"  ({horizon['rule']})",
        "",
        "Target (ready rows only)",
    ]
    lines += [f"  {key}: {value}" for key, value in report["target"].items()]

    lines += ["", "Missing feature values (share of rows; by month if any are missing)"]
    by_month = report["missing"]["by_month"]
    months = sorted(next(iter(by_month.values())).keys()) if by_month else []
    lines.append(
        f"  {'feature':<42} {'all':>6} " + " ".join(f"{m[2:]:>6}" for m in months)
    )
    for feature, share in report["missing"]["overall"].items():
        if feature in by_month:
            per_month = " ".join(f"{by_month[feature][m]:>6.3f}" for m in months)
            lines.append(f"  {feature:<42} {share:>6.3f} {per_month}")
    complete = [f for f, s in report["missing"]["overall"].items() if s == 0]
    lines.append(f"  never missing: {', '.join(complete)}")

    lines += ["", "New entities by month (count new / share of month's links on them)"]
    entities = report["new_entities"]
    months = list(next(iter(entities.values())).keys())
    lines.append(f"  {'entity':<12} " + " ".join(f"{m[2:]:>12}" for m in months))
    for entity, per_month in entities.items():
        cells = " ".join(
            f"{per_month[m]['new']:>4} / {per_month[m]['links_share']:>5.3f}"
            for m in months
        )
        lines.append(f"  {entity:<12} {cells}")

    lines += ["", "Checks"]
    for check in report["checks"]:
        lines.append(
            f"  [{check['status'].upper():<4}] {check['name']}: {check['detail']}"
        )
    return "\n".join(lines)


def has_failures(report: dict) -> bool:
    return any(check["status"] == "fail" for check in report["checks"])
