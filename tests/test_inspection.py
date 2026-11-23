import json
import subprocess
import sys
from pathlib import Path

import pytest

from linkcast.generate import GeneratorConfig, generate, write
from linkcast.inspection import format_report, has_failures, inspect_rows
from linkcast.schema import LinkRow, read_dataset

EXAMPLE = Path(__file__).parents[1] / "examples" / "link_rows_v1.jsonl"
SMALL = GeneratorConfig(days=70, n_stores=6, n_publishers=10, n_media=25)


@pytest.fixture(scope="module")
def rows():
    return generate(SMALL).rows


@pytest.fixture(scope="module")
def report(rows):
    return inspect_rows(rows)


def checks(report):
    return {check["name"]: check["status"] for check in report["checks"]}


def test_label_counts_cover_every_row(rows, report):
    total = sum(sum(month.values()) for month in report["labels_by_month"].values())
    assert total == len(rows) == report["overview"]["links"]
    months = list(report["labels_by_month"])
    assert report["labels_by_month"][months[0]]["pending"] == 0
    last = report["labels_by_month"][months[-1]]
    assert last["pending"] > last["ready"]  # recent links are not mature yet
    assert report["label_horizon"]["first_incomplete_day"] is not None


def test_target_summary_uses_ready_rows_only(rows, report):
    ready = [r.purchase_conversions_7d for r in rows if r.label_status == "ready"]
    target = report["target"]
    assert target["ready_links"] == len(ready)
    assert target["max"] == max(ready)
    assert target["zero_share"] == round(ready.count(0) / len(ready), 4)
    assert target["variance_to_mean"] > 1  # overdispersed counts


def test_first_week_history_gap_is_reported(report):
    first_month = next(iter(report["labels_by_month"]))
    by_month = report["missing"]["by_month"]["previous_posts_7d"]
    assert by_month[first_month] > 0
    assert report["missing"]["overall"]["store_id"] == 0


def test_generated_data_has_no_failing_checks(report):
    status = checks(report)
    assert not has_failures(report)
    assert status["model inputs exclude labels and audit fields"] == "pass"
    assert status["no stale pending labels"] == "pass"
    assert "Checks" in format_report(report)
    json.dumps(report)  # report must be serializable


def test_stale_pending_label_fails():
    record = json.loads(EXAMPLE.read_text().splitlines()[1])
    assert record["label_status"] == "pending"
    record["snapshot_at"] = "2026-03-01T00:00:00Z"  # long after the window closed
    report = inspect_rows([LinkRow.from_dict(record)])
    assert checks(report)["no stale pending labels"] == "fail"
    assert has_failures(report)


def test_inspect_command_writes_json(tmp_path):
    out = write(generate(SMALL), tmp_path / "data")
    report_path = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "linkcast.cli",
            "inspect",
            str(out / "link_rows.jsonl"),
            "--json",
            str(report_path),
        ],  # fmt: skip
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Labels by creation month" in result.stdout
    saved = json.loads(report_path.read_text())
    assert saved["overview"]["links"] == len(read_dataset(out / "link_rows.jsonl"))
