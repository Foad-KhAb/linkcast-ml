import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).parents[1] / "examples" / "link_rows_v1.jsonl"


def test_validate_command_reports_label_states():
    result = subprocess.run(
        [sys.executable, "-m", "linkcast.cli", "validate", str(EXAMPLE)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Valid schema 1.0: 3 links" in result.stdout
    assert "1 ready, 1 pending, 1 unknown" in result.stdout


def test_validate_command_rejects_invalid_dataset(tmp_path):
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text("{}\n")
    result = subprocess.run(
        [sys.executable, "-m", "linkcast.cli", "validate", str(dataset)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Validation failed" in result.stderr
