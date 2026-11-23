import dataclasses
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from linkcast.artifact import (
    METADATA_FILE,
    load,
    predict_rows,
    save,
    train_deployment,
)
from linkcast.generate import GeneratorConfig, generate, write
from linkcast.schema import read_dataset
from linkcast.splits import SplitConfig

DATA = GeneratorConfig(days=70, n_stores=6, n_publishers=10, n_media=25)
UTC = dt.UTC
CONFIG = SplitConfig(
    train_cutoff=dt.datetime(2026, 2, 1, tzinfo=UTC),
    validation_end=dt.datetime(2026, 2, 15, tzinfo=UTC),
    test_end=dt.datetime(2026, 3, 1, tzinfo=UTC),
)
GRID = {"depth": (4,), "learning_rate": (0.1,)}
EXAMPLES = Path(__file__).parents[1] / "examples" / "new_links_v1.jsonl"


@pytest.fixture(scope="module")
def rows():
    return generate(DATA).rows


@pytest.fixture(scope="module")
def trained(rows):
    return train_deployment(rows, CONFIG, GRID)


@pytest.fixture(scope="module")
def saved(trained, tmp_path_factory):
    return save(*trained, tmp_path_factory.mktemp("artifact") / "model")


def test_deployment_uses_every_label_available_at_snapshot(rows, trained):
    _, metadata = trained
    snapshot = max(row.snapshot_at for row in rows)
    eligible = [row for row in rows if row.eligible_for_training(snapshot)]
    assert metadata["training_links"] == len(eligible)
    assert metadata["training_cutoff"] == snapshot.isoformat()
    assert metadata["feature_set"] in {"default", "all"}
    assert metadata["schema_version"] == "1.0"


def test_saved_artifact_reloads_with_identical_predictions(rows, trained, saved):
    model, metadata = trained
    loaded, loaded_metadata = load(saved)
    assert loaded_metadata == json.loads(json.dumps(metadata))
    original = predict_rows(model, metadata, rows[:50])
    reloaded = predict_rows(loaded, loaded_metadata, rows[:50])
    np.testing.assert_array_equal(
        original.expected_conversions_7d, reloaded.expected_conversions_7d
    )


def test_predictions_ignore_labels_and_flag_unseen_entities(rows, saved):
    model, metadata = load(saved)
    ready = next(row for row in rows if row.label_status == "ready")
    relabeled = dataclasses.replace(ready, purchase_conversions_7d=999)
    renamed = dataclasses.replace(ready, link_id="other", media_id="unseen_media")
    result = predict_rows(model, metadata, [ready, relabeled, renamed])
    first, second, third = result.expected_conversions_7d
    assert first == second > 0
    assert result.new_media.tolist() == [False, False, True]


def test_load_rejects_incompatible_schema(saved, tmp_path):
    copy = save(*load(saved), tmp_path / "copy")
    metadata = json.loads((copy / METADATA_FILE).read_text())
    metadata["schema_version"] = "2.0"
    (copy / METADATA_FILE).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="schema"):
        load(copy)


def test_example_new_links_are_valid_and_predictable(saved):
    model, metadata = load(saved)
    rows = read_dataset(EXAMPLES)
    result = predict_rows(model, metadata, rows)
    assert len(result) == 2
    assert (result.expected_conversions_7d > 0).all()


def test_save_and_predict_commands(tmp_path):
    out = write(generate(DATA), tmp_path / "data")
    artifact = tmp_path / "artifact"
    dates = [
        "--train-cutoff", "2026-02-01",
        "--validation-end", "2026-02-15",
        "--test-end", "2026-03-01",
    ]  # fmt: skip
    subprocess.run(
        [sys.executable, "-m", "linkcast.cli", "save", str(out / "link_rows.jsonl")]
        + ["--out", str(artifact), *dates],
        check=True,
        capture_output=True,
    )
    metadata = json.loads((artifact / METADATA_FILE).read_text())
    assert metadata["data_manifest"]["config"]["days"] == 70
    predictions = tmp_path / "predictions.csv"
    subprocess.run(
        [sys.executable, "-m", "linkcast.cli", "predict", str(artifact)]
        + [str(EXAMPLES), "--out", str(predictions)],
        check=True,
        capture_output=True,
    )
    assert "expected_conversions_7d" in predictions.read_text().splitlines()[0]
