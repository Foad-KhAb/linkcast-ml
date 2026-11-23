"""Saved model artifacts and prediction for new links.

``train_deployment`` makes every choice on validation (as in evaluation), then
refits the chosen CatBoost model on every label available at the dataset's
snapshot time. That model predicts future links; its expected quality is the
test score from ``linkcast evaluate`` (it is not scored again here).

An artifact directory contains:

- ``model.cbm``: the CatBoost model in its native format;
- ``metadata.json``: schema and package versions, exact feature list and
  categorical features, settings, training cutoff and size, validation scores
  of the choice, entities seen in training, and the data manifest if present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

from linkcast import __version__
from linkcast.model import (
    FEATURE_SETS,
    GRID,
    Params,
    feature_frame,
    fit,
    model_data,
    predict,
    tune,
)
from linkcast.schema import CATEGORICAL_FEATURES, SCHEMA_VERSION, TARGET, LinkRow
from linkcast.splits import SplitConfig

ARTIFACT_VERSION = "1"
MODEL_FILE = "model.cbm"
METADATA_FILE = "metadata.json"
SEEN_ENTITIES = ("store_id", "media_id", "campaign_id", "creative_id")


def train_deployment(
    rows: list[LinkRow],
    config: SplitConfig | None = None,
    grid: dict = GRID,
    manifest: dict | None = None,
) -> tuple[CatBoostRegressor, dict]:
    config = config or SplitConfig()
    selection = {
        name: tune(model_data(rows, config, features), features, grid)["best"]
        for name, features in FEATURE_SETS.items()
    }
    chosen = min(selection, key=lambda name: selection[name]["poisson_deviance"])
    features = FEATURE_SETS[chosen]
    best = selection[chosen]

    cutoff = max(row.snapshot_at for row in rows)
    eligible = [row for row in rows if row.eligible_for_training(cutoff)]
    if not eligible:
        raise ValueError("no labels are available for training at the snapshot")
    frame = feature_frame(eligible, features)
    frame[TARGET] = [row.purchase_conversions_7d for row in eligible]
    params = Params(
        depth=best["depth"],
        learning_rate=best["learning_rate"],
        iterations=best["iterations"],
    )
    model = fit(frame, features, params)

    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "package_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "target": TARGET,
        "prediction": "expected purchase conversions in [created_at, +168h)",
        "loss": "Poisson",
        "feature_set": chosen,
        "features": list(features),
        "categorical_features": [f for f in features if f in CATEGORICAL_FEATURES],
        "params": params.__dict__,
        "training_cutoff": cutoff.isoformat(),
        "training_links": len(eligible),
        "first_training_link_created_at": min(
            r.created_at for r in eligible
        ).isoformat(),
        "last_training_link_created_at": max(
            r.created_at for r in eligible
        ).isoformat(),
        "selection": {
            "rule": "lowest validation Poisson deviance",
            "split_config": {k: v.isoformat() for k, v in config.__dict__.items()},
            "validation_scores": selection,
        },
        "seen_entities": {
            entity: sorted({getattr(row, entity) for row in eligible} - {None})
            for entity in SEEN_ENTITIES
        },
        "data_manifest": manifest,
    }
    return model, metadata


def save(model: CatBoostRegressor, metadata: dict, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_model(str(directory / MODEL_FILE))
    (directory / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
    return directory


def load(directory: str | Path) -> tuple[CatBoostRegressor, dict]:
    directory = Path(directory)
    metadata = json.loads((directory / METADATA_FILE).read_text())
    if metadata.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"unsupported artifact version {metadata.get('artifact_version')}"
        )
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"artifact expects schema {metadata.get('schema_version')}, "
            f"this package reads schema {SCHEMA_VERSION}"
        )
    model = CatBoostRegressor()
    model.load_model(str(directory / MODEL_FILE))
    return model, metadata


def predict_rows(
    model: CatBoostRegressor, metadata: dict, rows: list[LinkRow]
) -> pd.DataFrame:
    """Expected conversions per link, with flags for entities unseen in training.

    Rows are schema-v1 link rows at creation; any labels they carry are ignored.
    """
    features = tuple(metadata["features"])
    frame = feature_frame(rows, features)
    result = pd.DataFrame(
        {
            "link_id": frame.index,
            "created_at": [row.created_at.isoformat() for row in rows],
            "expected_conversions_7d": predict(model, frame, features).round(4),
        }
    )
    for entity, seen in metadata["seen_entities"].items():
        seen = set(seen)
        result[f"new_{entity.removesuffix('_id')}"] = [
            None if getattr(row, entity) is None else getattr(row, entity) not in seen
            for row in rows
        ]
    return result
