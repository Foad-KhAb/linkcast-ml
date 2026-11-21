# LinkCast

An end-to-end ML learning project for predicting seven-day purchase conversion
counts from information available immediately after a tracking link is created.
A prediction may be fractional because it represents an expected count.

## Current status

Implemented so far: standalone package and environment, versioned link
dataset contract, validation CLI, fictional example rows, and tests.
Generation, training, evaluation, and prediction are not implemented yet.

## Install and run

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run these commands from the repository root:

```sh
uv sync --locked
uv run --locked linkcast --help
uv run --locked linkcast --version
```

`uv sync` creates the project's `.venv` and installs the package in editable mode.
`uv.lock` records exact dependency versions and should be version controlled.
The `.venv` directory should not be version controlled.

To verify the setup:

```sh
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -c "import linkcast, catboost, numpy, pandas, sklearn; print('Imports OK:', linkcast.__version__)"
```

## Layout

```text
src/linkcast/      Reusable package, schema, and CLI
examples/         Hand-written fictional schema examples
tests/            Schema and CLI checks
pyproject.toml    Package metadata, dependencies, and tool settings
uv.lock           Exact dependency resolution
.python-version   Python version for the local environment
```

The ML dependencies are CatBoost for CPU tabular modeling, pandas and NumPy for
data handling, and scikit-learn for evaluation utilities. Ruff and pytest are
development tools. No GPU or application database is required.

## Dataset contract

Dataset schema 1.0 is defined and validated in `src/linkcast/schema.py`. Data is
stored as UTF-8 JSON Lines, one flat object per link. The main rules:

- One row per link, predicted at `created_at` using only information available
  by then (`features_available_at <= created_at`).
- The target `purchase_conversions_7d` counts approved purchase conversions in
  `[created_at, created_at + 168 hours)`.
- `label_status` is `ready` (complete tracking, label available after the window
  and reporting delay), `pending`, or `unknown`. Pending and unknown rows keep a
  null target. A 24-hour-old link stays in the dataset with partial observations,
  not a completed label.
- Unknown values are `null` and are never treated as zero. Unmeasured post views
  and unknown tracking coverage are not zero views or zero conversions.
- Model inputs come from an explicit allowlist (`DEFAULT_FEATURES`); labels and
  audit fields are never inputs. Unknown columns are rejected.

```sh
uv run --locked linkcast validate examples/link_rows_v1.jsonl
```

The three fictional examples contain one completed zero, one pending label, and
one unknown label. They demonstrate the contract; they are not generated training
data. Creation-time model inputs explicitly exclude partial outcomes and audit
fields.
