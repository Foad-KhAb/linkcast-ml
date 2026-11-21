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
docs/             Dataset contract and ML semantics
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

Read [dataset schema v1](docs/dataset-v1.md) for entity relationships, column
definitions, the initial feature allowlist, label maturity, and missingness rules.
One row describes one link at creation. A 24-hour-old link can remain in the
dataset with partial observations and a null seven-day target.

```sh
uv run --locked linkcast validate examples/link_rows_v1.jsonl
```

The three fictional examples contain one completed zero, one pending label, and
one unknown label. They demonstrate the contract; they are not generated training
data. Creation-time model inputs explicitly exclude partial outcomes and audit
fields.
