# LinkCast

An end-to-end ML learning project for predicting seven-day purchase conversion
counts from information available immediately after a tracking link is created.
A prediction may be fractional because it represents an expected count.

## Current status

Step 1 is implemented: standalone Python package, command-line entry point,
isolated environment, and dependency locking. Dataset generation, training,
evaluation, and prediction are planned next; they are not implemented yet.

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
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -c "import linkcast, catboost, numpy, pandas, sklearn; print('Imports OK:', linkcast.__version__)"
```

## Layout

```text
src/linkcast/      Reusable Python package and CLI
pyproject.toml    Package metadata, dependencies, and tool settings
uv.lock           Exact dependency resolution
.python-version   Python version for the local environment
```

The ML dependencies are CatBoost for CPU tabular modeling, pandas and NumPy for
data handling, and scikit-learn for evaluation utilities. Ruff and pytest are
development tools. No GPU or application database is required.
