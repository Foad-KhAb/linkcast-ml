# LinkCast

An end-to-end ML learning project for predicting seven-day purchase conversion
counts from information available immediately after a tracking link is created.
A prediction may be fractional because it represents an expected count.

## Current status

Implemented so far: standalone package and environment, versioned link
dataset contract, validation CLI, fictional example rows, a deterministic
synthetic data generator, dataset inspection report, and tests. Training,
evaluation, and prediction are not implemented yet.

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
src/linkcast/      Reusable package, schema, generator, inspection, and CLI
examples/         Hand-written fictional schema examples
tests/            Schema, generator, inspection, and CLI checks
pyproject.toml    Package metadata, dependencies, and tool settings
uv.lock           Exact dependency resolution
.python-version   Python version for the local environment
```

The ML dependencies are CatBoost for CPU tabular modeling, pandas and NumPy for
data handling, and scikit-learn for evaluation utilities. Ruff and pytest are
development tools. No GPU or application database is required; everything runs on a laptop CPU.

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

## Synthetic data

```sh
uv run --locked linkcast generate            # writes data/synthetic/ (seed 7)
uv run --locked linkcast validate data/synthetic/link_rows.jsonl
```

The same seed always produces identical output. The default run simulates 20
stores, 80 publishers, 150 media, and about 20,000 links over 182 days. Some
stores and media first appear late in the period so later evaluation can test
unseen entities. `--seed`, `--days`, and `--media` change the run.

The output directory (ignored by Git) contains:

- `sources/`: fictional source tables (stores, media, campaigns, creatives,
  landings, links, posts, post-link associations, view and audience
  observations, conversions, tracking reviews), each with occurrence,
  measurement, or receipt timestamps.
- `link_rows.jsonl`: one schema-1.0 row per link. Features use only source
  records available at link creation; labels use only matured, reviewed windows
  with complete tracking.
- `truth/`: hidden simulator variables, such as media quality and each link's
  expected seven-day count. They drive the simulation but never appear in
  sources or link rows; they are for checking the pipeline, not for training.
- `manifest.json`: configuration, generator version, and row counts.

Conversions follow a negative binomial (gamma-Poisson) process: most links get
zero or a few conversions and a few get many. Some post views are unmeasured,
some links have tracking outages or unverified coverage (label `unknown`), and
recent links remain `pending` with partial counts. Synthetic results show that
the pipeline behaves correctly; they say nothing about real business value.

## Inspecting a dataset

```sh
uv run --locked linkcast inspect data/synthetic/link_rows.jsonl
uv run --locked linkcast inspect data/synthetic/link_rows.jsonl --json report.json
```

Run this before splitting or training. The report shows:

- label status per creation month, and the first day whose labels are mostly
  still pending (the latest usable training data ends before it);
- the ready target's mean, variance, zero share, and upper quantiles;
- missing-value shares per feature and month;
- stores, media, campaigns, and creatives first seen each month;
- checks: model inputs exclude labels and audit fields, feature timestamps
  precede creation, one snapshot time, no stale pending labels, unknown-label
  share, constant features, and month-to-month missingness shifts.

Failing checks make the command exit with status 1; warnings are for review.
The report reads only link rows, never `truth/`.
