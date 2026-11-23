# LinkCast

An end-to-end ML learning project for predicting seven-day purchase conversion
counts from information available immediately after a tracking link is created.
A prediction may be fractional because it represents an expected count.

## Current status

Implemented so far: standalone package and environment, versioned link
dataset contract, validation CLI, fictional example rows, a deterministic
synthetic data generator, dataset inspection report, chronological splits, a
historical-mean baseline, a CatBoost count model compared on validation, and
tests. Final test evaluation, saved models, and prediction are not implemented
yet.

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
src/linkcast/      Package: schema, generator, inspection, splits, metrics, models, CLI
examples/         Hand-written fictional schema examples
tests/            Tests for each module and the CLI
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

## Chronological splits

```sh
uv run --locked linkcast split data/synthetic/link_rows.jsonl
```

This writes `splits.csv` next to the dataset, with one row per link. Defaults
match the default synthetic dataset and can be changed with `--train-cutoff`,
`--validation-end`, and `--test-end` (UTC dates):

| Set | Rule (defaults) | Use |
| --- | --- | --- |
| `train` | label available by 2026-05-01 | Fit models compared on validation |
| `validation` | ready label, created 2026-05-01 to 2026-05-31 | Compare and tune models |
| `in_final_train` | label available by 2026-06-01 | Refit once before testing |
| `test` | ready label, created 2026-06-01 to 2026-06-22 | Final score, used once |
| `excluded` | everything else, with a reason | Never fitted or scored |

Training eligibility depends on when the label became **available**, not on
when the link was created. Links created in the roughly ten days before a
cutoff are excluded (`label_after_cutoff`) because their seven-day outcomes were
not known yet. Pending and unknown labels are never trained on or scored.
Validation and test rows are flagged when their store, media, campaign, or
creative never appears in their training set.

## Historical-mean baseline

```sh
uv run --locked linkcast baseline data/synthetic/link_rows.jsonl
```

The baseline predicts a link's expected seven-day count from training-set
averages, falling back from media to store to platform to the global mean when
a group is absent. Each level is smoothed toward the level above:

```text
estimate = (group conversions + k * parent estimate) / (group links + k)
```

so a media with few links stays close to its store's estimate. The command
fits on `train`, scores `validation`, and reports every score below for the
default `k` (10), every `k` in `--grid`, and the one with the lowest Poisson
deviance. It also reports constant train-mean and train-median predictions and
scores by fallback level. The test split is not read.

MAE is minimized by a median, whereas this baseline predicts means. On skewed
counts, a constant median can therefore beat an informative mean-based model
on MAE alone.

## Scores

Predictions are expected counts, used to estimate total conversions. Every
model is scored with:

| Score | Meaning |
| --- | --- |
| MAE | Mean absolute error in conversions per link |
| Poisson deviance | Quality of predicted averages for counts; lower is better; used to choose settings |
| Bias | Mean prediction minus mean actual; near zero means totals are not systematically off |
| Total error | \|predicted total - actual total\| / actual total, overall and summed per store and per campaign |

## CatBoost model

```sh
uv run --locked linkcast train data/synthetic/link_rows.jsonl
```

This trains CatBoost with Poisson loss on two feature sets: the default 10
creation-time inputs and all 29 allowed candidates. For each set it tries tree
depths 4, 6, and 8 with learning rates 0.03 and 0.1, uses early stopping on
validation to choose the number of trees, and keeps the setting with the lowest
validation Poisson deviance. Identity fields are categorical; unknown
categories become an explicit missing category and unknown numbers stay
missing. The report compares both models with the baseline and lists feature
importances. The test split is not read. A run takes about a minute on a CPU.
