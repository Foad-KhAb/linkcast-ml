#!/usr/bin/env bash
# Run the full LinkCast pipeline on fictional data. Run from the repository root.
#
# Override defaults with environment variables, for example:
#   SEED=11 ./scripts/run_pipeline.sh
set -euo pipefail

DATA="${DATA:-data/synthetic}"
ARTIFACT="${ARTIFACT:-artifacts/linkcast-model}"
REPORTS="${REPORTS:-runs/latest}"
SEED="${SEED:-7}"

run() {
    echo
    echo "==> linkcast $*"
    uv run --locked linkcast "$@"
}

uv sync --locked
mkdir -p "$REPORTS"
ROWS="$DATA/link_rows.jsonl"

run generate --out "$DATA" --seed "$SEED"
run validate "$ROWS"
run inspect "$ROWS" --json "$REPORTS/inspection.json"
run split "$ROWS" --json "$REPORTS/splits.json"
run baseline "$ROWS" --json "$REPORTS/baseline.json"
run train "$ROWS" --json "$REPORTS/train.json"
run evaluate "$ROWS" --json "$REPORTS/evaluation.json"
run save "$ROWS" --out "$ARTIFACT"
run predict "$ARTIFACT" examples/new_links_v1.jsonl \
    --out "$REPORTS/example_predictions.csv"

echo
echo "Done. Data: $DATA  Model: $ARTIFACT  Reports: $REPORTS"
