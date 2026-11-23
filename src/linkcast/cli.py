"""Command-line entry point for LinkCast."""

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path

from linkcast import __version__
from linkcast.artifact import load as load_artifact
from linkcast.artifact import predict_rows, train_deployment
from linkcast.artifact import save as save_artifact
from linkcast.baseline import DEFAULT_K, K_GRID
from linkcast.baseline import evaluate as evaluate_baseline
from linkcast.baseline import format_report as format_baseline
from linkcast.evaluate import evaluate as evaluate_test
from linkcast.evaluate import format_report as format_evaluation
from linkcast.generate import GeneratorConfig, write
from linkcast.generate import generate as generate_data
from linkcast.inspection import format_report, has_failures, inspect_rows
from linkcast.model import evaluate as evaluate_model
from linkcast.model import format_report as format_model_report
from linkcast.schema import SCHEMA_VERSION, SchemaError, read_dataset
from linkcast.splits import (
    SplitConfig,
    assign_splits,
    format_summary,
    summarize,
    write_splits,
)


def _utc_date(value: str) -> dt.datetime:
    """Parse YYYY-MM-DD (or an ISO timestamp) as a UTC time."""
    parsed = dt.datetime.fromisoformat(value)
    return parsed.replace(tzinfo=dt.UTC) if parsed.tzinfo is None else parsed


def _float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def _add_split_dates(command: argparse.ArgumentParser) -> None:
    defaults = SplitConfig()
    for name in ("train_cutoff", "validation_end", "test_end"):
        command.add_argument(
            f"--{name.replace('_', '-')}",
            type=_utc_date,
            default=getattr(defaults, name),
            help=f"UTC date, default {getattr(defaults, name).date()}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="linkcast",
        description="Forecast seven-day purchase conversion counts per link.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command")
    validate = commands.add_parser(
        "validate", help="Validate a schema-v1 JSONL dataset"
    )
    validate.add_argument("dataset", help="Path to the JSONL file")
    inspect = commands.add_parser(
        "inspect", help="Report label maturity, target, missingness, and checks"
    )
    inspect.add_argument("dataset", help="Path to the JSONL file")
    inspect.add_argument("--json", help="Also write the full report as JSON")
    split = commands.add_parser(
        "split", help="Assign chronological train/validation/test splits"
    )
    split.add_argument("dataset", help="Path to the JSONL file")
    split.add_argument(
        "--out", help="CSV path (default: splits.csv next to the dataset)"
    )
    split.add_argument("--json", help="Also write the summary as JSON")
    _add_split_dates(split)
    baseline = commands.add_parser(
        "baseline", help="Fit the historical-mean baseline and score validation"
    )
    baseline.add_argument("dataset", help="Path to the JSONL file")
    baseline.add_argument(
        "--k", type=float, default=DEFAULT_K, help="Default smoothing strength"
    )
    baseline.add_argument(
        "--grid",
        type=_float_list,
        default=K_GRID,
        help="Comma-separated k values to try on validation",
    )
    baseline.add_argument("--json", help="Also write the report as JSON")
    _add_split_dates(baseline)
    train = commands.add_parser(
        "train", help="Tune CatBoost on validation and compare with the baseline"
    )
    train.add_argument("dataset", help="Path to the JSONL file")
    train.add_argument("--json", help="Also write the report as JSON")
    _add_split_dates(train)
    evaluate = commands.add_parser(
        "evaluate", help="Refit chosen models and score the test split once"
    )
    evaluate.add_argument("dataset", help="Path to the JSONL file")
    evaluate.add_argument("--json", help="Also write the report as JSON")
    _add_split_dates(evaluate)
    save = commands.add_parser(
        "save", help="Train the chosen model on all available labels and save it"
    )
    save.add_argument("dataset", help="Path to the JSONL file")
    save.add_argument(
        "--out", default="artifacts/linkcast-model", help="Artifact directory"
    )
    _add_split_dates(save)
    predict_cmd = commands.add_parser(
        "predict", help="Predict expected seven-day conversions for link rows"
    )
    predict_cmd.add_argument("artifact", help="Artifact directory from `save`")
    predict_cmd.add_argument("dataset", help="Schema-v1 JSONL rows at link creation")
    predict_cmd.add_argument("--out", help="CSV path (default: print to stdout)")
    generate = commands.add_parser(
        "generate", help="Generate deterministic fictional source data and link rows"
    )
    generate.add_argument("--out", default="data/synthetic", help="Output directory")
    generate.add_argument("--seed", type=int, default=GeneratorConfig.seed)
    generate.add_argument("--days", type=int, default=GeneratorConfig.days)
    generate.add_argument("--media", type=int, default=GeneratorConfig.n_media)
    args = parser.parse_args()
    if args.command == "validate":
        try:
            rows = read_dataset(args.dataset)
        except (OSError, SchemaError) as error:
            parser.exit(2, f"Validation failed: {error}\n")
        counts = Counter(row.label_status for row in rows)
        print(f"Valid schema {SCHEMA_VERSION}: {len(rows)} links")
        print(
            f"Labels: {counts['ready']} ready, {counts['pending']} pending, "
            f"{counts['unknown']} unknown"
        )
    elif args.command == "inspect":
        try:
            rows = read_dataset(args.dataset)
        except (OSError, SchemaError) as error:
            parser.exit(2, f"Validation failed: {error}\n")
        report = inspect_rows(rows)
        print(format_report(report))
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        if has_failures(report):
            parser.exit(1, "Inspection found failing checks\n")
    elif args.command == "split":
        try:
            rows = read_dataset(args.dataset)
            config = SplitConfig(args.train_cutoff, args.validation_end, args.test_end)
        except (OSError, SchemaError, ValueError) as error:
            parser.exit(2, f"Split failed: {error}\n")
        splits = assign_splits(rows, config)
        summary = summarize(splits, config)
        out = Path(args.out) if args.out else Path(args.dataset).with_name("splits.csv")
        write_splits(splits, out)
        print(format_summary(summary))
        print(f"\nWrote {out}")
        if args.json:
            Path(args.json).write_text(json.dumps(summary, indent=2) + "\n")
    elif args.command == "baseline":
        try:
            rows = read_dataset(args.dataset)
            config = SplitConfig(args.train_cutoff, args.validation_end, args.test_end)
            report = evaluate_baseline(rows, config, args.k, args.grid)
        except (OSError, SchemaError, ValueError) as error:
            parser.exit(2, f"Baseline failed: {error}\n")
        print(format_baseline(report))
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    elif args.command == "train":
        try:
            rows = read_dataset(args.dataset)
            config = SplitConfig(args.train_cutoff, args.validation_end, args.test_end)
        except (OSError, SchemaError, ValueError) as error:
            parser.exit(2, f"Training failed: {error}\n")
        report = evaluate_model(rows, config)
        print(format_model_report(report))
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    elif args.command == "evaluate":
        try:
            rows = read_dataset(args.dataset)
            config = SplitConfig(args.train_cutoff, args.validation_end, args.test_end)
            report = evaluate_test(rows, config)
        except (OSError, SchemaError, ValueError) as error:
            parser.exit(2, f"Evaluation failed: {error}\n")
        print(format_evaluation(report))
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    elif args.command == "save":
        try:
            rows = read_dataset(args.dataset)
            config = SplitConfig(args.train_cutoff, args.validation_end, args.test_end)
        except (OSError, SchemaError, ValueError) as error:
            parser.exit(2, f"Save failed: {error}\n")
        manifest_path = Path(args.dataset).with_name("manifest.json")
        manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        )
        model, metadata = train_deployment(rows, config, manifest=manifest)
        out = save_artifact(model, metadata, args.out)
        print(
            f"Saved {metadata['feature_set']} model "
            f"({len(metadata['features'])} features) to {out}"
        )
        print(
            f"Trained on {metadata['training_links']} links with labels available "
            f"by {metadata['training_cutoff']}"
        )
    elif args.command == "predict":
        try:
            model, metadata = load_artifact(args.artifact)
            rows = read_dataset(args.dataset)
        except (OSError, SchemaError, ValueError) as error:
            parser.exit(2, f"Prediction failed: {error}\n")
        predictions = predict_rows(model, metadata, rows)
        if args.out:
            predictions.to_csv(args.out, index=False)
            print(f"Wrote {len(predictions)} predictions to {args.out}")
        else:
            print(predictions.to_string(index=False))
    elif args.command == "generate":
        try:
            config = GeneratorConfig(seed=args.seed, days=args.days, n_media=args.media)
        except ValueError as error:
            parser.exit(2, f"Invalid configuration: {error}\n")
        out = write(generate_data(config), args.out)
        rows = read_dataset(out / "link_rows.jsonl")  # re-validate what was written
        counts = Counter(row.label_status for row in rows)
        print(f"Wrote {len(rows)} links to {out} (seed {config.seed})")
        print(
            f"Labels: {counts['ready']} ready, {counts['pending']} pending, "
            f"{counts['unknown']} unknown"
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
