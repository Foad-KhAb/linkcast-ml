"""Command-line entry point for LinkCast."""

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path

from linkcast import __version__
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
