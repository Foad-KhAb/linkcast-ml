"""Command-line entry point for LinkCast."""

import argparse
import json
from collections import Counter
from pathlib import Path

from linkcast import __version__
from linkcast.generate import GeneratorConfig, write
from linkcast.generate import generate as generate_data
from linkcast.inspection import format_report, has_failures, inspect_rows
from linkcast.schema import SCHEMA_VERSION, SchemaError, read_dataset


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
