"""Command-line entry point for LinkCast."""

import argparse
from collections import Counter

from linkcast import __version__
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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
