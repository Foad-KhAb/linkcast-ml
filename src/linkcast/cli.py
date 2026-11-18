"""Command-line entry point for LinkCast."""

import argparse

from linkcast import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="linkcast",
        description="Forecast seven-day purchase conversion counts per link.",
        epilog="Project setup is complete. Dataset and ML commands will follow.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
