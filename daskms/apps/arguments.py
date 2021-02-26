from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import sys


def add_convert_parser(subparser):
    p = subparser.add_parser("convert")
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("-f", "--format", choices=["ms", "zarr", "parquet"])


def parser_factory():
    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    sp = p.add_subparsers(help="command", dest="command")
    add_convert_parser(sp)
    return p


def parse_args(args):
    parser = parser_factory()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        sys.exit(0)

    return parsed_args
