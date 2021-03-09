from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import logging.config
from pathlib import Path
import sys


def main():
    return EntryPoint(sys.argv[1:]).execute()


class EntryPoint:
    LOGGING_INI = Path(__file__).parents[0] / "conf" / "logging.ini"

    def __init__(self, cmdline_args):
        self.cmdline_args = cmdline_args

    def execute(self):
        log = self._setup_logging()

        app_klasses = self._application_classes()
        parser = self._create_parsers(app_klasses)
        args = self._parse_args(parser, self.cmdline_args)

        try:
            cmd_klass = app_klasses[args.command]
        except KeyError:
            raise ValueError(f"No implementation class found "
                             f"for command {args.command}")

        cmd = cmd_klass(args, log)
        cmd.execute()

    @classmethod
    def _application_classes(cls):
        from daskms.apps.convert import Convert
        return {"convert": Convert}

    @classmethod
    def _create_parsers(cls, app_klasses):
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        subparsers = parser.add_subparsers(help="command", dest="command")

        for app_name, klass in app_klasses.items():
            app_parser = subparsers.add_parser(app_name)
            klass.setup_parser(app_parser)

        return parser

    @classmethod
    def _parse_args(cls, parser, args):
        parsed_args = parser.parse_args(args)

        if not parsed_args.command:
            parser.print_help()
            sys.exit(0)

        return parsed_args

    @classmethod
    def _setup_logging(cls):
        assert cls.LOGGING_INI.exists()

        logging.config.fileConfig(fname=cls.LOGGING_INI,
                                  disable_existing_loggers=False)
        return logging.getLogger(__name__)
