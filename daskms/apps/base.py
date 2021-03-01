from daskms.apps.arguments import parse_args
import logging
from pathlib import Path
from pprint import pformat
import sys


LOGGING_INI = Path(__file__).parents[0] / "conf" / "logging.ini"


def main():
    return Application(sys.argv[1:]).execute()


class Application:
    def __init__(self, cmdline_args):
        self.cmdline_args = cmdline_args

    def execute(self):
        assert LOGGING_INI.exists()

        logging.config.fileConfig(fname=LOGGING_INI,
                                  disable_existing_loggers=False)
        log = logging.getLogger(__name__)

        args = parse_args(self.cmdline_args)
        log.info(f"dask-ms {pformat(vars(args))}")