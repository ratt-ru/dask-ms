from daskms.apps.arguments import parse_args
import sys


def main():
    return Application(sys.argv[1:]).execute()


class Application:
    def __init__(self, cmdline_args):
        self.cmdline_args = cmdline_args

    def execute(self):
        args = parse_args(self.cmdline_args)

        from pprint import pprint
        pprint(vars(args))
