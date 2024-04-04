import logging

import click

from daskms.apps.convert import convert
from daskms.apps.katdal_import import _import


@click.group(name="dask-ms")
@click.pass_context
@click.option("--debug/--no-debug", default=False)
def main(ctx, debug):
    logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)
    ctx.ensure_object(dict)
    ctx.obj["DEBUG"] = debug


main.add_command(convert)
main.add_command(_import)
