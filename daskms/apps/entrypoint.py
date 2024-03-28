import logging

import click

from daskms.apps.convert import convert


@click.group()
@click.pass_context
@click.option("--debug/--no-debug", default=False)
def main(ctx, debug):
    logging.basicConfig(level=logging.INFO)
    ctx.ensure_object(dict)
    ctx.obj["DEBUG"] = debug


main.add_command(convert)
