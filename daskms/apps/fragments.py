import click
import dask
from daskms.fsspec_store import DaskMSStore
from daskms.experimental.fragments import get_ancestry
from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr


@click.group(help="Base command for interacting with fragments.")
def fragments():
    pass


@click.command(help="List fragment and parents.")
@click.argument(
    "fragment_path",
    type=DaskMSStore,
)
@click.option(
    "-p/-np",
    "--prune/--no-prune",
    default=False,
)
def stat(fragment_path, prune):
    ancestors = get_ancestry(fragment_path, only_required=prune)

    click.echo("Ancestry:")

    for i, fg in enumerate(ancestors):
        if i == 0:
            click.echo(f"   {fg.full_path} ---> root")
        elif i == len(ancestors) - 1:
            click.echo(f"   {fg.full_path} ---> target")
        else:
            click.echo(f"   {fg.full_path}")


@click.command(help="Change fragment parent.")
@click.argument(
    "fragment_path",
    type=DaskMSStore,
)
@click.argument(
    "parent_path",
    type=DaskMSStore,
)
def rebase(fragment_path, parent_path):
    xdsl = xds_from_zarr(fragment_path, columns=[])

    xdsl = [
        xds.assign_attrs({"__dask_ms_parent_url__": parent_path.url}) for xds in xdsl
    ]

    writes = xds_to_zarr(xdsl, fragment_path)

    dask.compute(writes)


fragments.add_command(stat)
fragments.add_command(rebase)


def main():
    fragments()
