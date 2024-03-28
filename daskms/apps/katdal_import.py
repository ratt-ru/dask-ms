import click


@click.group()
@click.pass_context
def katdal(ctx):
    """subgroup for katdal commands"""
    pass


class PolarisationListType(click.ParamType):
    name = "polarisation list"
    VALID = {"HH", "HV", "VH", "VV"}

    def convert(self, value, param, ctx):
        if isinstance(value, str):
            value = [p.strip() for p in value.split(",")]
        else:
            raise TypeError(
                f"{value} should be a comma separated string of polarisations"
            )

        if not set(value).issubset(self.VALID):
            raise ValueError(f"{set(value)} is not a subset of {self.VALID}")

        return value


@katdal.command(name="import")
@click.pass_context
@click.argument("rdb_url", required=True)
@click.option(
    "-a",
    "--no-auto",
    flag_value=True,
    default=False,
    help="Exclude auto-correlation data",
)
@click.option(
    "-o",
    "--output-store",
    help="Output store name. Will be derived from the rdb url if not provided.",
    default=None,
)
@click.option(
    "-p",
    "--pols-to-use",
    default="HH,HV,VH,VV",
    help="Select polarisation products to include in MS as "
    "a comma-separated list, containing values from [HH, HV, VH, VV].",
    type=PolarisationListType(),
)
@click.option(
    "--applycal",
    default="",
    help="List of calibration solutions to apply to data as "
    "a string of comma-separated names, e.g. 'l1' or "
    "'K,B,G'. Use 'default' for L1 + L2 and 'all' for "
    "all available products.",
)
def _import(ctx, rdb_url, no_auto, pols_to_use, applycal, output_store):
    """Export an observation in the SARAO archive to zarr formation

    RDB_URL is the SARAO archive link"""
    from daskms.experimental.katdal import katdal_import

    katdal_import(rdb_url, output_store, no_auto, applycal)
