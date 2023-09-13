from argparse import ArgumentParser
import logging

from daskms.apps.convert import Convert
from daskms import xds_from_storage_ms, xds_from_storage_table

import pytest

log = logging.getLogger(__file__)


@pytest.mark.applications
@pytest.mark.parametrize("format", ["ms", "zarr", "parquet"])
def test_convert_application(tau_ms, format, tmp_path_factory):
    OUTPUT = tmp_path_factory.mktemp(f"convert_{format}") / "output.{format}"

    exclude_columns = [
        "ASDM_ANTENNA::*",
        "ASDM_CALATMOSPHERE::*",
        "ASDM_CALWVR::*",
        "ASDM_RECEIVER::*",
        "ASDM_SOURCE::*",
        "ASDM_STATION::*",
        "POINTING::OVER_THE_TOP",
        "MODEL_DATA",
    ]

    args = [
        str(tau_ms),
        # "-g",
        # "FIELD_ID,DATA_DESC_ID,SCAN_NUMBER",
        "-x",
        ",".join(exclude_columns),
        "-o",
        str(OUTPUT),
        "--format",
        "zarr",
        "--force",
    ]

    p = ArgumentParser()
    Convert.setup_parser(p)
    args = p.parse_args(args)
    app = Convert(args, log)
    app.execute()

    datasets = xds_from_storage_ms(OUTPUT)

    for ds in datasets:
        assert "MODEL_DATA" not in ds.data_vars
        assert "FLAG" in ds.data_vars
        assert "ROWID" in ds.coords

    datasets = xds_from_storage_table(f"{str(OUTPUT)}::POINTING")

    for ds in datasets:
        assert "OVER_THE_TOP" not in ds.data_vars
        assert "NAME" in ds.data_vars
