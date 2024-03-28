import logging

from click.testing import CliRunner
from daskms.apps.entrypoint import main

from daskms import xds_from_storage_ms, xds_from_storage_table

import pytest

log = logging.getLogger(__file__)


@pytest.mark.applications
@pytest.mark.parametrize("format", ["zarr"])
def test_convert_application(tau_ms, format, tmp_path_factory):
    OUTPUT = tmp_path_factory.mktemp(f"convert_{format}") / f"output.{format}"

    exclude_columns = [
        "ASDM_ANTENNA::*",
        "ASDM_CALATMOSPHERE::*",
        "ASDM_CALWVR::*",
        "ASDM_RECEIVER::*",
        "ASDM_SOURCE::*",
        "ASDM_STATION::*",
        "POINTING::OVER_THE_TOP",
        "SPECTRAL_WINDOW::ASSOC_SPW_ID",
        "SPECTRAL_WINDOW::ASSOC_NATURE",
        "MODEL_DATA",
    ]

    args = [
        str(tau_ms),
        "-g",
        "FIELD_ID,DATA_DESC_ID,SCAN_NUMBER",
        "-x",
        ",".join(exclude_columns),
        "-o",
        str(OUTPUT),
        "--format",
        format,
        "--force",
    ]

    runner = CliRunner()
    result = runner.invoke(main, ["convert"] + args)
    assert result.exit_code == 0

    for ds in xds_from_storage_ms(OUTPUT):
        assert "MODEL_DATA" not in ds.data_vars
        assert "FLAG" in ds.data_vars
        assert "ROWID" in ds.coords

    for ds in xds_from_storage_table(f"{OUTPUT}::POINTING"):
        assert "OVER_THE_TOP" not in ds.data_vars
        assert "NAME" in ds.data_vars

    for ds in xds_from_storage_table(f"{OUTPUT}::SPECTRAL_WINDOW"):
        assert "CHAN_FREQ" in ds.data_vars
        assert "CHAN_WIDTH" in ds.data_vars
        assert "ASSOC_SPW_ID" not in ds.data_vars
        assert "ASSOC_NATURE" not in ds.data_vars
