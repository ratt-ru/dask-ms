from argparse import ArgumentParser
import logging

log = logging.getLogger(__file__)


@pytest.mark.applications
def test_convert_zarr(tau_ms, tmp_path_factory):
    from daskms.apps.convert import Convert

    OUTPUT = tmp_path_factory.mktemp("convert_zarr") / "output.zarr"
    args = [
        str(tau_ms),
        # "-g",
        # "FIELD_ID,DATA_DESC_ID,SCAN_NUMBER",
        "-x",
        "ASDM_ANTENNA::*,ASDM_CALATMOSPHERE::*,ASDM_CALWVR::*,ASDM_RECEIVER::*,ASDM_SOURCE::*,ASDM_STATION::*",
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
