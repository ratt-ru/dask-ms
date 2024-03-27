import pytest

from daskms.experimental.katdal.meerkat_antennas import MEERKAT_ANTENNA_DESCRIPTIONS
import numpy as np

NTIME = 20
NCHAN = 16
NANT = 4
DUMP_RATE = 8.0

DEFAULT_PARAM = {"ntime": NTIME, "nchan": NCHAN, "nant": NANT, "dump_rate": DUMP_RATE}


@pytest.fixture(scope="session", params=[DEFAULT_PARAM])
def dataset(request, tmp_path_factory):
    MockDataset = pytest.importorskip(
        "daskms.experimental.katdal.mock_dataset"
    ).MockDataset
    SpectralWindow = pytest.importorskip("katdal.spectral_window").SpectralWindow
    Target = pytest.importorskip("katpoint").Target

    DEFAULT_TARGETS = [
        # It would have been nice to have radec = 19:39, -63:42 but then
        # selection by description string does not work because the catalogue's
        # description string pads it out to radec = 19:39:00.00, -63:42:00.0.
        # (XXX Maybe fix Target comparison in katpoint to support this?)
        Target("J1939-6342 | PKS1934-638, radec bpcal, 19:39:25.03, -63:42:45.6"),
        Target("J1939-6342, radec gaincal, 19:39:25.03, -63:42:45.6"),
        Target("J0408-6545 | PKS 0408-65, radec bpcal, 4:08:20.38, -65:45:09.1"),
        Target("J1346-6024 | Cen B, radec, 13:46:49.04, -60:24:29.4"),
    ]
    targets = request.param.get("targets", DEFAULT_TARGETS)
    ntime = request.param.get("ntime", NTIME)
    nchan = request.param.get("nchan", NCHAN)
    nant = request.param.get("nant", NANT)
    dump_rate = request.param.get("dump_rate", DUMP_RATE)

    # Ensure that len(timestamps) is an integer multiple of len(targets)
    timestamps = 1234667890.0 + dump_rate * np.arange(ntime)
    assert ntime > len(targets)
    assert ntime % len(targets) == 0

    spw = SpectralWindow(
        centre_freq=1284e6,
        channel_width=0,
        num_chans=nchan,
        sideband=1,
        bandwidth=856e6,
    )

    return MockDataset(
        tmp_path_factory.mktemp("chunks"),
        targets,
        timestamps,
        antennas=MEERKAT_ANTENNA_DESCRIPTIONS[:nant],
        spw=spw,
    )
