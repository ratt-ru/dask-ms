import pytest

katdal = pytest.importorskip("katdal")

import numpy as np

from katdal.lazy_indexer import DaskLazyIndexer
from katdal.chunkstore_npy import NpyFileChunkStore
from katdal.dataset import Subarray
from katdal.spectral_window import SpectralWindow
from katdal.vis_flags_weights import ChunkStoreVisFlagsWeights
from katdal.test.test_vis_flags_weights import put_fake_dataset
from katdal.test.test_dataset import MinimalDataSet
from katpoint import Antenna, Target, Timestamp


from daskms.experimental.katdal.meerkat_antennas import MEERKAT_ANTENNA_DESCRIPTIONS

SPW = SpectralWindow(
    centre_freq=1284e6, channel_width=0, num_chans=16, sideband=1, bandwidth=856e6
)


class FakeDataset(MinimalDataSet):
    def __init__(
        self,
        path,
        targets,
        timestamps,
        antennas=MEERKAT_ANTENNA_DESCRIPTIONS,
        spw=SPW,
    ):
        antennas = list(map(Antenna, antennas))
        corr_products = [
            (a1.name + c1, a2.name + c2)
            for i, a1 in enumerate(antennas)
            for a2 in antennas[i:]
            for c1 in ("h", "v")
            for c2 in ("h", "v")
        ]

        subarray = Subarray(antennas, corr_products)
        assert len(subarray.ants) > 0

        store = NpyFileChunkStore(str(path))
        shape = (len(timestamps), spw.num_chans, len(corr_products))
        self._test_data, chunk_info = put_fake_dataset(
            store,
            "cb1",
            shape,
            chunk_overrides={
                "correlator_data": (1, spw.num_chans, len(corr_products)),
                "flags": (1, spw.num_chans, len(corr_products)),
                "weights": (1, spw.num_chans, len(corr_products)),
            },
        )
        self._vfw = ChunkStoreVisFlagsWeights(store, chunk_info)
        self._vis = None
        self._weights = None
        self._flags = None
        super().__init__(targets, timestamps, subarray, spw)

    def _set_keep(
        self,
        time_keep=None,
        freq_keep=None,
        corrprod_keep=None,
        weights_keep=None,
        flags_keep=None,
    ):
        super()._set_keep(time_keep, freq_keep, corrprod_keep, weights_keep, flags_keep)
        stage1 = (time_keep, freq_keep, corrprod_keep)
        self._vis = DaskLazyIndexer(self._vfw.vis, stage1)
        self._weights = DaskLazyIndexer(self._vfw.weights, stage1)
        self._flags = DaskLazyIndexer(self._vfw.flags, stage1)

    @property
    def vis(self):
        if self._vis is None:
            raise ValueError("Selection has not yet been performed")
        return self._vis

    @property
    def flags(self):
        if self._flags is None:
            raise ValueError("Selection has not yet been performed")
        return self._flags

    @property
    def weights(self):
        if self._weights is None:
            raise ValueError("Selection has not yet been performed")

        return self._weights


NTIME = 20
NCHAN = 16
NANT = 4
DUMP_RATE = 8.0
DEFAULT_PARAM = {"ntime": NTIME, "nchan": NCHAN, "nant": NANT, "dump_rate": DUMP_RATE}


@pytest.fixture(scope="session", params=[DEFAULT_PARAM])
def dataset(request, tmp_path_factory):
    path = tmp_path_factory.mktemp("chunks")
    targets = [
        # It would have been nice to have radec = 19:39, -63:42 but then
        # selection by description string does not work because the catalogue's
        # description string pads it out to radec = 19:39:00.00, -63:42:00.0.
        # (XXX Maybe fix Target comparison in katpoint to support this?)
        Target("J1939-6342 | PKS1934-638, radec bpcal, 19:39:25.03, -63:42:45.6"),
        Target("J1939-6342, radec gaincal, 19:39:25.03, -63:42:45.6"),
        Target("J0408-6545 | PKS 0408-65, radec bpcal, 4:08:20.38, -65:45:09.1"),
        Target("J1346-6024 | Cen B, radec, 13:46:49.04, -60:24:29.4"),
    ]
    ntime = request.param.get("ntime", NTIME)
    nchan = request.param.get("nchan", NCHAN)
    nant = request.param.get("nant", NANT)
    dump_rate = request.param.get("dump_rate", DUMP_RATE)

    # Ensure that len(timestamps) is an integer multiple of len(targets)
    timestamps = 1234667890.0 + dump_rate * np.arange(ntime)

    assert divmod(ntime, len(targets))[-1] == 0

    spw = SpectralWindow(
        centre_freq=1284e6,
        channel_width=0,
        num_chans=nchan,
        sideband=1,
        bandwidth=856e6,
    )

    return FakeDataset(
        path, targets, timestamps, antennas=MEERKAT_ANTENNA_DESCRIPTIONS[:nant], spw=spw
    )
