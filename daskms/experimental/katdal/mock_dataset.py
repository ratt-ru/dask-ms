from katdal.lazy_indexer import DaskLazyIndexer
from katdal.chunkstore_npy import NpyFileChunkStore
from katdal.dataset import Subarray
from katdal.spectral_window import SpectralWindow
from katdal.vis_flags_weights import ChunkStoreVisFlagsWeights
from katdal.test.test_vis_flags_weights import put_fake_dataset
from katdal.test.test_dataset import MinimalDataSet
from katpoint import Antenna


from daskms.experimental.katdal.meerkat_antennas import MEERKAT_ANTENNA_DESCRIPTIONS

SPW = SpectralWindow(
    centre_freq=1284e6, channel_width=0, num_chans=16, sideband=1, bandwidth=856e6
)


class MockDataset(MinimalDataSet):
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
