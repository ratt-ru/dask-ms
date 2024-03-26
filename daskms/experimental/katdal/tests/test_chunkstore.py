from functools import partial

import pytest

pytest.importorskip("dask.array")
pytest.importorskip("xarray")
pytest.importorskip("katdal")

import dask
import dask.array as da
import numba
import numpy as np
from numpy.testing import assert_array_equal
import xarray

from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr

from katdal.chunkstore_npy import NpyFileChunkStore
from katdal.dataset import Subarray
from katdal.lazy_indexer import DaskLazyIndexer
from katdal.spectral_window import SpectralWindow
from katdal.vis_flags_weights import ChunkStoreVisFlagsWeights
from katdal.test.test_vis_flags_weights import put_fake_dataset
from katdal.test.test_dataset import MinimalDataSet
from katpoint import Antenna, Target, Timestamp


from daskms.experimental.katdal.meerkat_antennas import MEERKAT_ANTENNA_DESCRIPTIONS
from daskms.experimental.katdal.transpose import transpose
from daskms.experimental.katdal.corr_products import corrprod_index
from daskms.experimental.katdal.uvw import uvw_coords

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


@pytest.mark.parametrize(
    "dataset", [{"ntime": 20, "nchan": 16, "nant": 4}], indirect=True
)
@pytest.mark.parametrize("include_auto_corrs", [True])
@pytest.mark.parametrize("row_dim", [True, False])
@pytest.mark.parametrize("out_store", ["output.zarr"])
def test_chunkstore(tmp_path_factory, dataset, include_auto_corrs, row_dim, out_store):
    # Example using an actual mvf4 dataset, make sure to replace the token with a real one
    # url = "https://archive-gw-1.kat.ac.za/1711249692/1711249692_sdp_l0.full.rdb?token=abcdef1234567890"
    # import katdal
    # dataset = katdal.open(url, applycal="l1")
    cp_info = corrprod_index(dataset, ["HH", "HV", "VH", "VV"], include_auto_corrs)
    all_antennas = dataset.ants

    xds = []

    for scan_index, scan_state, target in dataset.scans():
        # Extract numpy and dask products
        time_utc = dataset.timestamps
        t_chunks, chan_chunks, cp_chunks = dataset.vis.dataset.chunks

        # Modified Julian Date in Seconds
        time_mjds = np.asarray(
            [t.to_mjd() * 24 * 60 * 60 for t in map(Timestamp, time_utc)]
        )

        # Create a dask chunking transform
        rechunk = partial(da.rechunk, chunks=(t_chunks, chan_chunks, cp_chunks))

        # Transpose from (time, chan, corrprod) to (time, bl, chan, corr)
        cpi = cp_info.cp_index
        flag_transpose = partial(
            transpose,
            cp_index=cpi,
            data_type=numba.literally("flags"),
            row=row_dim,
        )
        weight_transpose = partial(
            transpose,
            cp_index=cpi,
            data_type=numba.literally("weights"),
            row=row_dim,
        )
        vis_transpose = partial(
            transpose, cp_index=cpi, data_type=numba.literally("vis"), row=row_dim
        )

        flags = DaskLazyIndexer(dataset.flags, (), (rechunk, flag_transpose))
        weights = DaskLazyIndexer(dataset.weights, (), (rechunk, weight_transpose))
        vis = DaskLazyIndexer(dataset.vis, (), transforms=(vis_transpose,))

        time = da.from_array(time_mjds[:, None], chunks=(t_chunks, 1))
        ant1 = da.from_array(cp_info.ant1_index[None, :], chunks=(1, cpi.shape[0]))
        ant2 = da.from_array(cp_info.ant2_index[None, :], chunks=(1, cpi.shape[0]))

        uvw = uvw_coords(
            target,
            da.from_array(time_utc, chunks=t_chunks),
            all_antennas,
            cp_info,
            row=row_dim,
        )

        time, ant1, ant2 = da.broadcast_arrays(time, ant1, ant2)

        if row_dim:
            primary_dims = ("row",)
            time = time.ravel().rechunk({0: vis.dataset.chunks[0]})
            ant1 = ant1.ravel().rechunk({0: vis.dataset.chunks[0]})
            ant2 = ant2.ravel().rechunk({0: vis.dataset.chunks[0]})
        else:
            primary_dims = ("time", "baseline")

        xds.append(
            xarray.Dataset(
                {
                    # Primary indexing columns
                    "TIME": (primary_dims, time),
                    "ANTENNA1": (primary_dims, ant1),
                    "ANTENNA2": (primary_dims, ant2),
                    "FEED1": (primary_dims, da.zeros_like(ant1)),
                    "FEED2": (primary_dims, da.zeros_like(ant1)),
                    # TODO(sjperkins)
                    # Fill these in with real values
                    "DATA_DESC_ID": (primary_dims, da.zeros_like(ant1)),
                    "FIELD_ID": (primary_dims, da.zeros_like(ant1)),
                    "STATE_ID": (primary_dims, da.zeros_like(ant1)),
                    "ARRAY_ID": (primary_dims, da.zeros_like(ant1)),
                    "OBSERVATION_ID": (primary_dims, da.zeros_like(ant1)),
                    "PROCESSOR_ID": (primary_dims, da.ones_like(ant1)),
                    "SCAN_NUMBER": (primary_dims, da.full_like(ant1, scan_index)),
                    "TIME_CENTROID": (primary_dims, time),
                    "INTERVAL": (primary_dims, da.full_like(time, dataset.dump_period)),
                    "EXPOSURE": (primary_dims, da.full_like(time, dataset.dump_period)),
                    "UVW": (primary_dims + ("uvw",), uvw),
                    "DATA": (primary_dims + ("chan", "corr"), vis.dataset),
                    "FLAG": (primary_dims + ("chan", "corr"), flags.dataset),
                    "WEIGHT_SPECTRUM": (
                        primary_dims + ("chan", "corr"),
                        weights.dataset,
                    ),
                    # Estimated RMS noise per frequency channel
                    # note this column is used when computing calibration weights
                    # in CASA - WEIGHT_SPECTRUM may be modified based on the
                    # values in this column. See
                    # https://casadocs.readthedocs.io/en/stable/notebooks/data_weights.html
                    # for further details
                    "SIGMA_SPECTRUM": (
                        primary_dims + ("chan", "corr"),
                        weights.dataset**-0.5,
                    ),
                }
            )
        )

    xds = xarray.concat(xds, dim=primary_dims[0])

    ant_xds = [
        xarray.Dataset(
            {
                "NAME": (("row",), np.asarray([a.name for a in all_antennas])),
                "OFFSET": (
                    ("row", "xyz"),
                    np.asarray(np.zeros((len(all_antennas), 3))),
                ),
                "POSITION": (
                    ("row", "xyz"),
                    np.asarray([a.position_ecef for a in all_antennas]),
                ),
                "DISH_DIAMETER": (
                    ("row",),
                    np.asarray([a.diameter for a in all_antennas]),
                ),
                # "FLAG_ROW": (("row","xyz"),
                #             np.zeros([a.flags for a in all_antennas],np.uint8)
                # )
            }
        )
    ]

    spw_xds = [
        xarray.Dataset(
            {
                "CHAN_FREQ": (("row", "chan"), dataset.channel_freqs[np.newaxis, :]),
                "CHAN_WIDTH": (
                    ("row", "chan"),
                    np.full_like(dataset.channel_freqs, dataset.channel_width)[
                        np.newaxis, :
                    ],
                ),
                "EFFECTIVE_BW": (
                    ("row", "chan"),
                    np.full_like(dataset.channel_freqs, dataset.channel_width)[
                        np.newaxis, :
                    ],
                ),
                "FLAG_ROW": (("row",), np.zeros(1, dtype=np.int32)),
                "NUM_CHAN": (("row",), np.array([SPW.num_chans], dtype=np.int32)),
            }
        )
    ]

    print(spw_xds)
    print(ant_xds)
    print(xds)

    # Reintroduce the shutil.rmtree and remote the tmp_path_factory
    # to test in the local directory
    # shutil.rmtree(out_store, ignore_errors=True)
    out_store = tmp_path_factory.mktemp("output") / out_store

    dask.compute(xds_to_zarr(xds, out_store))
    dask.compute(xds_to_zarr(ant_xds, f"{out_store}::ANTENNA"))

    # Compare visibilities, weights and flags
    (read_xds,) = dask.compute(xds_from_zarr(out_store))
    (read_xds,) = read_xds

    test_data = dataset._test_data["correlator_data"]
    # Defer to ChunkStoreVisWeights application of weight scaling
    test_weights = dataset._vfw.weights
    assert test_weights.shape == test_data.shape
    # Clamp test data to [0, 1]
    test_flags = np.where(dataset._test_data["flags"] != 0, 1, 0)
    ntime, nchan, _ = test_data.shape
    (nbl,) = cp_info.ant1_index.shape
    ncorr = read_xds.sizes["corr"]

    # This must hold for test_tranpose to work
    assert_array_equal(cp_info.cp_index.ravel(), np.arange(nbl * ncorr))

    def test_transpose(a):
        """Simple transpose of katdal (time, chan, corrprod) to
        (time, bl, chan, corr)."""
        o = a.reshape(ntime, nchan, nbl, ncorr).transpose(0, 2, 1, 3)
        return o.reshape(-1, nchan, ncorr) if row_dim else o

    assert_array_equal(test_transpose(test_data), read_xds.DATA.values)
    assert_array_equal(test_transpose(test_weights), read_xds.WEIGHT_SPECTRUM.values)
    assert_array_equal(test_transpose(test_flags), read_xds.FLAG.values)
