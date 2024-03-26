import pytest

xarray = pytest.importorskip("xarray")
katdal = pytest.importorskip("katdal")

import dask
import numpy as np
from numpy.testing import assert_array_equal

from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
from daskms.experimental.katdal.msv2_proxy import MSv2DatasetProxy


@pytest.mark.parametrize(
    "dataset", [{"ntime": 20, "nchan": 16, "nant": 4}], indirect=True
)
@pytest.mark.parametrize("auto_corrs", [True])
@pytest.mark.parametrize("row_dim", [True, False])
@pytest.mark.parametrize("out_store", ["output.zarr"])
def test_chunkstore(tmp_path_factory, dataset, auto_corrs, row_dim, out_store):
    proxy = MSv2DatasetProxy(dataset, auto_corrs, row_dim)
    all_antennas = proxy.ants
    xds = list(proxy.scans())

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

    spw = dataset.spectral_windows[dataset.spw]

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
                "NUM_CHAN": (("row",), np.array([spw.num_chans], dtype=np.int32)),
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
    (nbl,) = proxy.cp_info.ant1_index.shape
    ncorr = read_xds.sizes["corr"]

    # This must hold for test_tranpose to work
    assert_array_equal(proxy.cp_info.cp_index.ravel(), np.arange(nbl * ncorr))

    def assert_transposed_equal(a, e):
        """Simple transpose of katdal (time, chan, corrprod) to
        (time, bl, chan, corr)."""
        t = a.reshape(ntime, nchan, nbl, ncorr).transpose(0, 2, 1, 3)
        t = t.reshape(-1, nchan, ncorr) if row_dim else t
        return assert_array_equal(t, e)

    assert_transposed_equal(test_data, read_xds.DATA.values)
    assert_transposed_equal(test_weights, read_xds.WEIGHT_SPECTRUM.values)
    assert_transposed_equal(test_flags, read_xds.FLAG.values)
