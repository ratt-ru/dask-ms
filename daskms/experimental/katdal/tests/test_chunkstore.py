import pytest

xarray = pytest.importorskip("xarray")
katdal = pytest.importorskip("katdal")

import dask
import numpy as np
from numpy.testing import assert_array_equal

from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
from daskms.experimental.katdal.msv2_facade import XarrayMSV2Facade


@pytest.mark.parametrize(
    "dataset", [{"ntime": 20, "nchan": 16, "nant": 4}], indirect=True
)
@pytest.mark.parametrize("auto_corrs", [True])
@pytest.mark.parametrize("row_dim", [True, False])
@pytest.mark.parametrize("out_store", ["output.zarr"])
def test_chunkstore(tmp_path_factory, dataset, auto_corrs, row_dim, out_store):
    facade = XarrayMSV2Facade(dataset, not auto_corrs, row_dim)
    xds, sub_xds = facade.xarray_datasets()

    # Reintroduce the shutil.rmtree and remote the tmp_path_factory
    # to test in the local directory
    # shutil.rmtree(out_store, ignore_errors=True)
    out_store = tmp_path_factory.mktemp("output") / out_store

    writes = [
        xds_to_zarr(xds, out_store),
        *(xds_to_zarr(ds, f"{out_store}::{k}") for k, ds in sub_xds.items()),
    ]
    dask.compute(writes)

    # Compare visibilities, weights and flags
    (read_xds,) = dask.compute(xds_from_zarr(out_store))
    read_xds = xarray.concat(read_xds, dim="row" if row_dim else "time")

    test_data = dataset._test_data["correlator_data"]
    # Defer to ChunkStoreVisWeights application of weight scaling
    test_weights = dataset._vfw.weights
    assert test_weights.shape == test_data.shape
    # Clamp test data to [0, 1]
    test_flags = np.where(dataset._test_data["flags"] != 0, 1, 0)
    ntime, nchan, _ = test_data.shape
    (nbl,) = facade.cp_info.ant1_index.shape
    ncorr = read_xds.sizes["corr"]

    # This must hold for test_tranpose to work
    assert_array_equal(facade.cp_info.cp_index.ravel(), np.arange(nbl * ncorr))

    def assert_transposed_equal(a, e):
        """Simple transpose of katdal (time, chan, corrprod) to
        (time, bl, chan, corr)."""
        t = a.reshape(ntime, nchan, nbl, ncorr).transpose(0, 2, 1, 3)
        t = t.reshape(-1, nchan, ncorr) if row_dim else t
        return assert_array_equal(t, e)

    assert_transposed_equal(test_data, read_xds.DATA.values)
    assert_transposed_equal(test_weights, read_xds.WEIGHT_SPECTRUM.values)
    assert_transposed_equal(test_flags, read_xds.FLAG.values)
