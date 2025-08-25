import pytest

xarray = pytest.importorskip("xarray")
katdal = pytest.importorskip("katdal")
katpoint = pytest.importorskip("katpoint")

import dask
import numpy as np
from numpy.testing import assert_array_equal

from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
from daskms.experimental.katdal.msv2_facade import XArrayMSv2Facade


@pytest.mark.parametrize(
    "katdal_dataset",
    [
        {
            "ntime": 20,
            "nchan": 16,
            "nant": 4,
            "targets": [
                katpoint.Target(
                    "J1939-6342 | PKS1934-638, radec bpcal, 19:39:25.03, -63:42:45.6"
                )
            ],
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("auto_corrs", [True])
@pytest.mark.parametrize("row_dim", [True, False])
@pytest.mark.parametrize("out_store", ["output.zarr"])
def test_katdal_import(
    tmp_path_factory, katdal_dataset, auto_corrs, row_dim, out_store
):
    facade = XArrayMSv2Facade(katdal_dataset, not auto_corrs, row_dim)
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

    test_data = katdal_dataset._test_data["correlator_data"]
    # Defer to ChunkStoreVisWeights application of weight scaling
    test_weights = katdal_dataset._vfw.weights
    assert test_weights.shape == test_data.shape
    # Clamp test data to [0, 1]
    test_flags = np.where(katdal_dataset._test_data["flags"] != 0, 1, 0)
    ntime, nchan, _ = test_data.shape
    (nbl,) = facade.cp_info.ant1_index.shape
    ncorr = read_xds.sizes["corr"]

    # This must hold for test_tranpose to work
    assert_array_equal(facade.cp_info.cp_index.ravel(), np.arange(nbl * ncorr))

    def assert_transposed_equal(a, e):
        """Simple transpose of katdal (time, chan, corrprod) to
        (time, bl, chan, corr)."""
        # MinimalDataset uses 1 timestap as a slew scan
        # which is not returned here
        t = a.reshape(ntime - 1, nchan, nbl, ncorr).transpose(0, 2, 1, 3)
        t = t.reshape(-1, nchan, ncorr) if row_dim else t
        return assert_array_equal(t, e)

    assert_transposed_equal(test_data[1:, ...], read_xds.DATA.values)
    assert_transposed_equal(test_weights[1:, ...], read_xds.WEIGHT_SPECTRUM.values)
    assert_transposed_equal(test_flags[1:, ...], read_xds.FLAG.values)


@pytest.mark.parametrize(
    "katdal_dataset",
    [
        {
            "ntime": 100,
            "nchan": 16,
            "nant": 4,
            "targets": [
                katpoint.Target(
                    "J1939-6342 | PKS1934-638, radec bpcal, 19:39:25.03, -63:42:45.6"
                )
            ],
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "auto_corrs, chunks, expected_chunks",
    [
        # rows promoted to number of baselines if less than number of baselines
        (True, {"row": 1, "chan": 16}, {"row": 10, "chan": 16}),
        (False, {"row": 1, "chan": 16}, {"row": 6, "chan": 16}),
        # rows round down to a multiple of number of baselines
        (True, {"row": 11, "chan": 16}, {"row": 10, "chan": 16}),
        (True, {"row": 19, "chan": 16}, {"row": 10, "chan": 16}),
        (True, {"row": 21, "chan": 16}, {"row": 20, "chan": 16}),
        (False, {"row": 11, "chan": 16}, {"row": 6, "chan": 16}),
        (False, {"row": 12, "chan": 16}, {"row": 12, "chan": 16}),
        (False, {"row": 19, "chan": 16}, {"row": 18, "chan": 16}),
    ],
)
def test_facade_chunking(
    tmp_path_factory, katdal_dataset, auto_corrs, chunks, expected_chunks
):
    expected_chunks.update((("corr", 4), ("uvw", 3)))
    row_dim = "row" in chunks
    facade = XArrayMSv2Facade(katdal_dataset, not auto_corrs, row_dim, [chunks])
    xds, _ = facade.xarray_datasets()
    assert len(xds) == 1

    from dask.array.core import normalize_chunks

    for k, v in expected_chunks.items():
        (expected_chunks[k],) = normalize_chunks(v, (xds[0].sizes[k],))

    assert expected_chunks == xds[0].chunks, chunks
