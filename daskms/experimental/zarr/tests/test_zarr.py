import dask
from numpy.testing import assert_array_equal

from daskms import xds_from_ms
from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr


def test_xds_to_zarr(ms, tmp_path_factory):
    zarr_store = tmp_path_factory.mktemp("zarr_store") / "test.zarr"

    ms_datasets = xds_from_ms(ms)
    writes = xds_to_zarr(ms_datasets, zarr_store)
    dask.compute(writes)

    zarr_datasets = xds_from_zarr(zarr_store, chunks={"row": 1})

    for ms_ds, zarr_ds in zip(ms_datasets, zarr_datasets):
        for name, var in ms_ds.data_vars.items():
            assert_array_equal(var.data, getattr(zarr_ds, name).data)


def test_xds_from_zarr(zarr_store):
    zms = xds_from_zarr(zarr_store)  # noqa
