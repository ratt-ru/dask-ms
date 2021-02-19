import dask
import dask.array as da
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from daskms import xds_from_ms
from daskms.dataset import Dataset
from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr

try:
    import xarray
except ImportError:
    xarray = None


def test_zarr_string_array(tmp_path_factory):
    zarr_store = tmp_path_factory.mktemp("string-arrays") / "test.zarr"

    data = ["hello", "this", "strange new world",
            "full of", "interesting", "stuff"]
    data = np.array(data, dtype=np.object).reshape(3, 2)
    data = da.from_array(data, chunks=((1, 2), (1, 1)))

    datasets = [Dataset({"DATA": (("x", "y"), data)})]
    writes = xds_to_zarr(datasets, zarr_store)
    dask.compute(writes)

    new_datasets = xds_from_zarr(zarr_store)

    assert len(new_datasets) == len(datasets)

    for nds, ds in zip(new_datasets, datasets):
        assert_array_equal(nds.DATA.data, ds.DATA.data)


def test_xds_to_zarr(ms, tmp_path_factory):
    zarr_store = tmp_path_factory.mktemp("zarr_store") / "test.zarr"

    ms_datasets = xds_from_ms(ms)

    for i, ds in enumerate(ms_datasets):
        dims = ds.dims
        row, chan, corr = (dims[d] for d in ("row", "chan", "corr"))

        ms_datasets[i] = ds.assign_coords(**{
            "chan": (("chan",), np.arange(chan)),
            "corr": (("corr",), np.arange(corr)),
        })

    writes = xds_to_zarr(ms_datasets, zarr_store)
    dask.compute(writes)

    zarr_datasets = xds_from_zarr(zarr_store, chunks={"row": 1})

    for ms_ds, zarr_ds in zip(ms_datasets, zarr_datasets):
        # Check data variables
        assert ms_ds.data_vars, "MS Dataset has no variables"

        for name, var in ms_ds.data_vars.items():
            zdata = getattr(zarr_ds, name).data
            assert type(zdata) is type(var.data)  # noqa
            assert_array_equal(var.data, zdata)

        # Check coordinates
        assert ms_ds.coords, "MS Datset has no coordinates"

        for name, var in ms_ds.coords.items():
            zdata = getattr(zarr_ds, name).data
            assert type(zdata) is type(var.data)  # noqa
            assert_array_equal(var.data, zdata)

        # Check dataset attributes
        for k, v in ms_ds.attrs.items():
            zattr = getattr(zarr_ds, k)
            assert_array_equal(zattr, v)


def test_multiprocess_create(ms, tmp_path_factory):
    zarr_store = tmp_path_factory.mktemp("zarr_store") / "test.zarr"

    ms_datasets = xds_from_ms(ms)

    for i, ds in enumerate(ms_datasets):
        ms_datasets[i] = ds.chunk({"row": 1})

    writes = xds_to_zarr(ms_datasets, zarr_store)
    dask.compute(writes, scheduler="processes")

    zds = xds_from_zarr(zarr_store)

    for zds, msds in zip(zds, ms_datasets):
        for k, v in msds.data_vars.items():
            assert_array_equal(v, getattr(zds, k))

        for k, v in msds.coords.items():
            assert_array_equal(v, getattr(zds, k))

        for k, v in msds.attrs.items():
            assert_array_equal(v, getattr(zds, k))


@pytest.mark.optional
@pytest.mark.skipif(xarray is None, reason="depends on xarray")
def test_xarray_to_zarr(ms, tmp_path_factory):
    store = tmp_path_factory.mktemp("zarr_store")
    datasets = xds_from_ms(ms)

    for i, ds in enumerate(datasets):
        chunks = ds.chunks
        row = sum(chunks["row"])
        chan = sum(chunks["chan"])
        corr = sum(chunks["corr"])

        datasets[i] = ds.assign_coords(
            row=np.arange(row),
            chan=np.arange(chan),
            corr=np.arange(corr))

    for i, ds in enumerate(datasets):
        ds.to_zarr(str(store / f"ds-{i}.zarr"))
