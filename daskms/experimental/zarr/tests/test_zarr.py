import dask
import dask.array as da
import numpy as np
from numpy.testing import assert_array_equal

from daskms import xds_from_ms
from daskms.dataset import Dataset
from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr


def test_string_array(tmp_path_factory):
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

    ms_datasets = dask.persist(xds_from_ms(ms))[0]
    writes = xds_to_zarr(ms_datasets, zarr_store)
    dask.compute(writes)

    zarr_datasets = xds_from_zarr(zarr_store, chunks={"row": 1})

    for ms_ds, zarr_ds in zip(ms_datasets, zarr_datasets):
        for name, var in ms_ds.data_vars.items():
            assert_array_equal(var.data, getattr(zarr_ds, name).data)
