import pytest
import dask
from daskms import xds_to_storage_table, xds_from_ms
from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
from daskms.experimental.arrow import xds_from_parquet, xds_to_parquet

try:
    import xarray
except ImportError:
    xarray = None


@pytest.mark.skipif(xarray is None, reason="Need xarray to check equality.")
def test_storage_ms(ms):
    oxdsl = xds_from_ms(ms)

    writes = xds_to_storage_table(oxdsl, ms)

    oxdsl = dask.compute(oxdsl)[0]

    dask.compute(writes)

    xdsl = dask.compute(xds_from_ms(ms))[0]

    assert all([xds.equals(oxds) for xds, oxds in zip(xdsl, oxdsl)])


@pytest.mark.skipif(xarray is None, reason="Need xarray to check equality.")
def test_storage_zarr(ms, tmp_path_factory):
    zarr_store = tmp_path_factory.mktemp("zarr") / "test.zarr"

    oxdsl = xds_from_ms(ms)

    writes = xds_to_zarr(oxdsl, zarr_store)

    dask.compute(writes)

    oxdsl = xds_from_zarr(zarr_store)

    writes = xds_to_storage_table(oxdsl, zarr_store)

    oxdsl = dask.compute(oxdsl)[0]

    dask.compute(writes)

    xdsl = dask.compute(xds_from_zarr(zarr_store))[0]

    assert all([xds.equals(oxds) for xds, oxds in zip(xdsl, oxdsl)])


@pytest.mark.skipif(xarray is None, reason="Need xarray to check equality.")
def test_storage_parquet(ms, tmp_path_factory):
    parquet_store = tmp_path_factory.mktemp("parquet") / "test.parquet"

    oxdsl = xds_from_ms(ms)

    writes = xds_to_parquet(oxdsl, parquet_store)

    dask.compute(writes)

    oxdsl = xds_from_parquet(parquet_store)

    writes = xds_to_storage_table(oxdsl, parquet_store)

    oxdsl = dask.compute(oxdsl)[0]

    dask.compute(writes)

    xdsl = dask.compute(xds_from_parquet(parquet_store))[0]

    assert all([xds.equals(oxds) for xds, oxds in zip(xdsl, oxdsl)])
