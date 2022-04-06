import random

import dask
import dask.array as da
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from daskms import xds_from_storage_ms
from daskms.dataset import Dataset
from daskms.fsspec_store import DaskMSStore
from daskms.experimental.arrow.extension_types import TensorArray
from daskms.experimental.arrow.reads import xds_from_parquet
from daskms.experimental.arrow.reads import partition_chunking
from daskms.experimental.arrow.writes import xds_to_parquet
from daskms.constants import DASKMS_PARTITION_KEY

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

try:
    import xarray
except ImportError:
    xarray = None


def test_parquet_roundtrip(tmp_path_factory):
    """ Test round-trip via parquet file with custom Extension Types """
    time = np.linspace(0, 1.0, 10)
    ant1, ant2 = (a.astype(np.int32) for a in np.triu_indices(7, 1))

    ntime = time.shape[0]
    nbl = ant1.shape[0]

    time = np.tile(time, nbl)
    ant1 = np.repeat(ant1, ntime)
    ant2 = np.repeat(ant2, ntime)

    nrow = time.shape[0]
    nchan = 16
    ncorr = 4
    shape = (nrow, nchan, ncorr)
    data = np.random.random(shape) + np.random.random(shape)*1j
    uvw = np.random.random((nrow, 3))
    flag = np.random.randint(0, 2, shape).astype(np.bool_)

    columns = {
        "TIME": time,
        "ANTENNA1": ant1,
        "ANTENNA2": ant2,
        "UVW": uvw,
        "FLAG": flag,
        "DATA": data,
    }

    arrow_columns = {k: TensorArray.from_numpy(v) for k, v in columns.items()}
    table = pa.table(arrow_columns)
    filename = tmp_path_factory.mktemp("parquet_store") / "test.parquet"
    pq.write_table(table, filename)

    read_table = pq.read_table(filename)

    for c, v in columns.items():
        pqc = read_table.column(c)
        assert isinstance(pqc, pa.ChunkedArray) and pqc.num_chunks == 1
        parquet_array = next(iter(pqc.iterchunks())).to_numpy()
        assert_array_equal(v, parquet_array)


@pytest.mark.parametrize("row_chunks", [[2, 3, 4]])
@pytest.mark.parametrize("user_chunks", [{"row": 2, "chan": 4}])
def test_partition_chunks(row_chunks, user_chunks):
    expected = [(0, (0, 2)),
                (1, (0, 2)), (1, (2, 3)),
                (2, (0, 1)), (2, (1, 3)), (2, (3, 4))]

    assert partition_chunking(0, row_chunks, [user_chunks]) == expected


def test_xds_to_parquet_string(tmp_path_factory):
    store = tmp_path_factory.mktemp("parquet_store") / "string-dataset.parquet"

    datasets = []

    for i in range(3):
        names = random.choices([f"foo-{i}", f"bar-{i}", f"qux-{i}"], k=10)
        names = np.asarray(names, dtype=object)
        chunks = sorted([1, 2, 3, 4], key=lambda *a: random.random())
        names = da.from_array(names, chunks=chunks)
        datasets.append(Dataset({"NAME": (("row",), names)}))

    writes = xds_to_parquet(datasets, store)
    dask.compute(writes)

    parquet_datasets = xds_from_parquet(store)
    assert len(datasets) == len(parquet_datasets)

    for ds, pq_ds in zip(datasets, parquet_datasets):
        assert_array_equal(ds.NAME.data, pq_ds.NAME.data)


def parquet_tester(ms, store):
    datasets = xds_from_storage_ms(ms)

    # We can test row chunking if xarray is installed
    if xarray is not None:
        datasets = [ds.chunk({"row": 1}) for ds in datasets]

    # spw_datasets = xds_from_table(spw_table, group_cols="__row__")
    # ant_datasets = xds_from_table(ant_table, group_cols="__row__")

    writes = []
    writes.extend(xds_to_parquet(datasets, store))
    # TODO(sjperkins)
    # Fix arrow shape unification errors
    # writes.extend(xds_to_parquet(spw_datasets, spw_store))
    # writes.extend(xds_to_parquet(ant_datasets, antenna_store))
    dask.compute(writes)

    pq_datasets = xds_from_parquet(store, chunks={"row": 1})
    assert len(datasets) == len(pq_datasets)

    for ds, pq_ds in zip(datasets, pq_datasets):
        for column, var in ds.data_vars.items():
            pq_var = getattr(pq_ds, column)
            assert_array_equal(var.data, pq_var.data)
            assert var.dims == pq_var.dims

        for column, var in ds.coords.items():
            pq_var = getattr(pq_ds, column)
            assert_array_equal(var.data, pq_var.data)
            assert var.dims == pq_var.dims

        partitions = ds.attrs[DASKMS_PARTITION_KEY]
        pq_partitions = pq_ds.attrs[DASKMS_PARTITION_KEY]
        assert partitions == pq_partitions

        for field, dtype in partitions:
            assert getattr(ds, field) == getattr(pq_ds, field)


def test_xds_to_parquet_local(ms, tmp_path_factory, spw_table, ant_table):
    store = tmp_path_factory.mktemp("parquet_store") / "out.parquet"
    # antenna_store = store.parent / f"{store.name}::ANTENNA"
    # spw_store = store.parent / f"{store.name}::SPECTRAL_WINDOW"

    return parquet_tester(ms, store)


def test_xds_to_parquet_s3(ms, spw_table, ant_table,
                           py_minio_client, minio_user_key,
                           minio_url, s3_bucket_name):

    py_minio_client.make_bucket(s3_bucket_name)

    store = DaskMSStore(f"s3://{s3_bucket_name}/measurementset.MS",
                        key=minio_user_key,
                        secret=minio_user_key,
                        client_kwargs={
                          "endpoint_url": minio_url.geturl(),
                          "region_name": "af-cpt",
                        })

    # NOTE(sjperkins)
    # Review this interface
    # spw_store = store.subtable_store("SPECTRAL_WINDOW")
    # ant_store = store.subtable_store("ANTENNA")

    return parquet_tester(ms, store)
