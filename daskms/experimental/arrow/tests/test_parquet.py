import dask
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from daskms import xds_from_ms
from daskms.reads import PARTITION_KEY
from daskms.experimental.arrow.extension_types import TensorArray
from daskms.experimental.arrow.reads import xds_from_parquet
from daskms.experimental.arrow.reads import partition_chunking
from daskms.experimental.arrow.writes import xds_to_parquet

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


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

    columns = {
        "TIME": time,
        "ANTENNA1": ant1,
        "ANTENNA2": ant2,
        "UVW": uvw,
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


def test_xds_to_parquet(ms, tmp_path_factory):
    store = tmp_path_factory.mktemp("parquet_store") / "out.parquet"
    datasets = xds_from_ms(ms)
    writes = xds_to_parquet(datasets, store)
    dask.compute(writes)

    record_batches = pq.ParquetDataset(store).read().to_batches()

    for ds, batch in zip(datasets, record_batches):
        for column, array in zip(batch.schema.names, batch.columns):
            if column in ds.attrs:
                assert_array_equal(getattr(ds, column), array.to_numpy())
            else:
                var = getattr(ds, column).data
                expected_patype = TensorArray if var.ndim > 1 else pa.Array
                assert isinstance(array, expected_patype)
                assert_array_equal(var, array.to_numpy())

    pq_datasets = xds_from_parquet(store, chunks={"row": 1})
    assert len(datasets) == len(pq_datasets)

    for ds, pq_ds in zip(datasets, pq_datasets):
        for column, var in ds.data_vars.items():
            pq_var = getattr(pq_ds, column)
            assert_array_equal(var.data, pq_var.data)
            assert var.dims == pq_var.dims

        partitions = ds.attrs[PARTITION_KEY]
        pq_partitions = pq_ds.attrs[PARTITION_KEY]
        assert partitions == pq_partitions

        for field, dtype in partitions:
            assert getattr(ds, field) == getattr(pq_ds, field)
