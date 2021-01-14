import dask
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from daskms import xds_from_ms
from daskms.experimental.arrow.extension_types import TensorArray
from daskms.experimental.arrow.write import xds_to_parquet

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
        "DATA": data
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


def test_xds_to_parquet(ms, tmp_path_factory):
    store = tmp_path_factory.mktemp("parquet_store") / "out.parquet"
    datasets = xds_from_ms(ms)
    writes = xds_to_parquet(datasets, store, ["DATA_DESC_ID", "FIELD_ID"])
    dask.compute(writes)

    pq_dataset = pq.ParquetDataset(store)
    record_batches = pq_dataset.read().to_batches()

    for ds, batch in zip(datasets, record_batches):
        for column, array in zip(batch.schema.names, batch.columns):
            var = getattr(ds, column).data
            assert isinstance(array, TensorArray if var.ndim > 1 else pa.Array)
            assert_array_equal(var, array.to_numpy())
