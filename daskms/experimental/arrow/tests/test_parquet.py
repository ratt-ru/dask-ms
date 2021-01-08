import numpy as np
from numpy.testing import assert_array_equal
import pytest

from daskms.experimental.arrow.extension_types import TensorArray

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def test_parquet(tmp_path_factory):
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
    filename = tmp_path_factory.mktemp("test_parquest") / "test.parquet"
    pq.write_table(table, filename)

    read_table = pq.read_table(filename)

    for c, v in columns.items():
        pqc = read_table.column(c)
        assert isinstance(pqc, pa.ChunkedArray) and pqc.num_chunks == 1
        parquet_array = next(iter(pqc.iterchunks())).to_numpy()
        assert_array_equal(v, parquet_array)
