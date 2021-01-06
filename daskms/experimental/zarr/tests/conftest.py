from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session", params=[
    {"row": 1000, "chan": 4096, "corr": 4, "ant": 7},
])
def zarr_store(tmp_path_factory, request):
    row = request.param.get("row", 1000)
    chan = request.param.get("chan", 4096)
    corr = request.param.get("corr", 4)
    ant = request.param.get("ant", 7)

    rs = np.random.RandomState(42)
    data_shape = (row, chan, corr)
    data = rs.random_sample(data_shape) + rs.random_sample(data_shape)*1j
    data = data.astype(np.complex64)

    ant1, ant2 = (a.astype(np.int32) for a in np.triu_indices(ant, 1))
    bl = ant1.shape[0]
    ant1 = np.repeat(ant1, (row + bl - 1) // bl)
    ant2 = np.repeat(ant2, (row + bl - 1) // bl)

    # Common grouping columns
    field = [0,   0,   0,   1,   1,   1,   1,   2,   2,   2]
    ddid = [0,   0,   0,   0,   0,   0,   0,   1,   1,   1]
    scan = [0,   1,   0,   1,   0,   1,   0,   1,   0,   1]

    # Common indexing columns
    time = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    ant1 = [0,   0,   1,   1,   1,   2,   1,   0,   0,   1]
    ant2 = [1,   2,   2,   3,   2,   1,   0,   1,   1,   2]

    # Column we'll write to
    state = [0,   0,   0,   0,   0,   0,   0,   0,   0,   0]

    zarr = pytest.importorskip("zarr")
    store = zarr.DirectoryStore(tmp_path_factory.mktemp("zarr_test"))

    with zarr.group(store=store, overwrite=True) as root:
        root.array("ANTENNA1", ant1)
        root.array("ANTENNA2", ant2)
        root.array("TIME", time)
        root.array("DATA", data)

        root.zeros_like("FIELD_ID", np.asarray(field))
        root.zeros_like("DATA_DESC_ID", np.asarray(ddid))
        root.zeros_like("SCAN_NUMBER", np.asarray(scan))
        root.zeros_like("STATE_ID", np.asarray(state))

    yield Path(store.path)
