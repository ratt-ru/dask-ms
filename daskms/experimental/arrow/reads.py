from pathlib import Path

import numpy as np

from daskms.experimental.utils import DATASET_TYPE, DATASET_TYPES

try:
    import pyarrow as pa
except ImportError as e:
    pyarrow_import_error = e
else:
    pyarrow_import_error = None

try:
    import pyarrow.parquet as pq
except ImportError as e:
    pyarrow_import_error = e
else:
    pyarrow_import_error = None


def xds_from_parquet(store, chunks=None):
    if not isinstance(store, Path):
        store = Path(store)

    fragments = store.rglob("*.parquet")

    for fragment in fragments:
        *partitions, parquet_file = fragment.relative_to(store).parts
        partitions = [tuple(p.split("=")) for p in partitions]



