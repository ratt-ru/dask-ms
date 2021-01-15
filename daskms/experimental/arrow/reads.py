from collections import defaultdict
import json
from pathlib import Path

import dask.array as da
import numpy as np

from daskms.experimental.utils import DATASET_TYPE
from daskms.experimental.arrow.writes import DASKMS_METADATA
from daskms.experimental.arrow.extension_types import TensorType

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

_UNHANDLED_TYPES = (
    pa.DictionaryType,
    pa.ListType,
    pa.MapType,
    pa.StructType,
    pa.UnionType,
    pa.TimestampType,
    pa.Time32Type,
    pa.Time64Type,
    pa.FixedSizeBinaryType,
    pa.Decimal128Type,
)


def _column_getter(table, column):
    chunks = table.column(column).chunks
    assert len(chunks) == 1
    return chunks[0].to_numpy()


def xds_from_parquet(store, chunks=None):
    if not isinstance(store, Path):
        store = Path(store)

    if chunks is not None:
        raise NotImplementedError("Non-native chunking not yet implemented")

    fragments = store.rglob("*.parquet")
    ds_cfg = defaultdict(list)

    for fragment in sorted(fragments):
        *partitions, parquet_file = fragment.relative_to(store).parts
        fragment_meta = pq.read_metadata(fragment)
        metadata = json.loads(fragment_meta.metadata[DASKMS_METADATA.encode()])
        types = [np.dtype(dt).type for _, dt in metadata["partition"]]

        partitions = [tuple(p.split("=")) for p in partitions]
        assert len(partitions) == len(types)
        partitions = tuple((p, dt(v)) for (p, v), dt in zip(partitions, types))

        ds_cfg[partitions].append((fragment, fragment_meta))

    datasets = []

    for partition, values in ds_cfg.items():
        column_arrays = defaultdict(list)

        for fragment, fragment_meta in values:
            table = da.blockwise(pq.read_table, (),
                                 fragment, None,
                                 meta=np.empty((), np.object))

            rows = fragment_meta.num_rows
            schema = fragment_meta.schema.to_arrow_schema()

            for column in schema.names:
                field = schema.field_by_name(column)
                field_metadata = field.metadata[DASKMS_METADATA.encode()]
                field_metadata = json.loads(field_metadata)
                dims = tuple(field_metadata["dims"])

                if isinstance(field.type, TensorType):
                    shape = (rows,) + field.type.shape
                else:
                    shape = (rows,)

                meta = np.empty((0,)*len(dims), field.type.to_pandas_dtype())
                new_axes = {d: s for d, s in zip(dims, shape)}

                read = da.blockwise(_column_getter, dims,
                                    table, (),
                                    column, None,
                                    new_axes=new_axes,
                                    meta=meta)

                column_arrays[column].append((read, dims))

        data_vars = {}

        for column, values in column_arrays.items():
            arrays, array_dims = zip(*values)

            if not len(set(array_dims)) == 1:
                raise ValueError(f"{array_dims} don't match for {column}")

            data_vars[column] = (dims, da.concatenate(arrays))

        datasets.append(DATASET_TYPE(data_vars))

    return datasets
