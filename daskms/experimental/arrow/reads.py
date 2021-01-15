from collections import defaultdict
import json
from pathlib import Path

import dask.array as da
import numpy as np

from daskms.reads import PARTITION_KEY
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


def _partition_values(partition_strings, partition_meta):
    assert len(partition_strings) == len(partition_meta)
    partitions = []

    for ps, (pf, dt) in zip(partition_strings, partition_meta):
        field, value = (s.strip() for s in ps.split("="))

        if field != pf:
            raise ValueError(f"Column name {field} in partition string "
                             f"{partition_strings} does not match "
                             f"metadata column name {pf}")

        assert field == pf
        partitions.append((field, np.dtype(dt).type(value)))

    return tuple(partitions)


def xds_from_parquet(store, chunks=None):
    if not isinstance(store, Path):
        store = Path(store)

    if chunks is not None:
        raise NotImplementedError("Non-native chunking not yet implemented")

    fragments = store.rglob("*.parquet")
    ds_cfg = defaultdict(list)

    # Iterate over all parquet files in the directory tree
    # and group them by partition
    partition_schemas = set()

    for fragment in fragments:
        *partitions, parquet_file = fragment.relative_to(store).parts
        fragment_meta = pq.read_metadata(fragment)
        metadata = json.loads(fragment_meta.metadata[DASKMS_METADATA.encode()])
        partition_meta = metadata[PARTITION_KEY]
        partition_meta = tuple(tuple((f, v)) for f, v in partition_meta)
        partitions = _partition_values(partitions, partition_meta)
        partition_schemas.add(partition_meta)
        ds_cfg[partitions].append((fragment, fragment_meta))

    # Sanity check partition schemas of all parquet files
    if len(partition_schemas) != 1:
        raise ValueError(f"Multiple partitions discovered {partition_schemas}")

    partition_schemas = partition_schemas.pop()
    datasets = []

    # Now create a dataset per partition
    for partition, values in sorted(ds_cfg.items()):
        column_arrays = defaultdict(list)

        # For each parquet file in this partition
        for fragment, fragment_meta in values:
            table = da.blockwise(pq.read_table, (),
                                 fragment, None,
                                 meta=np.empty((), np.object))

            rows = fragment_meta.num_rows
            schema = fragment_meta.schema.to_arrow_schema()

            for column in schema.names:
                field = schema.field(column)
                field_metadata = field.metadata[DASKMS_METADATA.encode()]
                field_metadata = json.loads(field_metadata)
                dims = tuple(field_metadata["dims"])

                if isinstance(field.type, TensorType):
                    shape = (rows,) + field.type.shape
                else:
                    shape = (rows,)

                assert len(shape) == len(dims)

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
            array_dims = set(array_dims)

            if not len(array_dims) == 1:
                raise ValueError(f"Inconsistent array dimensions "
                                 f"{array_dims} for {column}")

            data_vars[column] = (array_dims.pop(), da.concatenate(arrays))

        attrs = dict(partition)
        attrs[PARTITION_KEY] = partition_schemas
        datasets.append(DATASET_TYPE(data_vars, attrs=attrs))

    return datasets
