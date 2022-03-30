from collections import defaultdict
import json
from pathlib import Path
from threading import Lock
import weakref
import warnings

import dask.array as da
from dask.array.core import normalize_chunks
import numpy as np

from daskms.dataset import Dataset
from daskms.fsspec_store import DaskMSStore
from daskms.experimental.utils import (promote_columns,
                                       column_iterator)
from daskms.experimental.arrow.arrow_schema import DASKMS_METADATA
from daskms.experimental.arrow.extension_types import TensorType
from daskms.experimental.arrow.require_arrow import requires_arrow
from daskms.constants import DASKMS_PARTITION_KEY
from daskms.patterns import Multiton
from daskms.utils import natural_order

try:
    import pyarrow as pa
except ImportError as e:
    pyarrow_import_error = e
    _UNHANDLED_TYPES = ()
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


try:
    import pyarrow.parquet as pq
except ImportError as e:
    pyarrow_import_error = e
else:
    pyarrow_import_error = None


_parquet_table_lock = Lock()
_parquet_table_cache = weakref.WeakValueDictionary()


class ParquetFileProxy(metaclass=Multiton):
    def __init__(self, store, key):
        self.store = store
        self.key = key
        self.lock = Lock()

    def __reduce__(self):
        return (ParquetFileProxy, (self.store, self.key))

    @property
    def file(self):
        try:
            return self._file
        except AttributeError:
            pass

        with self.lock:
            try:
                return self._file
            except AttributeError:
                pass

            sf = self.store.open_file(self.key, "rb")
            self._file = file_ = pq.ParquetFile(sf)
            return file_

    @property
    def metadata(self):
        return self.file.metadata

    def __eq__(self, other):
        return (isinstance(other, ParquetFileProxy) and
                self.store == other.store and
                self.key == other.key)

    def __lt__(self, other):
        return (isinstance(other, ParquetFileProxy) and
                self.store == other.store and
                natural_order(self.key) < natural_order(other.key))

    def read_column(self, column, start=None, end=None):
        chunks = self.file.read(columns=[column]).column(column).chunks
        assert len(chunks) == 1

        zero_copy = chunks[0].type not in (pa.string(), pa.bool_())
        return chunks[0].to_numpy(zero_copy_only=zero_copy)[start:end]


def _partition_values(partition_strings, partition_meta):
    assert len(partition_strings) == len(partition_meta)
    partitions = []

    for ps, (pf, dt) in zip(partition_strings, partition_meta):
        field, value = (s.strip() for s in ps.split("="))

        if field != pf:
            raise ValueError(f"Column name {field} in partition string "
                             f"{partition_strings} does not match "
                             f"metadata column name {pf}")

        partitions.append((field, np.dtype(dt).type(value)))

    return tuple(partitions)


def partition_chunking(partition, fragment_rows, chunks):
    partition_rows = sum(fragment_rows)

    if chunks is None:
        # Default to natural chunking determined from individual
        # parquet files in the dataset
        row_chunks = tuple(fragment_rows)
    else:
        try:
            partition_chunks = chunks[partition]
        except IndexError:
            partition_chunks = chunks[-1]

        # We only handle row chunking at present,
        # warn the user
        unhandled_dims = set(partition_chunks.keys()) - {"row"}

        if len(unhandled_dims) != 0:
            warnings.warn(f"{unhandled_dims} chunking dimensions are "
                          f"currently ignored for arrow", UserWarning)

        # Get any user specified row chunking, defaulting to
        row_chunks = partition_chunks.get("row", fragment_rows)

        if isinstance(row_chunks, list):
            row_chunks = tuple(row_chunks)

        row_chunks = normalize_chunks(row_chunks, (partition_rows,))[0]

    intervals = np.cumsum([0] + fragment_rows)
    chunk_intervals = np.cumsum((0,) + row_chunks)
    ranges = []
    it = zip(chunk_intervals, chunk_intervals[1:])

    for c, (lower, upper) in enumerate(it):
        si = np.searchsorted(intervals, lower, side='right') - 1
        ei = np.searchsorted(intervals, upper, side='left')

        if si == ei:
            raise ValueError("si == ei, arrays may have zero chunks")

        for s in range(si, ei):
            e = s + 1

            if lower <= intervals[s]:
                start = 0
            else:
                start = lower - intervals[s]

            if upper >= intervals[e]:
                end = intervals[e] - intervals[s]
            else:
                end = upper - intervals[s]

            ranges.append((s, (start, end)))

    return ranges


@requires_arrow(pyarrow_import_error)
def xds_from_parquet(store, columns=None, chunks=None, **kwargs):

    # If any kwargs are added, they should be popped prior to this check.
    if len(kwargs) > 0:
        warnings.warn(
            f"The following unsupported kwargs were ignored in "
            f"xds_from_parquet: {kwargs}",
            UserWarning,
        )

    if isinstance(store, DaskMSStore):
        pass
    elif isinstance(store, Path):
        store = DaskMSStore(f"file://{store}")
    elif isinstance(store, str):
        store = DaskMSStore(f"file://{store}")
    else:
        raise TypeError(f"store '{store}' must be "
                        f"Path, str or DaskMSStore")

    columns = promote_columns(columns)

    if chunks is None:
        pass
    elif isinstance(chunks, (tuple, list)):
        if len(chunks) == 0 or any(not isinstance(c, dict) for c in chunks):
            raise TypeError("chunks must be None or dict or list of dict")
    elif isinstance(chunks, dict):
        chunks = [chunks]
    else:
        raise TypeError("chunks must be None or dict or list of dict")

    fragments = list(map(Path, store.rglob("*.parquet")))
    ds_cfg = defaultdict(list)

    # Iterate over all parquet files in the directory tree
    # and group them by partition
    partition_schemas = set()

    for fragment in fragments:
        *partitions, parquet_file = fragment.relative_to(store.table).parts
        fragment = ParquetFileProxy(store, fragment)
        fragment_meta = fragment.metadata
        metadata = json.loads(fragment_meta.metadata[DASKMS_METADATA.encode()])
        partition_meta = metadata[DASKMS_PARTITION_KEY]
        partition_meta = tuple(tuple((f, v)) for f, v in partition_meta)
        partitions = _partition_values(partitions, partition_meta)
        partition_schemas.add(partition_meta)
        ds_cfg[partitions].append(fragment)

    # Sanity check partition schemas of all parquet files
    if len(partition_schemas) == 0:
        raise ValueError(f"No parquet files found in {store.path}")
    elif len(partition_schemas) != 1:
        raise ValueError(f"Multiple partitions discovered {partition_schemas}")

    partition_schemas = partition_schemas.pop()
    datasets = []

    # Now create a dataset per partition
    for p, (partition, fragments) in enumerate(sorted(ds_cfg.items())):
        fragments = list(sorted(fragments))
        column_arrays = defaultdict(list)
        fragment_rows = [f.metadata.num_rows for f in fragments]

        for (f, (start, end)) in partition_chunking(p, fragment_rows, chunks):
            fragment = fragments[f]
            fragment_meta = fragment.metadata
            rows = fragment_meta.num_rows
            schema = fragment_meta.schema.to_arrow_schema()
            fields = {n: schema.field(n) for n in schema.names}

            for column, field in column_iterator(fields, columns):
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

                read = da.blockwise(fragment.read_column, dims,
                                    column, None,
                                    start, None,
                                    end, None,
                                    adjust_chunks={"row": end - start},
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
        attrs[DASKMS_PARTITION_KEY] = partition_schemas
        datasets.append(Dataset(data_vars, attrs=attrs))

    return datasets
