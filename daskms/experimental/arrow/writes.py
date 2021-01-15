import json
from pathlib import Path
from threading import Lock
import weakref

import dask.array as da
import numpy as np

from daskms.experimental.arrow.schema import (dict_dataset_schema,
                                              DASKMS_METADATA)
from daskms.experimental.arrow.extension_types import TensorArray
from daskms.experimental.utils import DATASET_TYPES, DATASET_TYPE
from daskms.optimisation import inlined_array
from daskms.reads import PARTITION_KEY
from daskms.utils import freeze

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


_dataset_cache = weakref.WeakValueDictionary()
_dataset_lock = Lock()


class ParquetFragmentMetaClass(type):
    """
    https://en.wikipedia.org/wiki/Multiton_pattern

    """
    def __call__(cls, *args, **kwargs):
        key = freeze((cls,) + args + (kwargs,))

        try:
            return _dataset_cache[key]
        except KeyError:
            with _dataset_lock:
                try:
                    return _dataset_cache[key]
                except KeyError:
                    instance = type.__call__(cls, *args, **kwargs)
                    _dataset_cache[key] = instance
                    return instance


class ParquetFragment(metaclass=ParquetFragmentMetaClass):
    def __init__(self, path, schema, partition):
        path = Path(path)

        for field, value in partition:
            path /= f"{field}={value}"

        self.path = path
        self.schema = schema
        self.partition = partition
        self.lock = Lock()

    def __reduce__(self):
        return (ParquetFragment, (self.path, self.schema, self.partition))

    def write(self, chunk, *data):
        with self.lock:
            self.path.mkdir(parents=True, exist_ok=True)

        table_data = {}
        var_schema, table_meta = self.schema
        column_meta = {column: meta for (column, _, _, meta) in var_schema}
        fields = []

        for column, v in zip(data[::2], data[1::2]):
            while type(v) is list:
                assert len(v) == 1
                v = v[0]

            if v.ndim == 1:
                pa_data = pa.array(v)
            elif v.ndim > 1:
                pa_data = TensorArray.from_numpy(v)
            else:
                raise ValueError("Scalar arrays not yet handled")

            metadata = {DASKMS_METADATA: json.dumps(column_meta[column])}
            table_data[column] = pa_data
            fields.append(pa.field(column, pa_data.type,
                                   metadata=metadata,
                                   nullable=False))

        metadata = {DASKMS_METADATA: json.dumps(table_meta)}
        schema = pa.schema(fields, metadata=metadata)
        table = pa.table(table_data, schema=schema)
        pq.write_table(table, self.path / f"data-{chunk.item()}.parquet")

        return np.array([True], np.bool)


def xds_to_parquet(xds, path):
    if not isinstance(path, Path):
        path = Path(path)

    if isinstance(xds, DATASET_TYPES):
        xds = [xds]
    elif isinstance(xds, (tuple, list)):
        if not all(isinstance(ds, DATASET_TYPES) for ds in xds):
            raise TypeError("xds must be a Dataset or list of Datasets")
    else:
        raise TypeError("xds must be a Dataset or list of Datasets")

    schema = dict_dataset_schema(xds)
    datasets = []

    for i, ds in enumerate(xds):
        partition = ds.attrs.get(PARTITION_KEY, None)

        if not partition:
            ds_partition = (("DATASET", i),)
        else:
            ds_partition = tuple((p, getattr(ds, p)) for p, _ in partition)

        fragment = ParquetFragment(path, schema, ds_partition)
        chunk_ids = da.arange(len(ds.chunks["row"]), chunks=1)
        args = [chunk_ids, ("row",)]

        for column, variable in ds.data_vars.items():
            if not isinstance(variable.data, da.Array):
                raise ValueError(f"Column {column} does not "
                                 f"contain a dask Array")

            if len(variable.dims[0]) == 0 or variable.dims[0] != "row":
                raise ValueError(f"Column {column} dimensions "
                                 f"{variable.dims} don't start with 'row'")

            args.extend((column, None, variable.data, variable.dims))

            for dim, chunk in zip(variable.dims, variable.data.chunks):
                if len(chunk) != 1:
                    raise ValueError(f"Chunking in {dim} is not yet "
                                     f"supported.")

        writes = da.blockwise(fragment.write, ("row",),
                              *args,
                              adjust_chunks={"row": 1},
                              meta=np.empty((0,), np.bool))

        writes = inlined_array(writes, chunk_ids)

        datasets.append(DATASET_TYPE({"WRITE": (("row",), writes)}))

    return datasets
