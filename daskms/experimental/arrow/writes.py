import itertools
from pathlib import Path
from threading import Lock

import dask.array as da
import numpy as np

from daskms.dataset import Dataset
from daskms.optimisation import inlined_array
from daskms.constants import DASKMS_PARTITION_KEY
from daskms.fsspec_store import DaskMSStore
from daskms.patterns import Multiton
from daskms.experimental.arrow.arrow_schema import ArrowSchema
from daskms.experimental.arrow.extension_types import TensorArray
from daskms.experimental.arrow.require_arrow import requires_arrow
from daskms.experimental.utils import (promote_columns,
                                       column_iterator)

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


class ParquetFragment(metaclass=Multiton):
    def __init__(self, store, key, schema, dataset_id):
        partition = schema.attrs.get(DASKMS_PARTITION_KEY, False)

        if not partition:
            # There's no specific partitioning schema,
            # make one up
            partition = (("DATASET", dataset_id),)
            schema = ArrowSchema(
                schema.data_vars,
                schema.coords,
                {
                    **schema.attrs,
                    DASKMS_PARTITION_KEY: (("DATASET", "int32"),)
                })
        else:
            partition = tuple((p, schema.attrs[p]) for p, _ in partition)

        key = Path(key)

        # Add the partitioning to the path
        for field, value in partition:
            key /= f"{field}={value}"

        self.dataset_id = dataset_id
        self.store = store
        self.key = key
        self.schema = schema
        self.partition = partition
        self.lock = Lock()

    def __reduce__(self):
        return (ParquetFragment, (self.store, self.key,
                                  self.schema, self.dataset_id))

    def write(self, chunk, *data):
        with self.lock:
            self.store.makedirs(self.key, exist_ok=True)

        table_data = {}

        for column, var in zip(data[::2], data[1::2]):
            while type(var) is list:
                if len(var) != 1:
                    raise ValueError("Multiple chunks in blockwise "
                                     "on non-row dimension")
                var = var[0]

            if var.ndim == 1:
                pa_data = pa.array(var)
            elif var.ndim > 1:
                pa_data = TensorArray.from_numpy(var)
            else:
                raise NotImplementedError("Scalar array writing "
                                          "not implemented")

            table_data[column] = pa_data

        table = pa.table(table_data, schema=self.schema.to_arrow_schema())
        parquet_filename = self.key / f"{chunk.item()}.parquet"
        sf = self.store.open_file(parquet_filename, "wb")
        pq.write_table(table, sf)

        return np.array([True], bool)


@requires_arrow(pyarrow_import_error)
def xds_to_parquet(xds, store, columns=None):
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

    if isinstance(xds, Dataset):
        xds = [xds]
    elif isinstance(xds, (tuple, list)):
        if not all(isinstance(ds, Dataset) for ds in xds):
            raise TypeError("xds must be a Dataset or list of Datasets")
    else:
        raise TypeError("xds must be a Dataset or list of Datasets")

    datasets = []
    base_schema = ArrowSchema.from_datasets(xds)

    for ds_id, ds in enumerate(xds):
        arrow_schema = base_schema.with_attributes(ds)
        fragment = ParquetFragment(store, store.table, arrow_schema, ds_id)
        chunk_ids = da.arange(len(ds.chunks["row"]), chunks=1)
        args = [chunk_ids, ("row",)]

        data_var_it = column_iterator(ds.data_vars, columns)
        coord_it = column_iterator(ds.coords, columns)

        for column, variable in itertools.chain(data_var_it, coord_it):
            if not isinstance(variable.data, da.Array):
                raise ValueError(f"Column {column} does not "
                                 f"contain a dask Array")

            if len(variable.dims[0]) == 0 or variable.dims[0] != "row":
                raise ValueError(f"Column {column} dimensions "
                                 f"{variable.dims} don't start with 'row'")

            args.extend((column, None, variable.data, variable.dims))

            for dim, chunk in zip(variable.dims[1:], variable.data.chunks[1:]):
                if len(chunk) != 1:
                    raise ValueError(f"Chunking in {dim} is not yet "
                                     f"supported.")

        writes = da.blockwise(fragment.write, ("row",),
                              *args,
                              align_arrays=False,
                              adjust_chunks={"row": 1},
                              meta=np.empty((0,), bool))

        writes = inlined_array(writes, chunk_ids)

        # Transfer any partition information over to the write dataset
        partition = ds.attrs.get(DASKMS_PARTITION_KEY, False)

        if not partition:
            attrs = None
        else:
            attrs = {DASKMS_PARTITION_KEY: partition,
                     **{k: getattr(ds, k) for k, _ in partition}}

        datasets.append(Dataset({"WRITE": (("row",), writes)}, attrs=attrs))

    return datasets
