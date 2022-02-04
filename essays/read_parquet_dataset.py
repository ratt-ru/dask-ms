import argparse
import json
from pathlib import Path
from pprint import pprint

import dask.array as da
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

from daskms.patterns import LazyProxy
import daskms.experimental.arrow.extension_types
from daskms.experimental.arrow.arrow_schema import DASKMS_METADATA

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    p.add_argument("-c", "--column", default="TIME")
    return p

def read_column(table_proxy, column):
    arrow_col = table_proxy.item().column(column)
    chunks = arrow_col.chunks

    if not chunks:
        return np.array([])

    assert len(chunks) == 1
    zero_copy = chunks[0].type not in (pa.string(), pa.bool_())
    return chunks[0].to_numpy(zero_copy_only=zero_copy)

def column(path, column):
    fragments = list(sorted(path.rglob("*.parquet")))
    proxies = np.asarray([LazyProxy(pq.read_table, f, filters=[
        [("ANTENNA1", "<=", 3),
         ("ANTENNA2", ">=", 5)]]) for f in fragments])
    # NOTE: instantiates Tables's in the graph construction process
    # for purposes of reading metadata
    rows = np.asarray([p.num_rows for p in proxies])

    # Get the table schema from the first file, this should
    # be the same for all files
    schema = proxies[0].schema
    with open("schema.txt", "w") as f:
        f.write(str(schema))

    fields = {n: schema.field(n) for n in schema.names}

    try:
        field = fields[column]
    except KeyError:
        raise ValueError(f"Parquet dataset has no column {column}")

    field_metadata = field.metadata[DASKMS_METADATA.encode()]
    field_metadata = json.loads(field_metadata)
    dims = tuple(field_metadata["dims"])
    shape = (rows,) + field.type.shape

    assert len(shape) == len(dims)
    meta = np.empty((0,)*len(dims), field.type.to_pandas_dtype())
    new_axes = {d: s for d, s in zip(dims[1:], shape[1:])}

    dask_proxies = da.from_array(proxies, chunks=1)
    # dask_rows = da.from_array(rows, chunks=1)

    data = da.blockwise(read_column, dims,
                        dask_proxies, ("row",),
                        column, None,
                        new_axes=new_axes,
                        adjust_chunks={"row": tuple(rows.tolist())},
                        meta=meta)

    return data

if __name__ == "__main__":
    args = create_parser().parse_args()
    data = column(Path(args.ms), args.column)
    print(data, data.compute(scheduler="sync"))
    None