import argparse
from pathlib import Path
from pprint import pprint

import dask.array as da
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

from daskms.patterns import LazyProxy

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    return p

def read_column(file_proxy, column):
    chunks = file_proxy[0].read(columns=[column]).column(column).chunks
    assert len(chunks) == 1
    zero_copy = chunks[0].type not in (pa.string(), pa.bool_())
    return chunks[0].to_numpy(zero_copy_only=zero_copy)

def column(path, column):
    fragments = list(sorted(path.rglob("*.parquet")))
    proxies = np.asarray([LazyProxy(pq.ParquetFile, f) for f in fragments])
    # NOTE: instantiates ParquetFile's in the graph construction process
    # for purposes of reading metadata
    rows = np.asarray([p.metadata.num_rows for p in proxies])

    # Get the table schema from the first file, this should
    # be the same for all files
    schema = proxies[0].metadata.schema.to_arrow_schema()
    fields = {n: schema.field(n) for n in schema.names}

    try:
        field = fields[column]
    except KeyError:
        raise ValueError(f"Parquet dataset has no column {column}")


    dask_proxies = da.from_array(proxies, chunks=1)
    dask_rows = da.from_array(rows, chunks=1)


    data = da.blockwise(read_column, ("row",),
                        dask_proxies, ("row",),
                        column, None,
                        adjust_chunks={"row": tuple(rows.tolist())},
                        meta=np.empty((0,), dtype=field.type.to_pandas_dtype()))

    return data

if __name__ == "__main__":
    args = create_parser().parse_args()
    data = column(Path(args.ms), "TIME")
    print(data.compute())
