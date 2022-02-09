import argparse

import pyrap.tables as pt
import dask.array as da
from daskms.patterns import LazyProxy
import numpy as np

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    return p


def _read(proxy, column, start_len):
    proxy.lock(False)
    start, nrow = start_len[0][0]

    try:
        return proxy.getcol(column, startrow=start, nrow=nrow)
    finally:
        proxy.unlock()

def _write(proxy, column, start_len, data):
    proxy.lock(True)
    start, nrow = start_len[0][0]

    try:
        proxy.putcol(column, data, startrow=start, nrow=nrow)
        return np.array([True])
    except:
        return np.array([False])
    finally:
        proxy.unlock()


def main(ms, column="TIME"):
    read_proxy = LazyProxy(pt.table, ms, lockoptions="usernoread", readonly=True)
    write_proxy = LazyProxy(pt.table, ms, lockoptions="user", readonly=False)

    sl = np.array(([0, 10], [10, 10], [20, 10]))
    sl = da.from_array(sl, chunks=(1, 2))

    data = da.blockwise(_read, "r",
                        read_proxy, None,
                        column, None,
                        sl, "rb",
                        meta=np.empty((0,), np.float64))

    writes = da.blockwise(_write, "r",
                          write_proxy, None,
                          column, None,
                          sl, "rb",
                          data, "r",
                          meta=np.empty((0,), bool))

    print(writes.compute(scheduler="processes"))

if __name__ == "__main__":
    args = create_parser().parse_args()
    main(args.ms)
