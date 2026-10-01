# -*- coding: utf-8 -*-

import gc
import logging
import multiprocessing
import os
import socket
from uuid import uuid4

import numpy as np
import pytest

from daskms.casa_table import taql_table
from daskms.testing import mark_in_pytest


# content of conftest.py
def pytest_configure(config):
    mark_in_pytest(True)


def pytest_unconfigure(config):
    mark_in_pytest(False)


@pytest.fixture(autouse=True)
def xms_always_gc():
    """Force garbage collection after each test"""
    try:
        yield
    finally:
        gc.collect()


@pytest.fixture(autouse=True)
def xms_clear_table_cache():
    """Close any cached tables after each test.

    CasaTable instances live in a TTL cache, so without this a table
    written by one test would still be open when the next test reopens
    it.
    """
    from daskms.casa_table import clear_table_cache

    try:
        yield
    finally:
        clear_table_cache()


@pytest.fixture(scope="session")
def big_ms(tmp_path_factory, request):
    pytest.importorskip("arcae")
    msdir = tmp_path_factory.mktemp("big_ms_dir", numbered=False)
    fn = os.path.join(str(msdir), "big.ms")
    row = request.param
    chan = 4096
    corr = 4
    ant = 7

    create_table_query = f"""
    CREATE TABLE {fn}
    [FIELD_ID I4,
    TIME R8,
    ANTENNA1 I4,
    ANTENNA2 I4,
    DATA_DESC_ID I4,
    SCAN_NUMBER I4,
    STATE_ID I4,
    DATA C8 [NDIM=2, SHAPE=[{chan}, {corr}]]]
    LIMIT {row}
    """

    rs = np.random.RandomState(42)
    data_shape = (row, chan, corr)
    data = rs.random_sample(data_shape) + rs.random_sample(data_shape) * 1j

    # Create the table
    with taql_table(create_table_query) as ms:
        ant1, ant2 = (a.astype(np.int32) for a in np.triu_indices(ant, 1))
        bl = ant1.shape[0]
        ant1 = np.repeat(ant1, (row + bl - 1) // bl)
        ant2 = np.repeat(ant2, (row + bl - 1) // bl)

        zeros = np.zeros(row, np.int32)

        ms.putcol("ANTENNA1", ant1[:row])
        ms.putcol("ANTENNA2", ant2[:row])

        ms.putcol("FIELD_ID", zeros)
        ms.putcol("DATA_DESC_ID", zeros)
        ms.putcol("SCAN_NUMBER", zeros)
        ms.putcol("STATE_ID", zeros)
        ms.putcol("TIME", np.linspace(0, 1.0, row, dtype=np.float64))
        ms.putcol("DATA", data)

    yield fn

    # Remove the temporary directory
    # except it causes issues with casacore files on py3
    # https://github.com/ska-sa/dask-ms/issues/32
    # shutil.rmtree(str(msdir))


@pytest.fixture
def ms(tmp_path_factory):
    pytest.importorskip("arcae")
    msdir = tmp_path_factory.mktemp("msdir", numbered=True)
    fn = os.path.join(str(msdir), "test.ms")

    create_table_query = f"""
    CREATE TABLE {fn}
    [FIELD_ID I4,
    ANTENNA1 I4,
    ANTENNA2 I4,
    DATA_DESC_ID I4,
    SCAN_NUMBER I4,
    STATE_ID I4,
    UVW R8 [NDIM=1, SHAPE=[3]],
    TIME R8,
    DATA C8 [NDIM=2, SHAPE=[16, 4]]]
    LIMIT 10
    """

    # Common grouping columns
    i4 = np.int32
    field = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], i4)
    ddid = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], i4)
    scan = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], i4)

    # Common indexing columns
    time = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    ant1 = np.array([0, 0, 1, 1, 1, 2, 1, 0, 0, 1], i4)
    ant2 = np.array([1, 2, 2, 3, 2, 1, 0, 1, 1, 2], i4)

    # Column we'll write to
    state = np.zeros(10, i4)

    rs = np.random.RandomState(42)
    data_shape = (len(state), 16, 4)
    data = rs.random_sample(data_shape) + rs.random_sample(data_shape) * 1j
    uvw = rs.random_sample((len(state), 3)).astype(np.float64)

    # Create the table
    with taql_table(create_table_query) as ms:
        ms.putcol("FIELD_ID", field)
        ms.putcol("DATA_DESC_ID", ddid)
        ms.putcol("ANTENNA1", ant1)
        ms.putcol("ANTENNA2", ant2)
        ms.putcol("SCAN_NUMBER", scan)
        ms.putcol("STATE_ID", state)
        ms.putcol("UVW", uvw)
        ms.putcol("TIME", time)
        ms.putcol("DATA", data)

    yield fn

    # Remove the temporary directory
    # except it causes issues with casacore files on py3
    # https://github.com/ska-sa/dask-ms/issues/32
    # shutil.rmtree(str(msdir))


@pytest.fixture
def spw_chan_freqs():
    return (
        np.linspace(0.856e9, 2 * 0.856e9, 8),
        np.linspace(0.856e9, 2 * 0.856e9, 16),
        np.linspace(0.856e9, 2 * 0.856e9, 32),
    )


@pytest.fixture
def spw_table(tmp_path_factory, spw_chan_freqs):
    """Simulate a SPECTRAL_WINDOW table with two spectral windows"""
    pytest.importorskip("arcae")
    spw_dir = tmp_path_factory.mktemp("spw_dir", numbered=True)
    fn = os.path.join(str(spw_dir), "SPECTRAL_WINDOW")

    create_table_query = """
    CREATE TABLE %s
    [NUM_CHAN I4,
     CHAN_FREQ R8 [NDIM=1]]
    LIMIT %d
    """ % (
        fn,
        len(spw_chan_freqs),
    )

    with taql_table(create_table_query) as spw:
        # CHAN_FREQ is variably shaped, so each row is written separately
        for i, chan_freq in enumerate(spw_chan_freqs):
            index = (slice(i, i + 1),)
            spw.putcol("NUM_CHAN", np.array([chan_freq.shape[0]]), index=index)
            spw.putcol("CHAN_FREQ", chan_freq[None, :], index=index)

    yield fn

    # Remove the temporary directory
    # except it causes issues with casacore files on py3
    # https://github.com/ska-sa/dask-ms/issues/32
    # shutil.rmtree(str(spw_dir))


@pytest.fixture
def wsrt_antenna_positions():
    """Westerbork antenna positions"""
    return np.array(
        [
            [3828763.10544699, 442449.10566454, 5064923.00777],
            [3828746.54957258, 442592.13950824, 5064923.00792],
            [3828729.99081359, 442735.17696417, 5064923.00829],
            [3828713.43109885, 442878.2118934, 5064923.00436],
            [3828696.86994428, 443021.24917264, 5064923.00397],
            [3828680.31391933, 443164.28596862, 5064923.00035],
            [3828663.75159173, 443307.32138056, 5064923.00204],
            [3828647.19342757, 443450.35604638, 5064923.0023],
            [3828630.63486201, 443593.39226634, 5064922.99755],
            [3828614.07606798, 443736.42941621, 5064923.0],
            [3828609.94224429, 443772.19450029, 5064922.99868],
            [3828601.66208572, 443843.71178407, 5064922.99963],
            [3828460.92418735, 445059.52053929, 5064922.99071],
            [3828452.64716351, 445131.03744105, 5064922.98793],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def ant_table(tmp_path_factory, wsrt_antenna_positions):
    pytest.importorskip("arcae")
    ant_dir = tmp_path_factory.mktemp("ant_dir", numbered=True)
    fn = os.path.join(str(ant_dir), "ANTENNA")

    create_table_query = """
    CREATE TABLE %s
    [POSITION R8 [NDIM=1, SHAPE=[3]],
     NAME S]
    LIMIT %d
    """ % (
        fn,
        wsrt_antenna_positions.shape[0],
    )

    names = ["ANTENNA-%d" % i for i in range(wsrt_antenna_positions.shape[0])]

    with taql_table(create_table_query) as ant:
        ant.putcol("POSITION", wsrt_antenna_positions)
        ant.putcol("NAME", np.array(names))

    yield fn


S3_KEY = "abcdef1234567890"
S3_REGION = "af-cpt"


@pytest.fixture(scope="function")
def s3_bucket_name():
    return f"test-bucket-{uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def s3_server():
    """Starts a moto S3 server on a free local port, yielding its url"""
    moto_server = pytest.importorskip("moto.server")

    # The server logs every request, which is noisy under -s
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    server = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=port)
    server.start()

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.stop()


@pytest.fixture
def s3_url(s3_server):
    return s3_server


@pytest.fixture
def s3_key():
    # moto accepts any credentials
    return S3_KEY


@pytest.fixture
def s3_fs(s3_url, s3_key):
    s3fs = pytest.importorskip("s3fs")
    return s3fs.S3FileSystem(
        key=s3_key,
        secret=s3_key,
        client_kwargs={"endpoint_url": s3_url, "region_name": S3_REGION},
    )
