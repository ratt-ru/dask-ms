# -*- coding: utf-8 -*-

import gc
import multiprocessing
import os
from subprocess import Popen, PIPE
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np
import pyrap.tables as pt
import pytest

from daskms.testing import mark_in_pytest


# content of conftest.py
def pytest_configure(config):
    mark_in_pytest(True)


def pytest_unconfigure(config):
    mark_in_pytest(False)


@pytest.fixture(autouse=True)
def xms_always_gc():
    """ Force garbage collection after each test """
    try:
        yield
    finally:
        gc.collect()


@pytest.fixture(scope="session")
def big_ms(tmp_path_factory, request):
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
    data = rs.random_sample(data_shape) + rs.random_sample(data_shape)*1j

    # Create the table
    with pt.taql(create_table_query) as ms:
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
    field = [0,   0,   0,   1,   1,   1,   1,   2,   2,   2]
    ddid = [0,   0,   0,   0,   0,   0,   0,   1,   1,   1]
    scan = [0,   1,   0,   1,   0,   1,   0,   1,   0,   1]

    # Common indexing columns
    time = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    ant1 = [0,   0,   1,   1,   1,   2,   1,   0,   0,   1]
    ant2 = [1,   2,   2,   3,   2,   1,   0,   1,   1,   2]

    # Column we'll write to
    state = [0,   0,   0,   0,   0,   0,   0,   0,   0,   0]

    rs = np.random.RandomState(42)
    data_shape = (len(state), 16, 4)
    data = rs.random_sample(data_shape) + rs.random_sample(data_shape)*1j
    uvw = rs.random_sample((len(state), 3)).astype(np.float64)

    # Create the table
    with pt.taql(create_table_query) as ms:
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
    return (np.linspace(.856e9, 2*.856e9, 8),
            np.linspace(.856e9, 2*.856e9, 16),
            np.linspace(.856e9, 2*.856e9, 32))


@pytest.fixture
def spw_table(tmp_path_factory, spw_chan_freqs):
    """ Simulate a SPECTRAL_WINDOW table with two spectral windows """
    spw_dir = tmp_path_factory.mktemp("spw_dir", numbered=True)
    fn = os.path.join(str(spw_dir), "SPECTRAL_WINDOW")

    create_table_query = """
    CREATE TABLE %s
    [NUM_CHAN I4,
     CHAN_FREQ R8 [NDIM=1]]
    LIMIT %d
    """ % (fn, len(spw_chan_freqs))

    with pt.taql(create_table_query) as spw:
        spw.putvarcol("NUM_CHAN", {"r%d" % i: s.shape[0]
                                   for i, s
                                   in enumerate(spw_chan_freqs)})
        spw.putvarcol("CHAN_FREQ", {"r%d" % i: s[None, :]
                                    for i, s
                                    in enumerate(spw_chan_freqs)})

    yield fn

    # Remove the temporary directory
    # except it causes issues with casacore files on py3
    # https://github.com/ska-sa/dask-ms/issues/32
    # shutil.rmtree(str(spw_dir))


@pytest.fixture
def wsrt_antenna_positions():
    """ Westerbork antenna positions """
    return np.array([
        [3828763.10544699,   442449.10566454,  5064923.00777],
        [3828746.54957258,   442592.13950824,  5064923.00792],
        [3828729.99081359,   442735.17696417,  5064923.00829],
        [3828713.43109885,   442878.2118934,  5064923.00436],
        [3828696.86994428,   443021.24917264,  5064923.00397],
        [3828680.31391933,   443164.28596862,  5064923.00035],
        [3828663.75159173,   443307.32138056,  5064923.00204],
        [3828647.19342757,   443450.35604638,  5064923.0023],
        [3828630.63486201,   443593.39226634,  5064922.99755],
        [3828614.07606798,   443736.42941621,  5064923.],
        [3828609.94224429,   443772.19450029,  5064922.99868],
        [3828601.66208572,   443843.71178407,  5064922.99963],
        [3828460.92418735,   445059.52053929,  5064922.99071],
        [3828452.64716351,   445131.03744105,  5064922.98793]],
        dtype=np.float64)


@pytest.fixture
def ant_table(tmp_path_factory, wsrt_antenna_positions):
    ant_dir = tmp_path_factory.mktemp("ant_dir", numbered=True)
    fn = os.path.join(str(ant_dir), "ANTENNA")

    create_table_query = """
    CREATE TABLE %s
    [POSITION R8 [NDIM=1, SHAPE=[3]],
     NAME S]
    LIMIT %d
    """ % (fn, wsrt_antenna_positions.shape[0])

    names = ["ANTENNA-%d" % i for i in range(wsrt_antenna_positions.shape[0])]

    with pt.taql(create_table_query) as ant:
        ant.putcol("POSITION", wsrt_antenna_positions)
        ant.putcol("NAME", names)

    yield fn


MINIO_URL = urlparse("http://127.0.0.1:9000")


@pytest.fixture(scope="function")
def s3_bucket_name():
    return f"test-bucket-{uuid4().hex[:8]}"


@pytest.fixture
def minio_url():
    return MINIO_URL


def find_executable(executable, path=None):
    if not path:
        paths = os.environ["PATH"].split(os.pathsep)

        for path in map(Path, paths):
            result = find_executable(executable, path=path)

            if result:
                return result
    elif path.is_dir():
        for child in path.iterdir():
            result = find_executable(executable, child)

            if result:
                return result
    elif path.is_file():
        if path.stem == executable:
            return path
    else:
        return None


@pytest.fixture(scope="session")
def minio_server(tmp_path_factory):
    server_path = find_executable("minio")

    if not server_path:
        pytest.skip("Unable to find \"minio\" server binary")

    data_dir = tmp_path_factory.mktemp("data")
    args = [str(server_path), "server",
            str(data_dir), f"--address={MINIO_URL.netloc}"]

    # Start the server process and read a line from stdout so that we know
    # it's started
    ctx = multiprocessing.get_context("spawn")  # noqa
    server_process = Popen(args, shell=False, stdout=PIPE, stderr=PIPE)
    server_process.stdout.readline()

    try:
        retcode = server_process.poll()

        if retcode is not None:
            raise ValueError(f"Server failed to start "
                             f"with return code {retcode}")

        yield server_process
    finally:
        server_process.kill()


@pytest.fixture
def minio_alias():
    return "testcloud"


@pytest.fixture
def minio_client(minio_server, minio_alias):
    client_path = find_executable("mc")

    if not client_path:
        pytest.skip("Unable to find \"mc\" binary")

    # Set the server alias on the client
    args = [str(client_path), "alias", "set", minio_alias,
            MINIO_URL.geturl(), "minioadmin", "minioadmin"]

    ctx = multiprocessing.get_context("spawn")  # noqa
    with Popen(args, shell=False, stdout=PIPE, stderr=PIPE) as client_process:
        retcode = client_process.wait()

        if retcode != 0:
            raise ValueError(f"mc set alias failed with return code {retcode}")

    yield client_path


@pytest.fixture
def minio_user_key():
    return "abcdef1234567890"


@pytest.fixture
def minio_admin(minio_client, minio_alias, minio_user_key):
    minio = pytest.importorskip("minio")
    minio_admin = minio.MinioAdmin(minio_alias, binary_path=str(minio_client))
    # Add a user and give it readwrite access
    minio_admin.user_add(minio_user_key, minio_user_key)
    minio_admin.policy_set("readwrite", user=minio_user_key)
    yield minio_admin


@pytest.fixture
def py_minio_client(minio_client, minio_admin, minio_alias, minio_user_key):
    minio = pytest.importorskip("minio")
    yield minio.Minio(MINIO_URL.netloc,
                      access_key=minio_user_key,
                      secret_key=minio_user_key,
                      secure=MINIO_URL.scheme == "https")
