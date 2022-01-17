import multiprocessing
import os
from subprocess import Popen, PIPE
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest

from daskms.fsspec_store import Store


URL = urlparse("http://127.0.0.1:9000")


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
            str(data_dir), f"--address={URL.netloc}"]

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
            URL.geturl(), "minioadmin", "minioadmin"]

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
    yield minio.Minio(URL.netloc,
                      access_key=minio_user_key,
                      secret_key=minio_user_key,
                      secure=URL.scheme == "https")


def test_local_store(tmp_path):
    zarr = pytest.importorskip("zarr")
    payload = "How now brown cow"
    filename = "cow.txt"
    (tmp_path / filename).write_text(payload)
    (tmp_path / "foo.txt").write_text(payload)
    (tmp_path / "bar.txt").write_text(payload)
    (tmp_path / "qux.txt").write_text(payload)
    store = Store(str(tmp_path))

    assert store.map[filename] == payload.encode("utf-8")

    root = zarr.group(store=store.map)
    data = root.require_dataset("MODEL_DATA",  # noqa
                                shape=1000,
                                dtype=np.complex128)


def test_minio_server(tmp_path, py_minio_client,
                      minio_admin, minio_alias,
                      minio_user_key):
    payload = "How now brown cow"
    stuff = tmp_path / "stuff.txt"
    stuff.write_text(payload)

    py_minio_client.make_bucket("test-bucket")
    py_minio_client.fput_object("test-bucket",
                                "stuff.txt",
                                str(stuff))

    s3fs = pytest.importorskip("s3fs")
    s3 = s3fs.S3FileSystem(
        key=minio_user_key,
        secret=minio_user_key,
        client_kwargs={
            "endpoint_url": URL.geturl(),
            "region_name": "af-cpt"
        })

    with s3.open("test-bucket/stuff.txt", "rb") as f:
        assert f.read() == payload.encode("utf-8")
