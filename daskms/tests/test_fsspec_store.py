import os
from subprocess import Popen, PIPE
from pathlib import Path

import numpy as np
import pytest

from daskms.fsspec_store import Store


def find_executable(executable, path=None):
    if not path:
        paths = os.environ["PATH"].split(os.pathsep)

        for path in map(Path, paths):
            result = find_executable(executable, path=path)

            if result:
                return result

    if not path.is_dir():
        raise ValueError(f"{path} is not a directory")

    for child in path.iterdir():
        if child.is_file():
            if child.stem == executable:
                return child
        elif child.is_dir():
            result = find_executable(executable, path=child)

            if result:
                return result
        else:
            raise ValueError(f"Unhandled path {child}")

    return None


@pytest.fixture(scope="session")
def minio_server(tmp_path_factory):
    server_path = find_executable("minio")

    if not server_path:
        pytest.skip("Unable to find \"minio\" server binary")

    data_dir = tmp_path_factory.mktemp("data")
    args = [str(server_path), "server", str(data_dir)]

    # Start the server process and read a line from stdout so that we know
    # it's started
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
            "http://127.0.0.1:9000", "minioadmin", "minioadmin"]
    with Popen(args, shell=False, stdout=PIPE, stderr=PIPE) as client_process:
        retcode = client_process.wait()

        if retcode != 0:
            raise ValueError(f"mc set alias failed with return code {retcode}")

    yield client_path


@pytest.fixture
def minio_admin(minio_client, minio_alias):
    minio = pytest.importorskip("minio")
    yield minio.MinioAdmin(minio_alias, binary_path=str(minio_client))


def test_local_store(tmp_path, minio_admin):
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

    minio_admin.user_add("abcdef1234567890", "abcdef1234567890")

    print(minio_admin)

    print(store.ls())
