from io import BytesIO
import pickle

import numpy as np
import yaml
import pytest

from daskms.config import config
from daskms.fsspec_store import DaskMSStore, UnknownStoreTypeError

try:
    import s3fs
except ImportError:
    s3fs = None


def test_store_type(tmp_path_factory):
    path = tmp_path_factory.mktemp("casa")

    with open(path / "table.dat", "w") as f:
        f.write("dummy")

    assert DaskMSStore(path).type() == "casa"

    path = tmp_path_factory.mktemp("zarr")
    (path / "subdir").mkdir(parents=True, exist_ok=True)

    with open(path / "subdir" / ".zgroup", "w") as f:
        f.write("dummy")

    assert DaskMSStore(path).type() == "zarr"

    path = tmp_path_factory.mktemp("parquet")
    (path / "subdir").mkdir(parents=True, exist_ok=True)

    with open(path / "subdir" / "dummy.parquet", "w") as f:
        f.write("dummy")

    assert DaskMSStore(path).type() == "parquet"

    path = tmp_path_factory.mktemp("empty")

    with pytest.raises(UnknownStoreTypeError) as e:
        DaskMSStore(path).type()


def test_local_store(tmp_path):
    zarr = pytest.importorskip("zarr")
    payload = "How now brown cow"
    filename = "cow.txt"
    (tmp_path / filename).write_text(payload)
    (tmp_path / "foo.txt").write_text(payload)
    (tmp_path / "bar.txt").write_text(payload)
    (tmp_path / "qux.txt").write_text(payload)
    store = DaskMSStore(str(tmp_path))
    store.fs.mkdir(f"{store.full_path}{store.fs.sep}bob", exist_ok=True)

    assert store.map[filename] == payload.encode("utf-8")

    root = zarr.group(store=store.map)
    data = root.require_dataset("MODEL_DATA", shape=1000, dtype=np.complex128)  # noqa


def test_store_main_access(tmp_path_factory):
    store_dir = tmp_path_factory.mktemp("STORE0")

    store = DaskMSStore(f"file://{store_dir}")
    assert store.url == f"file://{store_dir}"
    assert store.full_path == str(store_dir)
    assert store.canonical_path == str(store_dir)
    assert store.table is None

    with store.open("foo.txt", "w") as f:
        f.write("How now brown cow")

    assert store.exists("foo.txt")
    assert (store_dir / "foo.txt").exists()


def test_store_subtable_access(tmp_path_factory):
    store_dir = tmp_path_factory.mktemp("STORE0")
    table_dir = store_dir / "TABLE"
    table_dir.mkdir()

    store = DaskMSStore(f"file://{store_dir}::TABLE")
    assert store.url == f"file://{store_dir}::TABLE"
    assert store.full_path == f"{store_dir}{store.fs.sep}TABLE"
    assert store.canonical_path == f"{store_dir}::TABLE"
    assert store.table == "TABLE"

    with store.open("foo.txt", "w") as f:
        f.write("How now brown cow")

    assert store.exists("foo.txt")
    assert (table_dir / "foo.txt").exists()


@pytest.mark.skipif(s3fs is None, reason="s3fs not installed")
def test_minio_server(
    tmp_path,
    py_minio_client,
    minio_user_key,
    minio_url,
    s3_bucket_name,
):
    payload = "How now brown cow"
    stuff = tmp_path / "stuff.txt"
    stuff.write_text(payload)

    py_minio_client.make_bucket(s3_bucket_name)
    py_minio_client.fput_object(s3_bucket_name, "stuff.txt", str(stuff))

    s3 = s3fs.S3FileSystem(
        key=minio_user_key,
        secret=minio_user_key,
        client_kwargs={"endpoint_url": minio_url, "region_name": "af-cpt"},
    )

    with s3.open(f"{s3_bucket_name}/stuff.txt", "rb") as f:
        assert f.read() == payload.encode("utf-8")


@pytest.mark.skipif(s3fs is None, reason="s3fs not installed")
def test_storage_options_from_config(
    tmp_path,
    py_minio_client,
    minio_user_key,
    minio_url,
    s3_bucket_name,
):
    filename = "test.txt"
    payload = "How now brown cow"
    py_minio_client.make_bucket(s3_bucket_name)
    py_minio_client.put_object(
        s3_bucket_name,
        f"subdir/{filename}",
        BytesIO(payload.encode("utf-8")),
        len(payload),
    )

    url = f"s3://{s3_bucket_name}"
    config_file = tmp_path / "config.yaml"
    opts = {
        "key": minio_user_key,
        "secret": minio_user_key,
        "client_kwargs": {
            "endpoint_url": minio_url,
            "region_name": "af-south-1",
        },
    }

    with open(config_file, "w") as f:
        yaml.safe_dump({"storage_options": {url: opts}}, f)

    config.refresh(paths=config.paths + [str(tmp_path)])

    try:
        store = DaskMSStore(f"{url}/subdir")
        assert store.storage_options == opts

        with store.open("test.txt", "rb") as f:
            assert f.read() == payload.encode("utf-8")
    finally:
        config.refresh()


@pytest.mark.skipif(s3fs is None, reason="s3fs not installed")
def test_store_pickle():
    store = DaskMSStore(
        "s3://binface",
        key="foo",
        secret="bar",
        client_kwargs={
            "endpoint_url": "http://127.0.0.1:9000",
            "region_name": "af-cpt",
        },
    )

    pstore = pickle.loads(pickle.dumps(store))
    assert pstore == store
