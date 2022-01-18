import numpy as np
import pytest

from daskms.fsspec_store import DaskMSStore


def test_local_store(tmp_path):
    zarr = pytest.importorskip("zarr")
    payload = "How now brown cow"
    filename = "cow.txt"
    (tmp_path / filename).write_text(payload)
    (tmp_path / "foo.txt").write_text(payload)
    (tmp_path / "bar.txt").write_text(payload)
    (tmp_path / "qux.txt").write_text(payload)
    store = DaskMSStore(str(tmp_path))
    store.fs.mkdir(f"{store.path}/bob", exist_ok=True)

    assert store.map[filename] == payload.encode("utf-8")

    root = zarr.group(store=store.map)
    data = root.require_dataset("MODEL_DATA",  # noqa
                                shape=1000,
                                dtype=np.complex128)


def test_minio_server(tmp_path, py_minio_client,
                      minio_admin, minio_alias,
                      minio_user_key, minio_url,
                      s3_bucket_name):
    payload = "How now brown cow"
    stuff = tmp_path / "stuff.txt"
    stuff.write_text(payload)

    py_minio_client.make_bucket(s3_bucket_name)
    py_minio_client.fput_object(s3_bucket_name,
                                "stuff.txt",
                                str(stuff))

    s3fs = pytest.importorskip("s3fs")
    s3 = s3fs.S3FileSystem(
        key=minio_user_key,
        secret=minio_user_key,
        client_kwargs={
            "endpoint_url": minio_url.geturl(),
            "region_name": "af-cpt"
        })

    with s3.open(f"{s3_bucket_name}/stuff.txt", "rb") as f:
        assert f.read() == payload.encode("utf-8")
