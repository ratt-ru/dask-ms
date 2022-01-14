import numpy as np
import pytest

from daskms.fsspec_store import Store

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
    data = root.require_dataset("MODEL_DATA", shape=1000, dtype=np.complex128)

    print(store.ls())
