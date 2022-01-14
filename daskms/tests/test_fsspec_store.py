from daskms.fsspec_store import Store

def test_local_store(tmp_path):
    payload = "How now brown cow"
    filename = "cow.txt"
    dummy = tmp_path / filename
    dummy.write_text(payload)
    store = Store(str(tmp_path))

    assert store.map[filename] == payload.encode("utf-8")