from daskms.apps.convert import parse_chunks


def test_chunk_parsing():
    assert parse_chunks("{row: 1000, chan: 16}") == {"row": 1000, "chan": 16}
    assert parse_chunks("{row: (1000, 1000, 10)}") == {"row": (1000, 1000, 10)}
