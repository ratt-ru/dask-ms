# -*- coding: utf-8 -*-

from daskms.table import table_exists


def test_table_exists(tmp_path):
    ms_path = tmp_path / "test.ms"
    ms_path.mkdir(parents=True, exist_ok=False)
    assert table_exists(str(ms_path)) is True

    ant_path = ms_path / "ANTENNA"
    ant_path.mkdir(parents=True, exist_ok=False)
    # Both the directory and canonical subtable access forms work
    assert table_exists(str(ant_path))
    assert table_exists('::'.join((str(ms_path), "ANTENNA")))
