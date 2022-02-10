# -*- coding: utf-8 -*-

import os
from pathlib import Path
import platform

import pytest

from daskms.utils import (promote_columns,
                          natural_order,
                          table_path_split,
                          requires)


def test_natural_order():
    data = [f"{i}.parquet" for i in reversed(range(20))]
    expected = [f"{i}.parquet" for i in range(20)]
    assert sorted(data, key=natural_order) == expected
    assert sorted(data) != expected


@pytest.mark.parametrize("columns", [["TIME", "ANTENNA1"]])
@pytest.mark.parametrize("default", [["DATA"]])
def test_promotion(columns, default):
    # Lists stay as lists
    assert promote_columns(columns, default) == columns

    # Tuples promoted to lists
    assert promote_columns(tuple(columns), default) == columns

    # Singleton promoted to list
    assert promote_columns(columns[0], default) == [columns[0]]

    # None gives us the default
    assert promote_columns(None, default) == default


_root_path = Path("C:/" if platform.system() == "Windows" else os.sep,
                  "home", "moriarty")


@pytest.mark.parametrize("path, root, table, subtable", [
    # Table access
    (_root_path / "test.ms",
     _root_path, "test.ms", ""),
    (_root_path / f"test.ms{os.sep}",
     _root_path, "test.ms", ""),
    (_root_path / "test.ms{s}{s}".format(s=os.sep),
     _root_path, "test.ms", ""),
    # Indirect subtable access
    (_root_path / "test.ms::SOURCE",
     _root_path, "test.ms", "SOURCE"),
    (_root_path / f"test.ms::SOURCE{os.sep}",
     _root_path, "test.ms", "SOURCE"),
    # Direct subtable access
    (_root_path / "test.ms" / "SOURCE",
     _root_path / "test.ms", "SOURCE", ""),
    (_root_path / "test.ms" / f"SOURCE{os.sep}",
     _root_path / "test.ms", "SOURCE", "")
])
def test_table_path_split(path, root, table, subtable):
    assert (root, table, subtable) == table_path_split(path)


def test_requires():
    def fn(*args, **kwargs):
        return 1

    decorator = requires(ImportError("foo"), ImportError("bar"),
                         "need foo", "need bar")(fn)

    with pytest.raises(ImportError) as e:
        decorator()

    assert "1. foo" in e.value.msg
    assert "2. bar" in e.value.msg
    assert "need foo" in e.value.msg
    assert "need bar" in e.value.msg

    assert requires("need foo", 1, None)(fn)() == 1
