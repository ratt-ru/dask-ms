# -*- coding: utf-8 -*-

from pathlib import Path
import os
import platform

import pytest

from daskms.descriptors.builder import DefaultDescriptorBuilder
from daskms.descriptors.ms import MSDescriptorBuilder
from daskms.descriptors.ms_subtable import MSSubTableDescriptorBuilder
from daskms.descriptors.builder_factory import (
    filename_builder_factory,
    string_builder_factory,
    parse_function_call_string,
)


_root = Path("C:/" if platform.system() == "Windows" else os.sep, "home", "moriarty")


@pytest.mark.parametrize(
    "filename, builder_cls",
    [
        (_root / "test.ms", MSDescriptorBuilder),
        (_root / f"test.ms{os.sep}", MSDescriptorBuilder),
        (_root / "test.ms{s}{s}".format(s=os.sep), MSDescriptorBuilder),
        # Indirectly accessed subtable correctly identified
        (_root / "test.ms::SOURCE", MSSubTableDescriptorBuilder),
        (_root / f"test.ms::SOURCE{os.sep}", MSSubTableDescriptorBuilder),
        # Directly accessed subtable not identified
        (_root / "test.ms" / "SOURCE", DefaultDescriptorBuilder),
        (_root / "test.ms" / f"SOURCE{os.sep}", DefaultDescriptorBuilder),
        # Default Table
        (_root / "test.table", DefaultDescriptorBuilder),
        # Default indirectly accessed Subtable
        (_root / "test.table::SUBTABLE", DefaultDescriptorBuilder),
        # Default directly accessed Subtable
        (_root / "test.table" / "SUBTABLE", DefaultDescriptorBuilder),
    ],
)
def test_filename_builder_factory(filename, builder_cls):
    assert isinstance(filename_builder_factory(filename), builder_cls)


@pytest.mark.parametrize(
    "builder, builder_cls",
    [
        ("ms", MSDescriptorBuilder),
        ("ms(False)", MSDescriptorBuilder),
        ("mssubtable('SOURCE')", MSSubTableDescriptorBuilder),
    ],
)
def test_string_builder_factory(builder, builder_cls):
    assert isinstance(string_builder_factory(builder), builder_cls)


@pytest.mark.parametrize(
    "fn_str, result",
    [
        ("fn(1, '2', c=3, d='fred')", ("fn", (1, "2"), {"c": 3, "d": "fred"})),
        ("fn", ("fn", (), {})),
    ],
)
def test_parse_function_call_string(fn_str, result):
    assert parse_function_call_string(fn_str) == result
