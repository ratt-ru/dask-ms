# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join as pjoin, sep

import pytest

from xarrayms.descriptors.builder import DefaultDescriptorBuilder
from xarrayms.descriptors.ms import MSDescriptorBuilder
from xarrayms.descriptors.ms_subtable import MSSubTableDescriptorBuilder
from xarrayms.descriptors.builder_factory import (filename_builder_factory,
                                                  string_builder_factory,
                                                  parse_function_call_string)


@pytest.mark.parametrize("filename, builder_cls", [
    (pjoin(sep, "tmp", "test.ms"), MSDescriptorBuilder),
    (pjoin(sep, "tmp", "test.ms", ""), MSDescriptorBuilder),
    (pjoin(sep, "tmp", "test.ms%s%s" % (sep, sep)), MSDescriptorBuilder),
    (pjoin(sep, "tmp", "test.ms::SOURCE"), MSSubTableDescriptorBuilder),
    (pjoin(sep, "tmp", "test.ms::SOURCE", ""), MSSubTableDescriptorBuilder),
    (pjoin(sep, "tmp", "test.table"), DefaultDescriptorBuilder),
    (pjoin(sep, "tmp", "test.table", ""), DefaultDescriptorBuilder)
])
def test_filename_builder_factory(filename, builder_cls):
    assert isinstance(filename_builder_factory(filename), builder_cls)


@pytest.mark.parametrize("builder, builder_cls", [
    ("ms", MSDescriptorBuilder),
    ("ms(False)", MSDescriptorBuilder),
    ("subtable('SOURCE')", MSSubTableDescriptorBuilder)])
def test_string_builder_factory(builder, builder_cls):
    assert isinstance(string_builder_factory(builder), builder_cls)


@pytest.mark.parametrize("fn_str, result", [
    ("fn(1, '2', c=3, d='fred')", ('fn', (1, '2'), {'c': 3, 'd': 'fred'})),
    ("fn", ('fn', (), {}))])
def test_parse_function_call_string(fn_str, result):
    assert parse_function_call_string(fn_str) == result
