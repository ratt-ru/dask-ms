# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import join as pjoin
import pytest

from daskms.table_schemas import (lookup_table_schema,
                                  MS_SCHEMA,
                                  ANTENNA_SCHEMA,
                                  FIELD_SCHEMA,
                                  SPECTRAL_WINDOW,
                                  POLARIZATION)


@pytest.mark.parametrize("filename, schema", [
    (pjoin("bob", "qux", "FRED.MS%s" % os.sep), MS_SCHEMA),
    ("test.ms", MS_SCHEMA),
    ("test.ms::ANTENNA", ANTENNA_SCHEMA),
    ("test.ms::FIELD", FIELD_SCHEMA),
    ("test.ms::SPECTRAL_WINDOW", SPECTRAL_WINDOW),
    ("test.ms::POLARIZATION", POLARIZATION)])
def test_table_suffix_lookup(filename, schema):
    assert schema == lookup_table_schema(filename, None)


@pytest.mark.parametrize("schema_name, schema", [
    ("MS", MS_SCHEMA)])
def test_table_schema_name_lookup(schema_name, schema):
    assert schema == lookup_table_schema("test.ms", schema_name)
