# -*- coding: utf-8 -*-

import os
from os.path import join as pjoin
import pytest

from daskms.table_schemas import (lookup_table_schema,
                                  MS_SCHEMA,
                                  ANTENNA_SCHEMA,
                                  FEED_SCHEMA,
                                  FIELD_SCHEMA,
                                  SPECTRAL_WINDOW_SCHEMA,
                                  OBSERVATION_SCHEMA,
                                  POLARIZATION_SCHEMA,
                                  POINTING_SCHEMA)


@pytest.mark.parametrize("filename, schema", [
    (pjoin("bob", "qux", f"FRED.MS{os.sep}"), MS_SCHEMA),
    ("test.ms", MS_SCHEMA),
    ("test.ms::ANTENNA", ANTENNA_SCHEMA),
    ("test.ms::FEED", FEED_SCHEMA),
    ("test.ms::FIELD", FIELD_SCHEMA),
    ("test.ms::OBSERVATION", OBSERVATION_SCHEMA),
    ("test.ms::POINTING", POINTING_SCHEMA),
    ("test.ms::POLARIZATION", POLARIZATION_SCHEMA),
    ("test.ms::SPECTRAL_WINDOW", SPECTRAL_WINDOW_SCHEMA)])
def test_table_suffix_lookup(filename, schema):
    assert schema == lookup_table_schema(filename, None)


@pytest.mark.parametrize("schema_name, schema", [
    ("MS", MS_SCHEMA)])
def test_table_schema_name_lookup(schema_name, schema):
    assert schema == lookup_table_schema("test.ms", schema_name)
