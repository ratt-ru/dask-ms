# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

from daskms.utils import promote_columns


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
