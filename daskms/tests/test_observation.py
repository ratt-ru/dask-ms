# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from daskms.observation import time


@pytest.mark.parametrize("date", [(2020, 1, 1)])
def test_time(date):
    year, month, day = date
    intervals = np.array([14, 15, 16, 17])
    res = time(year, month, day, 0, 0, 0, intervals)

    expected = 58849.0 + np.array([0, 14.5, 30, 46.5])
    assert_almost_equal(res, expected)
