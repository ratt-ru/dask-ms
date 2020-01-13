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
    start = 58849.0 * 86400.

    expected = start + np.array([0, 14.5, 30, 46.5])
    assert_almost_equal(res, expected)


def test_synthesize_uvw():
    from daskms.observation import synthesize_uvw, VLA_ANTENNA_POSITIONS
    synthesize_uvw
    intervals = np.full(60, 15.0, dtype=np.float64)
    t = time(2020, 1, 1, 0, 0, 0, intervals)
    synthesize_uvw(VLA_ANTENNA_POSITIONS, t, [0, 0])