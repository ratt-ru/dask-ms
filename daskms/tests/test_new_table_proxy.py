import casacore.tables as ct
import numpy as np
from numpy.testing import assert_array_almost_equal

from daskms.new_table_proxy import TableProxy


def test_new_table_proxy(ms):
    tp = TableProxy(ct.table, ms, readonly=True)
    assert tp.nrows() == 10
    assert_array_almost_equal(tp.getcol("TIME"), np.arange(1.0, 0.0, -0.1))
