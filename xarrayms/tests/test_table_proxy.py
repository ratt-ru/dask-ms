from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import cPickle as pickle
except ImportError:
    import pickle

from dask.sizeof import getsizeof, sizeof
import numpy as np
import pytest
from xarrayms import TableProxy


@pytest.mark.parametrize("table_kwargs", [
    {'readonly': True, 'ack': False},
    {'readonly': False, 'ack': False}])
def test_table_proxy_pickle(ms, table_kwargs):
    tp = TableProxy(ms, **table_kwargs)
    ntp = pickle.loads(pickle.dumps(tp))

    # Table name match
    assert ntp._table_name == tp._table_name
    # Table creation kwargs match
    assert ntp._table_kwargs == tp._table_kwargs

    # Table reads match
    with tp.read_locked() as table:
        ant1 = table.getcol("ANTENNA1")
        ant2 = table.getcol("ANTENNA2")

    with ntp.read_locked() as new_table:
        assert np.all(ant1 == new_table.getcol("ANTENNA1"))
        assert np.all(ant2 == new_table.getcol("ANTENNA2"))


def test_table_proxy_sizeof(ms):
    tp = TableProxy(ms, readonly=False, ack=False)

    size = getsizeof(tp._table_name)
    size += getsizeof(tp._table_kwargs)
    size += getsizeof(tp._table_key)

    assert sizeof(tp) == size
