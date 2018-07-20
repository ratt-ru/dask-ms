from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import pytest
from xarrayms import TableProxy


@pytest.mark.parametrize("table_kwargs", [
    {'readonly': True},
    {'readonly': False}])
def test_table_proxy_pickle(ms, table_kwargs):
    tp = TableProxy(ms, **table_kwargs)
    ntp = pickle.loads(pickle.dumps(tp))

    # Table object differs
    assert ntp._table != tp._table

    # Table name match
    assert ntp._table_name == tp._table_name
    # Table creation kwargs match
    assert ntp._kwargs == tp._kwargs

    # Table reads match
    ant1 = tp("getcol", "ANTENNA1")
    ant2 = tp("getcol", "ANTENNA2")

    assert np.all(ant1 == ntp("getcol", "ANTENNA1"))
    assert np.all(ant2 == ntp("getcol", "ANTENNA2"))
