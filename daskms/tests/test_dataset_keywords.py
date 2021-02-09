# -*- coding: utf-8 -*-

import dask
import pyrap.tables as pt
import pytest

from daskms.example_data import example_ms
from daskms.table_proxy import TableProxy
from daskms import xds_to_table, xds_from_ms
from daskms.dataset import Dataset


@pytest.fixture(scope='module')
def keyword_ms():
    try:
        yield example_ms()
    finally:
        pass


@pytest.mark.parametrize("table_kw", [True, False])
@pytest.mark.parametrize("column_kw", [True, False])
@pytest.mark.parametrize("table_proxy", [True, False])
def test_read_keywords(keyword_ms, table_kw, column_kw, table_proxy):
    # Create an example MS
    with pt.table(keyword_ms, ack=False, readonly=True) as T:
        desc = T._getdesc(actual=True)

    ret = xds_from_ms(keyword_ms,
                      table_keywords=table_kw,
                      column_keywords=column_kw,
                      table_proxy=table_proxy)

    if isinstance(ret, tuple):
        ret_pos = 1

        if table_kw is True:
            assert desc["_keywords_"] == ret[ret_pos]
            ret_pos += 1

        if column_kw is True:
            colkw = ret[ret_pos]

            for column, keywords in colkw.items():
                assert desc[column]['keywords'] == keywords

            ret_pos += 1

        if table_proxy is True:
            tp = ret[ret_pos]
            assert isinstance(tp, TableProxy)
            assert tp.nrows().result() == 10

            ret_pos += 1
    else:
        assert table_kw is False
        assert column_kw is False
        assert table_proxy is False
        assert isinstance(ret, list)


def test_write_keywords(ms):
    datasets = xds_from_ms(ms)

    # Add to table keywords
    writes = xds_to_table([], ms, [], table_keywords={'bob': 'qux'})
    assert isinstance(writes, Dataset)
    dask.compute(writes)

    with pt.table(ms, ack=False, readonly=True) as T:
        assert T.getkeywords()['bob'] == 'qux'

    # Add to column keywords
    writes = xds_to_table(datasets, ms, [],
                          column_keywords={'STATE_ID': {'bob': 'qux'}})
    assert isinstance(writes, list)
    assert all(isinstance(w, Dataset) for w in writes)
    dask.compute(writes)

    with pt.table(ms, ack=False, readonly=True) as T:
        assert T.getcolkeywords("STATE_ID")['bob'] == 'qux'

    # Remove from column and table keywords
    from daskms.writes import DELKW
    writes = xds_to_table(datasets, ms, [],
                          table_keywords={'bob': DELKW},
                          column_keywords={'STATE_ID': {'bob': DELKW}})
    assert isinstance(writes, list)
    assert all(isinstance(w, Dataset) for w in writes)
    dask.compute(writes)

    with pt.table(ms, ack=False, readonly=True) as T:
        assert 'bob' not in T.getkeywords()
        assert 'bob' not in T.getcolkeywords("STATE_ID")


def test_write_table_proxy_keyword(ms):
    datasets = xds_from_ms(ms)

    # Test that we get a TableProxy if requested
    writes, tp = xds_to_table(datasets, ms, [], table_proxy=True)
    assert isinstance(writes, list)
    assert all(isinstance(w, Dataset) for w in writes)
    assert isinstance(tp, TableProxy)
    assert tp.nrows().result() == 10

    writes = xds_to_table(datasets, ms, [], table_proxy=False)
    assert isinstance(writes, list)
    assert all(isinstance(w, Dataset) for w in writes)
