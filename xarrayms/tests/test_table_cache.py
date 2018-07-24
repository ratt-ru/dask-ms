from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from xarrayms.table_cache import TableCache


def test_table_cache(ms):
    with TableCache.instance().open(ms) as table:
        ant1 = table.getcol("ANTENNA1")  # noqa
        ant2 = table.getcol("ANTENNA2")  # noqa
