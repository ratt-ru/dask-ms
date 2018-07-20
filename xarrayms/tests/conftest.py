import shutil
import os

import pyrap.tables as pt
import pytest


@pytest.fixture(scope="session")
def ms(tmpdir_factory):
    msdir = tmpdir_factory.mktemp("msdir", numbered=False)
    fn = os.path.join(str(msdir), "test.ms")

    create_table_query = """
    CREATE TABLE %s
    [FIELD_ID I4,
    ANTENNA1 I4,
    ANTENNA2 I4,
    DATA_DESC_ID I4,
    SCAN_NUMBER I4,
    STATE_ID I4,
    TIME R8]
    LIMIT 10
    """ % fn

    # Common grouping columns
    field = [0,   0,   0,   1,   1,   1,   1,   2,   2,   2]
    ddid = [0,   0,   0,   0,   0,   0,   0,   1,   1,   1]
    scan = [0,   1,   0,   1,   0,   1,   0,   1,   0,   1]

    # Common indexing columns
    time = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    ant1 = [0,   0,   1,   1,   1,   2,   1,   0,   0,   1]
    ant2 = [1,   2,   2,   3,   2,   1,   0,   1,   1,   2]

    # Column we'll write to
    state = [0,   0,   0,   0,   0,   0,   0,   0,   0,   0]

    # Create the table
    with pt.taql(create_table_query) as ms:
        ms.putcol("TIME", time)
        ms.putcol("FIELD_ID", field)
        ms.putcol("DATA_DESC_ID", ddid)
        ms.putcol("ANTENNA1", ant1)
        ms.putcol("ANTENNA2", ant2)
        ms.putcol("SCAN_NUMBER", scan)
        ms.putcol("STATE_ID", state)

    yield fn

    # Remove the temporary directory
    shutil.rmtree(str(msdir))
