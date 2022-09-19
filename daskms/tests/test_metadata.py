import dask
import pyrap.tables as pt

from daskms.constants import DASKMS_METADATA
from daskms import xds_from_storage_ms, xds_to_table


def test_provenance(ms, tmp_path_factory):
    datasets = xds_from_storage_ms(ms)

    for ds in datasets:
        assert ds.attrs[DASKMS_METADATA]["provenance"] == [ms]

    data_dir = tmp_path_factory.mktemp("provenance")
    store = str(data_dir / "blah.ms")
    dask.compute(xds_to_table(datasets, store))

    with pt.table(str(store), ack=False) as T:
        assert T.getkeywords()[DASKMS_METADATA] == {"provenance": [ms, store]}
