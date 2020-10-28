from daskms import xds_from_ms
from daskms.expressions import data_column_expr

from numpy.testing import assert_array_equal


def test_expressions(ms):
    datasets = xds_from_ms(ms)

    for i, ds in enumerate(datasets):
        datasets[i] = ds.assign(DIR1_DATA=ds.DATA,
                                DIR2_DATA=ds.DATA,
                                DIR3_DATA=ds.DATA)

    results = [
        ds.DATA.data / (
            ds.DIR1_DATA.data +
            ds.DIR2_DATA.data +
            ds.DIR3_DATA.data) * 4
        for ds in datasets
    ]

    string = "DATA / (DIR1_DATA + DIR2_DATA + DIR3_DATA)*4"
    datasets = data_column_expr(string, datasets)

    for i, ds in enumerate(datasets):
        assert_array_equal(results[i], ds.DATA_EXPRESSION.data)
