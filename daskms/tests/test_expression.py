from daskms import xds_from_ms
from daskms.expressions import data_column_expr

from numpy.testing import assert_array_equal


def test_expressions(ms):
    datasets = xds_from_ms(ms)

    for i, ds in enumerate(datasets):
        dims = ds.DATA.dims
        datasets[i] = ds.assign(DIR1_DATA=(dims, ds.DATA.data),
                                DIR2_DATA=(dims, ds.DATA.data),
                                DIR3_DATA=(dims, ds.DATA.data))

    results = [
        ds.DATA.data / (
            -ds.DIR1_DATA.data +
            ds.DIR2_DATA.data +
            ds.DIR3_DATA.data) * 4
        for ds in datasets
    ]

    string = "DATA / (-DIR1_DATA + DIR2_DATA + DIR3_DATA)*4"
    expressions = data_column_expr(string, datasets)

    for i, (ds, expr) in enumerate(zip(datasets, expressions)):
        assert_array_equal(results[i], expr)
