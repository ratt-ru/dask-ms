# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from operator import mul
import os

from xarrayms.creation import empty_ms

import numpy as np
from numpy.testing import assert_array_almost_equal
import pyrap.tables as pt
import pytest
from six.moves import reduce


@pytest.mark.parametrize("nrow", [10])
@pytest.mark.parametrize("nchan", [16, None])
@pytest.mark.parametrize("ncorr", [1, 4, None])
@pytest.mark.parametrize("add_imaging_cols", [True, False])
def test_empty_ms(nrow, nchan, ncorr, add_imaging_cols, tmpdir):

    ms = str(tmpdir) + os.pathsep + "test.ms"
    empty_ms(ms, nchan, ncorr, add_imaging_cols=add_imaging_cols)
    imaging_cols = set(["MODEL_DATA", "CORRECTED_DATA", "IMAGING_WEIGHT"])

    with pt.table(ms, readonly=False, ack=False) as T:
        table_cols = set(T.colnames())

        if imaging_cols is True:
            assert imaging_cols in table_cols
        else:
            assert imaging_cols not in table_cols

        # Get the Data Manager info
        dminfo = {dm['NAME']: dm for dm in T.getdminfo().values()}

        # Columns in the Standard Storage Manager
        assert dminfo['StandardStMan']['COLUMNS'] == ['ANTENNA1',
                                                      'ANTENNA2',
                                                      'ARRAY_ID',
                                                      'DATA_DESC_ID',
                                                      'EXPOSURE',
                                                      'FEED1',
                                                      'FEED2',
                                                      'FIELD_ID',
                                                      'FLAG_ROW',
                                                      'INTERVAL',
                                                      'OBSERVATION_ID',
                                                      'PROCESSOR_ID',
                                                      'SCAN_NUMBER',
                                                      'STATE_ID',
                                                      'TIME',
                                                      'TIME_CENTROID']

        # Some others
        assert dminfo['FlagCategory']['COLUMNS'][0] == "FLAG_CATEGORY"
        assert dminfo['Data']['COLUMNS'][0] == "DATA"

        # Imaging columns
        if imaging_cols is True:
            assert dminfo['ImagingWeight']['COLUMNS'][0] == "IMAGING_WEIGHT"
            assert dminfo['ModelData']['COLUMNS'][0] == "MODEL_DATA"
            assert dminfo['CorrectedData']['COLUMNS'][0] == "CORRECTED_DATA"

        # Add rows, get data, put data
        T.addrows(nrow)
        print(T.getvarcol("DATA", startrow=0, nrow=1).popitem()[1].shape)
        shape = (nrow, nchan, ncorr)
        data = np.arange(reduce(mul, shape, 1), dtype=np.complex64)
        data = data.reshape(shape)
        dict_data = {'r%05d' % i: d[None, :, :] for i, d in enumerate(data)}
        from pprint import pprint
        pprint({k: v.shape for k, v in dict_data.items()})
        T.putvarcol("DATA", dict_data, startrow=0, nrow=nrow)
        new_data = T.getcol("DATA")
        assert_array_almost_equal(new_data, data)

