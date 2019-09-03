# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import product

import dask
import dask.array as da
import numpy as np
from numpy.testing import assert_array_equal
import pyrap.tables as pt
import pytest

from daskms import xds_to_table, xds_from_ms, Dataset

try:
    from xarray import Dataset as xrDataset
except ImportError:
    xrDataset = None
    xarray_param_marks = pytest.mark.skip(reason="xarray not installed")
else:
    xarray_param_marks = ()


@pytest.mark.parametrize("chunks", [{
    "row": (10,),
}])
@pytest.mark.parametrize("num_chans", [
    [16, 32],
])
@pytest.mark.parametrize("corr_types", [
    [[9, 10, 11, 12], [5, 8]],
    [[5, 6, 7, 8], [9, 12]]
])
@pytest.mark.parametrize("Dataset", [
    Dataset,
    pytest.param(xrDataset, marks=xarray_param_marks)
], ids=["daskms.Dataset", "xarray.Dataset"])
def test_ms_create(Dataset, tmp_path, chunks, num_chans, corr_types):
    # Set up
    rs = np.random.RandomState(42)

    ms_path = tmp_path / "create.ms"
    ms_table_name = str(ms_path)
    ant_table_name = str(ms_path / "ANTENNA")
    ddid_table_name = str(ms_path / "DATA_DESCRIPTION")
    pol_table_name = str(ms_path / "POLARIZATION")
    spw_table_name = str(ms_path / "SPECTRAL_WINDOW")

    ms_datasets = []
    ant_datasets = []
    ddid_datasets = []
    pol_datasets = []
    spw_datasets = []

    # For comparison
    all_data_desc_id = []
    all_data = []

    # Create ANTENNA dataset of 64 antennas
    # Each column in the ANTENNA has a fixed shape so we
    # can represent all rows with one dataset
    na = 64
    position = da.random.random((na, 3))*10000
    offset = da.random.random((na, 3))
    names = np.array(['ANTENNA-%d' % i for i in range(na)], dtype=np.object)
    ds = Dataset({
        'POSITION': (("row", "xyz"), position),
        'OFFSET': (("row", "xyz"), offset),
        'NAME': (("row",), da.from_array(names, chunks=na)),
    })
    ant_datasets.append(ds)

    # Create POLARISATION datasets.
    # Dataset per output row required because column shapes are variable
    for r, corr_type in enumerate(corr_types):
        dask_num_corr = da.full((1,), len(corr_type), dtype=np.int32)
        dask_corr_type = da.from_array(corr_type,
                                       chunks=len(corr_type))[None, :]
        ds = Dataset({
            "NUM_CORR": (("row",), dask_num_corr),
            "CORR_TYPE": (("row", "corr"), dask_corr_type),
        })

        pol_datasets.append(ds)

    # Create multiple MeerKAT L-band SPECTRAL_WINDOW datasets
    # Dataset per output row required because column shapes are variable
    for num_chan in num_chans:
        dask_num_chan = da.full((1,), num_chan, dtype=np.int32)
        dask_chan_freq = da.linspace(.856e9, 2*.856e9, num_chan,
                                     chunks=num_chan)[None, :]
        dask_chan_width = da.full((1, num_chan), .856e9/num_chan)

        ds = Dataset({
            "NUM_CHAN": (("row",), dask_num_chan),
            "CHAN_FREQ": (("row", "chan"), dask_chan_freq),
            "CHAN_WIDTH": (("row", "chan"), dask_chan_width),
        })

        spw_datasets.append(ds)

    # For each cartesian product of SPECTRAL_WINDOW and POLARIZATION
    # create a corresponding DATA_DESCRIPTION.
    # Each column has fixed shape so we handle all rows at once
    spw_ids, pol_ids = zip(*product(range(len(num_chans)),
                                    range(len(corr_types))))
    dask_spw_ids = da.asarray(np.asarray(spw_ids, dtype=np.int32))
    dask_pol_ids = da.asarray(np.asarray(pol_ids, dtype=np.int32))
    ddid_datasets.append(Dataset({
        "SPECTRAL_WINDOW_ID": (("row",), dask_spw_ids),
        "POLARIZATION_ID": (("row",), dask_pol_ids),
    }))

    # Now create the associated MS dataset
    for ddid, (spw_id, pol_id) in enumerate(zip(spw_ids, pol_ids)):
        # Infer row, chan and correlation shape
        row = sum(chunks['row'])
        chan = spw_datasets[spw_id].CHAN_FREQ.shape[1]
        corr = pol_datasets[pol_id].CORR_TYPE.shape[1]

        # Create some dask vis data
        dims = ("row", "chan", "corr")
        np_data = (rs.normal(size=(row, chan, corr)) +
                   1j*rs.normal(size=(row, chan, corr))).astype(np.complex64)

        data_chunks = tuple((chunks['row'], chan, corr))
        dask_data = da.from_array(np_data, chunks=data_chunks)
        # Create dask ddid column
        dask_ddid = da.full(row, ddid, chunks=chunks['row'], dtype=np.int32)
        dataset = Dataset({
            'DATA': (dims, dask_data),
            'DATA_DESC_ID': (("row",), dask_ddid)
        })
        ms_datasets.append(dataset)
        all_data.append(dask_data)
        all_data_desc_id.append(dask_ddid)

    ms_writes = xds_to_table(ms_datasets, ms_table_name, columns="ALL")
    ant_writes = xds_to_table(ant_datasets, ant_table_name, columns="ALL")
    pol_writes = xds_to_table(pol_datasets, pol_table_name, columns="ALL")
    spw_writes = xds_to_table(spw_datasets, spw_table_name, columns="ALL")
    ddid_writes = xds_to_table(ddid_datasets, ddid_table_name, columns="ALL")

    dask.compute(ms_writes, ant_writes, pol_writes, spw_writes, ddid_writes)

    # Check ANTENNA table correctly created
    with pt.table(ant_table_name, ack=False) as A:
        assert_array_equal(A.getcol("NAME"), names)
        assert_array_equal(A.getcol("POSITION"), position)
        assert_array_equal(A.getcol("OFFSET"), offset)

        required_desc = pt.required_ms_desc("ANTENNA")
        required_columns = set(k for k in required_desc.keys()
                               if not k.startswith("_"))

        assert set(A.colnames()) == set(required_columns)

    # Check POLARIZATION table correctly created
    with pt.table(pol_table_name, ack=False) as P:
        for r, corr_type in enumerate(corr_types):
            assert_array_equal(P.getcol("CORR_TYPE", startrow=r, nrow=1),
                               [corr_type])
            assert_array_equal(P.getcol("NUM_CORR", startrow=r, nrow=1),
                               [len(corr_type)])

        required_desc = pt.required_ms_desc("POLARIZATION")
        required_columns = set(k for k in required_desc.keys()
                               if not k.startswith("_"))

        assert set(P.colnames()) == set(required_columns)

    # Check SPECTRAL_WINDOW table correctly created
    with pt.table(spw_table_name, ack=False) as S:
        for r, num_chan in enumerate(num_chans):
            assert_array_equal(S.getcol("NUM_CHAN", startrow=r, nrow=1)[0],
                               num_chan)
            assert_array_equal(S.getcol("CHAN_FREQ", startrow=r, nrow=1)[0],
                               np.linspace(.856e9, 2*.856e9, num_chan))
            assert_array_equal(S.getcol("CHAN_WIDTH", startrow=r, nrow=1)[0],
                               np.full(num_chan, .856e9 / num_chan))

        required_desc = pt.required_ms_desc("SPECTRAL_WINDOW")
        required_columns = set(k for k in required_desc.keys()
                               if not k.startswith("_"))

        assert set(S.colnames()) == set(required_columns)

    # We should get a cartesian product out
    with pt.table(ddid_table_name, ack=False) as D:
        spw_id, pol_id = zip(*product(range(len(num_chans)),
                                      range(len(corr_types))))
        assert_array_equal(pol_id, D.getcol("POLARIZATION_ID"))
        assert_array_equal(spw_id, D.getcol("SPECTRAL_WINDOW_ID"))

        required_desc = pt.required_ms_desc("DATA_DESCRIPTION")
        required_columns = set(k for k in required_desc.keys()
                               if not k.startswith("_"))

        assert set(D.colnames()) == set(required_columns)

    with pt.table(str(ms_path), ack=False) as T:
        # DATA_DESC_ID's are all the same shape
        assert_array_equal(T.getcol("DATA_DESC_ID"),
                           da.concatenate(all_data_desc_id))

        # DATA is variably shaped (on DATA_DESC_ID) so we
        # compared each one separately.
        for ddid, data in enumerate(all_data):
            ms_data = T.getcol("DATA", startrow=ddid*row, nrow=row)
            assert_array_equal(ms_data, data)

        required_desc = pt.required_ms_desc()
        required_columns = set(k for k in required_desc.keys()
                               if not k.startswith("_"))

        # Check we have the required columns
        assert set(T.colnames()) == required_columns.union(["DATA",
                                                            "DATA_DESC_ID"])


@pytest.mark.parametrize("chunks", [{
    "row": (10,),
    "chan": (4, 4),
    "corr": (2, 2),
}])
@pytest.mark.parametrize("Dataset", [
    Dataset,
    pytest.param(xrDataset, marks=xarray_param_marks)
], ids=["daskms.Dataset", "xarray.Dataset"])
def test_ms_create_and_update(Dataset, tmp_path, chunks):
    """ Test that we can update and append at the same time """
    filename = str(tmp_path / "create-and-update.ms")

    rs = np.random.RandomState(42)

    # Create a dataset of 10 rows with DATA and DATA_DESC_ID
    dims = ("row", "chan", "corr")
    row, chan, corr = tuple(sum(chunks[d]) for d in dims)
    ms_datasets = []
    np_data = (rs.normal(size=(row, chan, corr)) +
               1j*rs.normal(size=(row, chan, corr))).astype(np.complex64)

    data_chunks = tuple((chunks['row'], chan, corr))
    dask_data = da.from_array(np_data, chunks=data_chunks)
    # Create dask ddid column
    dask_ddid = da.full(row, 0, chunks=chunks['row'], dtype=np.int32)
    dataset = Dataset({
        'DATA': (dims, dask_data),
        'DATA_DESC_ID': (("row",), dask_ddid),
    })
    ms_datasets.append(dataset)

    # Write it
    writes = xds_to_table(ms_datasets, filename, ["DATA", "DATA_DESC_ID"])
    dask.compute(writes)

    ms_datasets = xds_from_ms(filename)

    # Now add another dataset (different DDID), with no ROWID
    np_data = (rs.normal(size=(row, chan, corr)) +
               1j*rs.normal(size=(row, chan, corr))).astype(np.complex64)
    data_chunks = tuple((chunks['row'], chan, corr))
    dask_data = da.from_array(np_data, chunks=data_chunks)
    # Create dask ddid column
    dask_ddid = da.full(row, 1, chunks=chunks['row'], dtype=np.int32)
    dataset = Dataset({
        'DATA': (dims, dask_data),
        'DATA_DESC_ID': (("row",), dask_ddid),
    })
    ms_datasets.append(dataset)

    # Write it
    writes = xds_to_table(ms_datasets, filename, ["DATA", "DATA_DESC_ID"])
    dask.compute(writes)

    # Rows have been added and additional data is present
    with pt.table(filename, ack=False, readonly=True) as T:
        first_data_desc_id = da.full(row, ms_datasets[0].DATA_DESC_ID,
                                     chunks=chunks['row'])
        ds_data = da.concatenate([ms_datasets[0].DATA.data,
                                  ms_datasets[1].DATA.data])
        ds_ddid = da.concatenate([first_data_desc_id,
                                  ms_datasets[1].DATA_DESC_ID.data])
        assert_array_equal(T.getcol("DATA"), ds_data)
        assert_array_equal(T.getcol("DATA_DESC_ID"), ds_ddid)


