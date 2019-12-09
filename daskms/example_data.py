# -*- coding: utf-8 -*-

from tempfile import mkdtemp

import numpy as np
import pyrap.tables as pt


def _ms_factory_impl(ms_name):
    rs = np.random.RandomState(42)
    ant_name = "::".join((ms_name, "ANTENNA"))
    ddid_name = "::".join((ms_name, "DATA_DESCRIPTION"))
    field_name = "::".join((ms_name, "FIELD"))
    pol_name = "::".join((ms_name, "POLARIZATION"))
    spw_name = "::".join((ms_name, "SPECTRAL_WINDOW"))

    kw = {'ack': False, 'readonly': False}

    desc = {'DATA': {'_c_order': True,
                     'comment': 'DATA column',
                     'dataManagerGroup': 'StandardStMan',
                     'dataManagerType': 'StandardStMan',
                     'keywords': {},
                     'maxlen': 0,
                     'ndim': 2,
                     'option': 0,
                     # 'shape': ...,  # Variably shaped
                     'valueType': 'COMPLEX'}}

    na = 64
    corr_types = [[9, 10, 11, 12], [9, 12]]
    spw_chans = [16, 32]
    ddids = ([0, 0, 4], [1, 1, 6])

    with pt.default_ms(ms_name, desc) as ms:
        # Populate ANTENNA table
        with pt.table(ant_name, **kw) as A:
            A.addrows(na)
            A.putcol("POSITION", rs.random_sample((na, 3))*10000)
            A.putcol("OFFSET", rs.random_sample((na, 3)))
            A.putcol("NAME", ["ANT-%d" % i for i in range(na)])

        # Populate POLARIZATION table
        with pt.table(pol_name, **kw) as P:
            for r, corr_type in enumerate(corr_types):
                P.addrows(1)
                P.putcol("NUM_CORR", np.array(len(corr_type))[None],
                         startrow=r, nrow=1)
                P.putcol("CORR_TYPE", np.array(corr_type)[None, :],
                         startrow=r, nrow=1)

        # Populate SPECTRAL_WINDOW table
        with pt.table(spw_name, **kw) as SPW:
            freq_start = .856e9
            freq_end = 2*.856e9

            for r, nchan in enumerate(spw_chans):
                chan_width = (freq_end - freq_start) / nchan
                chan_width = np.full(nchan, chan_width)
                chan_freq = np.linspace(freq_start, freq_end, nchan)
                ref_freq = chan_freq[chan_freq.size // 2]

                SPW.addrows(1)
                SPW.putcol("NUM_CHAN", np.array(nchan)[None],
                           startrow=r, nrow=1)
                SPW.putcol("CHAN_WIDTH", chan_width[None, :],
                           startrow=r, nrow=1)
                SPW.putcol("CHAN_FREQ", chan_freq[None, :],
                           startrow=r, nrow=1)
                SPW.putcol("REF_FREQUENCY", np.array(ref_freq)[None],
                           startrow=r, nrow=1)

        # Populate FIELD table
        with pt.table(field_name, **kw) as F:
            fields = (['3C147', np.deg2rad([0, 60])],
                      ['3C147', np.deg2rad([30, 45])])

            npoly = 1

            for r, (name, phase_dir) in enumerate(fields):
                F.addrows(1)
                F.putcol("NAME", [name],
                         startrow=r, nrow=1)
                F.putcol("NUM_POLY", np.array(npoly)[None],
                         startrow=r, nrow=1)

                # Set all these to the phase centre
                for c in ["PHASE_DIR", "REFERENCE_DIR", "DELAY_DIR"]:
                    F.putcol(c, phase_dir[None, None, :],
                             startrow=r, nrow=1)

        # Populate DATA_DESCRIPTION table
        with pt.table(ddid_name, **kw) as D:
            for r, (spw_id, pol_id, _) in enumerate(ddids):
                D.addrows(1)
                D.putcol("SPECTRAL_WINDOW_ID", np.array(spw_id)[None],
                         startrow=r, nrow=1)

                D.putcol("POLARIZATION_ID", np.array(pol_id)[None],
                         startrow=r, nrow=1)

        startrow = 0

        # Add some data to the main table
        for ddid, (spw_id, pol_id, rows) in enumerate(ddids):
            ms.addrows(rows)
            ms.putcol("DATA_DESC_ID", np.full(rows, ddid),
                      startrow=startrow, nrow=rows)

            nchan = spw_chans[spw_id]
            ncorr = len(corr_types[pol_id])

            vis = (np.random.random((rows, nchan, ncorr)) +
                   np.random.random((rows, nchan, ncorr))*1j)

            ms.putcol("DATA", vis, startrow=startrow, nrow=rows)

            startrow += rows


def example_ms():
    ms_filename = mkdtemp('.ms')
    _ms_factory_impl(ms_filename)
    return ms_filename
