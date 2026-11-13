# -*- coding: utf-8 -*-

from tempfile import mkdtemp

import numpy as np

from daskms.casa_table import create_ms, open_table


def _row(r, n=1):
    """An arcae index selecting ``n`` rows starting at ``r``"""
    return (slice(r, r + n),)


def _ms_factory_impl(ms_name):
    rs = np.random.RandomState(42)
    ant_name = "::".join((ms_name, "ANTENNA"))
    ddid_name = "::".join((ms_name, "DATA_DESCRIPTION"))
    field_name = "::".join((ms_name, "FIELD"))
    pol_name = "::".join((ms_name, "POLARIZATION"))
    spw_name = "::".join((ms_name, "SPECTRAL_WINDOW"))

    kw = {"readonly": False}

    desc = {
        "DATA": {
            "_c_order": True,
            "comment": "DATA column",
            "dataManagerGroup": "StandardStMan",
            "dataManagerType": "StandardStMan",
            "keywords": {},
            "maxlen": 0,
            "ndim": 2,
            "option": 0,
            # 'shape': ...,  # Variably shaped
            "valueType": "COMPLEX",
        }
    }

    na = 64
    corr_types = [[9, 10, 11, 12], [9, 12]]
    spw_chans = [16, 32]
    ddids = ([0, 0, 4], [1, 1, 6])

    # NOTE(sjperkins)
    # arcae confines each table to its own thread and gives each a
    # thread-local casacore table cache, so the Measurement Set must be
    # closed before its subtables can be opened by path.
    create_ms(ms_name, table_desc=desc).close()

    # Populate ANTENNA table
    with open_table(ant_name, **kw) as A:
        A.addrows(na)
        A.putcol("POSITION", rs.random_sample((na, 3)) * 10000, index=_row(0, na))
        A.putcol("OFFSET", rs.random_sample((na, 3)), index=_row(0, na))
        A.putcol(
            "NAME",
            np.array(["ANT-%d" % i for i in range(na)]),
            index=_row(0, na),
        )

    # Populate POLARIZATION table
    with open_table(pol_name, **kw) as P:
        for r, corr_type in enumerate(corr_types):
            P.addrows(1)
            P.putcol("NUM_CORR", np.array(len(corr_type))[None], index=_row(r))
            P.putcol("CORR_TYPE", np.array(corr_type)[None, :], index=_row(r))

    # Populate SPECTRAL_WINDOW table
    with open_table(spw_name, **kw) as SPW:
        freq_start = 0.856e9
        freq_end = 2 * 0.856e9

        for r, nchan in enumerate(spw_chans):
            chan_width = (freq_end - freq_start) / nchan
            chan_width = np.full(nchan, chan_width)
            chan_freq = np.linspace(freq_start, freq_end, nchan)
            ref_freq = chan_freq[chan_freq.size // 2]

            SPW.addrows(1)
            SPW.putcol("NUM_CHAN", np.array(nchan)[None], index=_row(r))
            SPW.putcol("CHAN_WIDTH", chan_width[None, :], index=_row(r))
            SPW.putcol("CHAN_FREQ", chan_freq[None, :], index=_row(r))
            SPW.putcol("REF_FREQUENCY", np.array(ref_freq)[None], index=_row(r))

    # Populate FIELD table
    with open_table(field_name, **kw) as F:
        fields = (["3C147", np.deg2rad([0, 60])], ["3C147", np.deg2rad([30, 45])])

        npoly = 1

        for r, (name, phase_dir) in enumerate(fields):
            F.addrows(1)
            F.putcol("NAME", np.array([name]), index=_row(r))
            F.putcol("NUM_POLY", np.array(npoly)[None], index=_row(r))

            # Set all these to the phase centre
            for c in ["PHASE_DIR", "REFERENCE_DIR", "DELAY_DIR"]:
                F.putcol(c, phase_dir[None, None, :], index=_row(r))

    # Populate DATA_DESCRIPTION table
    with open_table(ddid_name, **kw) as D:
        for r, (spw_id, pol_id, _) in enumerate(ddids):
            D.addrows(1)
            D.putcol("SPECTRAL_WINDOW_ID", np.array(spw_id)[None], index=_row(r))
            D.putcol("POLARIZATION_ID", np.array(pol_id)[None], index=_row(r))

    # Add some data to the main table
    with open_table(ms_name, **kw) as ms:
        startrow = 0

        for ddid, (spw_id, pol_id, rows) in enumerate(ddids):
            ms.addrows(rows)
            ms.putcol("DATA_DESC_ID", np.full(rows, ddid), index=_row(startrow, rows))

            nchan = spw_chans[spw_id]
            ncorr = len(corr_types[pol_id])

            vis = (
                np.random.random((rows, nchan, ncorr))
                + np.random.random((rows, nchan, ncorr)) * 1j
            )

            ms.putcol("DATA", vis, index=_row(startrow, rows))

            startrow += rows


def example_ms():
    ms_filename = mkdtemp(".ms")
    _ms_factory_impl(ms_filename)
    return ms_filename
