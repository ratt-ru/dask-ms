import dask.array as da
from katpoint import Target
import numpy as np


def _uvw(target_description, time_utc, antennas, ant1, ant2, row):
    """Calculate UVW coordinates"""
    array_centre = antennas[0].array_reference_antenna()
    target = Target(target_description)
    uvw_ant = target.uvw(antennas, time_utc, array_centre)
    uvw_ant = np.transpose(uvw_ant, (1, 2, 0))
    # Compute baseline UVW coordinates from per-antenna coordinates.
    # The sign convention matches `CASA`_, rather than the
    # Measurement Set `definition`_.
    # .. _CASA: https://casa.nrao.edu/Memos/CoordConvention.pdf
    # .. _definition: https://casa.nrao.edu/Memos/229.html#SECTION00064000000000000000
    uvw_bl = np.take(uvw_ant, ant1, axis=1) - np.take(uvw_ant, ant2, axis=1)
    return uvw_bl.reshape(-1, 3) if row else uvw_bl


def uvw_coords(target, time_utc, antennas, cp_info, row=True):
    (ntime,) = time_utc.shape
    (nbl,) = cp_info.ant1_index.shape

    if row:
        out_dims = ("row", "uvw")
        new_axes = {"row": ntime * nbl, "uvw": 3}
    else:
        out_dims = ("time", "bl", "uvw")
        new_axes = {"uvw": 3}

    out = da.blockwise(
        _uvw,
        out_dims,
        target.description,
        None,
        time_utc,
        ("time",),
        antennas,
        ("ant",),
        cp_info.ant1_index,
        ("bl",),
        cp_info.ant2_index,
        ("bl",),
        row,
        None,
        concatenate=True,
        new_axes=new_axes,
        meta=np.empty((0,) * len(out_dims), np.float64),
    )

    if row:
        return out.rechunk({0: ntime * (nbl,)})

    return out
