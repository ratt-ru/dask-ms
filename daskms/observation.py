# -*- coding: utf-8 -*-

import numpy as np
from pyrap.measures import measures
from pyrap.quanta import quantity as q

VLA_ANTENNA_POSITIONS = np.asarray([
         [-1.60171e+06, -5.04201e+06, 3.5546e+06],
         [-1.60115e+06, -5.042e+06, 3.55486e+06],
         [-1.60072e+06, -5.04227e+06, 3.55467e+06],
         [-1.60119e+06, -5.042e+06, 3.55484e+06],
         [-1.60161e+06, -5.042e+06, 3.55465e+06],
         [-1.60116e+06, -5.04183e+06, 3.5551e+06],
         [-1.60101e+06, -5.04209e+06, 3.5548e+06],
         [-1.60119e+06, -5.04198e+06, 3.55488e+06],
         [-1.60095e+06, -5.04213e+06, 3.55477e+06],
         [-1.60118e+06, -5.04193e+06, 3.55495e+06],
         [-1.60107e+06, -5.04205e+06, 3.55482e+06],
         [-1.6008e+06, -5.04222e+06, 3.55471e+06],
         [-1.60116e+06, -5.04178e+06, 3.55516e+06],
         [-1.60145e+06, -5.04199e+06, 3.55474e+06],
         [-1.60123e+06, -5.04198e+06, 3.55486e+06],
         [-1.60153e+06, -5.042e+06, 3.5547e+06],
         [-1.60114e+06, -5.04168e+06, 3.55532e+06],
         [-1.60132e+06, -5.04199e+06, 3.55481e+06],
         [-1.60117e+06, -5.04187e+06, 3.55504e+06],
         [-1.60119e+06, -5.04202e+06, 3.55481e+06],
         [-1.60117e+06, -5.0419e+06, 3.55499e+06],
         [-1.60088e+06, -5.04217e+06, 3.55474e+06],
         [-1.60138e+06, -5.04199e+06, 3.55478e+06],
         [-1.60118e+06, -5.04195e+06, 3.55492e+06],
         [-1.60127e+06, -5.04198e+06, 3.55483e+06],
         [-1.60111e+06, -5.04202e+06, 3.55484e+06],
         [-1.60115e+06, -5.04173e+06, 3.55524e+06]])


def _julian_day(year, month, day):
    """
    Given a Anno Dominei date, computes the Julian Date in days.

    Parameters
    ----------
    year : int
    month : int
    day : float

    Returns
    -------
    float
        Julian Date
    """

    # Formula below from
    # http://scienceworld.wolfram.com/astronomy/JulianDate.html
    # Also agrees with https://gist.github.com/jiffyclub/1294443
    return (367*year - int(7*(year + int((month+9)/12))/4)
            - int((3*(int(year + (month - 9)/7)/100)+1)/4)
            + int(275*month/9) + day + 1721028.5)


def _modified_julian_date(year, month, day):
    """
    Given a Anno Dominei date, computes the Modified Julian Date in days.

    Parameters
    ----------
    year : int
    month : int
    day : int

    Returns
    -------
    float
        Modified Julian Date
    """

    return _julian_day(year, month, day) - 2400000.5


def time(year, month, day, hours, minutes, seconds, intervals):
    """
    Parameters
    ----------

    .. math::

        t_i = mjd +  \frac{interval_i}{2} + \frac{interval_{i + 1}}{2}}

    year : int
    month : int
    day : int
    hours : int
    minutes : int
    seconds : int
    interval : :class:`numpy.ndarray`
        A sequence of intervals **around** the time

    Returns
    -------
    time : :class:`numpy.ndarray`
        Array of time values
    """
    day += (hours / 24.) + (minutes / 1440.) + (seconds / 86400.)
    start = _modified_julian_date(year, month, day) * 86400.
    half = intervals / 2

    half[1:] += half[:-1]
    half[0] = 0

    return start + np.cumsum(half)


def synthesize_uvw(antenna_positions, time, phase_dir,
                   auto_correlations=True):
    """
    Synthesizes new UVW coordinates based on time according to
    NRAO CASA convention (same as in fixvis)
    User should check these UVW coordinates carefully:
    if time centroid was used to compute
    original uvw coordinates the centroids
    of these new coordinates may be wrong, depending on whether
    data timesteps were heavily flagged.
    """

    dm = measures()
    epoch = dm.epoch("UT1", q(time[0], "s"))
    ref_dir = dm.direction("j2000",
                           q(phase_dir[0], "rad"),
                           q(phase_dir[1], "rad"))
    ox, oy, oz = antenna_positions[0]
    obs = dm.position("ITRF", q(ox, "m"), q(oy, "m"), q(oz, "m"))

    # Setup local horizon coordinate frame with antenna 0 as reference position
    dm.do_frame(obs)
    dm.do_frame(ref_dir)
    dm.do_frame(epoch)

    ant1, ant2 = np.triu_indices(antenna_positions.shape[0],
                                 0 if auto_correlations else 1)

    ntime = time.shape[0]
    nbl = ant1.shape[0]
    rows = ntime * nbl
    uvw = np.empty((rows, 3), dtype=np.float64)

    # For each timestep
    for ti, t in enumerate(time):
        epoch = dm.epoch("UT1", q(t, "s"))
        dm.do_frame(epoch)

        ant_uvw = np.zeros_like(antenna_positions)

        # Calculate antenna UVW positions
        for ai, (x, y, z) in enumerate(antenna_positions):
            bl = dm.baseline("ITRF",
                             q([x, ox], "m"),
                             q([y, oy], "m"),
                             q([z, oz], "m"))

            ant_uvw[ai] = dm.to_uvw(bl)["xyz"].get_value()[0:3]

        # Now calculate baseline UVW positions
        # noting that ant1 - ant2 is the CASA convention
        base = ti*nbl
        uvw[base:base + nbl, :] = ant_uvw[ant1] - ant_uvw[ant2]

    return uvw
