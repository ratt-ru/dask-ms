# -*- coding: utf-8 -*-

import numpy as np


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
    start = _modified_julian_date(year, month, day)
    half = intervals / 2

    half[1:] += half[:-1]
    half[0] = 0

    return start + np.cumsum(half)
