# -*- coding: utf-8 -*-

import astropy.units as u
import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from daskms.observation import time as sim_time


@pytest.mark.parametrize("date", [(2020, 1, 1)])
def test_time(date):
    year, month, day = date
    intervals = np.array([14, 15, 16, 17])
    res = sim_time(year, month, day, 0, 0, 0, intervals)
    start = 58849.0 * 86400.

    expected = start + np.array([0, 14.5, 30, 46.5])
    assert_almost_equal(res, expected)


def xyz_to_uvw_rot_mat(hour_angle, declination):
    har = hour_angle.to(u.rad).value  # HA-radians
    decr = declination.to(u.rad).value  # Dec-radians

    ntime = hour_angle.shape[0]

    sin_har = np.sin(har)
    cos_har = np.cos(har)
    sin_decr = np.full(ntime, np.sin(decr))
    cos_decr = np.full(ntime, np.cos(decr))
    time_zeros = np.zeros(sin_har.shape, dtype=har.dtype)

    rotation = np.array(
        [[sin_har, cos_har, time_zeros],
         [-sin_decr * cos_har, sin_decr * sin_har, cos_decr],
         [cos_decr * cos_har, -cos_decr * sin_har, sin_decr]])

    return rotation


def z_rot_mat(rotation_angle):
    ar = rotation_angle.to(u.rad).value  # Angle in radians

    rotation = np.array(
        [[np.cos(ar), np.sin(ar), 0],
         [-np.sin(ar), np.cos(ar), 0],
         [0, 0, 1], ],
        dtype=np.float_)

    return rotation


@pytest.mark.parametrize("auto_corrs", [True])
def test_synthesize_uvw(wsrt_antenna_positions, auto_corrs):
    from daskms.observation import synthesize_uvw
    intervals = np.full(4, 15.0, dtype=np.float64)
    t = sim_time(2020, 1, 1, 0, 0, 0, intervals)
    phase_dir = np.deg2rad([30, 60])
    uvw = synthesize_uvw(wsrt_antenna_positions, t, phase_dir,
                         auto_correlations=auto_corrs)

    # print(uvw.shape)
    # print(wsrt_antenna_positions)

    from astropy.time import Time
    from astropy.coordinates import EarthLocation
    import astropy.units as u

    # mean_posn = np.mean(wsrt_antenna_positions, axis=0)
    mean_posn = wsrt_antenna_positions[0]
    centre = EarthLocation.from_geocentric(mean_posn[0],
                                           mean_posn[1],
                                           mean_posn[2],
                                           unit=u.m)

    lon, lat, height = centre.to_geodetic()

    mean_subbed_itrf = wsrt_antenna_positions - mean_posn
    obs_times = Time(t / 86400.00, format='mjd', scale='utc')

    rotation = z_rot_mat(lon)
    ant_local_xyz = np.dot(rotation, mean_subbed_itrf.T).T

    ant1, ant2 = np.triu_indices(wsrt_antenna_positions.shape[0],
                                 0 if auto_corrs else 1)

    baseline_local_xyz = ant_local_xyz[ant1] - ant_local_xyz[ant2]
    baseline_local_xyz *= u.m

    ntime = len(obs_times)
    nbl = len(baseline_local_xyz)
    uvw_array = np.zeros((ntime*nbl, 3), dtype=np.float_)
    uvw_array *= baseline_local_xyz.unit

    from astropy.coordinates import SkyCoord
    phase_dir = SkyCoord(ra=phase_dir[0], dec=phase_dir[1], unit=u.rad)

    lha = obs_times.sidereal_time('apparent', longitude=lon) - phase_dir.ra

    rotation = xyz_to_uvw_rot_mat(lha, phase_dir.dec)
    res = np.einsum("ijt,bj->tbi", rotation, baseline_local_xyz)
    res = res.reshape(-1, 3)

    uvw *= u.m
    v = res - uvw
    print(v)

    rmsd = np.sqrt(np.sum((res - uvw)**2, axis=0)/res.shape[0])
    print("RMSD", rmsd)
    print(uvw)
    print(res)
    print("max", (res - uvw).max(axis=0))
    assert_almost_equal(res, uvw)
