"""Focused checks for Brown-style TSRS wall-shear/velocity correlations.

Run directly with ``python test_tsrs_timeseries.py`` or with pytest when it is
available.  The data are synthetic so the derivative and lag conventions have
known exact answers.
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from postlib import tsrs_timeseries as tw  # noqa: E402


def test_nonuniform_wall_derivative():
    """The one-sided stencil differentiates a quadratic exactly at the wall."""
    y = np.array([0.0, 0.2, 0.65])
    nx, nz, nt = 2, 3, 11
    slope = np.arange(nx * nz * nt, dtype=float).reshape(nx, nz, nt) / 10.0
    curvature = -0.7
    fld = np.zeros((y.size, nx, nz, nt, 3))
    for iy, yy in enumerate(y):
        fld[iy, ..., 0] = slope * yy + curvature * yy ** 2
    data = {'fld': fld, 'names': np.array(['u', 'v', 'w']),
            'y': y, 'yplus': y * 100, 't': np.arange(nt, dtype=float)}

    got = tw.wall_shear(data, viscosity=0.25, component='u', npoints=3)
    assert np.allclose(got['gradient'], slope)
    assert np.allclose(got['tau'], 0.25 * slope)
    assert np.allclose(got['weights'] @ np.array([0.0, 0.2, 0.65]), 1.0)


def test_temporal_correlation_lag_and_no_wrap():
    """A known positive delay peaks at that delay and never wraps at the end."""
    rng = np.random.default_rng(123)
    a = rng.standard_normal((3, 4, 128))
    delay = 9
    b = np.zeros_like(a)
    b[..., delay:] = a[..., :-delay]

    corr = tw.temporal_cross_correlation(
        a, b, t=np.arange(a.shape[-1]) * 0.2, max_lag=20)
    assert corr['lag'][np.argmax(corr['rho'])] == delay
    assert corr['rho'].max() <= 1.0 + 1e-12
    assert corr['n_time_pairs'][0] == a.shape[-1] - 20
    assert corr['n_time_pairs'][-1] == a.shape[-1] - 20

    auto = tw.temporal_cross_correlation(a, a, t=np.arange(a.shape[-1]), max_lag=0)
    assert np.allclose(auto['rho'], [1.0])


if __name__ == '__main__':
    test_nonuniform_wall_derivative()
    test_temporal_correlation_lag_and_no_wrap()
    print('tsrs_timeseries: all tests passed')
