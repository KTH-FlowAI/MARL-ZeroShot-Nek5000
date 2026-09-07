"""Wall-shear / velocity correlations from stitched TSRS records.

The data layout used throughout :mod:`postlib.tsrs` is ``(ny, nx, nz, nt,
nfld)``.  This module implements the long-time correlation used by Brown and
Thomas (1977), with the probes replaced by the full sampled wall plane::

    R_tauw,u(dx, dz, T; y) =
        < tau_w'(x, z, t) u'(x + dx, z + dz, y, t + T) >
        -------------------------------------------------
                  tau_w,rms u_rms(y)

Spatial offsets are periodic because the TSRS grid is periodic in ``x`` and
``z``.  Time is deliberately *not* periodic: every lag is formed from its
overlapping samples only.  That distinction matters for a finite record.

TSRS files store velocities, not an explicit wall-shear field.  For a
no-slip wall we therefore reconstruct the signed shear with a one-sided,
non-uniform polynomial derivative at the wall.  The wall point plus y+ = 2
and 5, for example, give a second-order derivative without pretending that
the sensing planes are uniformly spaced.
"""

from __future__ import annotations

import math

import numpy as np

from . import tsrs


def derivative_weights(nodes, x0=None, order=1):
    """Finite-difference weights for the derivative at ``x0``.

    ``nodes`` may be non-uniform.  The returned weights differentiate exactly
    every polynomial of degree ``len(nodes) - 1``.  Keeping this small helper
    public makes the wall reconstruction independently testable and records
    the actual stencil rather than hiding it behind ``np.gradient``.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 1 or nodes.size <= order:
        raise ValueError('need a one-dimensional stencil with more nodes than '
                         f'derivative order {order}')
    if not np.isfinite(nodes).all() or np.unique(nodes).size != nodes.size:
        raise ValueError(f'derivative nodes must be finite and unique: {nodes}')
    x0 = float(nodes[0] if x0 is None else x0)
    dx = nodes - x0
    A = np.vstack([dx ** k for k in range(nodes.size)])
    rhs = np.zeros(nodes.size)
    rhs[order] = math.factorial(order)
    try:
        return np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f'singular derivative stencil {nodes}') from exc


def wall_shear(data, viscosity, component='u', wall_y=0.0, npoints=3,
               atol=None, dtype=np.float64):
    """Reconstruct one signed tangential wall-shear component.

    Parameters
    ----------
    data
        Stitched TSRS dictionary.  It must include ``component`` and the wall
        plane plus ``npoints - 1`` samples on the positive-normal side.
    viscosity
        Kinematic viscosity ``nu``.  The returned ``tau`` is ``nu*dq/dy``.
    component
        Tangential velocity component, normally ``'u'``.  ``'w'`` is also
        supported; a wall-normal component is intentionally rejected.
    wall_y, npoints
        Wall position and number of planes in the one-sided polynomial stencil.

    Returns a dictionary containing ``tau`` and ``gradient`` with shape
    ``(nx, nz, nt)`` plus the exact stencil metadata.  ``tau`` is signed;
    do not replace it by a magnitude before forming a correlation.
    """
    if component not in ('u', 'w'):
        raise ValueError("wall shear needs a tangential component 'u' or 'w', "
                         f'got {component!r}')
    if component not in data['names']:
        raise ValueError(f"record has no {component!r} field")
    if int(npoints) != npoints or npoints < 2:
        raise ValueError(f'npoints must be an integer >= 2, got {npoints!r}')
    npoints = int(npoints)
    viscosity = float(viscosity)
    if not np.isfinite(viscosity) or viscosity <= 0:
        raise ValueError(f'viscosity must be finite and positive, got {viscosity}')

    y = np.asarray(data['y'], dtype=float)
    if y.ndim != 1:
        raise ValueError('data["y"] must be a one-dimensional plane coordinate')
    if atol is None:
        spacing = np.diff(np.unique(np.sort(y)))
        positive = spacing[spacing > 0]
        atol = max(1e-12, 1e-9 * float(positive.min()) if positive.size else 1e-12)

    hit = np.flatnonzero(np.abs(y - float(wall_y)) <= atol)
    if hit.size != 1:
        raise ValueError(f'need exactly one plane at wall y={wall_y:g} '
                         f'(atol={atol:.3g}); stored y={y.tolist()}')
    wall_index = int(hit[0])
    # This post-processing is for the lower wall: use a forward stencil in its
    # positive-normal direction and fail rather than silently use the other wall.
    side = np.flatnonzero(y >= y[wall_index] - atol)
    side = side[np.argsort(y[side])]
    if side.size < npoints:
        raise ValueError(f'need {npoints} planes on the positive-normal side of '
                         f'y={wall_y:g}; have {side.size} in {y.tolist()}')
    indices = side[:npoints]
    nodes = y[indices]
    weights = derivative_weights(nodes, x0=y[wall_index])

    sample = tsrs.field(data, component, iy=int(indices[0]))
    gradient = np.zeros(sample.shape, dtype=dtype)
    for weight, iy in zip(weights, indices):
        gradient += weight * tsrs.field(data, component, iy=int(iy))

    yplus = np.asarray(data.get('yplus', np.full_like(y, np.nan)), dtype=float)
    return {
        'tau': viscosity * gradient,
        'gradient': gradient,
        'component': component,
        'viscosity': viscosity,
        'wall_index': wall_index,
        'wall_y': float(y[wall_index]),
        'indices': np.asarray(indices, dtype=int),
        'y': nodes,
        'yplus': yplus[indices],
        'weights': weights,
        'method': 'one-sided polynomial derivative',
    }


def _check_time(t):
    """Return the uniform sampling interval, refusing an ambiguous lag axis."""
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError('time coordinate must have at least two samples')
    dt = np.diff(t)
    if not np.isfinite(dt).all() or np.any(dt <= 0):
        raise ValueError('time coordinate must be finite and strictly increasing')
    step = float(np.median(dt))
    if not np.allclose(dt, step, rtol=1e-6, atol=max(1e-12, abs(step) * 1e-9)):
        raise ValueError('time coordinate is not uniform; resample before a '
                         'time-lag correlation')
    return step


def _max_lag(nt, dt, max_lag=None, max_lag_time=None):
    """Resolve a requested lag range to an integer number of snapshots."""
    if max_lag is not None and max_lag_time is not None:
        raise ValueError('pass max_lag or max_lag_time, not both')
    if max_lag_time is not None:
        if float(max_lag_time) < 0:
            raise ValueError('max_lag_time must be non-negative')
        max_lag = int(np.floor(float(max_lag_time) / dt + 1e-12))
    if max_lag is None:
        max_lag = nt // 4
    if int(max_lag) != max_lag or max_lag < 0 or max_lag >= nt:
        raise ValueError(f'max_lag must be in [0, {nt - 1}], got {max_lag!r}')
    return int(max_lag)


def temporal_cross_correlation(a, b, t, max_lag=None, max_lag_time=None,
                               x_shift=0, z_shift=0, mean='time'):
    """Long-time, normalized cross-correlation of two wall-plane series.

    ``a`` and ``b`` have shape ``(nx, nz, nt)``.  A positive lag implements
    ``a(x,z,t) * b(x+dx,z+dz,t+T)``.  ``x_shift`` and ``z_shift`` are integer
    periodic grid shifts; positive therefore means the second signal is sampled
    downstream/spanwise-positive of the first.

    The FFT computes all *linear* temporal lags at once.  Zero padding is the
    important part: it avoids the false end-to-start pairs that a circular
    correlation would introduce into a finite TSRS record.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 3 or b.ndim != 3 or a.shape != b.shape:
        raise ValueError('a and b must have the same (nx, nz, nt) shape')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('time-series correlation input contains non-finite values')
    nx, nz, nt = a.shape
    if int(x_shift) != x_shift or int(z_shift) != z_shift:
        raise ValueError('x_shift and z_shift are integer grid offsets')
    x_shift, z_shift = int(x_shift), int(z_shift)
    dt = _check_time(t)
    nlag = _max_lag(nt, dt, max_lag=max_lag, max_lag_time=max_lag_time)

    if mean == 'time':
        a = a - a.mean(axis=-1, keepdims=True)
        b = b - b.mean(axis=-1, keepdims=True)
    elif mean == 'global':
        a = a - a.mean()
        b = b - b.mean()
    elif mean is not None:
        raise ValueError("mean must be 'time', 'global' or None")
    if x_shift or z_shift:
        # np.roll(out)[i] is input[i - roll]; negative roll yields b[i + shift].
        b = np.roll(b, shift=(-x_shift, -z_shift), axis=(0, 1))

    ar, br = float(np.sqrt(np.mean(a ** 2))), float(np.sqrt(np.mean(b ** 2)))
    if ar == 0.0 or br == 0.0:
        raise ValueError('one input has zero RMS after mean removal')

    # A length >= 2*nt-1 turns circular FFT correlation into linear correlation.
    nfft = 1 << (2 * nt - 1).bit_length()
    fa = np.fft.rfft(a, n=nfft, axis=-1)
    fb = np.fft.rfft(b, n=nfft, axis=-1)
    raw = np.fft.irfft(np.conj(fa) * fb, n=nfft, axis=-1).sum(axis=(0, 1))

    lags = np.arange(-nlag, nlag + 1, dtype=int)
    sums = np.empty(lags.size, dtype=float)
    for i, lag in enumerate(lags):
        # FFT output indices 0..nt-1 are T >= 0; the final bins are T < 0.
        sums[i] = raw[lag if lag >= 0 else nfft + lag]
    nt_pairs = nt - np.abs(lags)
    # Brown and Thomas's definition has 1/T_s, not 1/(T_s-|T|).  The finite
    # record counterpart therefore retains the full record length here while
    # summing only overlapping pairs.  It is bounded by one (unlike the common
    # ``unbiased`` 1/(nt-|lag|) estimator, whose short-lag samples can exceed
    # one after normalisation by full-record RMS values).
    rho = sums / (nx * nz * nt * ar * br)
    return {
        'lag': lags,
        'lag_time': lags.astype(float) * dt,
        'rho': rho,
        'sum': sums,
        'n_time_pairs': nt_pairs,
        'normalization': 'full record length',
        'rms_a': ar,
        'rms_b': br,
        'dt': dt,
        'x_shift': x_shift,
        'z_shift': z_shift,
        'mean': 'none' if mean is None else mean,
    }


def wall_velocity_correlation(data, viscosity, velocity='u', plane_indices=None,
                              tau_component='u', wall_y=0.0, nwall=3,
                              max_lag=None, max_lag_time=None,
                              x_shift=0, z_shift=0, sensitivity=True):
    """Brown-style wall-shear/velocity correlations for selected heights.

    The output is a row per velocity plane.  ``peak_lag`` is the lag of the
    largest *positive* correlation, matching Brown and Thomas's use of the
    correlation maximum to identify the delayed velocity response.
    """
    if velocity not in data['names']:
        raise ValueError(f"record has no {velocity!r} field")
    shear = wall_shear(data, viscosity, component=tau_component, wall_y=wall_y,
                        npoints=nwall)
    ny = data['fld'].shape[0]
    if plane_indices is None:
        plane_indices = [i for i in range(ny) if i != shear['wall_index']]
    indices = np.asarray(plane_indices, dtype=int)
    if indices.ndim != 1 or not indices.size or np.any(indices < 0) or np.any(indices >= ny):
        raise ValueError(f'plane_indices must be non-empty valid indices, got {indices}')

    curves = []
    for iy in indices:
        curves.append(temporal_cross_correlation(
            shear['tau'], tsrs.field(data, velocity, iy=int(iy)), data['t'],
            max_lag=max_lag, max_lag_time=max_lag_time,
            x_shift=x_shift, z_shift=z_shift, mean='time'))
    rho = np.asarray([c['rho'] for c in curves])
    peak_index = np.argmax(rho, axis=1)

    out = {
        'lag': curves[0]['lag'],
        'lag_time': curves[0]['lag_time'],
        'rho': rho,
        'n_time_pairs': curves[0]['n_time_pairs'],
        'rms_tau': curves[0]['rms_a'],
        'rms_velocity': np.asarray([c['rms_b'] for c in curves]),
        'peak_index': peak_index,
        'peak_lag': curves[0]['lag'][peak_index],
        'peak_lag_time': curves[0]['lag_time'][peak_index],
        'peak_rho': rho[np.arange(indices.size), peak_index],
        'indices': indices,
        'y': np.asarray(data['y'], dtype=float)[indices],
        'yplus_nominal': np.asarray(data['yplus'], dtype=float)[indices],
        'velocity': velocity,
        'tau_component': tau_component,
        'wall_shear': shear,
        'x_shift': int(x_shift),
        'z_shift': int(z_shift),
        'dt': curves[0]['dt'],
    }

    # A two-point result is not a second estimate to average in; it is a
    # transparent resolution diagnostic for the reconstructed derivative.
    if sensitivity and nwall > 2:
        low = wall_shear(data, viscosity, component=tau_component, wall_y=wall_y,
                         npoints=2)
        low_rho = np.asarray([
            temporal_cross_correlation(
                low['tau'], tsrs.field(data, velocity, iy=int(iy)), data['t'],
                max_lag=max_lag, max_lag_time=max_lag_time,
                x_shift=x_shift, z_shift=z_shift, mean='time')['rho']
            for iy in indices
        ])
        out['rho_low_order'] = low_rho
        out['low_order_nwall'] = 2
        out['low_order_weights'] = low['weights']
        out['low_order_y'] = low['y']
    return out
