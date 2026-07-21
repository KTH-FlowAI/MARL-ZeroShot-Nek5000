"""
Spectra and two-point correlations from stitched tsrs data (see tsrs.py).

The channel is homogeneous and periodic in x and z, so wavenumber spectra are
exact DFTs -- no windowing, no detrending in those directions.  Correlations are
obtained from the same spectra by Wiener-Khinchin, which makes them consistent
with the spectra by construction and costs one inverse transform.

Two conventions are fixed throughout and asserted at runtime:

    sum_k E(k)      == <f'^2>        (E is energy per discrete wavenumber)
    R(0)            == <f'^2>        (R is the unnormalised autocovariance)

so `Phi = E / dk` is the spectral density with `integral Phi dk == <f'^2>`, and
the premultiplied spectrum is `k*Phi`, whose area against `d(ln k)` is the
variance -- the usual presentation in wall turbulence.

A note on the sampling.  writer_int_pos lays points out with
`np.linspace(0, L, N)`, i.e. **including both endpoints**, so x=0 and x=Lx are
the same physical point in a periodic channel and appear twice.  Feeding all N
points to an FFT declares the period to be N*dx = L*N/(N-1), which is wrong by
~5% at N=20.  `periodic_view()` drops the duplicate; everything here expects to
be given that view.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import tsrs

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------- preparation

def check_duplicate_endpoints(data, name='u', iy=0, atol=0.0):
    """
    Verify that x=0/x=Lx and z=0/z=Lz really are duplicated samples.

    Returns (dup_x, dup_z).  They should be bit-identical: the solver
    interpolates the same physical location twice.
    """
    f = tsrs.field(data, name, iy=iy)
    dup_x = bool(np.all(np.abs(f[0] - f[-1]) <= atol))
    dup_z = bool(np.all(np.abs(f[:, 0] - f[:, -1]) <= atol))
    return dup_x, dup_z


def periodic_view(data, drop=True):
    """
    Return a copy of `data` sampled on a genuinely periodic grid.

    Drops the duplicated last point in x and z, leaving N-1 samples spanning
    exactly one period at spacing L/(N-1).
    """
    if not drop:
        return dict(data)
    out = dict(data)
    out['fld'] = data['fld'][:, :-1, :-1]
    out['x'] = data['x'][:-1]
    out['z'] = data['z'][:-1]
    return out


def domain(data):
    """
    Periodic lengths (Lx, Lz) implied by a periodic_view.

    N samples at spacing dx span a period of N*dx, which is *not* x[-1].
    """
    x, z = data['x'], data['z']
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    return x.size * dx, z.size * dz


def fluctuation(data, name, iy, mean='instant', dtype=np.float64):
    """
    Fluctuation field of one variable on one plane, as (nx, nz, nt).

    mean='instant'  subtract <f>_xz(t) at each instant.  Immune to a drifting
                    bulk state, and the right choice for *spatial* statistics:
                    it is the usual definition of a spatial fluctuation and it
                    forces E(k=0) = 0 exactly.
    mean='global'   subtract the single <f>_xzt.  Required for *frequency*
                    spectra, which would otherwise have their low-frequency
                    content removed along with the drift.
    """
    f = tsrs.field(data, name, iy=iy).astype(dtype, copy=True)
    if mean == 'instant':
        f -= f.mean(axis=(0, 1), keepdims=True)
    elif mean == 'global':
        f -= f.mean()
    elif mean is not None:
        raise ValueError(f"mean must be 'instant', 'global' or None, got {mean!r}")
    return f


def trim_time(data, t0=None, t1=None):
    """Restrict the record to t0 <= t <= t1 (either may be None)."""
    t = data['t']
    keep = np.ones(t.size, dtype=bool)
    if t0 is not None:
        keep &= t >= t0
    if t1 is not None:
        keep &= t <= t1
    if keep.sum() < 2:
        raise ValueError(f'time window [{t0}, {t1}] keeps {keep.sum()} snapshots')
    out = dict(data)
    out['fld'] = data['fld'][:, :, :, keep, :]
    out['t'] = t[keep]
    return out


# --------------------------------------------------------------- diagnostics

def stationarity(data, name='u', iy=0):
    """
    How much the plane-averaged value drifts over the record.

    Spectra and correlations assume a statistically stationary record.  This
    reports the drift so a non-stationary segment is never used silently.
    """
    f = tsrs.field(data, name, iy=iy).astype(np.float64)
    m = f.mean(axis=(0, 1))
    n = max(1, m.size // 10)
    first, last = m[:n].mean(), m[-n:].mean()
    drift = last - first

    # Relative to the mean is the natural reading for u, but useless for a
    # zero-mean quantity such as v, where it explodes.  Scaling the drift by
    # the fluctuation level is meaningful for both, so that is what decides
    # whether the record counts as stationary.
    rms = float(np.sqrt(((f - f.mean()) ** 2).mean()))
    return {
        't': data['t'], 'mean': m, 'rms': rms,
        'first_decile': first, 'last_decile': last,
        'drift': drift,
        'drift_pct': 100.0 * drift / abs(first) if first else np.nan,
        'drift_rel_rms': drift / rms if rms > 0 else np.nan,
    }


def estimate_utau(data, utau_ref, name='u'):
    """
    Estimate the actual friction velocity from the viscous sublayer.

    In the sublayer u = u_tau^2 * y / nu, so u_tau = sqrt(u * nu / y) at each
    plane with y+ <~ 5.  nu is fixed by the *reference* scaling stored with the
    data, nu = utau_ref * h / Re_tau (h = 1).

    Under control the true u_tau shifts, so the stored y+ labels are nominal
    (fixed physical planes in reference wall units).  Returns None when no
    plane lies deep enough in the sublayer to support the estimate.
    """
    retau = float(np.asarray(data['retau']))
    nu = utau_ref / retau
    yp = np.asarray(data['yplus'], dtype=float)
    y = np.asarray(data['y'], dtype=float)

    est = []
    for i in np.flatnonzero(yp <= 5.0):
        if y[i] <= 0:
            continue
        um = float(tsrs.field(data, name, iy=int(i)).astype(np.float64).mean())
        est.append(np.sqrt(um * nu / y[i]))
    if not est:
        return None
    utau = float(np.mean(est))
    return {
        'utau': utau, 'utau_ref': utau_ref, 'nu': nu,
        'retau_actual': utau / nu,
        'per_plane': est,
        'drag_reduction_pct': 100.0 * (1.0 - (utau / utau_ref) ** 2),
    }


# ------------------------------------------------------------------- spectra

def _rfft_norm(f, axis):
    """Normalised half-spectrum: F_k = (1/N) sum_j f_j exp(-2 pi i k j / N)."""
    n = f.shape[axis]
    return np.fft.rfft(f, axis=axis) / n, n


def _fold(power, n, axis=0):
    """
    Fold a one-sided rfft power array so it sums to the full-spectrum total.

    Every wavenumber except k=0 (and the Nyquist mode when n is even) stands
    for a conjugate pair and therefore counts twice.
    """
    out = power.copy()
    idx = [slice(None)] * out.ndim
    idx[axis] = slice(1, None)
    out[tuple(idx)] *= 2.0
    if n % 2 == 0:
        idx[axis] = -1
        out[tuple(idx)] /= 2.0
    return out


def spectrum_1d(f, axis, L, check=True):
    """
    1-D wavenumber spectrum of a fluctuation field, averaged over the other axes.

    f    : (nx, nz, nt) fluctuation field
    axis : 0 for streamwise (x), 1 for spanwise (z)
    L    : the *period* in that direction (see domain())

    Returns dict with k, lam (wavelength), E (sums to the variance), Phi
    (density, integrates to the variance) and kPhi (premultiplied).
    """
    F, n = _rfft_norm(f, axis)
    power = np.abs(F) ** 2
    other = tuple(a for a in range(f.ndim) if a != axis)
    # averaging over the other axes leaves the transformed axis at position 0
    E = _fold(power.mean(axis=other), n)

    dk = TWO_PI / L
    k = dk * np.arange(E.size)
    with np.errstate(divide='ignore'):
        lam = np.where(k > 0, TWO_PI / np.where(k > 0, k, 1), np.inf)

    if check:
        var = float((f ** 2).mean())
        _assert_parseval(E.sum(), var, 'spectrum_1d')

    return {'k': k, 'lam': lam, 'E': E, 'Phi': E / dk, 'kPhi': k * E / dk,
            'dk': dk, 'L': L, 'n': n}


def cospectrum_1d(f, g, axis, L, check=True):
    """
    Co-spectrum of two fields, e.g. the uv co-spectrum whose sum is <u'v'>.

    Shows which scales carry the Reynolds shear stress -- the direct diagnostic
    for what a drag-reduction controller is suppressing.
    """
    F, n = _rfft_norm(f, axis)
    G, _ = _rfft_norm(g, axis)
    other = tuple(a for a in range(f.ndim) if a != axis)
    co = _fold(np.real(np.conj(F) * G).mean(axis=other), n)

    dk = TWO_PI / L
    k = dk * np.arange(co.size)
    with np.errstate(divide='ignore'):
        lam = np.where(k > 0, TWO_PI / np.where(k > 0, k, 1), np.inf)

    if check:
        _assert_parseval(co.sum(), float((f * g).mean()), 'cospectrum_1d')

    return {'k': k, 'lam': lam, 'E': co, 'Phi': co / dk, 'kPhi': k * co / dk,
            'dk': dk, 'L': L, 'n': n}


def spectrum_2d(f, Lx, Lz, check=True):
    """
    2-D spectrum E(kx, kz), averaged over time.  Sums to the variance.

    kx is the full (signed) axis; kz is one-sided and folded.
    """
    nx, nz = f.shape[0], f.shape[1]
    F = np.fft.rfft2(f, axes=(0, 1)) / (nx * nz)
    E = _fold(np.abs(F) ** 2, nz, axis=1).mean(axis=2)

    kx = TWO_PI * np.fft.fftfreq(nx, d=Lx / nx)
    kz = TWO_PI * np.arange(E.shape[1]) / Lz
    if check:
        _assert_parseval(E.sum(), float((f ** 2).mean()), 'spectrum_2d')
    return {'kx': np.fft.fftshift(kx), 'kz': kz,
            'E': np.fft.fftshift(E, axes=0)}


# -------------------------------------------------------------- correlations

def correlation_1d(f, axis, L, check=True):
    """
    Two-point correlation along a homogeneous direction, by Wiener-Khinchin.

    The inverse transform of the *unfolded* power spectrum is the
    autocovariance, so this is exactly consistent with spectrum_1d() and
    exactly periodic.  Returns separations over half the period (the
    correlation is symmetric) and the normalised rho = R/R(0).
    """
    F, n = _rfft_norm(f, axis)
    other = tuple(a for a in range(f.ndim) if a != axis)
    power = (np.abs(F) ** 2).mean(axis=other)

    R = np.fft.irfft(power, n=n) * n           # R[0] == <f'^2>
    if check:
        _assert_parseval(R[0], float((f ** 2).mean()), 'correlation_1d')

    half = n // 2 + 1
    sep = np.arange(half) * (L / n)
    rho = R[:half] / R[0] if R[0] > 0 else np.full(half, np.nan)
    return {'sep': sep, 'R': R[:half], 'rho': rho, 'var': R[0], 'L': L}


def correlation_y(data, name, mean='instant', dtype=np.float64):
    """
    Correlation coefficient between every pair of stored wall-normal planes.

    Coarse with only a handful of planes, but it shows how far the structures
    the controller acts on reach away from the wall.
    """
    ny = data['fld'].shape[0]
    flat = [fluctuation(data, name, iy, mean=mean, dtype=dtype).ravel()
            for iy in range(ny)]
    rms = np.array([np.sqrt((f ** 2).mean()) for f in flat])

    rho = np.eye(ny)
    for i in range(ny):
        for j in range(i + 1, ny):
            denom = rms[i] * rms[j]
            rho[i, j] = rho[j, i] = ((flat[i] * flat[j]).mean() / denom
                                     if denom > 0 else np.nan)
    return {'rho': rho, 'rms': rms, 'yplus': np.asarray(data['yplus'])}


# ----------------------------------------------------------------- frequency

def frequency_spectrum(f, dt, nperseg=None, overlap=0.5, detrend='constant',
                       check=True):
    """
    Temporal PSD, averaged over the homogeneous directions (Welch).

    Time is *not* periodic, so unlike the spatial directions this needs
    windowing and segment averaging.  Pass a field built with mean='global':
    with mean='instant' the low frequencies have already been removed.

    `var_ratio` in the result is integral(psd) df / <f'^2>.  It is normally
    below 1, and that is a property of the estimator rather than an error:
    Welch cannot represent periods longer than one segment (nperseg*dt), and
    with detrend='constant' it also discards each segment's mean.  Turbulence
    carries real energy there, so short segments lose it.  The trade-off is
    the usual one -- longer segments capture more low-frequency energy, fewer
    segments make a noisier estimate.  On a 75-time-unit channel record the
    measured ratios were 0.49 at nperseg=256 rising to 0.88 at nperseg=2500,
    and 1.00 with detrend=False.
    """
    from scipy import signal

    nt = f.shape[-1]
    if nperseg is None:
        # ~4 segments (7 with 50% overlap): enough averaging to be smooth
        # without throwing away most of the low-frequency content
        nperseg = max(64, nt // 4)
    nperseg = int(min(nperseg, nt))

    freq, psd = signal.welch(f, fs=1.0 / dt, axis=-1, window='hann',
                             nperseg=nperseg, noverlap=int(overlap * nperseg),
                             detrend=detrend, return_onesided=True,
                             scaling='density')
    psd = psd.mean(axis=(0, 1))

    area = float(np.trapz(psd, freq))
    var = float((f ** 2).mean())
    ratio = area / var if var > 0 else np.nan

    # Unlike the spatial transforms this is an *estimator*, so the variance is
    # not recovered exactly and a shortfall is not a code fault -- see the
    # docstring.  Warn only when most of the energy is missing, and say what
    # actually causes it.
    if check and np.isfinite(ratio) and ratio < 0.5:
        warnings.warn(
            f'frequency_spectrum: integral(psd) recovers only '
            f'{100 * ratio:.0f}% of the variance. Welch cannot represent '
            f'periods longer than nperseg*dt = {nperseg * dt:.3g}, and '
            f'detrend={detrend!r} also removes each segment mean, so the rest '
            f'is low-frequency energy. Increase nperseg for a larger share '
            f'(noisier estimate), or pass detrend=False.',
            RuntimeWarning, stacklevel=2)
    # Note: var_ratio cannot validate dt.  scipy's scaling='density' gives
    # psd ~ 1/fs while df ~ fs, so integral(psd) df is invariant to dt -- only
    # the frequency axis moves.  dt has to be right by construction.

    return {'f': freq, 'omega': TWO_PI * freq, 'psd': psd, 'var_ratio': ratio,
            'area': area, 'var': var,
            'nperseg': nperseg, 'nseg': max(1, nt // nperseg)}


def convection_velocity(f, Lx, dt, lag=1, weight='energy'):
    """
    Convection velocity from the phase of the space-time cross-spectrum.

    For a frozen pattern travelling at Uc, f(x,t) = g(x - Uc t), the cross
    spectrum between two instants separated by tau is

        C_k(tau) = <conj(F_k(t)) F_k(t+tau)> = |g_k|^2 exp(-i k Uc tau)

    so Uc(k) = -arg(C_k) / (k tau), giving a *scale-dependent* convection
    velocity directly.  This is preferred over tracking the peak of the
    space-time correlation, which is ambiguous for a narrowband field: any
    shift differing by one wavelength fits the peak equally well, and a global
    argmax can lock onto the aliased one.

    The phase is unambiguous only while |k Uc tau| < pi.  Beyond that it folds
    back into (-pi, pi], so the wrapping cannot be detected from the measured
    phase itself -- a badly wrapped estimate looks perfectly innocent.
    `wrap_risk` is therefore *predictive*: it is k_max |Uc| tau / pi evaluated
    with the lag-1 estimate (the safest available) and the highest energetic
    wavenumber.  Values >= 1 mean the result is aliased; keep `lag` small.
    """
    nx, nz, nt = f.shape
    if not 1 <= lag < nt:
        raise ValueError(f'lag {lag} out of range for {nt} snapshots')

    def _estimate(lg):
        F = np.fft.rfft(f, axis=0) / nx                 # (nkx, nz, nt)
        C = (np.conj(F[:, :, :nt - lg]) * F[:, :, lg:]).mean(axis=(1, 2))
        k = TWO_PI * np.arange(C.size) / Lx
        tau = lg * dt
        uc_k = np.full(C.size, np.nan)
        pos = k > 0
        uc_k[pos] = -np.angle(C)[pos] / (k[pos] * tau)

        energy = np.abs(C)
        w = energy.copy() if weight == 'energy' else np.ones_like(energy)
        w[~pos] = 0.0
        good = np.isfinite(uc_k) & (w > 0)
        uc = (float(np.sum(w[good] * uc_k[good]) / np.sum(w[good]))
              if good.any() else np.nan)
        return k, uc_k, uc, energy, tau

    k, uc_k, uc, energy, tau = _estimate(lag)

    # reference speed from the shortest lag, which is the least likely to alias
    uc_ref = uc if lag == 1 else _estimate(1)[2]
    # highest wavenumber carrying non-negligible energy
    sig = energy > 0.01 * energy[1:].max() if energy[1:].size else np.zeros_like(energy, bool)
    k_max = float(k[sig].max()) if sig.any() else float(k[-1])
    wrap_risk = (float(k_max * abs(uc_ref) * tau / np.pi)
                 if np.isfinite(uc_ref) else np.nan)

    return {'k': k, 'Uc_k': uc_k, 'Uc': uc, 'tau': tau, 'energy': energy,
            'Uc_ref': uc_ref, 'k_max': k_max, 'wrap_risk': wrap_risk}


# ------------------------------------------------------------------ internals

def _assert_parseval(got, want, where, rtol=1e-6, atol=1e-18):
    if not np.isclose(got, want, rtol=rtol, atol=atol):
        raise ValueError(
            f'{where}: Parseval check failed -- spectrum sums to {got:.12g} but '
            f'the variance is {want:.12g} (relative error '
            f'{abs(got - want) / max(abs(want), 1e-300):.3g}). '
            f'This means the normalisation or the folding is wrong.')


# ------------------------------------------------------------ wall-unit helpers

def to_wall_units(data, utau=None):
    """
    Scaling factors for wall units.

    Lengths use Re_tau from the file (reference scaling), which keeps different
    cases on a common axis.  Velocities use `utau` if given.
    """
    retau = float(np.asarray(data['retau']))
    out = {'retau': retau, 'len': retau}       # y+ = y * retau  (h = 1)
    if utau is not None:
        out['utau'] = utau
        out['vel'] = utau
        out['time'] = utau * retau             # t+ = t * utau^2/nu
    return out
