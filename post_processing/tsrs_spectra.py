#!/usr/bin/env python3
"""
Spectra and two-point correlations from stitched tsrs data.

Computes, per wall-normal plane: premultiplied 1-D spectra in x and z, the uv
co-spectrum, two-point correlations in x and z, the wall-normal correlation
matrix, temporal PSD and convection velocity.  Several cases can be given and
are overlaid, so controlled-vs-uncontrolled comparison is the normal mode of
use rather than an afterthought.

This script draws.  For record diagnostics (stationarity, u_tau, resolution,
noise floor) and for error bars on the map, use tsrs_diag.py.

With --results every spectrum is also written as its own self-contained .npz
under data/results/<solver_case>/<run_name>/spectra/, carrying its scaling
scalars so it can be replotted in other wall units later.

Everything is validated against Parseval at runtime -- see postlib/tsrs_spectra.py.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from postlib import tsrs_spectra as sp                    # noqa: E402
from postlib import tsrs_case as tc                       # noqa: E402
from postlib import results as res                        # noqa: E402

ANALYSES = ('spectra', 'map', 'correlations', 'ycorr', 'frequency', 'convection')


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('case_ids', type=str, nargs='+',
                    help='cases to compare; the first is the reference')
    ap.add_argument('--runs-dir', default='../runs')
    ap.add_argument('--env', type=int, default=0)
    ap.add_argument('--field', default='u',
                    help='variable for spectra/correlations (default: %(default)s)')
    ap.add_argument('--planes', default=None,
                    help='comma-separated nominal y+ values (default: all)')
    ap.add_argument('--t0', type=float, default=None,
                    help='discard snapshots before this time (transient)')
    ap.add_argument('--t1', type=float, default=None)
    ap.add_argument('--utau', type=float, default=None,
                    help='*reference* u_tau, the one the stored y+ labels and '
                         'nu were built with; default reads runner.u_tau from '
                         'current_conf.yml')
    ap.add_argument('--scale', default='actual', choices=tc.SCALES,
                    help="wall units the figures are drawn in: 'actual' (the "
                         "case's own u_tau, from the viscous sublayer) or "
                         "'reference' (the baseline u_tau, common to all "
                         "cases).  Default: %(default)s")
    ap.add_argument('--lag', type=int, default=1,
                    help='time lag for the convection velocity (default: %(default)s)')
    ap.add_argument('--nperseg', type=int, default=None,
                    help='Welch segment length (default: nt//4)')
    ap.add_argument('--what', default='all',
                    help=f'comma-separated subset of {",".join(ANALYSES)}')
    ap.add_argument('--map-quantity', default='uu', choices=('uu', 'uv', 'both'),
                    help="which premultiplied spectrum to map: 'uu' is the "
                         "auto-spectrum of --field, 'uv' its co-spectrum with "
                         "v.  The names are shorthand -- with --field w they "
                         "mean Phi_ww and Phi_wv (default: %(default)s)")
    ap.add_argument('--map-dir', default='z', choices=('z', 'x'),
                    help='wavelength axis of the map (default: %(default)s)')
    ap.add_argument('--map-levels', type=int, default=21,
                    help='number of filled contour levels (default: %(default)s)')
    ap.add_argument('--map-points', action='store_true',
                    help='overlay the actual (lambda, y+) sample locations')
    ap.add_argument('--map-style', default='contour', choices=('contour', 'mesh'),
                    help="'contour' interpolates between samples; 'mesh' draws "
                         'one cell per sample and claims no resolution it does '
                         'not have (default: %(default)s)')
    ap.add_argument('--outdir', default='Figs')
    ap.add_argument('--results', action='store_true',
                    help='also save each spectrum as its own .npz under '
                         'data/results/<solver_case>/<run_name>/spectra/')
    ap.add_argument('--results-root', default=None,
                    help='override the results base (default: <repo>/data/results)')
    ap.add_argument('--head', default=None,
                    help='figure filename prefix (default: the case ids)')
    return ap.parse_args()


def figure_prefix(case_ids, head=None):
    """
    Filename prefix for every figure.

    Defaults to the case ids themselves, so figures from different cases never
    collide in --outdir; --head overrides it when a shorter name is wanted.
    """
    if head:
        return head
    return '_vs_'.join(re.sub(r'[^\w.+-]', '-', str(c)) for c in case_ids)


# ------------------------------------------------------------------ datasets

def dataset(out, case, name, arrays, **meta):
    """
    Queue one spectrum to be saved as its own file.

    `arrays` holds everything needed to redraw it; the case's scaling scalars
    are attached at write time, so a saved file can be rescaled later without
    the original run.  Code-unit arrays are stored beside the plotted ones for
    exactly that reason -- see postlib/results.py.
    """
    out.append({'name': name, 'case': case, 'arrays': arrays, 'meta': meta})


def _stack(rows):
    """Per-plane lists -> (n_planes, n) array."""
    return np.asarray(rows, dtype=float)


def _plane_axes(case, idx):
    """The three ways a plane is labelled: physical, chosen units, nominal."""
    return {
        'y': np.asarray(case['data']['y'], dtype=float)[idx],
        'yplus': tc.yplus(case)[idx],
        'yplus_nominal': np.asarray(case['data']['yplus'], dtype=float)[idx],
    }


# ------------------------------------------------------------------ analyses

def do_spectra(cases, args, out):
    """Premultiplied 1-D spectra in z and x, plus the uv co-spectrum."""
    fig, axs = plt.subplots(len(cases), 3, squeeze=False,
                            figsize=(15, 3.6 * len(cases)))
    for i, case in enumerate(cases):
        d, retau, u2 = case['data'], tc.length(case), tc.norm(case)
        idx = tc.plane_indices(d, args.planes)
        cols = {'z': [], 'x': [], 'co': []}
        lam = {}
        for iy in idx:
            yp = tc.yplus(case)[iy]
            f = sp.fluctuation(d, args.field, iy)
            g = sp.fluctuation(d, 'v', iy) if 'v' in d['names'] else None

            sz = sp.spectrum_1d(f, 1, case['Lz'])
            sx = sp.spectrum_1d(f, 0, case['Lx'])
            lab = f'$y^+$={tc.ylab(yp)}'
            axs[i, 0].semilogx(sz['lam'][1:] * retau, sz['kPhi'][1:] / u2, 'o-', ms=3, label=lab)
            axs[i, 1].semilogx(sx['lam'][1:] * retau, sx['kPhi'][1:] / u2, 'o-', ms=3, label=lab)
            cols['z'].append(sz)
            cols['x'].append(sx)
            lam['z'], lam['x'] = sz['lam'][1:], sx['lam'][1:]
            if g is not None:
                co = sp.cospectrum_1d(f, g, 1, case['Lz'])
                axs[i, 2].semilogx(co['lam'][1:] * retau, co['kPhi'][1:] / u2, 'o-', ms=3, label=lab)
                cols['co'].append(co)

        ff, fv = args.field, f'{args.field}v'
        planes = _plane_axes(case, idx)
        for key, sub, lbl in (('z', 'z', f'{ff}{ff}'), ('x', 'x', f'{ff}{ff}'),
                              ('co', 'z', fv)):
            if not cols[key]:
                continue
            ss = cols[key]
            dataset(out, case, f'spec_{sub}_{lbl}', {
                **planes,
                'k': ss[0]['k'][1:], 'lam': ss[0]['lam'][1:],
                'lam_plus': ss[0]['lam'][1:] * retau,
                'E': _stack([s['E'][1:] for s in ss]),
                'Phi': _stack([s['Phi'][1:] for s in ss]),
                'kPhi': _stack([s['kPhi'][1:] for s in ss]),
                'kPhi_plus': _stack([s['kPhi'][1:] / u2 for s in ss]),
            }, kind='spectrum', direction=sub, quantity=lbl, field=args.field,
               xlabel=f'lambda_{sub}+', ylabel=f'k_{sub} Phi_{lbl}')

        u = tc.usym(case)
        axs[i, 0].set(xlabel=r'$\lambda_z^+$', ylabel=f'$k_z\\Phi_{{{ff}{ff}}}${u}')
        axs[i, 1].set(xlabel=r'$\lambda_x^+$', ylabel=f'$k_x\\Phi_{{{ff}{ff}}}${u}')
        axs[i, 2].set(xlabel=r'$\lambda_z^+$', ylabel=f'$k_z\\Phi_{{{fv}}}${u}')
        axs[i, 0].set_title(f'{case["id"]}  spanwise\n{tc.scale_note(case)}',
                            loc='left', fontsize=9)
        axs[i, 1].set_title('streamwise', loc='left')
        axs[i, 2].set_title(f'{fv} co-spectrum', loc='left')
        for ax in axs[i]:
            ax.grid(alpha=0.3)
            ax.axhline(0, color='k', lw=0.6)
        axs[i, 0].legend(fontsize=7, ncol=2)
    return fig, f'spectra_{args.field}'


def map_label(field, quantity):
    """
    Subscript for the mapped quantity, following the field.

    'uv' is the CLI name of the co-spectrum with v; with --field w it is really
    the wv co-spectrum, and the label has to say so.
    """
    return f'{field}{field}' if quantity == 'uu' else f'{field}v'


def map_planes(case, planes, verbose=True):
    """
    Plane indices to put on the map.

    Both axes are logarithmic, so y+ = 0 cannot be drawn; u is identically zero
    at the wall anyway, so nothing is lost but the note is printed.
    """
    yp_all = tc.yplus(case)
    idx = tc.plane_indices(case['data'], planes)
    usable = [k for k in idx if yp_all[k] > 0]
    dropped = [k for k in idx if yp_all[k] <= 0]
    if dropped and verbose:
        print(f'  [{case["id"]}] dropping y+ = '
              f'{[tc.ylab(yp_all[k]) for k in dropped]} (log axis)')
    if len(usable) < 2:
        raise SystemExit(f'case {case["id"]}: need at least 2 planes with '
                         f'y+ > 0 to draw a map, have {len(usable)}')
    return usable


def map_rows(case, field, quantity, map_dir, usable, data=None):
    """
    Premultiplied spectrum on every usable plane, as a (n_planes, n_lambda) map.

    Returns (lam_plus, yplus, M) with lam ascending, ready for contourf.
    `data` overrides the case's record, so a sub-block can be mapped with the
    case's own scaling -- that is what tsrs_diag.py uses for its error bars.
    """
    d = case['data'] if data is None else data
    retau, u2 = tc.length(case), tc.norm(case)
    axis = 1 if map_dir == 'z' else 0
    L = case['Lz'] if map_dir == 'z' else case['Lx']
    yp_all = tc.yplus(case)

    rows, lam = [], None
    for k in usable:
        f = sp.fluctuation(d, field, k)
        if quantity == 'uv':
            if 'v' not in d['names']:
                raise SystemExit(f'case {case["id"]} has no v; cannot map uv')
            s = sp.cospectrum_1d(f, sp.fluctuation(d, 'v', k), axis, L)
        else:
            s = sp.spectrum_1d(f, axis, L)
        rows.append(s['kPhi'][1:] / u2)          # drop k=0 (lambda = infinity)
        lam = s['lam'][1:] * retau

    # the FFT gives lambda descending; contourf wants an ascending axis
    order = np.argsort(lam)
    return lam[order], np.asarray([yp_all[k] for k in usable]), \
        np.asarray(rows)[:, order]


def map_levels(q, vmax, n):
    """Contour levels and colour map for one quantity."""
    if q == 'uv':                          # co-spectrum is negative: diverging
        return np.linspace(-vmax, vmax, n), 'RdBu_r', 'both'
    # energy is positive: sequential from 0, and never < 0, so no lower arrow
    return np.linspace(0, vmax, n), 'jet', 'max'


def log_axes(ax, xlim, ylim):
    """
    Log-log (lambda+, y+) axes with readable ticks.

    A log range spanning barely a decade is where matplotlib labels the minor
    ticks too and they collide ("3x10^1 4x10^1"); place plain 1/2/5 decade
    ticks instead.
    """
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    for axis_ in (ax.xaxis, ax.yaxis):
        axis_.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0),
                                                   numticks=12))
        axis_.set_major_formatter(mticker.ScalarFormatter())
        axis_.set_minor_formatter(mticker.NullFormatter())


def do_map(cases, args, out):
    """
    Premultiplied spectrum as a contour map over (lambda+, y+), both log.

    The standard wall-turbulence presentation: the ridge of the map traces the
    energetic scale at each height, so a controller's effect on the near-wall
    cycle shows up as a shift or a weakening of that ridge.
    """
    quantities = ('uu', 'uv') if args.map_quantity == 'both' else (args.map_quantity,)
    ncol, nrow = len(quantities), len(cases)
    fig, axs = plt.subplots(nrow, ncol, squeeze=False,
                            figsize=(5.8 * ncol, 4.4 * nrow))

    # the plane set does not depend on the quantity: pick it once, so a dropped
    # plane is reported once rather than once per column
    planes = {case['id']: map_planes(case, args.planes) for case in cases}

    for j, q in enumerate(quantities):
        # one colour scale per quantity across all cases, so rows compare
        mats = [map_rows(case, args.field, q, args.map_dir, planes[case['id']])
                for case in cases]
        vmax = max(np.abs(M).max() for _, _, M in mats)
        levels, cmap, extend = map_levels(q, vmax, args.map_levels)

        # rows are meant to be compared, so give them identical axes: different
        # plane sets would otherwise stretch each row differently and a shift
        # of the energy peak would be hard to read off
        ylim = (min(yp.min() for _, yp, _ in mats),
                max(yp.max() for _, yp, _ in mats))
        xlim = (min(lam.min() for lam, _, _ in mats),
                max(lam.max() for lam, _, _ in mats))

        for i, (case, (lam, yp, M)) in enumerate(zip(cases, mats)):
            ax = axs[i, j]
            if args.map_style == 'mesh':
                # one cell per sample: the map then claims exactly the
                # resolution it has, which for the outer planes is very little
                cf = ax.pcolormesh(lam, yp, M, cmap=cmap, shading='nearest',
                                   norm=mcolors.BoundaryNorm(levels,
                                                             plt.get_cmap(cmap).N,
                                                             extend=extend))
            else:
                cf = ax.contourf(lam, yp, M, levels=levels, cmap=cmap,
                                 extend=extend)
                ax.contour(lam, yp, M, levels=levels[::4], colors='k',
                           linewidths=0.4, alpha=0.5)

            # mark the peak: the most energetic (lambda, y+) pair
            pk = np.unravel_index(np.argmax(np.abs(M)), M.shape)
            ax.plot(lam[pk[1]], yp[pk[0]], 'w*', ms=13, mec='k', mew=0.8,
                    label=f'peak {M[pk]:.2f} at $\\lambda^+$={lam[pk[1]]:.0f}, '
                          f'$y^+$={tc.ylab(yp[pk[0]])}')
            if args.map_points:
                Lm, Ym = np.meshgrid(lam, yp)
                ax.plot(Lm.ravel(), Ym.ravel(), 'k.', ms=1.5, alpha=0.35)

            log_axes(ax, xlim, ylim)
            sub = args.map_dir
            lbl = map_label(args.field, q)
            ax.set_xlabel(f'$\\lambda_{sub}^+$')
            ax.set_ylabel('$y^+$')
            ax.set_title(f'{case["id"]}: $k_{sub}\\Phi_{{{lbl}}}${tc.usym(case)}\n'
                         f'{tc.scale_note(case)}', loc='left', fontsize=9)
            ax.legend(fontsize=7, loc='upper left')
            fig.colorbar(cf, ax=ax, pad=0.02)

            u2 = tc.norm(case)
            dataset(out, case, f'map{sub}_{lbl}', {
                **_plane_axes(case, planes[case['id']]),
                'lam': lam / tc.length(case), 'lam_plus': lam,
                'kPhi': M * u2, 'kPhi_plus': M,
                'peak': np.array([lam[pk[1]], yp[pk[0]], M[pk]]),
            }, kind='map', direction=sub, quantity=lbl, field=args.field,
               xlabel=f'lambda_{sub}+', ylabel='y+',
               zlabel=f'k_{sub} Phi_{lbl}',
               peak_note='lam_plus, yplus, kPhi_plus at the peak')

            print(f'  [{case["id"]}] {lbl}: {M.shape[0]} planes x {M.shape[1]} '
                  f'wavelengths, peak {M[pk]:.2f} at lam_{sub}+={lam[pk[1]]:.0f}, '
                  f'y+={tc.ylab(yp[pk[0]])}')

    # every switch that changes the picture has to be in the tag: without the
    # quantity a uv map overwrites the uu map of the same case, which looks
    # exactly like uv never having been written.  The tag names the *resolved*
    # pair, not the CLI shorthand -- '--field w --map-quantity uu' is a ww map
    # and the file has to say ww.
    tag = f'map{args.map_dir}_' + '-'.join(map_label(args.field, q)
                                           for q in quantities)
    return fig, tag + ('_mesh' if args.map_style == 'mesh' else '')


def do_correlations(cases, args, out):
    """Two-point correlations along z and x."""
    fig, axs = plt.subplots(len(cases), 2, squeeze=False,
                            figsize=(11, 3.6 * len(cases)))
    for i, case in enumerate(cases):
        d, retau = case['data'], tc.length(case)
        idx = tc.plane_indices(d, args.planes)
        cols = {'z': [], 'x': []}
        for iy in idx:
            yp = tc.yplus(case)[iy]
            f = sp.fluctuation(d, args.field, iy)
            cz = sp.correlation_1d(f, 1, case['Lz'])
            cx = sp.correlation_1d(f, 0, case['Lx'])
            lab = f'$y^+$={tc.ylab(yp)}'
            axs[i, 0].plot(cz['sep'] * retau, cz['rho'], 'o-', ms=3, label=lab)
            axs[i, 1].plot(cx['sep'] * retau, cx['rho'], 'o-', ms=3, label=lab)
            cols['z'].append(cz)
            cols['x'].append(cx)

        ff = args.field
        planes = _plane_axes(case, idx)
        for sub in ('z', 'x'):
            cs = cols[sub]
            dataset(out, case, f'corr_{sub}_{ff}{ff}', {
                **planes,
                'sep': cs[0]['sep'], 'sep_plus': cs[0]['sep'] * retau,
                'rho': _stack([c['rho'] for c in cs]),
                'R': _stack([c['R'] for c in cs]),
                'variance': np.array([c['var'] for c in cs]),
            }, kind='correlation', direction=sub, quantity=f'{ff}{ff}',
               field=ff, xlabel=f'delta {sub}+', ylabel=f'rho_{ff}{ff}')

        axs[i, 0].set(xlabel=r'$\Delta z^+$', ylabel=f'$\\rho_{{{ff}{ff}}}$')
        axs[i, 1].set(xlabel=r'$\Delta x^+$', ylabel=f'$\\rho_{{{ff}{ff}}}$')
        axs[i, 0].set_title(f'{case["id"]}\n{tc.scale_note(case)}', loc='left',
                            fontsize=9)
        for ax in axs[i]:
            ax.axhline(0, color='k', lw=0.6)
            ax.grid(alpha=0.3)
        axs[i, 0].legend(fontsize=7, ncol=2)
    return fig, f'corr_{args.field}'


def do_ycorr(cases, args, out):
    """Correlation between wall-normal planes."""
    fig, axs = plt.subplots(1, len(cases), squeeze=False,
                            figsize=(4.6 * len(cases), 4))
    for i, case in enumerate(cases):
        r = sp.correlation_y(case['data'], args.field)
        yp = tc.yplus(case)
        im = axs[0, i].imshow(r['rho'], vmin=-1, vmax=1, cmap='RdBu_r',
                              origin='lower')
        axs[0, i].set_xticks(range(len(yp)))
        axs[0, i].set_yticks(range(len(yp)))
        axs[0, i].set_xticklabels([tc.ylab(v) for v in yp], rotation=90, fontsize=7)
        axs[0, i].set_yticklabels([tc.ylab(v) for v in yp], fontsize=7)
        axs[0, i].set(xlabel='$y^+$', ylabel='$y^+$',
                      title=f'{case["id"]}: '
                            f'$\\rho_{{{args.field}{args.field}}}$\n'
                            f'{tc.scale_note(case)}')
        axs[0, i].title.set_fontsize(9)
        fig.colorbar(im, ax=axs[0, i], shrink=0.85)

        ff = args.field
        dataset(out, case, f'ycorr_{ff}{ff}', {
            **_plane_axes(case, list(range(len(yp)))),
            'rho': r['rho'], 'rms': r['rms'],
            'rms_plus': r['rms'] / (case['scale']['utau'] or 1.0),
        }, kind='ycorr', quantity=f'{ff}{ff}', field=ff,
           xlabel='y+', ylabel='y+', zlabel=f'rho_{ff}{ff}')
    return fig, f'ycorr_{args.field}'


def do_frequency(cases, args, out):
    """Temporal PSD, premultiplied against period."""
    fig, axs = plt.subplots(1, len(cases), squeeze=False,
                            figsize=(5.2 * len(cases), 4))
    for i, case in enumerate(cases):
        d, u2 = case['data'], tc.norm(case)
        idx = tc.plane_indices(d, args.planes)
        rows = []
        for iy in idx:
            yp = tc.yplus(case)[iy]
            f = sp.fluctuation(d, args.field, iy, mean='global')
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                fr = sp.frequency_spectrum(f, case['dt'], nperseg=args.nperseg)
            if w:
                print(f'  [{case["id"]} y+={tc.ylab(yp)}] {w[0].message}')
            m = fr['f'] > 0
            axs[0, i].loglog(fr['f'][m], fr['psd'][m] / u2,
                             label=f'$y^+$={tc.ylab(yp)}')
            rows.append(fr)
            print(f'  [{case["id"]} y+={tc.ylab(yp):>7}] nperseg={fr["nperseg"]} '
                  f'var_ratio={fr["var_ratio"]:.2f}')

        ff = args.field
        dataset(out, case, f'psd_{ff}', {
            **_plane_axes(case, idx),
            'f': rows[0]['f'], 'omega': rows[0]['omega'],
            'psd': _stack([r['psd'] for r in rows]),
            'psd_plus': _stack([r['psd'] / u2 for r in rows]),
            'var_ratio': np.array([r['var_ratio'] for r in rows]),
            'variance': np.array([r['var'] for r in rows]),
            'nperseg': np.array([r['nperseg'] for r in rows]),
        }, kind='psd', quantity=ff, field=ff, xlabel='f', ylabel=f'PSD_{ff}')

        axs[0, i].set(xlabel='$f$', ylabel='PSD' + tc.usym(case),
                      title=f'{case["id"]}\n{tc.scale_note(case)}')
        axs[0, i].title.set_fontsize(9)
        axs[0, i].grid(alpha=0.3, which='both')
        axs[0, i].legend(fontsize=7, ncol=2)
    return fig, f'freq_{args.field}'


def do_convection(cases, args, out):
    """Scale-dependent convection velocity."""
    fig, axs = plt.subplots(1, len(cases), squeeze=False,
                            figsize=(5.2 * len(cases), 4))
    for i, case in enumerate(cases):
        d, utau = case['data'], case['scale']['utau']
        idx = tc.plane_indices(d, args.planes)
        rows, lam = [], None
        for iy in idx:
            yp = tc.yplus(case)[iy]
            f = sp.fluctuation(d, args.field, iy)
            cv = sp.convection_velocity(f, case['Lx'], case['dt'], lag=args.lag)
            scale = utau or 1.0
            lam = np.where(cv['k'] > 0, 2 * np.pi / np.where(cv['k'] > 0, cv['k'], 1),
                           np.nan) * tc.length(case)
            axs[0, i].semilogx(lam[1:], cv['Uc_k'][1:] / scale, 'o-', ms=3,
                               label=f'$y^+$={tc.ylab(yp)}')
            flag = '  ALIASED -- lower --lag' if cv['wrap_risk'] >= 1 else ''
            plus = f' (Uc+={cv["Uc"] / scale:.2f})' if utau else ''
            print(f'  [{case["id"]} y+={tc.ylab(yp):>7}] Uc={cv["Uc"]:.4f}{plus}'
                  f'  wrap_risk={cv["wrap_risk"]:.2f}{flag}')
            rows.append(cv)

        ff, scale = args.field, utau or 1.0
        dataset(out, case, f'uc_{ff}', {
            **_plane_axes(case, idx),
            'k': rows[0]['k'][1:], 'lam_plus': lam[1:],
            'Uc_k': _stack([r['Uc_k'][1:] for r in rows]),
            'Uc_k_plus': _stack([r['Uc_k'][1:] / scale for r in rows]),
            'Uc': np.array([r['Uc'] for r in rows]),
            'Uc_plus': np.array([r['Uc'] / scale for r in rows]),
            'wrap_risk': np.array([r['wrap_risk'] for r in rows]),
        }, kind='convection', quantity=ff, field=ff, lag=args.lag,
           xlabel='lambda_x+', ylabel='Uc')

        axs[0, i].set(xlabel=r'$\lambda_x^+$',
                      ylabel='$U_c^+$' if utau else '$U_c$',
                      title=f'{case["id"]} (lag={args.lag})\n{tc.scale_note(case)}')
        axs[0, i].title.set_fontsize(9)
        axs[0, i].grid(alpha=0.3)
        axs[0, i].legend(fontsize=7, ncol=2)
    return fig, f'Uc_{args.field}'


# ---------------------------------------------------------------------- main

def main():
    args = parse_args()
    plt.rc('font', family='serif', size=11)
    plt.rc('axes', labelsize=11)

    what = ANALYSES if args.what == 'all' else tuple(
        w.strip() for w in args.what.split(',') if w.strip())
    unknown = set(what) - set(ANALYSES)
    if unknown:
        raise SystemExit(f'unknown analysis {sorted(unknown)}; '
                         f'choose from {ANALYSES}')

    cases = [tc.load_case(cid, runs_dir=args.runs_dir, env=args.env,
                          field=args.field, t0=args.t0, t1=args.t1,
                          utau=args.utau, scale=args.scale)
             for cid in args.case_ids]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    head = figure_prefix(args.case_ids, args.head)
    out, runners = [], {'spectra': do_spectra, 'map': do_map,
                        'correlations': do_correlations,
                        'ycorr': do_ycorr, 'frequency': do_frequency,
                        'convection': do_convection}

    # the scaling is baked into every number on every figure, so it belongs in
    # the filename -- the same figure in reference and in actual wall units
    # must not share a path
    suffix = 'uact' if all(c['scale']['kind'] == 'actual' for c in cases) else 'uref'

    # a map shares one colour scale across its rows, so rows normalised
    # differently would be read as if they were the same number
    units = {(c['scale']['kind'], c['scale']['utau'] is not None) for c in cases}
    if len(units) > 1:
        print('\nWARNING: the cases are not all in the same units '
              f'{sorted(k for k, _ in units)} -- a case without a reference '
              'u_tau stays in code units, and rows of a map share one colour '
              'scale.  Give every case a u_tau (--utau, or a current_conf.yml '
              'next to its data) before comparing rows.')

    print()
    for name in what:
        print(f'--- {name} ---')
        # the runner owns the whole tag, field included: a map tag already
        # names the resolved pair ('uu', 'wv'), and appending --field again
        # would give '..._mapz_uu_..._u', with 'u' meaning two different things
        fig, tag = runners[name](cases, args, out)
        fig.tight_layout()
        fig_out = outdir / f'{head}_{tag}_{suffix}.png'
        fig.savefig(fig_out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  -> {fig_out}')

    if args.results:
        write_results(out, args, suffix)
    return 0


def write_results(out, args, suffix):
    """
    One .npz per spectrum, under data/results/<solver_case>/<run_name>/spectra/.

    Each file carries its own scaling scalars, so it can be replotted in other
    wall units without the run it came from -- see postlib/results.py.
    """
    print('\n--- results ---')
    roots = {}
    for ds in out:
        case = ds['case']
        solver_case = res.solver_case_from_npz(case['path'])
        root = res.case_root(case['id'], solver_case, root=args.results_root)
        roots[case['id']] = (root, solver_case)
        target = root / 'spectra' / f'{case["id"]}_{ds["name"]}_{suffix}.npz'
        res.save_dataset(target, ds['arrays'], res.scalars_from_case(case),
                         ds['meta'])
        print(f'  -> {target.relative_to(root.parent.parent.parent)}')

    for cid, (root, solver_case) in roots.items():
        res.write_meta(root, solver_case=solver_case, run_name=str(cid),
                       spectra={'field': args.field, 'scale': args.scale,
                                'files': sorted(p.name for p in
                                                (root / 'spectra').glob('*.npz'))})
        print(f'  meta -> {root / "meta.yml"}')


if __name__ == '__main__':
    sys.exit(main())
