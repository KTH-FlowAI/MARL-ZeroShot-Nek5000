#!/usr/bin/env python3
"""
Spectra and two-point correlations from stitched tsrs data.

Computes, per wall-normal plane: premultiplied 1-D spectra in x and z, the uv
co-spectrum, two-point correlations in x and z, the wall-normal correlation
matrix, temporal PSD and convection velocity.  Several cases can be given and
are overlaid, so controlled-vs-uncontrolled comparison is the normal mode of
use rather than an afterthought.

Everything is validated against Parseval at runtime -- see postlib/tsrs_spectra.py.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from postlib import tsrs, tsrs_spectra as sp      # noqa: E402

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
                    help='comma-separated y+ values to analyse (default: all)')
    ap.add_argument('--t0', type=float, default=None,
                    help='discard snapshots before this time (transient)')
    ap.add_argument('--t1', type=float, default=None)
    ap.add_argument('--utau', type=float, default=None,
                    help='reference u_tau for wall units; default reads '
                         'runner.u_tau from current_conf.yml')
    ap.add_argument('--lag', type=int, default=1,
                    help='time lag for the convection velocity (default: %(default)s)')
    ap.add_argument('--nperseg', type=int, default=None,
                    help='Welch segment length (default: nt//4)')
    ap.add_argument('--what', default='all',
                    help=f'comma-separated subset of {",".join(ANALYSES)}')
    ap.add_argument('--map-quantity', default='uu', choices=('uu', 'uv', 'both'),
                    help='which premultiplied spectrum to map (default: %(default)s)')
    ap.add_argument('--map-dir', default='z', choices=('z', 'x'),
                    help='wavelength axis of the map (default: %(default)s)')
    ap.add_argument('--map-levels', type=int, default=21,
                    help='number of filled contour levels (default: %(default)s)')
    ap.add_argument('--map-points', action='store_true',
                    help='overlay the actual (lambda, y+) sample locations')
    ap.add_argument('--outdir', default='Figs')
    ap.add_argument('--save', default=None,
                    help='also write the computed curves to this .npz')
    ap.add_argument('--head', default='tsrs')
    return ap.parse_args()


# --------------------------------------------------------------------- input

def find_npz(runs_dir, case_id, env):
    base = Path(runs_dir) / str(case_id)
    data_path = base / 'eval' if (base / 'eval').is_dir() else base
    env_dirs = sorted(d for d in data_path.iterdir()
                      if d.is_dir() and 'env' in d.name)
    if not env_dirs:
        raise SystemExit(f'no env_* directories under {data_path}')
    if env >= len(env_dirs):
        raise SystemExit(f'case {case_id} has {len(env_dirs)} environments, '
                         f'--env {env} is out of range')
    hits = sorted(env_dirs[env].glob('tsrs_*.npz'))
    if not hits:
        raise SystemExit(f'no stitched .npz in {env_dirs[env]} -- '
                         f'run tsrs_stitch.py first')
    return hits[0], data_path


def read_utau(data_path):
    """Reference u_tau from the run config, if present."""
    cfg = data_path / 'current_conf.yml'
    if not cfg.exists():
        return None
    with open(cfg) as fh:
        conf = yaml.safe_load(fh) or {}
    return (conf.get('runner') or {}).get('u_tau')


def load_case(args, case_id):
    path, data_path = find_npz(args.runs_dir, case_id, args.env)
    data = tsrs.load(path)
    # validate before anything touches the field, so a typo gives this message
    # rather than a KeyError from deep inside the diagnostics
    if args.field not in data['names']:
        raise SystemExit(f'case {case_id} has no field {args.field!r}; '
                         f'available: {data["names"]}')

    dup = sp.check_duplicate_endpoints(data, args.field, 0)
    data = sp.periodic_view(data)
    if args.t0 is not None or args.t1 is not None:
        data = sp.trim_time(data, args.t0, args.t1)

    utau = args.utau if args.utau is not None else read_utau(data_path)
    Lx, Lz = sp.domain(data)
    dt = float(np.diff(data['t']).mean())
    retau = float(np.asarray(data['retau']))

    print(f'\ncase {case_id}: {path}')
    print(f'  periodic view {data["fld"].shape}  (endpoints duplicated: {dup})')
    print(f'  Lx={Lx:.4f} Lz={Lz:.4f}  Lx+={Lx * retau:.0f} Lz+={Lz * retau:.0f}'
          f'  dx+={Lx / data["x"].size * retau:.1f} dz+={Lz / data["z"].size * retau:.1f}')
    print(f'  t=[{data["t"][0]:.2f}, {data["t"][-1]:.2f}] n={data["t"].size} dt={dt:.4g}')

    st = sp.stationarity(data, args.field, min(1, data['fld'].shape[0] - 1))
    flag = '  <-- NOT stationary' if abs(st['drift_rel_rms']) > 0.5 else ''
    print(f'  drift of <{args.field}>_xz: {st["drift"]:+.4g} '
          f'({st["drift_pct"]:+.1f}% of the mean, '
          f'{st["drift_rel_rms"]:+.2f} x rms){flag}')

    if utau is not None:
        est = sp.estimate_utau(data, utau_ref=utau, name=args.field)
        if est:
            print(f'  u_tau: reference {utau:.4f}, actual {est["utau"]:.4f} '
                  f'(Re_tau {est["retau_actual"]:.0f}, '
                  f'drag reduction {est["drag_reduction_pct"]:+.1f}%)')
    return {'id': case_id, 'data': data, 'Lx': Lx, 'Lz': Lz, 'dt': dt,
            'retau': retau, 'utau': utau}


def plane_indices(data, spec):
    yp = np.asarray(data['yplus'], dtype=float)
    if spec is None:
        return list(range(yp.size))
    want = [float(v) for v in spec.split(',') if v.strip()]
    return [int(np.abs(yp - w).argmin()) for w in want]


# ------------------------------------------------------------------ analyses

def _norm(case):
    """Velocity scale squared, for normalising energies."""
    return (case['utau'] ** 2) if case['utau'] else 1.0


def do_spectra(cases, args, store):
    """Premultiplied 1-D spectra in z and x, plus the uv co-spectrum."""
    fig, axs = plt.subplots(len(cases), 3, squeeze=False,
                            figsize=(15, 3.6 * len(cases)))
    for i, case in enumerate(cases):
        d, retau, u2 = case['data'], case['retau'], _norm(case)
        idx = plane_indices(d, args.planes)
        for iy in idx:
            yp = d['yplus'][iy]
            f = sp.fluctuation(d, args.field, iy)
            g = sp.fluctuation(d, 'v', iy) if 'v' in d['names'] else None

            sz = sp.spectrum_1d(f, 1, case['Lz'])
            sx = sp.spectrum_1d(f, 0, case['Lx'])
            lab = f'$y^+$={yp:g}'
            axs[i, 0].semilogx(sz['lam'][1:] * retau, sz['kPhi'][1:] / u2, 'o-', ms=3, label=lab)
            axs[i, 1].semilogx(sx['lam'][1:] * retau, sx['kPhi'][1:] / u2, 'o-', ms=3, label=lab)
            if g is not None:
                co = sp.cospectrum_1d(f, g, 1, case['Lz'])
                axs[i, 2].semilogx(co['lam'][1:] * retau, co['kPhi'][1:] / u2, 'o-', ms=3, label=lab)
            store[f'{case["id"]}/y{yp:g}/lam_z+'] = sz['lam'] * retau
            store[f'{case["id"]}/y{yp:g}/kPhi_z'] = sz['kPhi']
            store[f'{case["id"]}/y{yp:g}/lam_x+'] = sx['lam'] * retau
            store[f'{case["id"]}/y{yp:g}/kPhi_x'] = sx['kPhi']

        u = '$/u_\\tau^2$' if case['utau'] else ''
        ff, fv = args.field, f'{args.field}v'
        axs[i, 0].set(xlabel=r'$\lambda_z^+$', ylabel=f'$k_z\\Phi_{{{ff}{ff}}}${u}')
        axs[i, 1].set(xlabel=r'$\lambda_x^+$', ylabel=f'$k_x\\Phi_{{{ff}{ff}}}${u}')
        axs[i, 2].set(xlabel=r'$\lambda_z^+$', ylabel=f'$k_z\\Phi_{{{fv}}}${u}')
        axs[i, 0].set_title(f'{case["id"]}  spanwise', loc='left')
        axs[i, 1].set_title('streamwise', loc='left')
        axs[i, 2].set_title(f'{fv} co-spectrum', loc='left')
        for ax in axs[i]:
            ax.grid(alpha=0.3)
            ax.axhline(0, color='k', lw=0.6)
        axs[i, 0].legend(fontsize=7, ncol=2)
    return fig, 'spectra'


def _map_rows(case, args, quantity):
    """
    Premultiplied spectrum on every usable plane, as a (n_planes, n_lambda) map.

    Returns (lam_plus, yplus, M) with lam ascending, ready for contourf.
    """
    d, retau, u2 = case['data'], case['retau'], _norm(case)
    axis = 1 if args.map_dir == 'z' else 0
    L = case['Lz'] if args.map_dir == 'z' else case['Lx']

    idx = plane_indices(d, args.planes)
    yp_all = np.asarray(d['yplus'], dtype=float)
    # y+ = 0 cannot go on a log axis, and at the wall u == 0 identically
    usable = [k for k in idx if yp_all[k] > 0]
    dropped = [k for k in idx if yp_all[k] <= 0]
    if dropped:
        print(f'  [{case["id"]}] dropping y+ = '
              f'{[f"{yp_all[k]:g}" for k in dropped]} (not representable on a '
              f'log axis)')
    if len(usable) < 2:
        raise SystemExit(f'case {case["id"]}: need at least 2 planes with y+ > 0 '
                         f'to draw a map, have {len(usable)}')

    rows, lam = [], None
    for k in usable:
        f = sp.fluctuation(d, args.field, k)
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


def do_map(cases, args, store):
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

    for j, q in enumerate(quantities):
        # one colour scale per quantity across all cases, so rows compare
        mats = [_map_rows(case, args, q) for case in cases]
        vmax = max(np.abs(M).max() for _, _, M in mats)
        if q == 'uv':                      # co-spectrum is negative: diverging
            levels = np.linspace(-vmax, vmax, args.map_levels)
            cmap, extend = 'RdBu_r', 'both'
        else:                              # energy is positive: sequential,
            levels = np.linspace(0, vmax, args.map_levels)   # and never < 0, so
            cmap, extend = 'jet', 'max'                  # no lower arrow

        # rows are meant to be compared, so give them identical axes: different
        # plane sets would otherwise stretch each row differently and a shift
        # of the energy peak would be hard to read off
        ylim = (min(yp.min() for _, yp, _ in mats),
                max(yp.max() for _, yp, _ in mats))
        xlim = (min(lam.min() for lam, _, _ in mats),
                max(lam.max() for lam, _, _ in mats))

        for i, (case, (lam, yp, M)) in enumerate(zip(cases, mats)):
            ax = axs[i, j]
            cf = ax.contourf(lam, yp, M, levels=levels, cmap=cmap, extend=extend)
            ax.contour(lam, yp, M, levels=levels[::4], colors='k',
                       linewidths=0.4, alpha=0.5)

            # mark the peak: the most energetic (lambda, y+) pair
            pk = np.unravel_index(np.argmax(np.abs(M)), M.shape)
            ax.plot(lam[pk[1]], yp[pk[0]], 'w*', ms=13, mec='k', mew=0.8,
                    label=f'peak $\\lambda^+$={lam[pk[1]]:.0f}, $y^+$={yp[pk[0]]:g}')
            if args.map_points:
                Lm, Ym = np.meshgrid(lam, yp)
                ax.plot(Lm.ravel(), Ym.ravel(), 'k.', ms=1.5, alpha=0.35)

            ax.set_xscale('log')
            # ax.set_yscale('log') # we do not log the y+
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            # both ranges span barely a decade, where matplotlib labels the
            # minor ticks too and they collide ("3x10^1 4x10^1"); place plain
            # 1/2/5 decade ticks instead
            for axis_ in (ax.xaxis, ax.yaxis):
                axis_.set_major_locator(mticker.LogLocator(base=10.0,
                                                           subs=(1.0, 2.0, 5.0),
                                                           numticks=12))
                axis_.set_major_formatter(mticker.ScalarFormatter())
                axis_.set_minor_formatter(mticker.NullFormatter())
            sub = args.map_dir
            u = '$/u_\\tau^2$' if case['utau'] else ''
            ax.set_xlabel(f'$\\lambda_{sub}^+$')
            ax.set_ylabel('$y^+$')
            ax.set_title(f'{case["id"]}: $k_{sub}\\Phi_{{{q}}}${u}', loc='left')
            ax.legend(fontsize=7, loc='upper left')
            fig.colorbar(cf, ax=ax, pad=0.02)

            store[f'{case["id"]}/map_{q}_{sub}/lam+'] = lam
            store[f'{case["id"]}/map_{q}_{sub}/yplus'] = yp
            store[f'{case["id"]}/map_{q}_{sub}/kPhi'] = M
            print(f'  [{case["id"]}] {q}: {M.shape[0]} planes x {M.shape[1]} '
                  f'wavelengths, peak at lam_{sub}+={lam[pk[1]]:.0f}, '
                  f'y+={yp[pk[0]]:g}')
    return fig, f'map{args.map_dir}'


def do_correlations(cases, args, store):
    """Two-point correlations along z and x."""
    fig, axs = plt.subplots(len(cases), 2, squeeze=False,
                            figsize=(11, 3.6 * len(cases)))
    for i, case in enumerate(cases):
        d, retau = case['data'], case['retau']
        for iy in plane_indices(d, args.planes):
            yp = d['yplus'][iy]
            f = sp.fluctuation(d, args.field, iy)
            cz = sp.correlation_1d(f, 1, case['Lz'])
            cx = sp.correlation_1d(f, 0, case['Lx'])
            lab = f'$y^+$={yp:g}'
            axs[i, 0].plot(cz['sep'] * retau, cz['rho'], 'o-', ms=3, label=lab)
            axs[i, 1].plot(cx['sep'] * retau, cx['rho'], 'o-', ms=3, label=lab)
            store[f'{case["id"]}/y{yp:g}/dz+'] = cz['sep'] * retau
            store[f'{case["id"]}/y{yp:g}/rho_z'] = cz['rho']
            store[f'{case["id"]}/y{yp:g}/dx+'] = cx['sep'] * retau
            store[f'{case["id"]}/y{yp:g}/rho_x'] = cx['rho']
        axs[i, 0].set(xlabel=r'$\Delta z^+$', ylabel=r'$\rho_{uu}$')
        axs[i, 1].set(xlabel=r'$\Delta x^+$', ylabel=r'$\rho_{uu}$')
        axs[i, 0].set_title(f'{case["id"]}', loc='left')
        for ax in axs[i]:
            ax.axhline(0, color='k', lw=0.6)
            ax.grid(alpha=0.3)
        axs[i, 0].legend(fontsize=7, ncol=2)
    return fig, 'corr'


def do_ycorr(cases, args, store):
    """Correlation between wall-normal planes."""
    fig, axs = plt.subplots(1, len(cases), squeeze=False,
                            figsize=(4.6 * len(cases), 4))
    for i, case in enumerate(cases):
        r = sp.correlation_y(case['data'], args.field)
        yp = r['yplus']
        im = axs[0, i].imshow(r['rho'], vmin=-1, vmax=1, cmap='RdBu_r',
                              origin='lower')
        axs[0, i].set_xticks(range(len(yp)))
        axs[0, i].set_yticks(range(len(yp)))
        axs[0, i].set_xticklabels([f'{v:g}' for v in yp], rotation=90, fontsize=7)
        axs[0, i].set_yticklabels([f'{v:g}' for v in yp], fontsize=7)
        axs[0, i].set(xlabel='$y^+$', ylabel='$y^+$',
                      title=f'{case["id"]}: $\\rho_{{{args.field}{args.field}}}$')
        fig.colorbar(im, ax=axs[0, i], shrink=0.85)
        store[f'{case["id"]}/ycorr'] = r['rho']
        store[f'{case["id"]}/ycorr_yplus'] = yp
    return fig, 'ycorr'


def do_frequency(cases, args, store):
    """Temporal PSD, premultiplied against period."""
    fig, axs = plt.subplots(1, len(cases), squeeze=False,
                            figsize=(5.2 * len(cases), 4))
    for i, case in enumerate(cases):
        d, u2 = case['data'], _norm(case)
        for iy in plane_indices(d, args.planes):
            yp = d['yplus'][iy]
            f = sp.fluctuation(d, args.field, iy, mean='global')
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                fr = sp.frequency_spectrum(f, case['dt'], nperseg=args.nperseg)
            if w:
                print(f'  [{case["id"]} y+={yp:g}] {w[0].message}')
            m = fr['f'] > 0
            axs[0, i].loglog(fr['f'][m], fr['psd'][m] / u2, label=f'$y^+$={yp:g}')
            store[f'{case["id"]}/y{yp:g}/freq'] = fr['f']
            store[f'{case["id"]}/y{yp:g}/psd'] = fr['psd']
            print(f'  [{case["id"]} y+={yp:6g}] nperseg={fr["nperseg"]} '
                  f'var_ratio={fr["var_ratio"]:.2f}')
        axs[0, i].set(xlabel='$f$', ylabel='PSD' + ('$/u_\\tau^2$' if case['utau'] else ''),
                      title=f'{case["id"]}')
        axs[0, i].grid(alpha=0.3, which='both')
        axs[0, i].legend(fontsize=7, ncol=2)
    return fig, 'freq'


def do_convection(cases, args, store):
    """Scale-dependent convection velocity."""
    fig, axs = plt.subplots(1, len(cases), squeeze=False,
                            figsize=(5.2 * len(cases), 4))
    for i, case in enumerate(cases):
        d = case['data']
        for iy in plane_indices(d, args.planes):
            yp = d['yplus'][iy]
            f = sp.fluctuation(d, args.field, iy)
            cv = sp.convection_velocity(f, case['Lx'], case['dt'], lag=args.lag)
            scale = case['utau'] or 1.0
            lam = np.where(cv['k'] > 0, 2 * np.pi / np.where(cv['k'] > 0, cv['k'], 1),
                           np.nan) * case['retau']
            axs[0, i].semilogx(lam[1:], cv['Uc_k'][1:] / scale, 'o-', ms=3,
                               label=f'$y^+$={yp:g}')
            flag = '  ALIASED -- lower --lag' if cv['wrap_risk'] >= 1 else ''
            plus = f' (Uc+={cv["Uc"] / scale:.2f})' if case['utau'] else ''
            print(f'  [{case["id"]} y+={yp:6g}] Uc={cv["Uc"]:.4f}{plus}'
                  f'  wrap_risk={cv["wrap_risk"]:.2f}{flag}')
            store[f'{case["id"]}/y{yp:g}/Uc_lam+'] = lam
            store[f'{case["id"]}/y{yp:g}/Uc_k'] = cv['Uc_k']
        axs[0, i].set(xlabel=r'$\lambda_x^+$',
                      ylabel='$U_c^+$' if case['utau'] else '$U_c$',
                      title=f'{case["id"]} (lag={args.lag})')
        axs[0, i].grid(alpha=0.3)
        axs[0, i].legend(fontsize=7, ncol=2)
    return fig, 'Uc'


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

    cases = [load_case(args, cid) for cid in args.case_ids]
    for case in cases:
        if args.field not in case['data']['names']:
            raise SystemExit(f'case {case["id"]} has no field {args.field!r}; '
                             f'available: {case["data"]["names"]}')

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    store, runners = {}, {'spectra': do_spectra, 'map': do_map,
                          'correlations': do_correlations,
                          'ycorr': do_ycorr, 'frequency': do_frequency,
                          'convection': do_convection}

    print()
    for name in what:
        print(f'--- {name} ---')
        fig, tag = runners[name](cases, args, store)
        fig.tight_layout()
        out = outdir / f'{args.head}_{tag}_{args.field}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  -> {out}')

    if args.save:
        np.savez(args.save, **store)
        print(f'\ncurves -> {args.save}  ({len(store)} arrays)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
