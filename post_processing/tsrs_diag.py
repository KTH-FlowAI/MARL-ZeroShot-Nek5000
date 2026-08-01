#!/usr/bin/env python3
"""
Is this record good enough to draw conclusions from?

Four questions, one section each:

  record      is it long enough, and is it stationary?
  scaling     what is the real u_tau, and what do the nominal y+ labels mean?
  profiles    do the fluctuation profiles and the stress balance make sense?
  floor       below which wavelength is the spectrum numerical rather than flow?
  error       how big is the standard error of each value on the spectral map?

The figures live in tsrs_spectra.py; nothing here draws unless --plot is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from postlib import tsrs, tsrs_spectra as sp               # noqa: E402
from postlib import tsrs_case as tc                        # noqa: E402
from tsrs_spectra import map_planes, map_rows, map_label    # noqa: E402

SECTIONS = ('record', 'scaling', 'profiles', 'floor', 'error')


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('case_ids', type=str, nargs='+')
    ap.add_argument('--runs-dir', default='../runs')
    ap.add_argument('--env', type=int, default=0)
    ap.add_argument('--field', default='u')
    ap.add_argument('--planes', default=None,
                    help='comma-separated nominal y+ values (default: all)')
    ap.add_argument('--t0', type=float, default=None)
    ap.add_argument('--t1', type=float, default=None)
    ap.add_argument('--utau', type=float, default=None,
                    help='*reference* u_tau; default from current_conf.yml')
    ap.add_argument('--scale', default='actual', choices=tc.SCALES)
    ap.add_argument('--what', default='all',
                    help=f'comma-separated subset of {",".join(SECTIONS)}')
    ap.add_argument('--blocks', type=int, default=6,
                    help='blocks the record is split into for the error '
                         'analysis (default: %(default)s).  Keep each block '
                         'at least one eddy turnover long')
    ap.add_argument('--quantity', default='uu', choices=('uu', 'uv'),
                    help='which map the error analysis is of (default: %(default)s)')
    ap.add_argument('--map-dir', default='z', choices=('z', 'x'))
    ap.add_argument('--modes', type=int, default=8,
                    help='how many low modes to tabulate (default: %(default)s)')
    return ap.parse_args()


# --------------------------------------------------------------------- record

def sec_record(case, args):
    """Box, resolution and whether the record is long enough and settled."""
    d, Lx, Lz = case['data'], case['Lx'], case['Lz']
    retau, utau = tc.length(case), case['scale']['utau']
    nx, nz, nt = d['x'].size, d['z'].size, d['t'].size
    T = float(d['t'][-1] - d['t'][0])

    print('\n--- record ---')
    print(f'  periodic view {d["fld"].shape}   endpoints duplicated: {case["dup"]}')
    print(f'  box    Lx={Lx:.3f}h Lz={Lz:.3f}h   '
          f'Lx+={Lx * retau:.0f} Lz+={Lz * retau:.0f}')
    print(f'  sample dx+={Lx / nx * retau:.1f} dz+={Lz / nz * retau:.1f}   '
          f'{nx // 2} x-modes, {nz // 2} z-modes')
    print(f'         largest lam_z+={Lz * retau:.0f} (the box), '
          f'smallest {Lz * retau / (nz // 2):.1f}')
    print(f'  time   t=[{d["t"][0]:.1f}, {d["t"][-1]:.1f}]  n={nt}  '
          f'dt={case["dt"]:.4g}')
    if utau:
        print(f'         T={T:.0f} time units = {T * utau:.1f} eddy turnovers '
              f'(h/u_tau = {1 / utau:.1f})')

    st = sp.stationarity(d, args.field, min(1, d['fld'].shape[0] - 1))
    flag = '   <-- NOT stationary' if abs(st['drift_rel_rms']) > 0.5 else '   ok'
    print(f'  drift of <{args.field}>_xz: {st["drift"]:+.4g} '
          f'({st["drift_pct"]:+.1f}% of the mean, '
          f'{st["drift_rel_rms"]:+.2f} x rms){flag}')
    if abs(st['drift_rel_rms']) > 0.5:
        print('         the record is still evolving; trim it with --t0 before '
              'reading any spectrum from it')


# -------------------------------------------------------------------- scaling

def sec_scaling(case, args):
    """The two u_tau, and what the stored y+ labels really mean."""
    est, s = case['est'], case['scale']
    print('\n--- scaling ---')
    if not est:
        print('  no sublayer estimate available; everything is in '
              f'{s["kind"]} units' + ('' if s['utau'] else ' (code units)'))
        return
    print(f'  u_tau  reference {est["utau_ref"]:.5f}   '
          f'actual {est["utau"]:.5f}   nu={est["nu"]:.6g}')
    print(f'  Re_tau reference {case["retau"]:.0f}   '
          f'actual {est["retau_actual"]:.0f}')
    print(f'  drag reduction {est["drag_reduction_pct"]:+.1f}%   '
          f'(per-plane u_tau: ' +
          ', '.join(f'{v:.5f}' for v in est['per_plane']) + ')')
    print(f'  drawing in {s["kind"]} units: lengths x'
          f'{s["retau"] / case["retau"]:.3f}, energies x'
          f'{(case["utau"] / s["utau"]) ** 2:.3f} versus reference')
    nom = np.asarray(case['data']['yplus'], dtype=float)
    print(f'    y+ nominal {[f"{v:g}" for v in nom]}')
    print(f'    y+ in use  {[tc.ylab(v) for v in tc.yplus(case)]}')


# ------------------------------------------------------------------- profiles

def sec_profiles(case, args):
    """
    Fluctuation profiles and the stress balance.

    In a channel the total stress -<u'v'> + nu dU/dy, in units of u_tau^2, is
    1 - y/h.  Checking it is the sharpest test that u_tau is right: it uses the
    whole profile rather than the two sublayer planes the estimate came from.
    """
    d, s = case['data'], case['scale']
    if not s['utau']:
        print('\n--- profiles ---\n  no u_tau, nothing to normalise by')
        return
    utau, u2, retau = s['utau'], s['vel2'], s['retau']
    nu = case['utau'] / case['retau'] if case['utau'] else np.nan
    y, yp = np.asarray(d['y']), tc.yplus(case)
    U = np.array([tsrs.field(d, 'u', iy=i).mean() for i in range(y.size)])

    print('\n--- profiles ---')
    print(f'{"y+":>8} {"y/h":>6} {"U+":>7} {"u_rms+":>7} {"v_rms+":>7} '
          f'{"-<uv>+":>7} {"visc+":>6} {"total+":>7} {"1-y/h":>7}')
    for i in range(y.size):
        up = sp.fluctuation(d, 'u', i)
        vp = sp.fluctuation(d, 'v', i) if 'v' in d['names'] else None
        uv = float((up * vp).mean()) if vp is not None else np.nan
        j0, j1 = max(0, i - 1), min(y.size - 1, i + 1)
        dUdy = (U[j1] - U[j0]) / (y[j1] - y[j0])
        visc = nu * dUdy / u2
        print(f'{yp[i]:8.2f} {y[i]:6.3f} {U[i] / utau:7.3f} '
              f'{np.sqrt((up ** 2).mean()) / utau:7.3f} '
              + (f'{np.sqrt((vp ** 2).mean()) / utau:7.3f} ' if vp is not None
                 else f'{"-":>7} ')
              + f'{-uv / u2:7.3f} {visc:6.3f} {-uv / u2 + visc:7.3f} '
                f'{1 - y[i]:7.3f}')
    print('  total+ should follow 1-y/h; a systematic offset means u_tau is off '
          '(or the two walls are not equivalent)')


# ---------------------------------------------------------------------- floor

def sec_floor(case, args):
    """
    Where the spectrum stops being flow and starts being discretisation.

    A resolved spectrum decays monotonically.  Where it flattens -- or turns
    back up -- the sampling has run past what the solver holds, and those modes
    are the interpolant, not turbulence.
    """
    d = case['data']
    L = case['Lz'] if args.map_dir == 'z' else case['Lx']
    axis = 1 if args.map_dir == 'z' else 0
    n = (d['z'] if args.map_dir == 'z' else d['x']).size
    retau, u2 = tc.length(case), tc.norm(case)

    idx = tc.plane_indices(d, args.planes)
    # the plane with the most energy is where the floor is easiest to see
    iy = max(idx, key=lambda k: float((sp.fluctuation(d, args.field, k) ** 2).mean()))
    E = sp.spectrum_1d(sp.fluctuation(d, args.field, iy), axis, L)['E'] / u2
    pk = E[1:].max()

    print(f'\n--- floor ({args.map_dir}, y+={tc.ylab(tc.yplus(case)[iy])}) ---')
    print(f'{"m":>4} {"lam+":>8} {"E/Epk":>10}   ratio to previous')
    prev, floor_at = None, None
    for m in range(1, n // 2 + 1):
        r = E[m] / prev if prev else np.nan
        # a decayed spectrum keeps falling; the first place it stops is the floor
        if prev is not None and E[m] / pk < 1e-2 and r > 0.95 and floor_at is None:
            floor_at = m
        if m <= 5 or m % max(1, (n // 2) // 12) == 0 or m > n // 2 - 6:
            mark = '  <-- floor' if m == floor_at else ''
            print(f'{m:4d} {L / m * retau:8.1f} {E[m] / pk:10.6f}   '
                  f'{r:5.3f}{mark}' if prev else
                  f'{m:4d} {L / m * retau:8.1f} {E[m] / pk:10.6f}')
        prev = E[m]
    if floor_at:
        print(f'  spectrum stops decaying at m={floor_at}, '
              f'lam+={L / floor_at * retau:.1f} -- trust nothing below that')
    else:
        print('  no floor detected: the spectrum decays over the whole range')


# ---------------------------------------------------------------------- error

def sec_error(case, args):
    """
    Standard error of every map value, from the scatter of block means.

    The largest scales have the fewest independent realisations -- only Lz/lam
    of them fit across the span, and they decorrelate slowest -- so a map value
    out there can be a factor of two from its converged value on a short record.
    This is a *lower* bound: blocks shorter than the integral time are not
    independent, so keep each block at least one eddy turnover long.
    """
    if args.blocks < 2:
        print('\n--- error ---\n  --blocks < 2, skipped')
        return
    d, q = case['data'], args.quantity
    usable = map_planes(case, args.planes, verbose=False)
    lam, yp, M = map_rows(case, args.field, q, args.map_dir, usable)

    t = d['t']
    edges = np.linspace(t[0], t[-1], args.blocks + 1)
    blocks = [map_rows(case, args.field, q, args.map_dir, usable,
                       data=sp.trim_time(d, lo, hi))[2]
              for lo, hi in zip(edges[:-1], edges[1:])]
    sem = np.asarray(blocks).std(axis=0, ddof=1) / np.sqrt(args.blocks)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.abs(sem) / np.abs(M)
    rel[~np.isfinite(rel)] = np.inf

    span = (t[-1] - t[0]) / args.blocks
    utau = case['scale']['utau']
    print(f'\n--- error ({map_label(args.field, q)}, {args.blocks} blocks of '
          f'{span:.1f} time units'
          + (f' = {span * utau:.2f} turnovers' if utau else '') + ') ---')
    if utau and span * utau < 1.0:
        print('  WARNING: blocks are shorter than one eddy turnover, so they '
              'are not independent and this underestimates the error')

    pk = np.unravel_index(np.argmax(np.abs(M)), M.shape)
    print(f'  peak {M[pk]:+.3f} +/- {sem[pk]:.3f} ({100 * rel[pk]:.0f}%) at '
          f'lam+={lam[pk[1]]:.0f}, y+={tc.ylab(yp[pk[0]])}')
    for p in (50, 90, 99):
        print(f'  {p}th percentile of the relative error: '
              f'{100 * np.nanpercentile(rel[np.isfinite(rel)], p):.0f}%')
    for thr in (0.10, 0.25, 0.50):
        print(f'  {100 * np.mean(rel > thr):5.1f}% of the map is above '
              f'{100 * thr:.0f}% relative error')

    # the error is strongly scale-dependent, so show it against lambda
    print(f'\n  relative error against wavelength (worst plane per column)')
    print(f'{"lam+":>8} {"max|kPhi|":>10} {"rel err":>9}')
    for j in range(len(lam) - 1, -1, -1):
        i = int(np.argmax(np.abs(M[:, j])))
        if j % max(1, len(lam) // 12) == 0 or j >= len(lam) - 2:
            print(f'{lam[j]:8.0f} {M[i, j]:10.3f} {100 * rel[i, j]:8.0f}%')


# ------------------------------------------------------------------------ main

def main():
    args = parse_args()
    what = SECTIONS if args.what == 'all' else tuple(
        w.strip() for w in args.what.split(',') if w.strip())
    unknown = set(what) - set(SECTIONS)
    if unknown:
        raise SystemExit(f'unknown section {sorted(unknown)}; '
                         f'choose from {SECTIONS}')

    runners = {'record': sec_record, 'scaling': sec_scaling,
               'profiles': sec_profiles, 'floor': sec_floor,
               'error': sec_error}
    for cid in args.case_ids:
        case = tc.load_case(cid, runs_dir=args.runs_dir, env=args.env,
                            field=args.field, t0=args.t0, t1=args.t1,
                            utau=args.utau, scale=args.scale)
        for name in what:
            runners[name](case, args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
