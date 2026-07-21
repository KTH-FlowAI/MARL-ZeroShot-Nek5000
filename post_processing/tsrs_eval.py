#!/usr/bin/env python3
"""
Plot instantaneous fluctuation fields from stitched tsrs data (see tsrs_stitch.py).

Coordinates and times come from the .npz, never from hardcoded domain sizes, so
comparing cases with different boxes or sampling rates is safe.  Snapshots are
selected by physical time, not by array index.

Note on plane 0 (the wall): userbc sets ux = uz = 0 there and drives uy with the
control signal, so at y = 0 only v (the actuation) and p carry signal; u and w
are roundoff.  --fields defaults accordingly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from postlib import tsrs                             # noqa: E402

WALL_DEFAULT_FIELDS = 'v,p'
OFFWALL_DEFAULT_FIELDS = 'u,v,w'


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # case ids are directory names: numeric ('302000') or not ('oc-mc-dr')
    ap.add_argument('case_ids', type=str, nargs='+',
                    help='cases to compare, one row per case')
    ap.add_argument('--runs-dir', default='../runs')
    ap.add_argument('--env', type=int, default=0,
                    help='which env_* directory to plot (default: %(default)s)')
    plane = ap.add_mutually_exclusive_group()
    plane.add_argument('--plane', type=int, default=None,
                       help='wall-normal plane index; 0 = wall (default: the '
                            'first plane above the wall)')
    plane.add_argument('--yplus', type=float, default=None,
                       help='select the plane nearest this y+ instead of by index')
    ap.add_argument('--fields', default=None,
                    help='comma-separated fields (default: '
                         f'{WALL_DEFAULT_FIELDS} at the wall, '
                         f'{OFFWALL_DEFAULT_FIELDS} above it)')
    ap.add_argument('--time', type=float, default=None,
                    help='physical time to plot; default is the last common time')
    ap.add_argument('--head', default='TCF',
                    help='figure filename prefix (default: %(default)s)')
    ap.add_argument('--outdir', default='Figs')
    return ap.parse_args()


def find_npz(runs_dir, case_id, env):
    base = Path(runs_dir) / str(case_id)
    data_path = base / 'eval' if (base / 'eval').is_dir() else base
    env_dirs = sorted(d for d in data_path.iterdir()
                      if d.is_dir() and 'env' in d.name)
    if not env_dirs:
        raise FileNotFoundError(f'no env_* directories under {data_path}')
    if env >= len(env_dirs):
        raise IndexError(f'case {case_id} has {len(env_dirs)} environments, '
                         f'--env {env} is out of range')
    hits = sorted(env_dirs[env].glob('tsrs_*.npz'))
    if not hits:
        raise FileNotFoundError(
            f'no stitched .npz in {env_dirs[env]} -- run tsrs_stitch.py first')
    return hits[0]


def fmt_yplus(yp):
    """Compact 'y+ = [0, 15, 30]' rendering, without numpy scalar noise."""
    return 'unknown' if yp is None else '[' + ', '.join(f'{v:g}' for v in yp) + ']'


def plane_yplus(data):
    """y+ of each plane, falling back to y*Re_tau for older .npz files."""
    if 'yplus' in data:
        return np.asarray(data['yplus'], dtype=float)
    if 'retau' in data:
        return np.asarray(data['y'], dtype=float) * float(data['retau'])
    return None


def select_plane(data, plane, yplus):
    """Resolve --plane / --yplus to a plane index, or None if unspecified."""
    nplane = data['fld'].shape[0]
    yp = plane_yplus(data)

    if yplus is not None:
        if yp is None:
            raise SystemExit('--yplus needs y+ metadata; this .npz predates it, '
                             'restitch or use --plane')
        idx = int(np.abs(yp - yplus).argmin())
        if abs(yp[idx] - yplus) > 1e-6:
            print(f'note: no y+ = {yplus:g} plane in this data '
                  f'({fmt_yplus(yp)}); using the nearest, y+ = {yp[idx]:g}')
        return idx

    if plane is None:
        # default: the first plane above the wall, or the only plane there is
        plane = 1 if nplane > 1 else 0
    if not 0 <= plane < nplane:
        raise SystemExit(f'--plane {plane} out of range; the data has {nplane} '
                         f'plane(s), y+ = {fmt_yplus(yp)}')
    return plane


def plot_plane(ax, data, name, iy, it):
    """Contour the fluctuation of one field on one plane at one snapshot."""
    fld = tsrs.field(data, name, iy=iy)                  # (nx, nz, nt)
    fluc = fld[:, :, it] - fld.mean(axis=-1)             # about the time mean

    # data is (x, z); contourf wants (rows, cols) = (z, x)
    X, Z = np.meshgrid(data['x'], data['z'], indexing='xy')
    lim = np.abs(fluc).max()
    # explicit symmetric levels: passing vmin/vmax alongside an integer `levels`
    # does NOT centre the map, matplotlib just re-derives levels from the data
    # range, so a fluctuation that happens to be one-signed reads as if it
    # straddled zero
    levels = np.linspace(-lim, lim, 101) if lim > 0 else None
    contour = ax.contourf(X, Z, fluc.T, levels=levels, cmap='turbo')
    ax.set_aspect('equal')
    # printing the amplitude makes an all-roundoff panel (e.g. u at the wall)
    # obvious instead of rendering it as a vivid turbo field
    ax.set_title(f"{name}'   max|·| = {lim:.1e}", fontsize=10)
    return contour


def main():
    args = parse_args()
    plt.rc('font', family='serif', size=12)
    plt.rc('axes', labelsize=12, linewidth=1)
    plt.rc(('xtick', 'ytick'), labelsize=12)

    cases = []
    for case_id in args.case_ids:
        path = find_npz(args.runs_dir, case_id, args.env)
        data = tsrs.load(path)
        print(f'case {case_id}: {path}')
        print(f'  fld{data["fld"].shape}  fields = {data["names"]}')
        print(f'  planes y+ = {fmt_yplus(plane_yplus(data))}')
        print(f'  t = [{data["t"][0]:.4f}, {data["t"][-1]:.4f}]'
              f' ({data["t"].size} snapshots)')
        cases.append((case_id, data))

    # resolve the plane once, on the first case, and reuse it for the rest so
    # every row of the figure is the same physical height
    iy = select_plane(cases[0][1], args.plane, args.yplus)
    yp = plane_yplus(cases[0][1])
    yp_sel = None if yp is None else yp[iy]
    is_wall = (yp_sel == 0.0) if yp_sel is not None else (iy == 0)

    fields = args.fields or (WALL_DEFAULT_FIELDS if is_wall
                             else OFFWALL_DEFAULT_FIELDS)
    fields = [f.strip() for f in fields.split(',') if f.strip()]

    # one physical time for every case, so the rows are actually comparable
    t_target = args.time
    if t_target is None:
        t_target = min(data['t'][-1] for _, data in cases)
    label = f'y+ = {yp_sel:g}' if yp_sel is not None else f'plane {iy}'
    print(f'-> {label} (y = {cases[0][1]["y"][iy]:.5f}), '
          f'fields {fields}, t = {t_target:.4f}')

    nrow, ncol = len(cases), len(fields)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                            sharex=True, sharey=True, squeeze=False)

    for i, (case_id, data) in enumerate(cases):
        it = int(np.abs(data['t'] - t_target).argmin())
        drift = abs(data['t'][it] - t_target)
        if drift > np.diff(data['t']).mean():
            print(f'  warning: case {case_id} nearest snapshot is '
                  f't = {data["t"][it]:.4f} ({drift:.4f} away from target)')

        # match on y+ rather than index: cases may have been stitched with
        # different plane sets, and index 1 need not be the same height
        iy_case, yp_case = iy, plane_yplus(data)
        if yp_sel is not None and yp_case is not None:
            iy_case = int(np.abs(yp_case - yp_sel).argmin())
            if abs(yp_case[iy_case] - yp_sel) > 1e-6:
                print(f'  warning: case {case_id} has no y+ = {yp_sel:g} plane; '
                      f'using the nearest, y+ = {yp_case[iy_case]:g}')
        elif iy >= data['fld'].shape[0]:
            raise SystemExit(f'case {case_id} has only {data["fld"].shape[0]} '
                             f'plane(s); cannot plot plane {iy}')

        for j, name in enumerate(fields):
            if name not in data['names']:
                raise SystemExit(f'case {case_id} has no field {name!r}; '
                                 f'available: {data["names"]}')
            contour = plot_plane(axs[i, j], data, name, iy_case, it)
            fig.colorbar(contour, ax=axs[i, j], orientation='vertical',
                         pad=0.02, shrink=1)
        axs[i, 0].set_ylabel(f'case {case_id}\nz')
    for ax in axs[-1, :]:
        ax.set_xlabel('x')

    fig.suptitle(label, y=0.995)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ttag = f'{t_target:.2f}'.replace('.', 'p')
    ptag = f'yp{yp_sel:g}' if yp_sel is not None else f'plane{iy}'
    out = outdir / f'{args.head}_tsrs_{ptag}_t{ttag}.png'
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
