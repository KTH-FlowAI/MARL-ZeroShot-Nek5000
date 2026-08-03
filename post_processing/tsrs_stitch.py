#!/usr/bin/env python3
"""
Stitch the tsrs point time-series of a run into one .npz per environment.

The wall-normal / streamwise / spanwise ordering is not inferred from the data:
it comes from writer_int_pos.channel_grid(), the same function that generated
the int_pos file the solver sampled.  The MPI point shuffle is undone with the
global point ids stored in the file.  See postlib/tsrs.py.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, REPO_ROOT / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lib.writer_int_pos import channel_grid, resolve_yplus   # noqa: E402
from postlib import tsrs                             # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case_id', type=str, nargs='?', default=302000)
    ap.add_argument('--runs-dir', default='../runs',
                    help='directory holding <case_id>/ (default: %(default)s)')
    ap.add_argument('--fields', default='u,v,w,p',
                    help='comma-separated subset of '
                         f'{",".join(tsrs.FIELD_NAMES_3D)} (default: %(default)s)')
    ap.add_argument('--dtype', default='float32', choices=('float32', 'float64'),
                    help='storage precision (default: %(default)s)')
    ap.add_argument('--overwrite', action='store_true',
                    help='restitch environments that already have a .npz')
    ap.add_argument('--env-start', type=int, default=None, metavar='N',
                    help='first environment to stitch, by the number in its '
                         'directory name: --env-start 3 starts at env_003 '
                         '(default: the lowest one present)')
    ap.add_argument('--nenv', type=int, default=None, metavar='N',
                    help='stitch at most N environments from --env-start '
                         '(default: all of them)')
    args = ap.parse_args()
    if args.nenv is not None and args.nenv < 1:
        ap.error('--nenv must be at least 1')
    if args.env_start is not None and args.env_start < 0:
        ap.error('--env-start must not be negative')
    return args


def env_number(name):
    """The number in an env directory name: env_003 -> 3, else None."""
    match = re.search(r'env_?(\d+)', name)
    return int(match.group(1)) if match else None


def select_envs(env_dirs, env_start, nenv):
    """
    The subset of environments to stitch.

    Selecting on the number in the directory name rather than on position keeps
    `--env-start 3` meaning env_003 even when an earlier environment is missing
    or was never written.
    """
    if env_start is None and nenv is None:
        return env_dirs
    if env_start is not None:
        env_dirs = [d for d in env_dirs
                    if (n := env_number(d.name)) is not None and n >= env_start]
    return env_dirs[:nenv] if nenv is not None else env_dirs


def resolve_case(runs_dir, case_id):
    """Return (data_path, config) for a case, preferring the eval/ subdirectory."""
    base = Path(runs_dir) / str(case_id)
    data_path = base / 'eval' if (base / 'eval').is_dir() else base
    config_path = data_path / 'current_conf.yml'
    if not config_path.exists():
        raise FileNotFoundError(f'no current_conf.yml under {data_path}')
    with open(config_path) as fh:
        return data_path, yaml.safe_load(fh)


def main():
    args = parse_args()
    fields = [f.strip() for f in args.fields.split(',') if f.strip()]
    unknown = set(fields) - set(tsrs.FIELD_NAMES_3D)
    if unknown:
        raise SystemExit(f'unknown field(s): {sorted(unknown)}')

    data_path, config = resolve_case(args.runs_dir, args.case_id)
    sim = config['simulation']
    case_name = sim['CASENAME']

    # the grid the solver was told to sample -- not guessed from the data.
    # y_planes (optional) overrides the historical [0, y_sensing] pair.
    retau = sim['retau']
    yplus = sim.get('y_planes') or sim['y_sensing']
    y, x, z = channel_grid(Ret=retau, yplus=yplus,
                           Lx=sim['Lx'], Lz=sim['Lz'],
                           Nx=sim['Nx'], Nz=sim['Nz'], lx1=sim['lx1'])
    shape = (len(y), len(x), len(z))

    all_envs = sorted(d for d in data_path.iterdir()
                      if d.is_dir() and 'env' in d.name)
    if not all_envs:
        raise SystemExit(f'no env_* directories under {data_path}')

    env_dirs = select_envs(all_envs, args.env_start, args.nenv)
    if not env_dirs:
        raise SystemExit(
            f'--env-start {args.env_start} / --nenv {args.nenv} selects none of '
            f'the {len(all_envs)} environments under {data_path}: '
            f'{", ".join(d.name for d in all_envs)}')

    print(f'Case {args.case_id} ({case_name}) in {data_path}')
    print(f'  Re_tau = {retau}, planes y+ = {resolve_yplus(yplus)} -> y = {y}')
    print(f'  grid (ny, nx, nz) = {shape}, Lx = {sim["Lx"]}, Lz = {sim["Lz"]}')
    print(f'  fields = {fields} as {args.dtype}')
    print(f'  environments = {env_dirs[0].name} .. {env_dirs[-1].name} '
          f'({len(env_dirs)} of {len(all_envs)})')
    print('-' * 32)

    written, skipped, failed = 0, 0, []
    for env_dir in env_dirs:
        out = env_dir / f'tsrs_{case_name}.npz'
        if out.exists() and not args.overwrite:
            print(f'{env_dir.name}: {out.name} exists, skipping (--overwrite to redo)')
            skipped += 1
            continue

        pts = sorted(env_dir.glob(f'pts{case_name}0.f*'))
        if not pts:
            print(f'{env_dir.name}: no pts files')
            skipped += 1
            continue

        print(f'{env_dir.name}: {len(pts)} files')
        # one unusable environment (e.g. stale files from an earlier config)
        # must not cost us the environments that follow it
        try:
            data = tsrs.stitch([str(p) for p in pts], shape, y, x, z,
                               fields=fields, dtype=np.dtype(args.dtype))
        except (ValueError, NotImplementedError) as exc:
            print(f'  SKIPPED: {exc}')
            failed.append(env_dir.name)
            continue

        # keep the wall units with the data so plots can label planes by y+
        data['yplus'] = np.asarray(resolve_yplus(yplus))
        data['retau'] = np.asarray(retau)
        np.savez(out, **data)
        nt = data['t'].size
        print(f'  -> {out}  fld{data["fld"].shape}  '
              f'nt = {nt}, t = [{data["t"][0]:.4f}, {data["t"][-1]:.4f}], '
              f'{out.stat().st_size / 1e6:.1f} MB')
        written += 1

    print('-' * 32)
    print(f'Stitched {written}, skipped {skipped}, failed {len(failed)} '
          f'of {len(env_dirs)} environments')
    if failed:
        print(f'Failed: {", ".join(failed)}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
