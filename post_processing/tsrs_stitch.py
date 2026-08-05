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
    ap.add_argument('--force', action='store_true',
                    help='with --overwrite, restitch even when it would shrink '
                         'the record.  The raw pts files are routinely cleaned '
                         'up after stitching, so an existing .npz is often the '
                         'only copy of the snapshots in it')
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
    """
    Return (data_path, config) for a case, preferring the eval/ subdirectory.

    The config is looked for in the data directory and then in the case root
    above it: a solo-nek run stages current_conf.yml next to runs/<case>/
    rather than inside eval/, and only the data lives in eval/.
    """
    base = Path(runs_dir) / str(case_id)
    data_path = base / 'eval' if (base / 'eval').is_dir() else base
    for config_path in (data_path / 'current_conf.yml',
                        data_path.parent / 'current_conf.yml'):
        if config_path.exists():
            with open(config_path) as fh:
                return data_path, yaml.safe_load(fh)
    raise FileNotFoundError(f'no current_conf.yml under {data_path} '
                            f'or {data_path.parent}')


def sampling_grid(config, y_planes=None):
    """
    The (y, x, z) the solver was told to sample, and the plane heights in y+.

    `y_planes` overrides `simulation.y_planes` from the config.  That override
    is not cosmetic: a config edited after the run (an emptied `y_planes`, a
    changed Nx) no longer describes the `pts` files on disk, and stitching with
    it raises `grid mismatch` from tsrs.to_grid().  Passing the planes the run
    actually used is the way to stitch such a case without editing the config
    back.
    """
    sim = config['simulation']
    retau = sim['retau']
    # y_planes (optional) overrides the historical [0, y_sensing] pair
    yplus = y_planes if y_planes is not None else (sim.get('y_planes')
                                                   or sim['y_sensing'])
    y, x, z = channel_grid(Ret=retau, yplus=yplus,
                           Lx=sim['Lx'], Lz=sim['Lz'],
                           Nx=sim['Nx'], Nz=sim['Nz'], lx1=sim['lx1'])
    return y, x, z, resolve_yplus(yplus), retau


def record_nt(npz_path):
    """Snapshot count of an existing stitched record, without reading it."""
    import zipfile
    try:
        with zipfile.ZipFile(npz_path) as zf, zf.open('t.npy') as fh:
            return int(np.load(fh).size)
    except (OSError, KeyError, zipfile.BadZipFile, ValueError):
        return None


def incoming_nt(pts):
    """Snapshots the `pts` files would contribute, from their headers alone."""
    try:
        return sum(tsrs.read_header(p)['ntsnap'] for p in pts)
    except (OSError, ValueError):
        return None


def stitch_env(env_dir, case_name, grid, fields=('u', 'v', 'w', 'p'),
               dtype='float32', overwrite=False, force=False, verbose=True):
    """
    Stitch one environment's `pts` files into `tsrs_<case_name>.npz` beside them.

    `grid` is what sampling_grid() returns.  Returns (path, status) with status
    one of 'written', 'exists', 'no-pts', 'shrink', 'failed' -- the caller
    decides what a failure costs, so one stale environment never stops the ones
    after it.

    Overwriting is refused when it would *shrink* the record.  Raw `pts` files
    get cleaned up once they have been stitched, so an existing record is
    routinely the only copy of snapshots no longer on disk; overwriting it with
    what the surviving files can supply destroys them for good.  `force=True`
    proceeds anyway.
    """
    env_dir = Path(env_dir)
    y, x, z, yplus, retau = grid
    out = env_dir / f'tsrs_{case_name}.npz'

    if out.exists() and not overwrite:
        if verbose:
            print(f'{env_dir.name}: {out.name} exists, skipping '
                  f'(--overwrite / overwrite=True to redo)')
        return out, 'exists'

    pts = sorted(env_dir.glob(f'pts{case_name}0.f*'))
    if not pts:
        if verbose:
            print(f'{env_dir.name}: no pts files')
        return out, 'no-pts'

    if out.exists() and not force:
        have, coming = record_nt(out), incoming_nt(pts)
        if have is not None and coming is not None and coming < have:
            print(f'{env_dir.name}: REFUSING to overwrite {out.name} -- it '
                  f'holds {have} snapshots and the {len(pts)} pts file(s) here '
                  f'supply only {coming}.\n'
                  f'  The raw files this record was built from are gone, so '
                  f'this would destroy {have - coming} snapshots that exist '
                  f'nowhere else.\n'
                  f'  Pass --force / force=True if that is really what you want.')
            return out, 'shrink'

    if verbose:
        print(f'{env_dir.name}: {len(pts)} files')
    try:
        data = tsrs.stitch([str(p) for p in pts], (len(y), len(x), len(z)),
                           y, x, z, fields=list(fields),
                           dtype=np.dtype(dtype), verbose=verbose)
    except (ValueError, NotImplementedError) as exc:
        if verbose:
            print(f'  SKIPPED: {exc}')
        return out, 'failed'

    # keep the wall units with the data so plots can label planes by y+
    data['yplus'] = np.asarray(yplus)
    data['retau'] = np.asarray(retau)
    np.savez(out, **data)
    if verbose:
        print(f'  -> {out}  fld{data["fld"].shape}  '
              f'nt = {data["t"].size}, '
              f't = [{data["t"][0]:.4f}, {data["t"][-1]:.4f}], '
              f'{out.stat().st_size / 1e6:.1f} MB')
    return out, 'written'


def stitch_case(case_id, runs_dir='../runs', envs=None, fields=('u', 'v', 'w', 'p'),
                dtype='float32', overwrite=False, force=False, y_planes=None,
                verbose=True):
    """
    Stitch a whole case.  The importable form of this script -- see main().

    `envs` selects environments by directory name (`['env_001']`); None takes
    all of them.  Returns {env_name: (path, status)}.
    """
    data_path, config = resolve_case(runs_dir, case_id)
    case_name = config['simulation']['CASENAME']
    grid = sampling_grid(config, y_planes)
    y, x, z, yplus, retau = grid

    all_envs = sorted(d for d in Path(data_path).iterdir()
                      if d.is_dir() and 'env' in d.name)
    if not all_envs:
        raise SystemExit(f'no env_* directories under {data_path}')
    env_dirs = ([d for d in all_envs if d.name in set(envs)] if envs is not None
                else all_envs)
    if not env_dirs:
        raise SystemExit(f'{envs} matches none of the environments under '
                         f'{data_path}: {", ".join(d.name for d in all_envs)}')

    if verbose:
        print(f'Case {case_id} ({case_name}) in {data_path}')
        print(f'  Re_tau = {retau}, planes y+ = {yplus} -> y = {y}')
        print(f'  grid (ny, nx, nz) = {(len(y), len(x), len(z))}, '
              f'Lx = {config["simulation"]["Lx"]}, '
              f'Lz = {config["simulation"]["Lz"]}')
        print(f'  fields = {list(fields)} as {dtype}')
        print(f'  environments = {", ".join(d.name for d in env_dirs)} '
              f'(of {len(all_envs)})')
        print('-' * 32)

    return {d.name: stitch_env(d, case_name, grid, fields=fields, dtype=dtype,
                               overwrite=overwrite, force=force, verbose=verbose)
            for d in env_dirs}


def main():
    args = parse_args()
    fields = [f.strip() for f in args.fields.split(',') if f.strip()]
    unknown = set(fields) - set(tsrs.FIELD_NAMES_3D)
    if unknown:
        raise SystemExit(f'unknown field(s): {sorted(unknown)}')

    data_path, _ = resolve_case(args.runs_dir, args.case_id)
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

    results = stitch_case(args.case_id, runs_dir=args.runs_dir,
                          envs=[d.name for d in env_dirs], fields=fields,
                          dtype=args.dtype, overwrite=args.overwrite,
                          force=args.force)

    status = [s for _, s in results.values()]
    failed = [name for name, (_, s) in results.items()
              if s in ('failed', 'shrink')]
    print('-' * 32)
    print(f'Stitched {status.count("written")}, '
          f'skipped {status.count("exists") + status.count("no-pts")}, '
          f'failed {len(failed)} of {len(results)} environments')
    if failed:
        print(f'Failed: {", ".join(failed)}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
