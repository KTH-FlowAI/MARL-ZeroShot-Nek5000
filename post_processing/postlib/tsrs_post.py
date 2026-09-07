"""
Notebook-facing post-processing of stitched tsrs records.

`tsrs_cases.ipynb` is the thin layer on top of this; `tsrs_spectra.py` and
`tsrs_diag.py` remain the command-line paths.  Everything here works on one
environment of one case (`env_001` by default) -- the other environments are
independent samples of the same policy and are not what these figures are for.

Two phases, with `data/results/` as the boundary between them
-------------------------------------------------------------
A stitched record is up to several GB, so it is read **once**::

    tsrs_post.process(cases)          # phase A: stitch, extract, spectra, archive

writes, under ``data/results/<solver_case>/<run_name>/``::

    tsrs/<env>/  snap_<plane>_<field>.npy    (nx, nz, nt) float32 fluctuation
                 mean_<plane>_<field>.npy    (nx, nz)     the time mean removed
                 axes.npz, snap_meta.yml
                 tsrs_<solver_case>.npz      the stitched record, archived
    spectra/     one .npz per spectrum, named as `tsrs_spectra.py --results`
                 writes them, so the two are interchangeable

and every figure afterwards reads *those*::

    tsrs_post.plot_snapshot_grid(cases, yplus=15)

The snapshots are plain ``.npy`` rather than a bundle so that phase B can open
them with ``mmap_mode='r'``: drawing one frame of a 3529-snapshot record costs
one page read instead of 81 MB.  Re-running a figure cell is therefore free,
and only `process()` is expensive.

Fluctuations are `f' = f - <f>_t(x, z)`, the Reynolds decomposition about the
time mean of each point (`postlib.tsrs_spectra.fluctuation`, mean='time').
The mean is stored beside the fluctuation, so the raw field is recoverable
exactly as ``snap + mean[:, :, None]``.
"""

from __future__ import annotations

import datetime
import gc
import shutil
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import yaml

from . import (results as res, tsrs, tsrs_case as tc, tsrs_spectra as sp,
               tsrs_timeseries as tsp)

#: Default environment.  One case, one policy, one sample -- see the note above.
DEFAULT_ENV = 'env_001'

#: The two planes the design asks for: the wall and the sensing plane.
DEFAULT_PLANES = (0.0, 15.0)

#: userbc holds ux = uz = 0 at the wall and drives uy with the control signal,
#: so at y = 0 only v (the actuation) and p carry signal; u and w are roundoff.
WALL_FIELDS = ('v', 'p')
OFFWALL_FIELDS = ('u', 'v', 'w', 'p')

SEQ_CMAP = 'jet'          # positive quantity: sequential
DIV_CMAP = 'RdBu_r'       # signed quantity: diverging about zero
SNAP_CMAP = 'turbo'


# --------------------------------------------------------------- case lookup

def _data_path(case_path):
    """Evaluation artefacts live under eval/ when it exists (see evaluate.py)."""
    case_path = Path(case_path)
    return case_path / 'eval' if (case_path / 'eval').is_dir() else case_path


def _read_config(data_path):
    """current_conf.yml from the data directory, or the case root above it."""
    for cand in (Path(data_path) / 'current_conf.yml',
                 Path(data_path).parent / 'current_conf.yml'):
        if cand.exists():
            with open(cand) as fh:
                return yaml.safe_load(fh) or {}, cand
    return {}, None


def resolve(case, label=None, style=None, runs_dir='../runs', env=DEFAULT_ENV,
            y_planes=None, results_root=None):
    """
    Everything about one case that can be read off the disk, in one dict.

    Environments are addressed by **directory name** (`env_001`), not by
    position in a sorted list the way `tsrs_case.find_npz` does it: with
    `env_005` missing, position 4 is `env_006`, and a figure captioned env_005
    would then be of something else.
    """
    case_path = Path(runs_dir) / str(case)
    if not case_path.is_dir():
        raise SystemExit(f'no such run: {case_path}')
    data_path = _data_path(case_path)
    env_dir = data_path / env
    if not env_dir.is_dir():
        have = sorted(d.name for d in data_path.iterdir()
                      if d.is_dir() and 'env' in d.name)
        raise SystemExit(f'case {case} has no {env} under {data_path}; '
                         f'have {have or "no environments at all"}')

    config, conf_path = _read_config(data_path)
    sim = config.get('simulation', {})
    solver_case = sim.get('CASENAME') or res.solver_case_from_npz(
        next(iter(sorted(env_dir.glob('tsrs_*.npz'))), env_dir / 'tsrs_unknown'))
    root = res.case_root(case, solver_case, root=results_root)

    return {
        'id': str(case), 'label': label or str(case), 'style': style,
        'runs_dir': Path(runs_dir), 'case_path': case_path,
        'data_path': data_path, 'env': env, 'env_dir': env_dir,
        'config': config, 'config_path': conf_path, 'sim': sim,
        'solver_case': str(solver_case),
        'npz': env_dir / f'tsrs_{solver_case}.npz',
        'root': root, 'tsrs_dir': root / 'tsrs' / env,
        'spectra_dir': root / 'spectra', 'figs_dir': root / 'figs',
        'y_planes': y_planes,
        'utau_ref': (config.get('runner') or {}).get('u_tau'),
    }


def resolve_all(case_table, runs_dir='../runs', env=DEFAULT_ENV,
                results_root=None, verbose=True):
    """
    Resolve a notebook case table into {case_id: entry}, order preserved.

    Each row is a dict: ``{'case': ..., 'label': ..., 'style': ...}`` plus any
    of `env` and `y_planes` to override the defaults for that case alone.
    """
    out = {}
    for row in case_table:
        row = dict(row)
        cid = row.pop('case')
        row.setdefault('env', env)
        entry = resolve(cid, runs_dir=runs_dir, results_root=results_root, **row)
        out[cid] = entry
        if verbose:
            exists = 'stitched' if entry['npz'].exists() else 'NOT stitched'
            print(f"{cid:<28} {entry['env']}  case={entry['solver_case']}  "
                  f"{exists}  -> {entry['root']}")
    return out


# ------------------------------------------------------------------ planes

def yp_tag(yplus):
    """Filename-safe tag for a plane: 15.0 -> 'yp15', 4.5 -> 'yp4p5'."""
    return 'yp' + f'{float(yplus):g}'.replace('.', 'p').replace('-', 'm')


def stored_yplus(data):
    """Nominal y+ of every stored plane."""
    return np.asarray(data['yplus'], dtype=float)


def resolve_planes(data, want, atol=1e-6, verbose=True, case=''):
    """
    Match requested y+ values against the planes a record actually holds.

    Returns [(iy, yp_stored, tag)] for the ones that are there.  A miss is
    **dropped with a note**, never silently replaced by the nearest plane:
    `oc-mc-dr` starts at y+ = 5, and plotting that as "the wall" would be a
    quietly wrong figure rather than a missing one.
    """
    yp = stored_yplus(data)
    found = []
    for w in want:
        i = int(np.abs(yp - float(w)).argmin())
        if abs(yp[i] - float(w)) <= atol:
            found.append((i, float(yp[i]), yp_tag(yp[i])))
        elif verbose:
            print(f'  [{case}] no y+ = {float(w):g} plane '
                  f'(has {", ".join(f"{v:g}" for v in yp)}) -- skipped')
    return found


# --------------------------------------------------------------- phase A

def stitch(entry, fields=('u', 'v', 'w', 'p'), dtype='float32',
           overwrite=False, force=False, verbose=True):
    """
    Stitch this case's `pts` files, unless the `.npz` is already there.

    `overwrite` here is **destructive and separate** from the one on
    `process()`, deliberately.  Raw `pts` files get cleaned up once they have
    been stitched, so an existing record is routinely the only copy of the
    snapshots in it -- restitching then replaces months of run with whatever
    the surviving files can supply.  Recomputing snapshots and spectra costs
    only time; this can cost data, so it is never swept along with them.
    stitch_env() refuses a restitch that would shrink the record unless
    `force=True`.

    `y_planes` on the entry is passed through: a config whose `y_planes` was
    edited after the run no longer describes the files on disk, and stitching
    with it fails with `grid mismatch`.  Setting it on the case row is how such
    a case is stitched without editing the config back.
    """
    if entry['npz'].exists() and not overwrite:
        # nothing to do, and saying so needs neither the config nor the grid --
        # which a case whose config has since moved or changed may no longer
        # have.  Only an actual stitch depends on them.
        if verbose:
            print(f"{entry['env']}: {entry['npz'].name} exists, skipping "
                  f"(overwrite=True to redo)")
        return entry['npz'], 'exists'

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import tsrs_stitch                                     # noqa: E402

    result = tsrs_stitch.stitch_case(
        entry['id'], runs_dir=entry['runs_dir'], envs=[entry['env']],
        fields=fields, dtype=dtype, overwrite=overwrite, force=force,
        y_planes=entry.get('y_planes'), verbose=verbose)
    path, status = result[entry['env']]
    if status == 'failed':
        raise SystemExit(
            f"{entry['id']}/{entry['env']}: stitching failed -- see the grid "
            f"mismatch above.  The config's y_planes no longer match the pts "
            f"files; pass the planes the run used, e.g. "
            f"y_planes=[0, 2, 5, ...] on the case row.")
    if status == 'shrink':
        raise SystemExit(f"{entry['id']}/{entry['env']}: restitch refused, see "
                         f"above.  Nothing was written.")
    return path, status


def extract_snapshots(entry, planes=DEFAULT_PLANES, fields=None, data=None,
                      overwrite=False, verbose=True):
    """
    Fluctuation time series of every requested (plane, field), one `.npy` each.

    `fields` is a list applied to every plane, or a {y+: list} mapping; None
    takes WALL_FIELDS at y+ = 0 and OFFWALL_FIELDS above it.  The per-point
    time mean that was removed is written beside each one, so nothing is lost.
    """
    out_dir = entry['tsrs_dir']
    meta_path = out_dir / 'snap_meta.yml'
    if meta_path.exists() and not overwrite:
        if verbose:
            print(f'  snapshots: {meta_path.parent} exists, skipping '
                  f'(overwrite=True to redo)')
        return read_snap_meta(entry)

    if data is None:
        data = tsrs.load(entry['npz'])
    out_dir.mkdir(parents=True, exist_ok=True)

    found = resolve_planes(data, planes, verbose=verbose, case=entry['id'])
    if not found:
        raise SystemExit(f"{entry['id']}: none of the planes {list(planes)} are "
                         f"in the record (has "
                         f"{', '.join(f'{v:g}' for v in stored_yplus(data))})")

    def fields_for(yp):
        if isinstance(fields, dict):
            return list(fields.get(yp, fields.get(float(yp), OFFWALL_FIELDS)))
        if fields is not None:
            return list(fields)
        return list(WALL_FIELDS if yp == 0.0 else OFFWALL_FIELDS)

    plane_meta = {}
    for iy, yp, tag in found:
        want = [f for f in fields_for(yp) if f in data['names']]
        for name in want:
            f = tsrs.field(data, name, iy=iy).astype(np.float32)   # (nx, nz, nt)
            # a field the solver never filled comes through as NaN, and a NaN
            # propagates into every mean and every colour limit it touches.
            # Say it here, where the cause is still visible, rather than
            # letting a figure four cells later be blank
            finite = np.isfinite(f)
            if not finite.all() and verbose:
                share = 100.0 * (1.0 - finite.mean())
                print(f"  [{entry['id']}] WARNING: {name!r} at y+ = {yp:g} is "
                      f"{share:.1f}% non-finite in the source record"
                      + (' -- the whole field' if share > 99.99 else ''))
            mean = f.mean(axis=-1)
            np.save(out_dir / f'snap_{tag}_{name}.npy', f - mean[:, :, None])
            np.save(out_dir / f'mean_{tag}_{name}.npy', mean)
            del f
        plane_meta[tag] = {'yplus': float(yp), 'y': float(data['y'][iy]),
                           'index': int(iy), 'fields': want}
        if verbose:
            nx, nz, nt = data['fld'].shape[1], data['fld'].shape[2], data['t'].size
            mb = 4 * nx * nz * nt * len(want) / 1e6
            print(f"  snapshots: y+ = {yp:g} ({tag})  {want}  "
                  f"({nx}, {nz}, {nt})  {mb:.0f} MB")

    np.savez(out_dir / 'axes.npz', x=data['x'], z=data['z'], t=data['t'],
             y=data['y'], yplus=stored_yplus(data),
             retau=np.asarray(data['retau']))

    meta = {
        'case': entry['id'], 'env': entry['env'],
        'solver_case': entry['solver_case'], 'source': str(entry['npz']),
        'mean': 'time', 'dtype': 'float32',
        'nx': int(data['x'].size), 'nz': int(data['z'].size),
        'nt': int(data['t'].size),
        't0': float(data['t'][0]), 't1': float(data['t'][-1]),
        'retau': float(np.asarray(data['retau'])),
        'utau_ref': (float(entry['utau_ref']) if entry['utau_ref'] else None),
        'Lx': float(entry['sim'].get('Lx', np.nan)),
        'Lz': float(entry['sim'].get('Lz', np.nan)),
        'planes': plane_meta,
        'written': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(meta_path, 'w') as fh:
        yaml.safe_dump(meta, fh, sort_keys=False, default_flow_style=False)
    return meta


def archive_stitched(entry, overwrite=False, verbose=True):
    """
    Copy the stitched `.npz` into the case results folder.

    A copy, not a link: this tree is what survives the run directory being
    cleared, and on HPC storage the duplication is affordable.  The manifest
    records size and mtime, so a re-run can tell whether the archive is still
    the file that produced everything beside it.
    """
    src = entry['npz']
    if not src.exists():
        raise SystemExit(f'{entry["id"]}: nothing to archive, {src} is missing')
    dst = entry['tsrs_dir'] / src.name
    man_path = entry['tsrs_dir'] / 'stitch.yml'
    stat = src.stat()
    record = {'source': str(src), 'name': src.name,
              'bytes': int(stat.st_size),
              'mtime': datetime.datetime.fromtimestamp(
                  stat.st_mtime).isoformat(timespec='seconds')}

    if dst.exists() and man_path.exists() and not overwrite:
        with open(man_path) as fh:
            old = yaml.safe_load(fh) or {}
        if (old.get('bytes') == record['bytes']
                and old.get('mtime') == record['mtime']
                and dst.stat().st_size == stat.st_size):
            if verbose:
                print(f'  archive: {dst.name} is current, skipping')
            return dst
        if verbose:
            print(f'  archive: {dst.name} is stale (source changed), replacing')

    dst.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f'  archive: copying {stat.st_size / 1e9:.2f} GB -> {dst}')
    shutil.copy2(src, dst)
    record['archived'] = datetime.datetime.now().isoformat(timespec='seconds')
    with open(man_path, 'w') as fh:
        yaml.safe_dump(record, fh, sort_keys=False, default_flow_style=False)
    return dst


def process(cases, planes=DEFAULT_PLANES, fields=None, field='u',
            quantities=('uu', 'uv'), map_dir='z', scale='reference',
            wall_velocity=None, do_stitch=True, do_snapshots=True,
            do_spectra=True, do_archive=True,
            overwrite=False, overwrite_stitch=False, verbose=True):
    """
    Phase A for every case: stitch, extract, spectra, archive.

    One case at a time, and the record is dropped before the next one starts --
    holding three of them would be 5 GB of nothing but raw input.  Everything
    is skipped when its output is already there, so this is cheap to re-run.

    `overwrite` recomputes the snapshots, the spectra and the archive copy.  It
    deliberately does **not** restitch: that is `overwrite_stitch`, which is
    destructive -- see stitch().  Recomputing costs time, restitching can cost
    data, and one flag must not mean both.

    `scale` defaults to 'reference' rather than the 'actual' of
    `tsrs_spectra.py`: these figures put cases side by side, and only the
    common reference u_tau puts them on one energy and one y+ axis.  Both sets
    can live in the archive at once -- the units are in the file name.

    ``wall_velocity`` optionally enables the Brown-style correlation between
    reconstructed streamwise wall shear and streamwise velocity away from the
    wall.  It is a dictionary passed to :func:`compute_wall_velocity_correlation`,
    for example ``{'planes': (2, 5, 15, 30), 'max_lag_time': 20.0}``.  Leaving
    it ``None`` preserves the inexpensive historical processing path.
    """
    entries = cases.values() if isinstance(cases, dict) else cases
    done = {}
    for entry in entries:
        print(f"\n=== {entry['id']} ({entry['label']}) / {entry['env']} ===")
        if do_stitch:
            stitch(entry, overwrite=overwrite_stitch, verbose=verbose)

        needed = do_snapshots and not (
            (entry['tsrs_dir'] / 'snap_meta.yml').exists() and not overwrite)
        needed |= do_spectra and not (
            _spectra_present(entry, field, quantities, map_dir, scale)
            and not overwrite)
        needed |= wall_velocity is not None and not (
            _wall_velocity_present(entry, wall_velocity, scale) and not overwrite)

        data = None
        if needed:
            if verbose:
                print(f"  reading {entry['npz']} "
                      f"({entry['npz'].stat().st_size / 1e9:.2f} GB)")
            data = tsrs.load(entry['npz'])

        if do_snapshots:
            extract_snapshots(entry, planes=planes, fields=fields, data=data,
                              overwrite=overwrite, verbose=verbose)
        if do_spectra:
            compute_spectra(entry, field=field, quantities=quantities,
                            map_dir=map_dir, scale=scale, data=data,
                            overwrite=overwrite, verbose=verbose)
        if wall_velocity is not None:
            options = dict(wall_velocity)
            options.setdefault('scale', scale)
            compute_wall_velocity_correlation(entry, data=data,
                                              overwrite=overwrite,
                                              verbose=verbose, **options)
        del data
        gc.collect()

        if do_archive:
            archive_stitched(entry, overwrite=overwrite, verbose=verbose)
        done[entry['id']] = entry
    return done


# ------------------------------------------------------------------ spectra

def map_label(field, quantity):
    """
    Subscript for the mapped quantity, following the field.

    'uv' is the shorthand for the co-spectrum with v; with field='w' it is
    really the wv co-spectrum, and the label has to say so.
    """
    return f'{field}{field}' if quantity == 'uu' else f'{field}v'


def parse_quantity(spec):
    """'-uv' -> ('uv', -1.0).  The sign is a plotting choice, not stored data."""
    spec = str(spec)
    if spec.startswith('-'):
        return spec[1:], -1.0
    return spec, 1.0


def map_planes(case, planes=None, verbose=True):
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


def map_levels(q, vmax, n, vmin=None):
    """Contour levels and colour map for one quantity."""
    if q == 'uv' and (vmin is None or vmin < 0):   # signed: diverging about 0
        return np.linspace(-vmax, vmax, n), DIV_CMAP, 'both'
    # energy is positive: sequential from 0, and never < 0, so no lower arrow
    return np.linspace(0, vmax, n), SEQ_CMAP, 'max'


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


def scale_suffix(case):
    """The tag `tsrs_spectra.py` puts on a file to name the wall units in it."""
    return 'uact' if case['scale']['kind'] == 'actual' else 'uref'


def _spectra_present(entry, field, quantities, map_dir, scale):
    """
    True when every dataset compute_spectra() would write is already there.

    The wall units are part of the file name, so a file in the other units does
    not count -- switching `scale` must recompute, not silently reuse.
    'actual' also accepts 'uref', because a case with no plane in the viscous
    sublayer legitimately falls back to reference units (see tc.set_scale).
    """
    ok = ('uref',) if scale == 'reference' else ('uact', 'uref')
    want = [f'map{map_dir}_{map_label(field, parse_quantity(q)[0])}'
            for q in quantities]
    want += [f'corr_{d}_{field}{field}' for d in ('z', 'x')]
    for name in want:
        if not any((entry['spectra_dir'] / f'{entry["id"]}_{name}_{s}.npz').exists()
                   for s in ok):
            return False
    return True


def load_case_scaled(entry, field='u', scale='actual', utau=None, data=None,
                     verbose=True):
    """The `tsrs_case` view of this entry: periodic grid, resolved wall units."""
    return tc.load_case(entry['id'], runs_dir=entry['runs_dir'], field=field,
                        utau=utau if utau is not None else entry['utau_ref'],
                        scale=scale, verbose=verbose, data=data,
                        npz_path=entry['npz'], data_path=entry['data_path'])


def compute_spectra(entry, field='u', quantities=('uu', 'uv'), map_dir='z',
                    planes=None, scale='actual', utau=None, data=None,
                    overwrite=False, verbose=True):
    """
    The two spectral products the design asks for, saved as self-contained .npz.

    * `map<dir>_<qty>`: premultiplied spectrum over (lambda+, y+), every plane
      above the wall.
    * `corr_z_<ff>` / `corr_x_<ff>`: two-point correlations, every plane.

    File names and contents match `tsrs_spectra.py --results` exactly, so the
    notebook and the command line fill one archive rather than two.
    """
    if _spectra_present(entry, field, quantities, map_dir, scale) and not overwrite:
        if verbose:
            print(f'  spectra: already in {entry["spectra_dir"]}, skipping '
                  f'(overwrite=True to redo)')
        return {}

    if data is None:
        data = tsrs.load(entry['npz'])
    case = load_case_scaled(entry, field=field, scale=scale, utau=utau,
                            data=data, verbose=verbose)
    suffix = scale_suffix(case)
    written = {}

    usable = map_planes(case, planes, verbose=verbose)
    for q in quantities:
        qty, _ = parse_quantity(q)
        lam, yp, M = map_rows(case, field, qty, map_dir, usable)
        pk = np.unravel_index(np.argmax(np.abs(M)), M.shape)
        name = f'map{map_dir}_{map_label(field, qty)}'
        arrays = {**_plane_axes(case, usable),
                  'lam': lam / tc.length(case), 'lam_plus': lam,
                  'kPhi': M * tc.norm(case), 'kPhi_plus': M,
                  'peak': np.array([lam[pk[1]], yp[pk[0]], M[pk]])}
        written[name] = _save(entry, case, name, suffix, arrays,
                              kind='map', direction=map_dir,
                              quantity=map_label(field, qty), field=field,
                              xlabel=f'lambda_{map_dir}+', ylabel='y+',
                              zlabel=f'k_{map_dir} Phi_{map_label(field, qty)}',
                              peak_note='lam_plus, yplus, kPhi_plus at the peak')
        if verbose:
            print(f'  map {map_label(field, qty)}: {M.shape[0]} planes x '
                  f'{M.shape[1]} wavelengths, peak {M[pk]:.3g} at '
                  f'lam_{map_dir}+={lam[pk[1]]:.0f}, y+={tc.ylab(yp[pk[0]])}')

    idx = tc.plane_indices(case['data'], planes)
    retau = tc.length(case)
    cols = {'z': [], 'x': []}
    for iy in idx:
        f = sp.fluctuation(case['data'], field, iy)
        cols['z'].append(sp.correlation_1d(f, 1, case['Lz']))
        cols['x'].append(sp.correlation_1d(f, 0, case['Lx']))
    for sub in ('z', 'x'):
        cs = cols[sub]
        name = f'corr_{sub}_{field}{field}'
        arrays = {**_plane_axes(case, idx),
                  'sep': cs[0]['sep'], 'sep_plus': cs[0]['sep'] * retau,
                  'rho': np.asarray([c['rho'] for c in cs], dtype=float),
                  'R': np.asarray([c['R'] for c in cs], dtype=float),
                  'variance': np.array([c['var'] for c in cs])}
        written[name] = _save(entry, case, name, suffix, arrays,
                              kind='correlation', direction=sub,
                              quantity=f'{field}{field}', field=field,
                              xlabel=f'delta {sub}+',
                              ylabel=f'rho_{field}{field}')
    if verbose:
        print(f'  correlations: {len(idx)} planes, '
              f'half-period {cs[0]["sep"][-1] * retau:.0f} wall units')

    res.write_meta(entry['root'], solver_case=entry['solver_case'],
                   run_name=entry['id'],
                   spectra={'field': field, 'scale': scale,
                            'files': sorted(p.name for p in
                                            entry['spectra_dir'].glob('*.npz'))})
    return written


def _plane_axes(case, idx):
    """The three ways a plane is labelled: physical, chosen units, nominal."""
    return {
        'y': np.asarray(case['data']['y'], dtype=float)[idx],
        'yplus': tc.yplus(case)[idx],
        'yplus_nominal': np.asarray(case['data']['yplus'], dtype=float)[idx],
    }


def _save(entry, case, name, suffix, arrays, **meta):
    target = entry['spectra_dir'] / f'{entry["id"]}_{name}_{suffix}.npz'
    return res.save_dataset(target, arrays, res.scalars_from_case(case), meta)


# ----------------------------------------------------- wall shear / velocity

def _wall_velocity_name(velocity='u', tau_component='u'):
    """Archive name for a signed wall-shear/velocity correlation."""
    direction = _tau_direction(tau_component)
    return f'wallcorr_tau{direction}w_{velocity}'


def _tau_direction(tau_component):
    """Cartesian direction of a tangential velocity component at the wall."""
    direction = {'u': 'x', 'w': 'z'}.get(str(tau_component))
    if direction is None:
        raise ValueError("tau_component must be 'u' (streamwise) or 'w' "
                         f"(spanwise), got {tau_component!r}")
    return direction


def _wall_velocity_present(entry, options, scale):
    """Whether this analysis has already produced its requested unit set."""
    options = dict(options or {})
    name = _wall_velocity_name(options.get('velocity', 'u'),
                               options.get('tau_component', 'u'))
    requested = options.get('scale', scale)
    suffixes = ('uref',) if requested == 'reference' else ('uact', 'uref')
    return any((entry['spectra_dir'] /
                f'{entry["id"]}_{name}_{suffix}.npz').exists()
               for suffix in suffixes)


def _kinematic_viscosity(entry, case):
    """Viscosity from the Nek config, with a documented wall-unit fallback."""
    raw = entry.get('sim', {}).get('viscosity')
    if raw is not None:
        raw = float(raw)
        if raw == 0.0 or not np.isfinite(raw):
            raise ValueError(f'{entry["id"]}: invalid simulation.viscosity={raw!r}')
        # Nek's channel inputs use a negative reciprocal Reynolds number.
        return 1.0 / abs(raw) if raw < 0.0 else raw
    if case['utau'] is not None and case['retau'] > 0:
        return float(case['utau']) / float(case['retau'])
    raise ValueError(f'{entry["id"]}: no simulation.viscosity and no reference '
                     'u_tau/Re_tau from which to derive it')


def compute_wall_velocity_correlation(entry, planes=None, velocity='u',
                                      tau_component='u', wall_y=0.0, nwall=3,
                                      max_lag=None, max_lag_time=None,
                                      x_shift=0, z_shift=0, scale='reference',
                                      data=None, overwrite=False, strict=False,
                                      verbose=True):
    """Archive Brown-style wall-shear/streamwise-velocity correlations.

    The wall-shear signal is reconstructed from the no-slip tangential
    velocity with a one-sided derivative.  A record without the wall plus the
    requested number of near-wall planes cannot support that reconstruction;
    by default such a case is reported and skipped, which keeps a multi-case
    notebook honest rather than substituting a different observable.  Set
    ``strict=True`` when an absent wall correlation should stop a batch.

    ``planes`` are nominal y+ requests and must exist exactly (within the
    normal ``resolve_planes`` tolerance).  ``None`` analyses every off-wall
    stored plane.  The result contains both the primary ``nwall`` derivative
    and a two-point correlation diagnostic when ``nwall > 2``.
    """
    name = _wall_velocity_name(velocity, tau_component)
    tau_direction = _tau_direction(tau_component)
    if _wall_velocity_present(entry, {
            'velocity': velocity, 'tau_component': tau_component, 'scale': scale},
            scale) and not overwrite:
        if verbose:
            print(f'  wall correlation: {name} already in {entry["spectra_dir"]}, '
                  'skipping (overwrite=True to redo)')
        return None

    if data is None:
        data = tsrs.load(entry['npz'])
    try:
        case = load_case_scaled(entry, field=velocity, scale=scale, data=data,
                                verbose=False)
        d = case['data']
        nu = _kinematic_viscosity(entry, case)
        if planes is None:
            indices = [int(i) for i, y in enumerate(d['y'])
                       if not np.isclose(y, wall_y, atol=1e-12)]
        else:
            found = resolve_planes(d, planes, verbose=verbose, case=entry['id'])
            indices = [int(i) for i, _, _ in found]
        if not indices:
            raise ValueError('none of the requested velocity planes is present')

        corr = tsp.wall_velocity_correlation(
            d, viscosity=nu, velocity=velocity, plane_indices=indices,
            tau_component=tau_component, wall_y=wall_y, nwall=nwall,
            max_lag=max_lag, max_lag_time=max_lag_time,
            x_shift=x_shift, z_shift=z_shift, sensitivity=True)
    except ValueError as exc:
        message = f'  [{entry["id"]}] wall correlation skipped: {exc}'
        if strict:
            raise ValueError(message) from exc
        if verbose:
            print(message)
        return None

    shear = corr['wall_shear']
    time_factor = ((case['scale']['utau'] ** 2 / nu)
                   if case['scale']['utau'] is not None else np.nan)
    nx, nz = d['fld'].shape[1:3]
    dx, dz = case['Lx'] / nx, case['Lz'] / nz
    yplus = tc.yplus(case)[corr['indices']]
    arrays = {
        'lag': corr['lag'],
        'lag_time': corr['lag_time'],
        'lag_plus': corr['lag_time'] * time_factor,
        'rho': corr['rho'],
        'n_time_pairs': corr['n_time_pairs'],
        'rms_tau': corr['rms_tau'],
        'rms_velocity': corr['rms_velocity'],
        'peak_index': corr['peak_index'],
        'peak_lag': corr['peak_lag'],
        'peak_lag_time': corr['peak_lag_time'],
        'peak_lag_plus': corr['peak_lag_time'] * time_factor,
        'peak_rho': corr['peak_rho'],
        'plane_index': corr['indices'],
        'y': corr['y'],
        'yplus': yplus,
        'yplus_nominal': corr['yplus_nominal'],
        'wall_index': shear['wall_index'],
        'wall_y': shear['wall_y'],
        'stencil_index': shear['indices'],
        'stencil_y': shear['y'],
        'stencil_yplus_nominal': shear['yplus'],
        'stencil_weights': shear['weights'],
        'viscosity': nu,
        'nwall': int(nwall),
        'x_shift': int(x_shift),
        'z_shift': int(z_shift),
        'x_offset': float(x_shift) * dx,
        'z_offset': float(z_shift) * dz,
    }
    if 'rho_low_order' in corr:
        arrays.update({
            'rho_low_order': corr['rho_low_order'],
            'low_order_nwall': corr['low_order_nwall'],
            'low_order_y': corr['low_order_y'],
            'low_order_weights': corr['low_order_weights'],
        })
    target = _save(
        entry, case, name, scale_suffix(case), arrays,
        kind='time correlation', quantity=f'tau_{tau_direction}w,{velocity}',
        field=velocity, tau_component=tau_component,
        mean='time at each x,z', time_pairs='overlapping only; not periodic',
        derivative=shear['method'],
        xlabel='T+', ylabel=f'R_tau{tau_direction}w,{velocity}')
    res.write_meta(entry['root'], solver_case=entry['solver_case'],
                   run_name=entry['id'],
                   wall_velocity_correlation={
                       'velocity': velocity, 'tau_component': tau_component,
                       'scale': scale, 'file': target.name,
                       'stencil': f'{nwall}-point one-sided derivative',
                   })
    if verbose:
        unit = '+' if np.isfinite(time_factor) else ''
        print(f'  wall correlation tau_{tau_direction}w,{velocity}: '
              f'{len(indices)} planes, |T| <= {abs(corr["lag_time"][-1]):.3g} '
              f'time units ({abs(arrays["lag_plus"][-1]):.0f} T{unit})')
    return target


def load_wall_velocity_correlation(entry, velocity='u', tau_component='u',
                                   scale=None, verbose=True):
    """Load one archived wall-shear/velocity correlation, or return ``None``."""
    name = _wall_velocity_name(velocity, tau_component)
    suffix = {'actual': 'uact', 'reference': 'uref', None: '*'}[scale]
    hits = sorted(entry['spectra_dir'].glob(f'{entry["id"]}_{name}_{suffix}.npz'))
    if not hits:
        if verbose:
            print(f'  [{entry["id"]}] no {name} ({scale or "any"} units); '
                  'the case may lack the wall/y+<=5 stencil')
        return None
    if len(hits) > 1 and verbose:
        print(f'  [{entry["id"]}] {name}: both wall unit sets on disk, '
              f'using {hits[0].name}')
    with np.load(hits[0], allow_pickle=False) as fh:
        return {k: fh[k] for k in fh.files}


def plot_wall_velocity_correlation(cases, yplus=(2.0, 5.0, 15.0, 30.0),
                                   velocity='u', tau_component='u',
                                   scale='reference', save=None, verbose=True):
    """Plot Brown-style ``tau_w``--velocity lag curves for every available case."""
    entries = _cases(cases)
    tau_direction = _tau_direction(tau_component)
    fig, axs = plt.subplots(1, len(entries), squeeze=False,
                            figsize=(5.0 * len(entries), 4.2))
    for j, entry in enumerate(entries):
        ax = axs[0, j]
        d = load_wall_velocity_correlation(entry, velocity=velocity,
                                           tau_component=tau_component,
                                           scale=scale, verbose=verbose)
        if d is None:
            ax.set_title(entry['label'], loc='left', fontsize=11)
            ax.text(0.5, 0.5, 'wall correlation unavailable', ha='center', va='center',
                    transform=ax.transAxes, color='0.35')
            ax.set_axis_off()
            continue
        yp = np.asarray(d['yplus'], dtype=float)
        requested = yp if yplus is None else np.asarray(yplus, dtype=float)
        color = plt.get_cmap('viridis', max(1, requested.size))
        x = np.asarray(d['lag_plus'], dtype=float)
        xlabel = r'$T^+$'
        if not np.isfinite(x).all():
            x = np.asarray(d['lag_time'], dtype=float)
            xlabel = r'$T$'
        plotted = 0
        for i, wanted in enumerate(requested):
            hit = np.flatnonzero(np.isclose(yp, wanted, rtol=0.0, atol=1e-6))
            if not hit.size:
                if verbose and yplus is not None:
                    print(f'  [{entry["id"]}] no archived wall correlation at '
                          f'y+ = {wanted:g}; skipped')
                continue
            k = int(hit[0])
            c = color(i)
            ax.plot(x, d['rho'][k], color=c, lw=1.8,
                    label=fr'$y^+={yp[k]:.4g}$')
            pk = int(d['peak_index'][k])
            ax.plot(x[pk], d['peak_rho'][k], 'o', color=c, ms=4)
            plotted += 1
        ax.axhline(0.0, color='k', lw=0.8, ls=(0, (4, 3)))
        ax.axvline(0.0, color='k', lw=0.8, ls=(0, (4, 3)))
        ax.set(xlabel=xlabel,
               ylabel=fr'$R_{{\tau_{{{tau_direction}w}},{velocity}}}$'
                      if j == 0 else None)
        ax.set_title(entry['label'], loc='left', fontsize=11)
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend(fontsize=8, title='velocity plane', title_fontsize=8)
    fig.tight_layout()
    _save_fig(fig, save, verbose)
    return fig


# --------------------------------------------------------------- phase B

def read_snap_meta(entry):
    """What extract_snapshots() wrote for this case."""
    path = entry['tsrs_dir'] / 'snap_meta.yml'
    if not path.exists():
        raise SystemExit(f'{entry["id"]}: no snapshots at {path.parent} -- '
                         f'run tsrs_post.process() first')
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_axes(entry):
    """x, z, t, y, yplus of the saved snapshots."""
    with np.load(entry['tsrs_dir'] / 'axes.npz') as fh:
        return {k: fh[k] for k in fh.files}


def plane_tag(entry, yplus, meta=None):
    """The tag of the saved plane nearest `yplus`, or None if it was not saved."""
    meta = meta or read_snap_meta(entry)
    planes = meta.get('planes') or {}
    if not planes:
        return None
    tag = min(planes, key=lambda k: abs(planes[k]['yplus'] - float(yplus)))
    return tag if abs(planes[tag]['yplus'] - float(yplus)) <= 1e-6 else None


def load_snapshot(entry, yplus, field, mmap=True):
    """
    One saved fluctuation series as (nx, nz, nt), memory-mapped by default.

    Returns None when that plane or that field was not extracted -- the wall
    plane does not exist in every case, and u at the wall is not saved.
    """
    meta = read_snap_meta(entry)
    tag = plane_tag(entry, yplus, meta)
    if tag is None:
        return None
    path = entry['tsrs_dir'] / f'snap_{tag}_{field}.npy'
    if not path.exists():
        return None
    return np.load(path, mmap_mode='r' if mmap else None)


def load_mean(entry, yplus, field):
    """The per-point time mean removed from that snapshot series, as (nx, nz)."""
    tag = plane_tag(entry, yplus)
    if tag is None:
        return None
    path = entry['tsrs_dir'] / f'mean_{tag}_{field}.npy'
    return np.load(path) if path.exists() else None


def load_spectrum(entry, name, scale=None, verbose=True):
    """
    One saved spectrum by name, e.g. 'mapz_uu' or 'corr_z_uu'.

    `scale` picks the wall units ('actual' / 'reference').  The units are in
    every number in the file, so when both are on disk and neither is asked
    for, say which one was taken rather than letting alphabetical order decide.
    """
    suffix = {'actual': 'uact', 'reference': 'uref', None: '*'}[scale]
    hits = sorted(entry['spectra_dir'].glob(f'{entry["id"]}_{name}_{suffix}.npz'))
    if not hits:
        raise SystemExit(f'{entry["id"]}: no {name} ({scale or "any"} units) in '
                         f'{entry["spectra_dir"]} -- run tsrs_post.process() '
                         f'with scale={scale!r} first')
    if len(hits) > 1 and verbose:
        print(f'  [{entry["id"]}] {name}: both wall unit sets on disk, '
              f'using {hits[0].name}')
    with np.load(hits[0], allow_pickle=False) as fh:
        return {k: fh[k] for k in fh.files}


def time_index(entry, t, axes=None):
    """Index of the snapshot nearest physical time `t`, and its actual time."""
    axes = axes or load_axes(entry)
    it = int(np.abs(axes['t'] - float(t)).argmin())
    return it, float(axes['t'][it])


def common_time(cases, t=None):
    """
    A physical time every case has: the last one they share, unless given.

    Different runs cover different windows -- 399-474 for the mc cases,
    476-716 for the lc one -- so a shared *index* would compare different
    instants of different flows.
    """
    entries = list(cases.values()) if isinstance(cases, dict) else list(cases)
    spans = [load_axes(e)['t'] for e in entries]
    lo, hi = max(s[0] for s in spans), min(s[-1] for s in spans)
    if t is not None:
        return float(t)
    if hi < lo:
        print(f'[WARN] the cases do not overlap in time '
              f'(latest start {lo:.1f} > earliest end {hi:.1f}); '
              f'each panel will use its own nearest snapshot')
        return float(min(s[-1] for s in spans))
    return float(hi)


# --------------------------------------------------------------- figures

def _cases(cases):
    return list(cases.values()) if isinstance(cases, dict) else list(cases)


def plot_snapshot_grid(cases, yplus=15.0, fields=None, t=None, clim='shared',
                       cmap=SNAP_CMAP, save=None, verbose=True):
    """
    One instant of the fluctuation field: rows are cases, columns are fields.

    `clim='shared'` gives every case in a column the same symmetric colour
    limit, which is the point of the figure: a controlled case is supposed to
    look *weaker* than the uncontrolled one, and per-panel limits normalise
    exactly that difference away.  `clim='panel'` restores the per-panel scale.
    """
    entries = _cases(cases)
    t_target = common_time(entries, t)

    if fields is None:
        fields = list(WALL_FIELDS if float(yplus) == 0.0 else OFFWALL_FIELDS)
    fields = list(fields)

    # gather first: the shared colour limit has to be known before anything is
    # drawn, and a missing plane must leave a labelled hole rather than shift
    # the grid
    panels = {}
    for e in entries:
        axes = load_axes(e)
        it, t_act = time_index(e, t_target, axes)
        for name in fields:
            arr = load_snapshot(e, yplus, name)
            panels[(e['id'], name)] = (
                None if arr is None else np.asarray(arr[:, :, it]),
                axes, t_act)

    # a NaN would poison the shared limit of a whole column, so the limit is
    # taken over the finite samples only
    lims = {}
    for name in fields:
        vals = [np.nanmax(np.abs(f)) for (f, _, _) in
                (panels[(e['id'], name)] for e in entries)
                if f is not None and np.isfinite(f).any()]
        lims[name] = max(vals) if vals else 0.0

    # panels are drawn to scale, so the figure has to carry the box aspect:
    # the channel is 3.3x longer than it is wide and square axes would waste
    # most of the page on margins
    boxes = [(float(ax['x'][-1] - ax['x'][0]), float(ax['z'][-1] - ax['z'][0]))
             for (_, ax, _) in panels.values() if ax is not None]
    aspect = (boxes[0][1] / boxes[0][0]) if boxes else 0.5
    # only share the axes when the cases are actually the same box: a minimal
    # channel forced onto a large box's axes shrinks into one corner and
    # becomes unreadable, and reading it is the point
    same_box = len({(round(w, 6), round(h, 6)) for w, h in boxes}) <= 1
    if not same_box:
        aspect = max(h / w for w, h in boxes)
    nrow, ncol = len(entries), len(fields)
    fig, axs = plt.subplots(nrow, ncol, squeeze=False,
                            sharex=same_box, sharey=same_box,
                            layout='constrained',
                            figsize=(4.6 * ncol, (4.0 * aspect + 0.9) * nrow))

    drawn = {}
    for i, e in enumerate(entries):
        for j, name in enumerate(fields):
            ax = axs[i, j]
            f, axes, t_act = panels[(e['id'], name)]
            if j == 0:
                ax.set_ylabel(f"{e['label']}\n$z$", fontsize=10)
            if i == nrow - 1:
                ax.set_xlabel('$x$')
            # a panel with nothing to draw still carries the case's box, so a
            # row of the grid keeps one geometry whatever is missing from it
            if axes is not None:
                ax.set_xlim(axes['x'][0], axes['x'][-1])
                ax.set_ylim(axes['z'][0], axes['z'][-1])

            if f is None:
                ax.text(0.5, 0.5, f"no $y^+$ = {float(yplus):g} plane\n"
                                  f"in this case",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=9, color='0.45')
                continue

            if not np.isfinite(f).any():
                ax.text(0.5, 0.5, f"${name}'$ is not finite\nin this record",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=9, color='0.45')
                ax.set_title(f"${name}'$   $t$={t_act:.1f}", fontsize=9)
                continue

            peak = float(np.nanmax(np.abs(f)))
            # v and w at the wall are held at zero by userbc unless the case
            # actuates: a flat panel under a 1e-13 colour scale reads as
            # structure, so say what it is instead of drawing it
            if peak <= 1e-12 * max(lims[name], 1e-30) or peak == 0.0:
                ax.text(0.5, 0.5, f"${name}'\\equiv 0$ at the wall\n"
                                  f"(max$|\\cdot|$ = {peak:.1e})",
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=9, color='0.45')
                ax.set_title(f"${name}'$   $t$={t_act:.1f}", fontsize=9)
                continue

            lim = lims[name] if clim == 'shared' else peak
            X, Z = np.meshgrid(axes['x'], axes['z'], indexing='xy')
            cf = ax.contourf(X, Z, f.T, levels=np.linspace(-lim, lim, 101),
                             cmap=cmap, extend='both')
            ax.set_aspect('equal')
            ax.set_title(f"${name}'$   $t$={t_act:.1f}   "
                         f"max$|\\cdot|$={peak:.1e}", fontsize=9)
            drawn.setdefault(j, []).append((cf, ax))

    # one colourbar per column when the column shares a scale, one per panel
    # when it does not -- a colourbar that does not apply to its neighbour
    # would be worse than none
    for j, items in drawn.items():
        if clim == 'shared':
            fig.colorbar(items[0][0], ax=[axs[i, j] for i in range(nrow)],
                         pad=0.02, shrink=0.85, label=f"${fields[j]}'$")
        else:
            for cf, ax in items:
                fig.colorbar(cf, ax=ax, pad=0.02, shrink=0.85)

    fig.suptitle(f'$y^+ = {float(yplus):g}$, fluctuation about the time mean'
                 + ('  (one colour scale per column)' if clim == 'shared' else ''),
                 fontsize=11)
    _save_fig(fig, save, verbose)
    return fig


def plot_map(cases, quantities=('uu', '-uv'), map_dir='z', field='u',
             scale='reference', nlevels=21, style='contour', save=None,
             verbose=True):
    """
    Premultiplied spectra over (lambda+, y+): rows are quantities, columns cases.

    One colour scale per row, so the columns are comparable -- which is the
    whole reason to draw them together.  That is also why `scale` defaults to
    'reference' here and not to the 'actual' of `tsrs_spectra.py`: in actual
    units every case is divided by its own u_tau and placed on its own y+ axis,
    so a 30% drag reduction is scaled out of the picture it should dominate.

    A leading '-' on a quantity flips its sign for the picture only: `-uv`
    makes the co-spectrum positive, the usual presentation, while the archived
    data stays sign-true.
    """
    entries = _cases(cases)
    nrow, ncol = len(quantities), len(entries)
    fig, axs = plt.subplots(nrow, ncol, squeeze=False,
                            figsize=(5.2 * ncol, 4.2 * nrow))

    for i, spec in enumerate(quantities):
        qty, sign = parse_quantity(spec)
        lbl = map_label(field, qty)
        loaded = [load_spectrum(e, f'map{map_dir}_{lbl}', scale, verbose)
                  for e in entries]
        mats = [(d['lam_plus'], d['yplus'], sign * d['kPhi_plus']) for d in loaded]

        vmax = max(np.abs(M).max() for _, _, M in mats)
        vmin = min(M.min() for _, _, M in mats)
        levels, cmap, extend = map_levels('uv' if vmin < -1e-12 * vmax else 'uu',
                                          vmax, nlevels, vmin)
        ylim = (min(yp.min() for _, yp, _ in mats),
                max(yp.max() for _, yp, _ in mats))
        xlim = (min(lam.min() for lam, _, _ in mats),
                max(lam.max() for lam, _, _ in mats))

        for j, (e, (lam, yp, M)) in enumerate(zip(entries, mats)):
            ax = axs[i, j]
            if style == 'mesh':
                # one cell per sample: the map then claims exactly the
                # resolution it has, which for the outer planes is very little
                cf = ax.pcolormesh(lam, yp, M, cmap=cmap, shading='nearest',
                                   norm=mcolors.BoundaryNorm(
                                       levels, plt.get_cmap(cmap).N, extend=extend))
            else:
                cf = ax.contourf(lam, yp, M, levels=levels, cmap=cmap,
                                 extend=extend)
                ax.contour(lam, yp, M, levels=levels[::4], colors='k',
                           linewidths=0.4, alpha=0.5)
            pk = np.unravel_index(np.argmax(np.abs(M)), M.shape)
            ax.plot(lam[pk[1]], yp[pk[0]], 'w*', ms=13, mec='k', mew=0.8,
                    label=f'peak {M[pk]:.3g} at $\\lambda^+$={lam[pk[1]]:.0f}, '
                          f'$y^+$={yp[pk[0]]:.4g}')
            log_axes(ax, xlim, ylim)
            sgn = '-' if sign < 0 else ''
            ax.set_xlabel(f'$\\lambda_{map_dir}^+$')
            ax.set_ylabel('$y^+$' if j == 0 else '')
            ax.set_title(f"{e['label']}: ${sgn}k_{map_dir}\\Phi_{{{lbl}}}^+$",
                         loc='left', fontsize=10)
            ax.legend(fontsize=7, loc='upper left')
            fig.colorbar(cf, ax=ax, pad=0.02)
    fig.tight_layout()
    _save_fig(fig, save, verbose)
    return fig


def plot_correlation(cases, yplus=15.0, field='u', directions=('z', 'x'),
                     extra_planes=(), scale='reference', save=None, verbose=True):
    """
    Two-point correlation, one subfigure per case, y+ = 15 in the foreground.

    `extra_planes` draws further heights faintly for context.  The separation
    axis stops at half the period, which is all a periodic box can say; where
    rho is still well above zero there, the structure is longer than the box
    and the curve is the box talking (see README_tsrs.md, box-size caveat).
    """
    entries = _cases(cases)
    fig, axs = plt.subplots(1, len(entries), squeeze=False,
                            figsize=(5.0 * len(entries), 4.2))
    ls = {'z': '-', 'x': '--'}

    for j, e in enumerate(entries):
        ax = axs[0, j]
        color = (e['style'] or {}).get('c', 'C0')
        for sub in directions:
            d = load_spectrum(e, f'corr_{sub}_{field}{field}', scale, verbose)
            yp = d['yplus']
            for w in extra_planes:
                k = int(np.abs(yp - float(w)).argmin())
                ax.plot(d['sep_plus'], d['rho'][k], ls[sub], color='0.75',
                        lw=1.0, zorder=1)
            k = int(np.abs(yp - float(yplus)).argmin())
            if abs(yp[k] - float(yplus)) > 1e-6 and verbose:
                print(f"  [{e['id']}] no y+ = {float(yplus):g} plane; "
                      f"using y+ = {yp[k]:.4g}")
            ax.plot(d['sep_plus'], d['rho'][k], ls[sub], color=color, lw=2.0,
                    marker=(e['style'] or {}).get('marker', 'o'), ms=4,
                    markevery=max(1, d['sep_plus'].size // 10), zorder=3,
                    label=f'$\\Delta {sub}^+$  ($y^+$={yp[k]:.4g})')
        ax.axhline(0.0, color='k', lw=0.8, ls=(0, (4, 3)))
        ax.set(xlabel=r'$\Delta^+$',
               ylabel=f'$\\rho_{{{field}{field}}}$' if j == 0 else None)
        ax.set_title(e['label'], loc='left', fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    _save_fig(fig, save, verbose)
    return fig


def animate(entry, yplus=15.0, field='u', nframes=200, t0=None, t1=None,
            fps=15, cmap=SNAP_CMAP, out=None, verbose=True):
    """
    A gif of one plane of one case, over a fixed colour scale.

    The scale is set once over the whole window rather than per frame: a
    per-frame scale makes a turbulent field pulse in and out of saturation and
    hides exactly the intensity changes the animation is for.  Frames are
    subsampled to `nframes` -- 3529 of them is a 100 MB gif no one opens.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    arr = load_snapshot(entry, yplus, field)
    if arr is None:
        raise SystemExit(f"{entry['id']}: no saved y+ = {float(yplus):g} / "
                         f"{field!r} snapshots")
    axes = load_axes(entry)
    t = axes['t']
    lo = 0 if t0 is None else int(np.abs(t - t0).argmin())
    hi = t.size - 1 if t1 is None else int(np.abs(t - t1).argmin())
    frames = np.unique(np.linspace(lo, hi, min(nframes, hi - lo + 1)).astype(int))

    # one pass over the chosen frames only: reading all 3529 off the memmap to
    # find a limit would defeat the point of not loading the whole array
    lim = max(float(np.abs(arr[:, :, k]).max()) for k in frames)
    out = Path(out or entry['figs_dir'] / 'anim' /
               f'{entry["id"]}_{yp_tag(yplus)}_{field}.gif')
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    im = ax.pcolormesh(axes['x'], axes['z'], np.asarray(arr[:, :, frames[0]]).T,
                       cmap=cmap, vmin=-lim, vmax=lim, shading='gouraud')
    ax.set_aspect('equal')
    ax.set(xlabel='$x$', ylabel='$z$')
    title = ax.set_title('')
    fig.colorbar(im, ax=ax, pad=0.02, label=f"${field}'$")

    def draw(k):
        im.set_array(np.asarray(arr[:, :, k]).T.ravel())
        title.set_text(f"{entry['label']}  ${field}'$  $y^+$={float(yplus):g}  "
                       f"$t$={t[k]:.2f}")
        return im, title

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    if verbose:
        print(f'  -> {out}  ({len(frames)} frames, '
              f'{out.stat().st_size / 1e6:.1f} MB)')
    return out


def _save_fig(fig, save, verbose=True):
    if not save:
        return
    path = Path(save)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    if verbose:
        print(f'  -> {path}')
