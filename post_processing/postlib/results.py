"""
Where post-processed results live, and what a saved dataset looks like.

    data/results/<solver_case>/<run_name>/
        meta.yml       what was archived, from where, when
        config/        current_conf.yml exactly as the run used it
        model/         eval_model_<run_name>.zip -- the evaluated checkpoint
        drl/           reward / action / observation records, one per env
        spectra/       one .npz per spectrum
        figs/          figures

The root mirrors ``utils/mv-data``, which archives the *raw* solver output to
``data/results/<solver_case>/<run_name>/env_<id>/res_N/``.  Derived results sit
beside it in named directories, so the two never collide and one case has one
home.

``<solver_case>`` is the Nek session name (``tcf`` for the channel), and
``<run_name>`` is the ``runs/`` folder.  Both come off the data itself --
``tsrs_tcf.npz`` in ``runs/lc_omega1_SS_oc_solonek/`` gives both -- so nothing
has to be typed twice.

A saved dataset
---------------
One ``.npz`` per spectrum, self-contained: the arrays as plotted, the same
arrays in code units, and every scaling scalar as its own variable.  Storing
both costs a few kB and means a file can be replotted in different wall units
years later without the original run::

    d = np.load('..._mapz_uu_uact.npz')
    d['kPhi_plus']                       # as plotted, / u_tau^2
    d['kPhi'] * (1 / other_utau ** 2)    # rescaled to any other u_tau
    d['utau'], d['utau_ref'], d['retau'], d['retau_ref'], d['nu']
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import yaml

#: Directories under one case root, and what belongs in each.
SUBDIRS = {
    'config': 'current_conf.yml as the run used it',
    'model': 'the evaluated checkpoint, eval_model_<run>.zip',
    'drl': 'reward / action / observation records',
    'spectra': 'one .npz per spectrum',
    'figs': 'figures',
}


def find_repo_root(start=None):
    """
    Walk up from `start` to the repository root.

    The root is the directory holding both `src/` and `data/`; looking for two
    markers rather than one keeps it from stopping at a stray `data` folder.
    """
    here = Path(start or __file__).resolve()
    for cand in [here, *here.parents]:
        if (cand / 'src').is_dir() and (cand / 'data').is_dir():
            return cand
    raise SystemExit(f'no repository root above {here} (need src/ and data/)')


def solver_case_from_npz(npz_path):
    """
    Nek session name from a stitched file name: tsrs_<case>.npz -> <case>.

    Falls back to the SESSION.NAME beside it, then to 'unknown'.
    """
    stem = Path(npz_path).stem
    if stem.startswith('tsrs_') and len(stem) > 5:
        return stem[5:]
    session = Path(npz_path).parent / 'SESSION.NAME'
    if session.exists():
        first = session.read_text().splitlines()
        if first and first[0].strip():
            return first[0].strip()
    return 'unknown'


def case_root(run_name, solver_case, root=None, create=True):
    """The one directory that holds everything derived from one run."""
    base = Path(root) if root else find_repo_root() / 'data' / 'results'
    path = base / str(solver_case) / str(run_name)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def subdir(run_name, solver_case, which, root=None, create=True):
    """One of SUBDIRS under a case root."""
    if which not in SUBDIRS:
        raise SystemExit(f'unknown results subdir {which!r}; '
                         f'choose from {sorted(SUBDIRS)}')
    path = case_root(run_name, solver_case, root, create) / which
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------ datasets

def scalars_from_case(case):
    """
    Every scaling scalar of one case, as plain numbers.

    These are what makes a saved spectrum rescalable: `utau`/`retau` are the
    units it was written in, `utau_ref`/`retau_ref` the nominal ones stored with
    the data, and `nu` ties them together.
    """
    s = case['scale']
    est = case.get('est')
    out = {
        'scale': s['kind'],
        'utau': float(s['utau']) if s['utau'] else np.nan,
        'retau': float(s['retau']),
        'utau_ref': float(case['utau']) if case['utau'] else np.nan,
        'retau_ref': float(case['retau']),
        'nu': float(est['nu']) if est else np.nan,
        'drag_reduction_pct': float(est['drag_reduction_pct']) if est else np.nan,
        'Lx': float(case['Lx']),
        'Lz': float(case['Lz']),
        'dt': float(case['dt']),
        't0': float(case['data']['t'][0]),
        't1': float(case['data']['t'][-1]),
        'nt': int(case['data']['t'].size),
        'nx': int(case['data']['x'].size),
        'nz': int(case['data']['z'].size),
        'case': str(case['id']),
        'source': str(case['path']),
    }
    return out


def save_dataset(path, arrays, scalars=None, meta=None):
    """
    Write one spectrum to one .npz: arrays, scaling scalars, and labels.

    Everything lands at the top level of the file, so `np.load(f)['utau']` works
    without unpacking anything.  Strings are stored as 0-d unicode arrays, which
    `np.load` returns unchanged.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    for k, v in (scalars or {}).items():
        payload[k] = np.asarray(v)
    for k, v in (meta or {}).items():
        payload[k] = np.asarray(str(v))
    payload['written'] = np.asarray(datetime.datetime.now().isoformat(timespec='seconds'))
    np.savez(path, **payload)
    return path


def describe(path):
    """One-line summary of a saved dataset, for logs and for sanity."""
    with np.load(path, allow_pickle=False) as d:
        shapes = ', '.join(f'{k}{d[k].shape}' for k in d.files
                           if d[k].ndim >= 1 and d[k].dtype.kind == 'f')
    return f'{Path(path).name}: {shapes}'


# ---------------------------------------------------------------------- meta

def write_meta(root, **entries):
    """
    Record what is in a case root and where it came from.

    Merged with anything already there, so each producer (spectra, drl records,
    the model copy) can add its own section without clobbering the others.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'meta.yml'
    meta = {}
    if path.exists():
        with open(path) as fh:
            meta = yaml.safe_load(fh) or {}
    meta.update(entries)
    meta['updated'] = datetime.datetime.now().isoformat(timespec='seconds')
    with open(path, 'w') as fh:
        yaml.safe_dump(meta, fh, sort_keys=False, default_flow_style=False)
    return path
