"""
Locating, loading and scaling one stitched tsrs case.

Shared by ``tsrs_spectra.py`` (figures) and ``tsrs_diag.py`` (diagnostics and
error analysis), so both speak about the same case in the same wall units.

The wall units are the subtle part.  The ``yplus`` and ``retau`` stored in the
``.npz`` are *nominal*: they were built from the reference u_tau of the
uncontrolled flow.  A controlled case runs at a different u_tau, so those labels
are wrong for it by exactly the drag-reduction factor.  ``set_scale`` resolves
which one is in force and every consumer reads the scale off the case dict
rather than reaching for ``data['yplus']`` directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from . import tsrs, tsrs_spectra as sp

SCALES = ('actual', 'reference')


# --------------------------------------------------------------------- input

def find_npz(runs_dir, case_id, env=0):
    """Locate the stitched .npz of one environment, and its case directory."""
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
    """
    Reference u_tau from the run config.

    Looks in the data directory and then in the case root above it: a solo-nek
    run stages current_conf.yml next to runs/<case>/ rather than inside eval/,
    and without it every energy silently stays in code units.
    """
    for cand in (Path(data_path) / 'current_conf.yml',
                 Path(data_path).parent / 'current_conf.yml'):
        if cand.exists():
            with open(cand) as fh:
                conf = yaml.safe_load(fh) or {}
            utau = (conf.get('runner') or {}).get('u_tau')
            if utau is not None:
                return utau
    return None


def load_case(case_id, runs_dir='../runs', env=0, field='u', t0=None, t1=None,
              utau=None, scale='actual', verbose=True):
    """
    Load one case, trimmed and scaled, ready for either script.

    Returns a dict with the periodic-view data, the box lengths, dt, the
    reference scaling as stored, the sublayer u_tau estimate and the resolved
    `scale` (see set_scale).
    """
    path, data_path = find_npz(runs_dir, case_id, env)
    data = tsrs.load(path)
    # validate before anything touches the field, so a typo gives this message
    # rather than a KeyError from deep inside the diagnostics
    if field not in data['names']:
        raise SystemExit(f'case {case_id} has no field {field!r}; '
                         f'available: {data["names"]}')

    dup = sp.check_duplicate_endpoints(data, field, 0)
    data = sp.periodic_view(data)
    if t0 is not None or t1 is not None:
        data = sp.trim_time(data, t0, t1)

    if utau is None:
        utau = read_utau(data_path)
    Lx, Lz = sp.domain(data)

    # the sublayer relation u = u_tau^2 y / nu is a statement about the
    # *streamwise* component, so the estimate is always made on u, whatever
    # field the spectra are of
    est = (sp.estimate_utau(data, utau_ref=utau, name='u')
           if utau is not None and 'u' in data['names'] else None)

    case = {'id': case_id, 'path': path, 'data': data, 'dup': dup,
            'Lx': Lx, 'Lz': Lz,
            'dt': float(np.diff(data['t']).mean()),
            'retau': float(np.asarray(data['retau'])),
            'utau': utau, 'est': est}
    set_scale(case, scale, verbose=verbose)

    if verbose:
        print(f'\ncase {case_id}: {path}')
        print(f'  {data["fld"].shape}  t=[{data["t"][0]:.1f}, {data["t"][-1]:.1f}]'
              f'  n={data["t"].size}  {scale_note(case, tex=False)}')
    return case


# --------------------------------------------------------------------- scale

def set_scale(case, scale='actual', verbose=True):
    """
    Fix the wall units this case is expressed in.

    'actual'    the case's own u_tau, from the viscous sublayer.  Lengths and
                energies both move; this is what "does the near-wall cycle look
                canonical" wants.
    'reference' the nominal scaling stored with the data, common to every case,
                which is what a case-to-case energy comparison wants.

    Falls back to reference (and says so) when no sublayer estimate is possible.
    """
    if scale not in SCALES:
        raise SystemExit(f'unknown scale {scale!r}; choose from {SCALES}')

    est = case['est']
    if scale == 'actual' and est:
        utau, retau, kind = est['utau'], est['retau_actual'], 'actual'
    else:
        if scale == 'actual' and verbose:
            why = ('no plane at y+ <= 5' if case['utau'] is not None
                   else 'no reference u_tau, so energies stay in code units')
            print(f'  [{case["id"]}] no sublayer estimate ({why}) -- '
                  f'falling back to reference scaling')
        utau, retau, kind = case['utau'], case['retau'], 'reference'

    case['scale'] = {'kind': kind, 'utau': utau, 'retau': retau,
                     'vel2': utau ** 2 if utau else 1.0,
                     'yplus': np.asarray(case['data']['y'], dtype=float) * retau}
    return case['scale']


def norm(case):
    """Velocity scale squared, for normalising energies."""
    return case['scale']['vel2']


def length(case):
    """Length scale of the chosen wall units: y+ = y * this."""
    return case['scale']['retau']


def yplus(case):
    """Plane heights in the chosen wall units."""
    return case['scale']['yplus']


def ylab(yp):
    """
    y+ as a short label.

    The nominal planes are round numbers, but rescaling by the actual u_tau
    turns y+ = 23 into 19.06, and '%g' would print all of it.
    """
    return f'{yp:.4g}'


def usym(case):
    """
    Normalising-velocity symbol for an axis label, '' when there is none.

    u_tau is the case's own friction velocity, u_tau,0 the reference one -- a
    plot in reference units has to say so, or a 30% drag reduction silently
    becomes a 44% error in the colour scale.
    """
    if not case['scale']['utau']:
        return ''
    return ('$/u_\\tau^2$' if case['scale']['kind'] == 'actual'
            else '$/u_{\\tau,0}^2$')


def scale_note(case, tex=True):
    """Compact statement of the wall units in force, for a title or a log line."""
    s = case['scale']
    if not s['utau']:
        return 'code units'
    if tex:
        return f'{s["kind"]} $u_\\tau$={s["utau"]:.4f}, $Re_\\tau$={s["retau"]:.0f}'
    return f'{s["kind"]} u_tau={s["utau"]:.4f}, Re_tau={s["retau"]:.0f}'


def plane_indices(data, spec):
    """
    Planes to analyse, selected on the *nominal* y+ stored with the data.

    Selection stays nominal under every scaling: those are the numbers in
    `y_planes` in the config, and they are what one types.  Only the labels move
    when the scaling changes.
    """
    yp = np.asarray(data['yplus'], dtype=float)
    if spec is None:
        return list(range(yp.size))
    want = [float(v) for v in spec.split(',') if v.strip()]
    return [int(np.abs(yp - w).argmin()) for w in want]
