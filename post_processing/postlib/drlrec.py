"""
Multi-case post-processing of embedded-policy (`drlrec*.bin`) runs.

`post_processing/read_drlrec.py` reads *one* run; this module reads a *list* of
them, resolves the physical scalars each run actually used, derives the reward
decomposition, and writes the outcome to the results tree that
`utils/collect-results` and `postlib/results.py` define::

    data/results/<solver_case>/<run_name>/
        config/   current_conf.yml, env<NNN>_drl_policy.in
        drl/      <run_name>_env<NNN>_drlrec.mat
        meta.yml

It backs `post_processing/drlrec_cases.ipynb` the way `determine.py` backs
`deterministic.ipynb`: the notebook holds the case table and the plots, the
reading and the arithmetic live here.

Typical use
-----------
    from postlib.drlrec import read_drlrec_cases, summary_table, save_case_mat

    case_dict = read_drlrec_cases('../runs/', ['solo_timing_cmp', 'oc_solonek'])
    summary_table(case_dict)
    for case in case_dict:
        save_case_mat(case_dict, case)

What the reward columns mean
----------------------------
`drl/pol_IO.f` writes three reward slots, but their meaning depends on the mode
the solver ran in (`drl_policy.in`, first field of the reward line):

    net_gain : (tau_w, |p'v|, 0.5|v^3|)   reference tau_ref = nu * dUdy_ref
    dudy     : (dUdy,  0,     0     )     reference          = dUdy_ref

so the decomposition here branches on the mode rather than assuming net_gain --
reading a `dudy` run as if it were net_gain silently scales the drag by nu.

@yuningw
"""

from __future__ import annotations

import glob
import os
import re
import shlex
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import yaml

from . import results as res

# read_drlrec.py sits in post_processing/, one level above this package, and is
# not importable as `postlib.read_drlrec`. Putting its directory on the path is
# what the single-case notebook does too.
_POST = Path(__file__).resolve().parent.parent
if str(_POST) not in sys.path:
    sys.path.insert(0, str(_POST))
from read_drlrec import load_run                      # noqa: E402
from read_drlrec import _discover as _discover_records  # noqa: E402

#: `env_001` is an env; `env_001.saved_20260730` is a backup and is not.
ENV_DIR_RE = re.compile(r'^env_(\d+)$')

#: First field of the reward line in drl_policy.in.
REWARD_MODES = {0: 'dudy', 1: 'net_gain'}

#: What the three reward slots hold, per mode.
RWD_NAMES = {'net_gain': ('tau_w', 'pw', 'v3'),
             'dudy': ('dudy', 'unused', 'unused')}

#: Categorical slots, in fixed order and never cycled (postlib.plot.colorplate:
#: blue, red, deepgreen, deeppurple, brown). Validated for CVD separation; the
#: blue/red/green triple sits in the 6-8 dE floor band, so every plot that uses
#: it also carries a marker or a direct label as secondary encoding.
CASE_COLORS = ('#2E59A7', '#D23918', '#057748', '#674196', '#9F6027')
CASE_MARKERS = ('o', 'X', 'D', 's', '^')


# --------------------------------------------------------------- discovery

def find_env_dirs(case_path):
    """
    Every env folder of one case that actually holds records.

    Three layouts are in the wild: `runs/<case>/eval/env_XXX` (what evaluate.py
    writes), `runs/<case>/env_XXX` (hand-staged and smoke runs), and the case
    folder itself (v1 files, no env level). Returns `[(env_id, path), ...]`.
    """
    case_path = str(case_path)
    for parent in (os.path.join(case_path, 'eval'), case_path):
        if not os.path.isdir(parent):
            continue
        found = []
        for name in sorted(os.listdir(parent)):
            m = ENV_DIR_RE.match(name)
            if not m:
                continue
            path = os.path.join(parent, name)
            if os.path.isdir(path) and _discover_records(path):
                found.append((int(m.group(1)), path))
        if found:
            return found
    if _discover_records(case_path):
        return [(1, case_path)]
    return []


def read_conf(case_path):
    """`current_conf.yml` of one case: under eval/ first, then the case root."""
    for cand in (os.path.join(case_path, 'eval', 'current_conf.yml'),
                 os.path.join(case_path, 'current_conf.yml')):
        if os.path.isfile(cand):
            with open(cand) as fh:
                return yaml.safe_load(fh), cand
    return None, None


def read_policy_cfg(env_path):
    """
    Parse `drl_policy.in`, the file the solver itself read.

    Machine-written by `src/lib/pol_export.py:write_run_config`, so the layout
    is fixed: three scalar lines then one line per policy region. Comments and
    blanks are dropped, which is what makes the positional read safe.
    """
    path = os.path.join(env_path, 'drl_policy.in')
    if not os.path.isfile(path):
        return None
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.split('#', 1)[0].strip()
            if line:
                rows.append(line)
    if len(rows) < 3:
        return None

    nb, recf, bufsz, iprec = rows[0].split()[:4]
    mode, dudy_ref, alpha, beta, gamma = rows[1].split()[:5]
    npol = int(rows[2].split()[0])

    policies = []
    for row in rows[3:3 + npol]:
        tok = shlex.split(row)          # the .pol file name is quoted
        policies.append(dict(xmin=float(tok[0]), xmax=float(tok[1]),
                             side=int(tok[2]), utau=float(tok[3]),
                             amp=float(tok[4]), nupd=int(tok[5]),
                             file=tok[6] if len(tok) > 6 else ''))

    return dict(nb_interactions=int(nb), rec_freq=int(recf),
                rec_bufsize=int(bufsz), iprec=int(iprec),
                reward_mode=int(mode), dudy_ref=float(dudy_ref),
                alpha=float(alpha), beta=float(beta), gamma=float(gamma),
                npol=npol, policies=policies, path=path)


def _nu_from_viscosity(value):
    """Nek's sign convention: a negative `viscosity` is 1/nu, not nu."""
    value = float(value)
    return 1.0 / abs(value) if value < 0 else value


def _nu_from_par(env_path):
    """Kinematic viscosity from the `.par` beside the records, if there is one."""
    for par in sorted(glob.glob(os.path.join(env_path, '*.par'))):
        with open(par) as fh:
            for line in fh:
                key, sep, val = line.partition('=')
                if sep and key.strip().lower() == 'viscosity':
                    try:
                        return _nu_from_viscosity(val.split('#')[0].strip())
                    except ValueError:
                        pass
    return np.nan


def _nu_from_rea(env_path):
    """
    Kinematic viscosity from a `.rea`, for the v17-era cases that carry no
    `.par` -- p002 holds it under the same sign convention.
    """
    for rea in sorted(glob.glob(os.path.join(env_path, '*.rea'))):
        with open(rea) as fh:
            for il, line in enumerate(fh):
                if il > 200:                     # the parameter block is first
                    break
                tok = line.split()
                if len(tok) >= 2 and tok[1].lower() == 'p002':
                    try:
                        return _nu_from_viscosity(tok[0])
                    except ValueError:
                        pass
    return np.nan


def solver_case_name(conf, env_paths, case_path):
    """Nek session name: the config first, then SESSION.NAME beside the data."""
    if conf:
        name = (conf.get('simulation') or {}).get('CASENAME')
        if name:
            return str(name)
    for path in list(env_paths) + [case_path]:
        session = os.path.join(path, 'SESSION.NAME')
        if os.path.isfile(session):
            with open(session) as fh:
                first = fh.readline().strip()
            if first:
                return first
    return 'unknown'


# ------------------------------------------------------------------ scales

def resolve_scale(conf, polcfg, env_path, override=None):
    """
    The physical scalars one run used, and where each one came from.

    `drl_policy.in` wins for the reward definition -- it is literally the file
    the solver parsed -- while `u_tau` and `nu` come from the YAML, because the
    `utau` field of `drl_policy.in` is the export normalisation baked into the
    `.pol` (1.0 in every run so far), not the physical friction velocity. An
    explicit `override` beats both, and is the only way to scale a smoke run
    that was staged without a config.

    The `src` entry records the winner for each key, so a plot can say where
    its numbers came from instead of asking you to trust them.
    """
    override = dict(override or {})
    src = {}
    out = dict(nu=np.nan, utau=np.nan, dudy_ref=np.nan,
               alpha=1.0, beta=1.0, gamma=1.0, reward_mode=None,
               ctrl_max_amp=np.nan, normalize_input=None, retau=np.nan)

    if conf:
        sim = conf.get('simulation') or {}
        run = conf.get('runner') or {}
        if sim.get('viscosity') is not None:
            out['nu'] = _nu_from_viscosity(sim['viscosity'])
            src['nu'] = 'current_conf.yml'
        if run.get('u_tau'):
            out['utau'] = float(run['u_tau'])
            src['utau'] = 'current_conf.yml'
        if run.get('dUdy'):
            out['dudy_ref'] = float(run['dUdy'])
            src['dudy_ref'] = 'current_conf.yml'
        for key, conf_key in (('alpha', 'reward_alpha'), ('beta', 'reward_beta'),
                              ('gamma', 'reward_gamma')):
            if run.get(conf_key) is not None:
                out[key] = float(run[conf_key])
                src[key] = 'current_conf.yml'
        if run.get('reward_fn'):
            out['reward_mode'] = str(run['reward_fn'])
            src['reward_mode'] = 'current_conf.yml'
        if run.get('ctrl_max_amp') is not None:
            out['ctrl_max_amp'] = float(run['ctrl_max_amp'])
        out['normalize_input'] = run.get('normalize_input')
        if sim.get('retau'):
            out['retau'] = float(sim['retau'])

    if polcfg:
        out['reward_mode'] = REWARD_MODES.get(polcfg['reward_mode'], 'unknown')
        out['dudy_ref'] = polcfg['dudy_ref']
        out['alpha'], out['beta'] = polcfg['alpha'], polcfg['beta']
        out['gamma'] = polcfg['gamma']
        for key in ('reward_mode', 'dudy_ref', 'alpha', 'beta', 'gamma'):
            src[key] = 'drl_policy.in'

    if not np.isfinite(out['nu']):
        for reader, tag in ((_nu_from_par, 'par'), (_nu_from_rea, 'rea')):
            nu = reader(env_path)
            if np.isfinite(nu):
                out['nu'], src['nu'] = nu, tag
                break

    for key, value in override.items():
        if key in out:
            out[key] = value
            src[key] = 'override'

    if out['reward_mode'] is None:
        out['reward_mode'] = 'net_gain'
        src.setdefault('reward_mode', 'default')

    # u_tau of the *uncontrolled* flow is sqrt(tau_ref) at unit density, and
    # tau_ref is nu * dUdy_ref -- the same reference the reward is built on.
    # That is the right scale for a t+ axis, so a run staged without a config
    # still gets one instead of falling back to raw solver time.
    if not np.isfinite(out['utau']) and np.isfinite(out['nu']) \
            and np.isfinite(out['dudy_ref']) and out['dudy_ref'] > 0:
        out['utau'] = float(np.sqrt(out['nu'] * out['dudy_ref']))
        src['utau'] = 'derived sqrt(nu*dUdy_ref)'

    # The drag reference is tau_w in net_gain mode and dUdy in dudy mode; the
    # rest of the module only ever divides by `ref`, so the two modes share one
    # code path from here on.
    out['tau_ref'] = out['nu'] * out['dudy_ref']
    out['ref'] = out['tau_ref'] if out['reward_mode'] == 'net_gain' \
        else out['dudy_ref']
    out['t_star'] = out['nu'] / out['utau'] ** 2 \
        if np.isfinite(out['nu']) and np.isfinite(out['utau']) \
        and out['utau'] > 0 else np.nan
    out['src'] = src
    return out


def missing_scalars(scale):
    """
    Which scalars a case still needs, given its reward mode.

    `nu` is only needed in net_gain mode (the reference is nu * dUdy there) and
    for the t+ axis; saying so keeps a dudy run from being flagged for a number
    it never uses.
    """
    need = []
    if not np.isfinite(scale['dudy_ref']):
        need.append('dudy_ref')
    if scale['reward_mode'] == 'net_gain' and not np.isfinite(scale['nu']):
        need.append('nu')
    if not np.isfinite(scale['t_star']):
        need.append('utau (t+ axis; falls back to solver time)')
    return need


# ------------------------------------------------------------------ derive

def active_agents(rec):
    """
    The agents that are actually inside a control region.

    `pol_id == 0` marks a wall point that no policy claims: `pol_net.f` leaves
    its action at zero but the solver still records its wall quantities. On the
    wing that is half of them (816 of 1632 in `small_wing_nes`), so counting
    them would halve every action statistic and put a spike at zero in the
    distribution, and would mix uncontrolled wall into the drag average. On the
    channel one policy covers everything and this selects all of them.
    """
    ipol = rec.agents['ipol'].to_numpy()
    sel = np.where(ipol > 0)[0]
    return sel if sel.size else np.arange(len(ipol))


def select_agents(rec, how='active'):
    """Resolve 'active' / 'all' / an explicit index array to indices."""
    if how is None or isinstance(how, str) and how == 'all':
        return np.arange(rec.nagent)
    if isinstance(how, str):
        if how != 'active':
            raise ValueError(f"agents must be 'active', 'all' or indices, "
                             f"got {how!r}")
        return active_agents(rec)
    return np.asarray(how)


def derive_terms(rec, scale, agents=None):
    """
    Reward decomposition and t+ axis of one record, averaged over agents.

    `agents` restricts the average to a subset (an index array), which is how
    the per-policy breakdown on the wing is built; the default is every agent.

    Returns a dict of (nrec,) series. In dudy mode the two penalty terms are
    identically zero -- the solver never filled those slots -- and R_tot is the
    bare drag term, matching `nek_marl._normalize_reward`.
    """
    sel = slice(None) if agents is None else np.asarray(agents)
    ref = scale['ref']

    q = rec.rwd[:, sel, 0].mean(axis=1)          # tau_w, or dUdy in dudy mode
    R_tau = 1.0 - q / ref
    if scale['reward_mode'] == 'net_gain':
        pw = rec.rwd[:, sel, 1].mean(axis=1)
        v3 = rec.rwd[:, sel, 2].mean(axis=1)
        R_pw, R_v3 = -pw / ref, -v3 / ref
        R_tot = scale['alpha'] * R_tau + scale['beta'] * R_pw \
            + scale['gamma'] * R_v3
    else:
        pw = v3 = np.zeros_like(q)
        R_pw = R_v3 = np.zeros_like(q)
        R_tot = R_tau

    t = rec.time - rec.time[0]
    if np.isfinite(scale['t_star']):
        tp, tp_unit = t / scale['t_star'], 't+'
    else:
        tp, tp_unit = t, 't'

    return dict(time=rec.time, tp=tp, tp_unit=tp_unit,
                q=q, pw=pw, v3=v3,
                R_tau=R_tau, R_pw=R_pw, R_v3=R_v3, R_tot=R_tot,
                dr_t=100.0 * R_tau)


def terms_by_policy(rec, scale):
    """
    The same decomposition, one entry per policy region (`ipol`).

    On the channel `rwd_xavg`/`rwd_zavg` make every agent carry the same reward
    and this returns one group identical to the global mean. On the wing the
    regions sit at different chord stations and a global mean smears them, so
    the per-region series is the one worth reading.
    """
    ipol = rec.agents['ipol'].to_numpy()
    return {int(p): derive_terms(rec, scale, np.where(ipol == p)[0])
            for p in np.unique(ipol)}


# -------------------------------------------------------------------- read

def read_drlrec_cases(run_path, case_list, overrides=None, verbose=True,
                      stitch=True, results_root=None, agents='active'):
    """
    Read a list of embedded-policy runs into one dict, keyed by case.

    `case_list` holds `runs/` folder names (or absolute paths). `overrides` is
    `{case: {'utau': ..., 'nu': ..., 'dudy_ref': ...}}` for runs staged without
    a config -- the smoke runs have only `drl_policy.in`, which fixes the reward
    definition but not the flow scales.

    `agents` selects which wall points every derived quantity is built from:
    'active' (the default, `ipol > 0`) drops the wall points no policy claims,
    'all' keeps them. See `active_agents` for why the default is not 'all'.

    Every env folder of a case is loaded separately and kept separate: the
    reward is an ensemble quantity worth averaging over envs, but actions and
    observations are not, and merging them here would make that impossible.
    """
    overrides = overrides or {}
    case_dict = {}

    for case in case_list:
        if verbose:
            print(f"---- Case {case} ----")

        case_path = case if os.path.isabs(str(case)) \
            else os.path.join(run_path, str(case))
        if not os.path.isdir(case_path):
            raise FileNotFoundError(f"Case {case} not found: {case_path}")

        env_dirs = find_env_dirs(case_path)
        if not env_dirs:
            raise FileNotFoundError(
                f"no drlrec files under {case_path} (looked in eval/env_XXX, "
                f"env_XXX and the case root). Embedded runs only write these "
                f"when embedded.enabled is set.")

        conf, conf_path = read_conf(case_path)
        env_paths = [p for _, p in env_dirs]
        polcfg = read_policy_cfg(env_paths[0])
        scale = resolve_scale(conf, polcfg, env_paths[0], overrides.get(case))
        solver_case = solver_case_name(conf, env_paths, case_path)

        entry = dict(case=str(case), path=case_path, conf=conf,
                     conf_path=conf_path, polcfg=polcfg, scale=scale,
                     solver_case=solver_case, run_name=str(case), envs=[])
        entry['name'] = _name_case(conf) or str(case)

        for env_id, env_path in env_dirs:
            rec = load_run(env_path, stitch=stitch)
            sel = select_agents(rec, agents)
            entry['envs'].append(dict(env_id=env_id, path=env_path, rec=rec,
                                      agent_sel=sel,
                                      series=derive_terms(rec, scale, sel)))

        recs = [e['rec'] for e in entry['envs']]
        entry.update(nenv=len(recs), nagent=recs[0].nagent,
                     nactive=len(entry['envs'][0]['agent_sel']),
                     agents=agents, nfld=recs[0].obs.shape[2],
                     nrec=min(r.nrec for r in recs))

        # Averaging over envs assumes they share a time base. They do not when
        # one env restarted from a different checkpoint, and the mean would then
        # be taken across records that are not contemporaneous.
        if entry['nenv'] > 1:
            ends = [e['series']['tp'][entry['nrec'] - 1] for e in entry['envs']]
            span = max(ends) - min(ends)
            if span > 0.01 * max(abs(max(ends)), 1e-30):
                print(f"[WARN] {case}: envs end {span:.3g} apart on the time "
                      f"axis; the ensemble mean averages records that are not "
                      f"contemporaneous. Read them one at a time if that "
                      f"matters.")

        # Resolved, not created: reading a case should leave no trace in the
        # results tree. save_case_mat is what makes the directories.
        root = res.case_root(entry['run_name'], solver_case, root=results_root,
                             create=False)
        entry['result_root'] = str(root)
        entry['save_path'] = str(root / 'drl')

        if verbose:
            miss = missing_scalars(scale)
            drop = entry['nagent'] - entry['nactive']
            print(f"[IO] {entry['nenv']} env(s), {entry['nactive']} agents "
                  f"({drop} uncontrolled dropped), {entry['nrec']} records, "
                  f"{entry['nfld']} obs components")
            print(f"[IO] mode={scale['reward_mode']} "
                  f"dUdy_ref={scale['dudy_ref']:g} nu={scale['nu']:g} "
                  f"u_tau={scale['utau']:g}  "
                  f"({', '.join(f'{k}<-{v}' for k, v in scale['src'].items())})")
            span = entry['envs'][0]['series']
            print(f"[IO] {span['tp_unit']} span: {span['tp'][0]:.3g} .. "
                  f"{span['tp'][-1]:.3g}")
            if miss:
                print(f"[WARN] unresolved for {case}: {', '.join(miss)}. "
                      f"Pass overrides={{'{case}': {{'utau': ..., 'nu': ...}}}} "
                      f"to fix.")

        case_dict[case] = entry

    return case_dict


def _name_case(conf):
    """`determine.name_case` when there is a config to name the run from."""
    if not conf:
        return None
    run = conf.get('runner') or {}
    sim = conf.get('simulation') or {}
    if not run.get('RL_algorithm'):
        return None
    return (f"{run['RL_algorithm']}_Retau{int(sim.get('retau', 0))}"
            f"_{run.get('agent_run_name', '')}")


# ------------------------------------------------------ across-env helpers

def env_stack(case_entry, key):
    """
    One series from every env of a case, stacked and truncated to the shortest.

    Returns `(tp, arr)` with `arr` of shape `(nenv, nmin)`. Envs differ in
    length when one of them was killed early; taking the common part is what
    `determine.load_reward_monitor_case` does for the reward monitor.
    """
    parts = [e['series'][key] for e in case_entry['envs']]
    nmin = min(len(p) for p in parts)
    tp = case_entry['envs'][0]['series']['tp'][:nmin]
    return tp, np.stack([p[:nmin] for p in parts], axis=0)


def env_mean_std(case_entry, key):
    """`(tp, mean, std)` across envs -- the ensemble view of a reward series."""
    tp, arr = env_stack(case_entry, key)
    return tp, arr.mean(axis=0), arr.std(axis=0)


def pool(case_entry, what, agents=None, mask=None):
    """
    Exact samples from every env, concatenated -- no averaging anywhere.

    `what` is 'act' or 'obs'. This is what the action and observation
    distributions are built from: an env-mean would invent a sample that no run
    ever produced, whereas pooling keeps every value the solver wrote.
    `agents` defaults to the case's own selection; `mask` is a per-record
    boolean, e.g. the evaluation window.
    Returns (N,) for 'act' and (N, nfld) for 'obs'.
    """
    out = []
    for env in case_entry['envs']:
        rec = env['rec']
        sel = env['agent_sel'] if agents is None else np.asarray(agents)
        arr = rec.act[:, sel] if what == 'act' else rec.obs[:, sel, :]
        if mask is not None:
            arr = arr[mask[:arr.shape[0]]]
        out.append(arr.reshape(-1) if what == 'act'
                   else arr.reshape(-1, arr.shape[-1]))
    return np.concatenate(out, axis=0)


def eval_mask(tp, trans_time=500.0, eval_time=1500.0, min_count=10):
    """
    The averaging window, and an honest label for it.

    The control is switched on abruptly at the restart, so the first few hundred
    viscous time units are transient and must not enter a reported mean. Short
    runs never reach the window at all; rather than report a transient-free
    number that quietly included the transient, this falls back to the whole
    record and says so in the label it returns.
    """
    sel = (tp > trans_time) & (tp < eval_time)
    if sel.sum() >= min_count:
        return sel, f"{trans_time:g} < t+ < {eval_time:g}", True
    return np.ones_like(tp, dtype=bool), "whole record (transient included)", False


# --------------------------------------------------------------- summaries

def case_summary(case_entry, trans_time=500.0, eval_time=1500.0):
    """One row of the comparison table: what each case actually achieved."""
    scale = case_entry['scale']
    tp, dr = env_mean_std(case_entry, 'dr_t')[:2]
    sel, window, windowed = eval_mask(tp, trans_time, eval_time)

    terms = {}
    for key in ('R_tot', 'R_tau', 'R_pw', 'R_v3'):
        terms[key] = env_mean_std(case_entry, key)[1][sel].mean()

    act = pool(case_entry, 'act')
    amax = np.abs(act).max()
    sat = 100.0 * np.mean(np.abs(act) > 0.99 * amax) if amax > 0 else 0.0

    return {
        'case': case_entry['case'],
        'solver_case': case_entry['solver_case'],
        'mode': scale['reward_mode'],
        'envs': case_entry['nenv'],
        'agents': case_entry['nactive'],
        'records': case_entry['nrec'],
        't_unit': case_entry['envs'][0]['series']['tp_unit'],
        't_end': tp[-1],
        'window': window,
        'DR_eval_pct': dr[sel].mean(),
        'DR_full_pct': dr.mean(),
        'R_tot': terms['R_tot'],
        'R_tau': terms['R_tau'],
        'R_pw': terms['R_pw'],
        'R_v3': terms['R_v3'],
        'act_max': amax,
        'act_sat_pct': sat,
        'windowed': windowed,
    }


def summary_table(case_dict, trans_time=500.0, eval_time=1500.0, labels=None):
    """The comparison table over every case, as a DataFrame."""
    import pandas as pd
    rows = []
    for il, (case, entry) in enumerate(case_dict.items()):
        row = case_summary(entry, trans_time, eval_time)
        row['label'] = labels[il] if labels else entry['case']
        rows.append(row)
    cols = ['label', 'case', 'solver_case', 'mode', 'envs', 'agents',
            'records', 't_unit', 't_end', 'window', 'DR_eval_pct',
            'DR_full_pct', 'R_tot', 'R_tau', 'R_pw', 'R_v3', 'act_max',
            'act_sat_pct']
    return pd.DataFrame(rows)[cols]


# ------------------------------------------------------------------ export

def _mat_scalars(scale):
    """The scale block of a .mat, as plain floats MATLAB can read."""
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else
                ('' if v is None else str(v)))
            for k, v in scale.items() if k != 'src'}


def save_case_mat(case_dict, case, save_full=True, compress=True,
                  trans_time=500.0, eval_time=1500.0, warn_gb=1.0,
                  verbose=True):
    """
    Write one `.mat` per env of a case into the results tree.

        data/results/<solver_case>/<run_name>/drl/<run_name>_env<NNN>_drlrec.mat

    The path and the `<run>_env<NNN>_` prefix are the ones `utils/collect-results`
    already uses for the Python-side records, so the embedded records land beside
    them instead of in a second, competing tree.

    Each file is self-contained and flat: the full per-agent arrays, the
    agent-averaged series, every reward term as its own variable, the agent
    table, and the scalars the terms were derived with -- so it can be re-scaled
    years later without the original run. `save_full=False` drops the three big
    per-agent arrays and keeps everything else, for runs where they do not fit.
    """
    entry = case_dict[case]
    scale = entry['scale']
    root = Path(entry['result_root'])
    drl_dir = Path(entry['save_path'])
    drl_dir.mkdir(parents=True, exist_ok=True)

    # Provenance: the two files the scalars above were resolved from.
    config_dir = root / 'config'
    config_dir.mkdir(parents=True, exist_ok=True)
    if entry['conf_path']:
        shutil.copy(entry['conf_path'], config_dir / 'current_conf.yml')
    for env in entry['envs']:
        pol = os.path.join(env['path'], 'drl_policy.in')
        if os.path.isfile(pol):
            shutil.copy(pol, config_dir / f"env{env['env_id']:03d}_drl_policy.in")

    names = RWD_NAMES.get(scale['reward_mode'], ('rwd1', 'rwd2', 'rwd3'))
    written = []

    for env in entry['envs']:
        rec, ser = env['rec'], env['series']
        sel, window, _ = eval_mask(ser['tp'], trans_time, eval_time)

        payload = {
            'time': rec.time,
            'tplus': ser['tp'],
            'istep': rec.istep.astype(np.int32),
            'icycle': rec.icycle.astype(np.int32),
            # agent-averaged raw reward buffers, in solver units
            'rwd_mean': np.stack([ser['q'], ser['pw'], ser['v3']], axis=1),
            'rwd_names': np.array(list(names), dtype=object),
            # the decomposition, one variable per term
            'R_tau': ser['R_tau'], 'R_pw': ser['R_pw'], 'R_v3': ser['R_v3'],
            'R_tot': ser['R_tot'], 'dr_t': ser['dr_t'],
            'act_mean': rec.act[:, env['agent_sel']].mean(axis=1),
            'act_std': rec.act[:, env['agent_sel']].std(axis=1),
            'eval_mask': sel.astype(np.int8),
            # `used` flags the agents every derived series above was built
            # from, so the full arrays below can be re-reduced the same way.
            'agents': dict({c: rec.agents[c].to_numpy()
                            for c in rec.agents.columns},
                           used=np.isin(np.arange(rec.nagent),
                                        env['agent_sel']).astype(np.int8)),
            'scale': _mat_scalars(scale),
            'summary': {k: v for k, v in
                        case_summary(entry, trans_time, eval_time).items()
                        if k not in ('case', 'solver_case', 'mode', 'window')},
            'meta': {
                'case': entry['case'],
                'run_name': entry['run_name'],
                'solver_case': entry['solver_case'],
                'name': entry['name'],
                'env_id': env['env_id'],
                'source': env['path'],
                'files': np.array([os.path.basename(f) for f in rec.files],
                                  dtype=object),
                'nseg_files': len(rec.files),
                'dt': float(rec.dt), 'ndrl': int(rec.ndrl),
                'rec_freq': int(rec.rec_freq),
                'reward_mode': scale['reward_mode'],
                'eval_window': window,
                'agent_selection': str(entry['agents']),
                'nagent_total': int(rec.nagent),
                'nagent_used': int(len(env['agent_sel'])),
                'scalar_source': {k: str(v) for k, v in scale['src'].items()},
                'time_unit': ser['tp_unit'],
                'writer': 'postlib.drlrec.save_case_mat',
            },
        }

        if save_full:
            nbytes = (rec.act.nbytes + rec.obs.nbytes + rec.rwd.nbytes)
            if nbytes > warn_gb * (1 << 30):
                print(f"[WARN] {case} env{env['env_id']:03d}: per-agent arrays "
                      f"are {nbytes / (1 << 30):.1f} GB before compression; "
                      f"pass save_full=False to keep only the averaged series.")
            payload['act'] = rec.act
            payload['obs'] = rec.obs
            payload['rwd'] = rec.rwd

        path = drl_dir / f"{entry['run_name']}_env{env['env_id']:03d}_drlrec.mat"
        sio.savemat(path, payload, do_compression=compress)
        written.append(path.name)
        if verbose:
            print(f"[IO] {path}  ({path.stat().st_size / 1e6:.1f} MB, "
                  f"full={save_full})")

    res.write_meta(root, solver_case=entry['solver_case'],
                   run_name=entry['run_name'],
                   drlrec={'files': written,
                           'envs': entry['nenv'],
                           'agents': int(entry['nactive']),
                           'agents_total': int(entry['nagent']),
                           'records': int(entry['nrec']),
                           'reward_mode': scale['reward_mode'],
                           'dudy_ref': float(scale['dudy_ref']),
                           'nu': float(scale['nu']),
                           'utau': float(scale['utau']),
                           'source': entry['path'],
                           'full_arrays': bool(save_full)})
    return written


def save_all(case_dict, **kwargs):
    """`save_case_mat` over every case; returns {case: [file names]}."""
    return {case: save_case_mat(case_dict, case, **kwargs)
            for case in case_dict}
