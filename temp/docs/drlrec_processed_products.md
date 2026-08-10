# drlrec: plot-ready products archived beside the full record

**Date:** 2026-08-10
**Goal:** Archive the four things `drlrec_cases.ipynb` actually draws — the
reward decomposition, the observation density, the action signal and the
learned control law — as small self-contained `.mat` files, so the figures can
be redrawn, shared or replotted in MATLAB without the run they came from.

Companion to `temp/docs/tsrs_case_postprocessing.md`, which does the same for
the `tsrs` field data. Usage is in the notebook itself, sections 8.1 and 8.2;
this file is the rationale.

---

## 1. What was missing

`postlib/drlrec.save_case_mat` already wrote one `.mat` per env holding
everything, including the full `act`/`obs`/`rwd` per-agent arrays. That file is
the archive of the *run*. What it is not is the archive of the *figures*: the
2-D histograms behind the observation density and the control law were computed
inline in the notebook, plotted, and thrown away.

The size gap is the point:

| | phill DDPG | wing NES | a real collected record |
|---|---|---|---|
| full per-env record | 285 kB | 982 kB | 369 MB (`*_vars_record_*.mat`) |
| all four products | 53 kB | 154 kB | — |

The per-agent arrays scale with `nrec x nagent`; the products do not. On a
production run the full record is gigabytes and the products stay tens of
kilobytes.

## 2. The four products

`obs` carries **two** components — `(nrec, nagent, 2)`, wall-tangential and
wall-normal — so "all three variables" in the request is the meshgrid triple
needed to draw a surface: `X`, `Y` and the value array, all the same shape,
straight into `pcolor`/`surf`/`contourf`.

```
data/results/<solver_case>/<run_name>/drl/
    <run_name>_env<NNN>_drlrec.mat        the full record (8.1, unchanged)
    processed_data/                       the products (8.2, new)
        <run_name>_reward_terms.mat
        <run_name>_observation.mat
        <run_name>_action.mat
        <run_name>_control_law.mat
```

One folder, not four: the file names already separate the products, and a
deeper tree buys nothing.

| File | Key variables |
|---|---|
| `reward_terms` | `R_tau`, `R_pw`, `R_v3`, `R_tot`, `dr_t` as `(nenv, nt)` plus `<term>_mean` / `<term>_std` across envs; the raw `q`, `pw`, `v3`; `eval_mask`, `eval_window` |
| `observation` | `X`, `Y`, `pdf`, `counts` on the `(u_t+, v_n+)` plane; `pdf1`/`pdf2` + edges for the marginals; `range`, `clip`, `utau_applied` |
| `action` | `traces` `(nenv, nt, n_show)`, `agent_index`, `agent_xyz_ipol`, `band_mean`/`band_std` over every active agent, action `pdf` + edges, `act_max`, `act_sat_pct` |
| `control_law` | `X`, `Y`, `act_mean`, `counts`, `min_count`; `sweep_x1/a1` and `sweep_x2/a2` for the 1-D sensitivity |

Every file also carries the `scale` block (`nu`, `utau`, `dudy_ref`, `alpha`,
`beta`, `gamma`, `reward_mode`) and a `meta` block with provenance and a `note`
field saying what its own variables mean.

## 3. The design point: compute once, plot and save the same object

If the writers recomputed the histograms, a saved file could silently disagree
with the figure above it — different bins, a different percentile clip, a
different agent selection. So the computations moved out of the notebook and
into the library, and the notebook now builds them **once**:

```python
PRODUCTS = {case: {'reward_terms': dr.reward_terms(entry, TRANS_TIME, EVAL_TIME),
                   'observation':  dr.observation_pdf(entry, nbin=N_MESH),
                   'action':       dr.action_traces(entry, n_show=N_AGENT_SHOW),
                   'control_law':  dr.control_law(entry, nbin=N_BIN,
                                                  min_count=MIN_COUNT),
                   'sensitivity':  dr.sensitivity_sweep(entry)}
            for case, entry in case_dict.items()}
```

Seven figure cells plot from `PRODUCTS`; section 8.2 writes it:

```python
dr.save_light(case_dict, precomputed=PRODUCTS, by_policy='auto')
```

`precomputed` is the mechanism — anything not supplied is computed with the
arguments given, so `save_light(case_dict)` alone still works from a script.

## 4. New and changed

### `post_processing/postlib/drlrec.py`

Each of the first four was previously computed inline in the notebook section
named beside it, and discarded once drawn.

| Added | What | Came from |
|---|---|---|
| `observation_pdf` | joint + marginal densities | §4 Observation |
| `control_law` | mean action per observation bin | §5 The learned control law |
| `sensitivity_sweep` | one component swept, the other held near its median | §5 Sensitivity to each input |
| `action_traces` | traces of a few agents + envelope + action PDF | §3 Action |
| `reward_terms` | per-env stack with ensemble mean/std | `env_mean_std`, at plot time |
| `env_agents`, `policies` | per-env agent resolution, and the `ipol` regions present |
| `save_light`, `LIGHT_DIR`, `LIGHT_PRODUCTS` | the four writers |
| `pool(..., policy=)` | restrict pooling to one `ipol` region |

### Elsewhere

| File | Change |
|---|---|
| `drlrec_cases.ipynb` | new "Plot-ready products" cell after the provenance table; seven figure cells read `PRODUCTS`; section 8 split into 8.1 (full record) / 8.2 (products); binning constants hoisted to the case-table cell |
| `postlib/results.py` | `drl` description mentions `processed_data/` |
| `utils/collect-results` | `drl/` and `tsrs/` inventoried at `-maxdepth 2`, since both now hold subdirectories; header comment updated |

## 5. Decisions

### 5.1 `.mat`, not `.npz`

Matches what is already in `drl/`, and these are the files most likely to be
handed to MATLAB plotting. `scipy.io.savemat` turns the nested `scale` / `meta`
dicts into structs, so `d.meta.note` works on both sides.

### 5.2 Pooled across envs — except traces

Distributions pool every sample of every env: an env-mean would invent a sample
no run produced. The reward is genuinely an ensemble quantity, so it is stored
as `(nenv, nt)` with the mean and std the figures draw.

Action *traces* are stacked, not pooled, to `(nenv, nt, n_show)` — concatenating
two time series would invent a trajectory neither run produced.

### 5.3 Per-region breakdown on `auto`

A case with more than one `ipol` region also gets `pol1`..`polN` structs holding
the same fields. On the channel one policy covers the whole wall and the
breakdown would be a copy of the pooled version, so it is skipped; on the wing
the four regions sit at different chord stations and pooling smears them.

**The regions share the pooled axes.** The first implementation let each region
clip to its own percentiles. Two things were wrong with that: the region counts
summed to 32078 against a pooled 31994 (samples outside the pooled clip fell
inside a region's), and — the real defect — auto-zooming each region centres it
in its own axes and normalises away the fact that the regions occupy *different*
parts of the observation plane, which is the entire reason to break them out.
`observation_pdf` and `control_law` therefore take `rng=`, and `save_light`
passes the pooled range down. Counts now sum exactly and the axes are identical,
so the panels compare.

This is the same failure mode as a per-panel colour limit, and it is worth
naming: **a comparison figure must not let each member choose its own scale.**

### 5.4 `counts` beside every histogram

A density says nothing about how many samples produced it. At `min_count = 1` a
control-law bin holding one sample is drawn exactly as confidently as one
holding a thousand, and a single railed action paints a saturated cell — which
is why the wing map looks speckled on a 40-record smoke run. The default stays
1 so the figure is what it always was, `counts` is archived so a reader can be
stricter without recomputing, and the notebook now says so where the speckle is
visible rather than leaving it to be read as structure.

### 5.5 Agent traces are identified by their real index

The figure used to label the three traces `agent 0/1/2` by position among those
shown. They are now labelled with their index into the agent table (`0`, `679`,
`1359` on the wing), which is what `agent_index` in the saved file holds, and
`agent_xyz_ipol` puts each one back on the wall.

## 6. Verification

| Check | Result |
|---|---|
| Every notebook cell | executes clean in one namespace, both cases |
| Saved vs replotted | `act_mean` and `pdf` bit-identical to a fresh recompute |
| Normalisation | the 2-D observation PDF integrates to 1.0000 |
| Per-region consistency | `pol1..pol4` counts sum to 31994 = pooled; edges identical |
| Round-trip | all four load in `scipy.io.loadmat`; structs, notes and scalars intact |
| `collect-results` | reports `drl: 5` for `small_wing_nes` (1 record + 4 products) |

## 7. Reading one back

```python
import scipy.io as sio
d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
d['X'], d['Y'], d['act_mean']      # the control-law surface
d['counts']                        # samples per bin -- check before believing
d['pol2'].act_mean                 # one policy region, where present
d['meta'].note                     # what this file's variables mean
d['scale'].utau, d['scale'].alpha  # rescale without the original run
```

```matlab
d = load(path);  pcolor(d.X, d.Y, d.act_mean);  shading flat
```

`squeeze_me=True` drops singleton axes, so a single-env run reads `R_tot` back
as `(nt,)` rather than `(1, nt)`. Pass `squeeze_me=False` to keep the env axis.

## 8. Known gaps

- The products are written for the case's own agent selection (`agents='active'`
  by default, i.e. `ipol > 0`). Re-running with `agents='all'` overwrites them;
  the selection is recorded in `meta.agent_selection` but not in the filename.
- `sensitivity_sweep` holds the other component within `0.1 sigma` of its
  median. On a short record that band can be empty, in which case the arrays are
  empty and the notebook prints a warning rather than drawing a line through
  nothing.
- Both cases exercised here are single-env smoke runs (30 and 40 records), so
  the `_std` fields are zero and the histograms are sparse. The shapes and the
  code paths are right; the statistics are not, and should be re-checked on a
  production run.
