# tsrs case post-processing: snapshots, spectra and one results tree

**Date:** 2026-08-05
**Goal:** Turn the stitched `pts` time series of several runs into one notebook
that compares them — wall and sensing-plane fluctuation fields, per-variable
snapshot archives, spectral maps and two-point correlations — with every derived
product landing in the case results tree so nothing is recomputed twice.

Implements `temp/docs/prompt030_tsrsPostprocessing.md`. Usage lives in
`post_processing/README_tsrs.md` (sections 5 and 6); this file is the rationale
and the record of what changed.

---

## 1. The shape of the problem

Three facts about the data decided the architecture. They were measured, not
assumed:

| Case | `fld` shape `(ny,nx,nz,nt,4)` | y⁺ planes stored | size |
|---|---|---|---|
| `mc-noctrl` | `(12, 20, 20, 2500)` | 0,2,5,7,10,15,30,45,50,75,100,150 | 184 MB |
| `oc-mc-dr` | `(7, 20, 20, 2500)` | **5**,10,15,30,50,100,150 | 107 MB |
| `lc_omega1_SS_oc_solonek` | `(15, 80, 72, 3529)` | 0,3,5,8,10,12,15,20,23,26,30,40,50,100,200 | **4.9 GB** |

1. **A record can be 4.9 GB.** Holding several the way `drlrec_cases.ipynb`
   holds `case_dict` does not survive re-running cells. Measured peak RSS for
   one such case is 5.8 GB.
2. **Plane sets differ per case.** `oc-mc-dr` has no wall plane at all. Nearest
   matching would draw its y⁺ = 5 plane as "the wall".
3. **Time windows do not overlap.** The mc cases cover `t = 399–474`, the lc
   case `476–716`. A shared snapshot *index* would compare different instants of
   different flows.

## 2. Two phases, with `data/results/` as the boundary

```
tsrs_post.process(cases)          ← phase A: expensive, idempotent, run once
        │  reads each stitched record ONCE
        ▼
data/results/<solver_case>/<run_name>/
    tsrs/<env>/  snap_<plane>_<field>.npy   (nx, nz, nt) float32 fluctuation
                 mean_<plane>_<field>.npy   (nx, nz)     the time mean removed
                 axes.npz        x, z, t, y, yplus, retau
                 snap_meta.yml   planes, fields, shapes, scaling, provenance
                 stitch.yml      the archived record's size and mtime
                 tsrs_<solver_case>.npz     the stitched record, copied
    spectra/     one .npz per spectrum, named exactly as
                 `tsrs_spectra.py --results` writes them
        │
        ▼
tsrs_post.plot_*(cases)           ← phase B: reads the above, memory-mapped
```

**Why `.npy` and not a bundle.** Phase B opens the snapshots with
`mmap_mode='r'`, so drawing one frame of a 3529-snapshot record costs one page
read instead of 81 MB. That is the entire reason the design brief's "convert the
stacked data into a `.npy` file" is worth honouring: a `.npz` is a zip and cannot
be mapped. Re-running a figure cell is therefore free, and only `process()` is
expensive.

**Why the mean is stored beside the fluctuation.** `mean_*.npy` is `(nx, nz)`,
~23 kB, and makes the extraction lossless: `raw = snap[:, :, it] + mean`.

**Memory discipline.** `process()` handles one case at a time and drops the
record before the next starts. `tsrs_case.load_case` gained `data=`, `npz_path=`
and `data_path=` parameters so the spectra can be computed from the record
already in hand instead of reading it a second time.

## 3. New and changed files

### New

| File | What |
|---|---|
| `post_processing/postlib/tsrs_post.py` | All of the logic: case resolution, phase-A pipeline, loaders, the four figure functions, and the map geometry shared with `tsrs_spectra.py` |
| `post_processing/tsrs_cases.ipynb` | The thin notebook layer, laid out like `drlrec_cases.ipynb` |

### Changed

| File | Change |
|---|---|
| `postlib/tsrs_spectra.py` | `fluctuation()` gains `mean='time'` and **defaults to it**; `correlation_y()` follows. See §5 |
| `postlib/tsrs.py` | New `read_header()` — the 132-byte header without the data behind it |
| `postlib/tsrs_case.py` | `load_case()` accepts a preloaded `data` dict / explicit paths |
| `postlib/results.py` | `tsrs` added to `SUBDIRS` |
| `tsrs_stitch.py` | Importable `sampling_grid()`, `stitch_env()`, `stitch_case()`; `main()` is now a thin CLI wrapper. Config lookup falls back to the case root. **Refuses a shrinking restitch** (§6). New `--force` |
| `tsrs_spectra.py` | `map_label`/`map_planes`/`map_rows`/`map_levels`/`log_axes`/`_plane_axes` moved into `postlib/tsrs_post.py` and re-exported from here, so the notebook and the CLI draw the same map. `tsrs_diag.py`'s imports resolve unchanged |
| `utils/collect-results` | Creates and inventories `tsrs/` (counted one level deeper, since it is one directory per env) |
| `README_tsrs.md` | New section 5 (the notebook), section 6 renumbered, mean-removal table |

## 4. The API

```python
from postlib import tsrs_post as tp

cases = tp.resolve_all([
    dict(case='mc-noctrl', label='uncontrolled', style=STYLE_A),
    dict(case='oc-mc-dr',  label='opposition',   style=STYLE_B),
], runs_dir='../runs/')            # optional per row: env=..., y_planes=[...]

tp.process(cases, planes=(0.0, 15.0), field='u', scale='reference')

tp.plot_snapshot_grid(cases, yplus=15.0)      # rows cases, cols variables
tp.plot_map(cases, quantities=('uu', '-uv'))  # rows quantities, cols cases
tp.plot_correlation(cases, yplus=15.0)        # one subfigure per case
tp.animate(cases['mc-noctrl'], yplus=15.0, field='u')

up = tp.load_snapshot(cases['mc-noctrl'], 15, 'u')   # memmap (nx, nz, nt)
d  = tp.load_spectrum(cases['mc-noctrl'], 'mapz_uu', 'reference')
```

`process()` composes `stitch` → `extract_snapshots` → `compute_spectra` →
`archive_stitched`, each of which is callable on its own and each of which skips
work whose output already exists.

## 5. Decisions, and what they cost

### 5.1 `mean='time'` is the default fluctuation — a repo-wide change

`f' = f - ⟨f⟩_t(x, z)`, the Reynolds decomposition about the time mean of each
point.

| `mean` | Subtracts | Note |
|---|---|---|
| `'time'` | `⟨f⟩_t(x, z)` | **new default.** The only one that removes a *stationary spatial pattern* |
| `'instant'` | `⟨f⟩_xz(t)` | the previous default; forces `E(k=0) = 0` exactly |
| `'global'` | `⟨f⟩_xzt` | unchanged requirement for frequency spectra |

The physical argument: the actuators sit at fixed `(x, z)`, so a controlled case
carries a *steady* imprint of the control. `'instant'` leaves that imprint in the
fluctuation and counts it as turbulent energy.

**This changes what `tsrs_spectra.py` and `tsrs_diag.py` produce**, since both
call `fluctuation()` without naming the convention. Both still run; pass
`mean='instant'` to reproduce earlier figures. One consequence worth stating:
under `'time'` the `k=0` bin holds the variance of the instantaneous plane mean
about its own time average and is no longer identically zero. Every plot drops
`k=0` and Parseval is checked against whatever field it is given, so nothing is
inconsistent — the premultiplied spectrum simply integrates to slightly less than
the full variance.

### 5.2 Reference wall units by default, where the CLI uses actual

`tsrs_spectra.py` defaults to `scale='actual'` — each case's own `u_τ` from the
viscous sublayer, which is the right single-case view of the near-wall cycle. The
notebook defaults to `'reference'`, because in actual units every case is divided
by its own `u_τ` **and** placed on its own y⁺ axis, which scales a 32% drag
reduction out of the picture meant to show it. Both sets coexist in the archive;
the units are in the filename (`_uact` / `_uref`), and `_spectra_present()` is
suffix-aware so switching `scale` recomputes rather than silently reusing.

### 5.3 Shared colour scales

One scale per column in the snapshot grid, one per row in the map. `tsrs_eval.py`
sets its limit per panel from that panel's own max; comparing a controlled and an
uncontrolled case that way normalises away exactly the effect being looked for.
`clim='panel'` restores the old behaviour.

### 5.4 A missing plane is dropped, never substituted

`resolve_planes()` matches on nominal y⁺ with `atol=1e-6` and reports misses.
`oc-mc-dr` therefore gets a labelled empty panel at y⁺ = 0 rather than its y⁺ = 5
plane relabelled. Panels also detect and label two other honest blanks: a field
held identically at zero (`v'` at the wall in an uncontrolled case, which would
otherwise render roundoff under a `1e-13` colour scale as vivid structure) and a
field that is not finite (§7).

### 5.5 Environments by name, not by position

`tsrs_case.find_npz` indexes a sorted env list positionally, so with `env_005`
missing, position 4 is `env_006`. `tsrs_post.resolve()` keys on the directory
name. Default `env_001`.

### 5.6 `-uv` is a plotting sign only

`parse_quantity('-uv') → ('uv', -1.0)`. The co-spectrum is negative and `−uv` is
the usual presentation; the archived `.npz` stays sign-true so it remains
interchangeable with what the CLI writes.

### 5.7 Mixed box sizes

Axes are shared only when every case is the same box. A minimal channel forced
onto a large box's axes shrinks into one corner, and reading it is the point.
Panels with nothing to draw still carry their case's box, so a row keeps one
geometry.

## 6. The restitch guard — a behaviour change to an existing CLI

**Raw `pts` files get cleaned up once a run has been stitched.**
`runs/lc_omega1_SS_oc_solonek/eval/env_001` keeps 1 pts file (100 snapshots)
behind a record of 3529. Restitching such a case replaces the record with what
the surviving files supply and destroys the rest.

This was not hypothetical: it happened during development of this work, on the
4.9 GB lc record, and was recoverable only because item 7 of the brief — archive
the stitched `.npz` — had already run. Two guards followed:

1. **`overwrite` no longer reaches the stitcher.** `process()` takes a separate
   `overwrite_stitch`, default `False`. Recomputing snapshots and spectra costs
   time; restitching can cost data, and one flag must not mean both.
2. **`stitch_env()` refuses a shrinking restitch.** Before overwriting it
   compares the existing record's `t.size` against the snapshots the surviving
   `pts` headers would supply (`tsrs.read_header`, 132 bytes per file) and
   returns status `'shrink'` with an explicit message. `--force` / `force=True`
   overrides.

```
env_001: REFUSING to overwrite tsrs_tcf.npz -- it holds 3529 snapshots and the
1 pts file(s) here supply only 100.
  The raw files this record was built from are gone, so this would destroy 3429
  snapshots that exist nowhere else.
  Pass --force / force=True if that is really what you want.
```

The archive is a **copy** (`shutil.copy2`), not a link, per the design decision
for HPC storage: the results tree is what survives the run directory being
cleared. `stitch.yml` records the source's size and mtime, so a re-run can tell
whether the archive is still the file that produced everything beside it.

## 7. Two data findings

**`p` is 100% NaN in the lc record.** Every plane, all 3529 snapshots of
`runs/lc_omega1_SS_oc_solonek/eval/env_001/tsrs_tcf.npz`. `u`, `v`, `w` are
clean. This is the source record, not an extraction artefact.
`extract_snapshots()` now warns at extraction time (where the cause is still
visible) and the panels are labelled, but the pressure diagnostics for that case
are unavailable until it is re-run. Worth checking whether other solo-nek `tcf`
runs share the fault.

**The box-size caveat is now visible rather than described.** With `Lz⁺ = 144`
and 19 periodic points, the minimal channel leaves **nine** spanwise
wavenumbers, and its premultiplied spectrum runs into the fundamental at
`λ_z⁺ = 144` with no closed peak. The large box closes a peak at
`λ_z⁺ ≈ 130, y⁺ ≈ 23` and its `ρ_uu(Δz)` reaches a proper negative minimum near
`Δz⁺ ≈ 50`. Draw conclusions from `lc_*`. `style='mesh'` on the map claims no
resolution the data does not have and is the honest view of the mc cases.

## 8. Verification

Run end to end on all three cases, `env_001`:

| Check | Result |
|---|---|
| `process()` on the mc cases | snapshots, spectra, archive written; second run skips all four steps |
| `process()` on the lc case | 40 s, peak RSS 5.77 GB, 4.88 GB archived |
| Every notebook cell | executes clean in one namespace |
| Restitch guard | `process(overwrite=True)` does not restitch; `overwrite_stitch=True` is refused with the message above; record verified intact afterwards |
| `tsrs_stitch.py --case_id <case>` | unchanged. The env_005/env_006 failures on `oc-mc-dr` are pre-existing and unrelated: those files predate a config change (`grid mismatch`, 1152 points laid out as `(2, 24, 24)` against the config's `(7, 20, 20)`) and have no `.npz` for the new guard to protect |
| `tsrs_eval.py`, `tsrs_spectra.py` (all six analyses), `tsrs_diag.py` | all still run after the helper move |
| `utils/collect-results --run_name mc-noctrl` | inventories `tsrs: 16` |

Physical sanity, `mc-noctrl` → `oc-mc-dr`: peak `k_zΦ⁺_uu` 3.55 → 1.25, sublayer
`u_τ` 0.0657 → 0.0535, DR 32.2%, and at the compared instant `max|v'|` at
y⁺ = 15 falls from `1.8e-1` to `6.7e-3`. The controller is doing what it should,
and the figures show it.

## 9. Known gaps

- `env_001` only. Ensemble spread across environments is a question for the
  reward records (`drlrec_cases.ipynb`), not for these figures, but nothing
  prevents adding it.
- Correlations are computed for every stored plane; the map drops y⁺ = 0
  (log axis). Neither is configurable from the notebook beyond `planes=`.
- Animations are `.gif` via `PillowWriter`. `ffmpeg` is on PATH, so `.mp4`
  would be a small addition if the frame counts grow.
- `imageio` is not installed and is not used.
