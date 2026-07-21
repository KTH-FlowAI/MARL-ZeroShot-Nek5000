# tsrs time-series: glid-based ordering and multi-`y+` sensing planes

**Date:** 2026-07-20
**Goal:** Rewrite the `pts*` time-series post-processing so the structured grid
is reconstructed from the **global point ids** stored in the file instead of by
sorting unique coordinates, and generalise the `int_pos` writer from a fixed
wall + one-plane pair to an arbitrary list of `y+` planes.

Code changes in `src/` are tagged with `#[MOD]`. The post-processing scripts were
rewritten wholesale, so they are not individually tagged.

## Background: why the old reader needed the coordinate sort

`post_processing/tsrs_stitch.py` used to rebuild the 2D mesh by extracting
`np.unique` coordinate values and matching each point back with
`np.where(x_coords == x_val)`. That looks like an odd way to do it, and the
instinct that it should just follow the point order is half right: the reason
the sort existed is real, but the sort was the wrong fix.

**The order of points in the file is genuinely not the order they were written
to `int_pos`:**

| Source | What it does |
|---|---|
| `Toolbox/tools/tsrs/pts_redistribute.f` | Hands every point to the rank that owns its element, with overflow/spill handling |
| `Toolbox/tools/tsrs/tsrs_IO.f:222-250` | Each rank writes its own `tsrs_ipts` chunk; the file is those chunks concatenated rank by rank |

So the point axis is permuted by the MPI decomposition, and the permutation
depends on the mesh partition and the process count. A naive positional read
would silently scramble the field.

**But the file already stores the inverse.** `tsrs_ipts` is the 1-based global
point id, written alongside the data — the old reader read it
(`tsrs_stitch.py:93`) and then discarded it. The whole `unique`/`where` machinery
was a workaround for information already on disk:

```python
perm = np.argsort(glid)          # undo the MPI shuffle
fld  = fld[perm].reshape(ny, nx, nz, ntsnap, nfld)
```

**Confirmed empirically** on `runs/oc-mc/env_001/ptsphill0.f00001`: `glid` is a
valid permutation of `1..1152` but is **not** the identity. After `argsort`,
`pos[:3]` shows `z` varying fastest at fixed `x, y` — exactly the writer's
`(jy, ix, iz)` nesting.

### Other defects removed

| Defect | Where | Consequence |
|---|---|---|
| Triple loop over `(jy, ix, iz)` that only incremented a counter | old `read_data` | Dead code; looked like it applied the ordering but was the identity map |
| Float equality on coordinates (`x_coords == x_val`, `pos[:,1] == 0.0`) | old `wall_data` | Worked only because `linspace` round-trips exactly in float64; breaks at `wdsize=4` or a non-zero wall plane |
| `np.where` over N uniques inside a loop over N points | old `wall_data` | Quadratic; plus per-point `struct.unpack` building N Python objects |
| `npoints` read from header field 4 (`nptot`) instead of 3 (`nelo`) | old `read_int_fld` | Equal under `ifmpiio`, but would run off the end of a multi-file dump |
| `tmlist` read and thrown away | old `read_int_fld` | No time axis anywhere; no check that stitched files were contiguous |
| Domain hardcoded as `linspace(0, 2π) × linspace(0, π)` | old `tsrs_eval.plot_data` | Ignored the stored `x`/`z`; silently mislabels any case with a different box |
| `vars = ['u','v','w']` on the wall plane | old `tsrs_eval` | `userbc` sets `ux = uz = 0` at `y = 0`, so two of three panels rendered pure roundoff as a vivid turbo field |
| `ID=200` bare snapshot index | old `tsrs_eval` | Snapshot 200 of case A is not necessarily the same physical time as case B |
| `Head`/`case_ids` overwritten on the next line | old `tsrs_eval:44-45` | Configuration by comment-editing; `argparse` was set up and never read |
| Only `y15_path[0]` used | old `tsrs_eval:78-83` | Every environment but the first silently discarded |

## New data model

One `.npz` per environment, replacing the `wall`/`y15` `.mat` pair:

| Key | Shape | Meaning |
|---|---|---|
| `fld` | `(ny, nx, nz, nt, nfld)` | Field data, `float32` by default |
| `y`, `x`, `z` | `(ny,)`, `(nx,)`, `(nz,)` | Real coordinates, from `channel_grid()` |
| `t` | `(nt,)` | Snapshot times, from `tmlist`, checked strictly increasing |
| `names` | `(nfld,)` | Field names, so `nfld` is not positional-by-comment |
| `yplus` | `(ny,)` | Plane heights in wall units |
| `retau` | scalar | `Re_tau`, for reconstructing `y+` |

Selecting the wall plane is now `iy = 0` rather than a float-equality mask.

### File layout reference (`tsrs_IO.f`)

```
[   132 B ] ascii header: '#std' wdsize ldim nelo nptot ntsnap nfld time fid0 nfileo
[     4 B ] float32 endianness tag (6.54321)
[ wdsize * ntsnap ] snapshot time list
[      4 * nptot  ] int32 global point id (1-based)
[ wdsize * nptot * ldim        ] point coordinates
[ wdsize * nptot * ntsnap*nfld ] interpolated fields
```

Field order is `u, v, w, p, omega_x, omega_y, omega_z` (`nfld = 2*ldim + 1`), set
by `tsrs_interpolate` (`tsrs.f:534-563`). The field block is
`(npoints, ntsnap, nfld)` (`tsrs.f:613-617`). Output is forced to double
precision (`tsrs_IO.f:63`).

## Multi-`y+` planes

`write_channel` previously hardcoded `linspace(0, yplus/Ret, 2)`. `yplus` now
accepts a scalar or a sequence, normalised by the new `resolve_yplus()`:

- `15.0` → `[0, 15]` — the historical wall + one-plane pair
- `[0, 15, 30, 50]` → four planes, sorted and de-duplicated

Rejected: empty lists, duplicates, negative `y+`, and `y+ > Re_tau` (outside the
channel — `findpts` would otherwise fail obscurely).

### Why this is safe for training

`tsrs` is compiled under `#ifdef TSRS` and is **purely diagnostic**. The DRL
sensing plane reaches the solver as a **UPARAM** (`src/lib/nek_utils.py:287`),
not through `int_pos`; the RL state is produced by `DRL_main` under `#ifdef DRL`.
Adding interpolation planes therefore cannot change the RL state.

This is why the plane list is a **separate** config key rather than an overload
of `y_sensing` — overloading `y_sensing` would have moved the DRL sensing plane
as a side effect of a diagnostic change:

```yaml
simulation:
    y_sensing : 15.0              # DRL sensing plane — unchanged, still scalar
    y_planes  : [0, 15, 30, 50]   # tsrs diagnostic planes; omit → [0, y_sensing]
```

### The `lhis` constraint

`lhis` in `SIZE` (100 in `envs/cases/lc_n7_onlyV/SIZE:34`) bounds the tsrs arrays
**per rank**, so the plane count is limited by `npoints / nproc`. `write_channel`
now parses `SIZE` from the compile path and reports:

```
 4 planes, 512 ranks: [int_pos] 23040 points, >= 45/rank, lhis = 100
 4 planes,  64 ranks: [int_pos] WARNING: ... needs at least 360 points/rank, lhis = 100
12 planes, 512 ranks: [int_pos] WARNING: ... needs at least 135 points/rank, lhis = 100
```

> The reported figure is a **lower bound**, assuming perfect balance. Real
> redistribution is lumpier — all near-wall planes fall inside the same element
> layer — so headroom under `lhis` is necessary, not sufficient. If the solver
> aborts in `tsrs`, raise `lhis` and recompile, or use fewer planes.

## Changes applied

### Source

| File | Change |
|---|---|
| `src/lib/writer_int_pos.py` | New `channel_grid()` — single source of truth for the sensing-point coordinates and their `(y, x, z)` nesting; `write_channel` now calls it. New `resolve_yplus()` for scalar-or-list plane specs. New `_check_lhis()` warning. `write_channel` gains `nproc=None` |
| `src/configs.py` | New `y_planes: Any = None` beside `y_sensing` (`#[MOD]`) |
| `src/lib/nek_utils.py` | `write_timeSeries` passes `y_planes` (falling back to `y_sensing`) and `nproc` to `write_channel` (`#[MOD]`) |

### Post-processing

| File | Change |
|---|---|
| `post_processing/postlib/tsrs.py` | **New.** `read_pts` (three bulk `frombuffer` reads + exact byte-length check), `sort_by_glid`, `to_grid`, `validate_grid`, `stitch`, `load`, `field` |
| `post_processing/tsrs_stitch.py` | Rewritten. Grid from `channel_grid()` via config; stitches per environment; writes `.npz` with coordinates, `t`, `names`, `yplus`, `retau` |
| `post_processing/tsrs_eval.py` | Rewritten. Coordinates and times read from the file; snapshots chosen by **physical time**; `--env`, `--plane`, `--yplus`, `--fields` |

`post_processing/stats/writer_int_pos.py` is a separate older copy and was left
untouched.

## Guard rails

The reader now turns silent mis-ordering into a loud failure:

- `sort_by_glid` asserts `glid` is a permutation of `1..npoints`
- `to_grid` asserts the grid size matches the point count
- `validate_grid` asserts the recovered coordinates equal `channel_grid()`'s, and
  that each coordinate is constant along the axes it should be
- `read_pts` asserts the file length matches the parsed header exactly
- `read_pts` raises on multi-file output (`nelo != nptot`) instead of misreading
- `stitch` asserts the concatenated time vector is strictly increasing, catching
  overlapping, repeated or out-of-order files
- `tsrs_eval` prints `max|·|` per panel, so an all-roundoff plane (e.g. `u` at
  the wall) is obvious instead of rendering as a vivid turbo field

A grid mismatch is **self-diagnosing**: `infer_shape()` reads the point
coordinates out of the offending file and reports the layout and plane heights it
actually contains, rather than just a point total.

```
env_005: 1 files
  SKIPPED: grid mismatch: the config asks for (ny, nx, nz) = (7, 20, 20) = 2800
  points, but this file holds 1152 points laid out as (2, 24, 24).
    the file's own y planes are [0.0, 0.083333333]
    this usually means the file predates a change to y_planes / Nx / Nz / lx1
    -- check whether it is stale
```

This is how stale data is caught in practice: `[0.0, 0.0833]` at `Re_tau = 180`
is `y+ = 0, 15`, i.e. the old two-plane configuration, in a run now configured
for seven planes. A failing environment is **skipped, not fatal** — the
environments after it are still stitched, and `tsrs_stitch.py` exits `1` if any
environment failed, so it stays usable in a script.

## Usage

```bash
# stitch every environment of a case  (reads runs/<id>/eval/ or runs/<id>/)
python tsrs_stitch.py 302000
python tsrs_stitch.py 302000 --fields u,v,w,p,omega_x --dtype float64 --overwrite

# plot: select the plane by height, compare cases at one physical time
python tsrs_eval.py 302000 402032 --yplus 15
python tsrs_eval.py 302000 --yplus 0                 # wall → defaults to v,p
python tsrs_eval.py 302000 --plane 1 --time 412.5 --env 2
```

Plane selection across cases is matched on **`y+`, not index**: comparing a
`[0,15,30,50]` case against a `[0,30]` case at `--yplus 30` picks index 2 and
index 1 respectively. Index-based selection would have silently plotted different
physical heights in the two rows. A requested `y+` that is absent produces a
warning naming the substitute.

## Backward compatibility

- **`int_pos` output is byte-identical.** A scalar `yplus=15.0` reproduces the
  pre-change file exactly (`cmp`-verified for the `lc-omega2` parameters), so
  existing runs and restarts are unaffected.
- `y_planes` unset, `None` or `[]` all fall back to `[0, y_sensing]`.
- `tsrs_eval.py` still plots `.npz` files that predate the `yplus`/`retau` keys,
  via `--plane`; `--yplus` then reports that the file must be restitched.
- The `.mat` output is **gone**. There were no downstream MATLAB consumers —
  `tsrs_eval.py` was the only reader, and it re-saved a near-duplicate of its own
  input into `datas/` with regenerated (wrong-domain) coordinates. Old `.mat`
  files are not read by the new scripts; restitch from the `pts*` files.

## Verification performed

| Check | Result |
|---|---|
| `glid` is a non-identity permutation on real data | Confirmed (`ptsphill0.f00001`, 1152 points) |
| Regrid through a random permutation, 2 planes | `pos` and `fld` recovered exactly (`array_equal`) |
| Regrid through a random permutation, 4 planes | `pos` and `fld` recovered exactly; plane `y` values correct |
| `int_pos` byte-identical for scalar `yplus` | Confirmed via `cmp` |
| `lhis` warning at 4/12 planes × 512/64 ranks | Fires correctly, stays quiet when there is headroom |
| `resolve_yplus` rejects empty/duplicate/negative/`> Re_tau` | All four rejected with specific messages |
| Overlapping files, wrong grid shape, missing field, out-of-range plane | All rejected with specific messages |
| Cross-case plane matching by `y+` | Correct indices chosen in a `[0,15,30,50]` vs `[0,30]` comparison |
| Legacy `.npz` without `y+` metadata | Still plots via `--plane` |
| End-to-end stitch + plot | Works; axes span the true `Lx`, `Lz` from the file |

### Validated on real data (`runs/oc-mc-dr`, 2026-07-20)

Seven planes, `(7, 20, 20)`, 2500 snapshots, `Re_tau = 180`. Per-plane statistics
from the stitched `.npz`:

| `y+` | `<u>` | `rms(u')` | `rms(v')` |
|---|---|---|---|
| 5 | 0.2201 | 0.0483 | 0.0088 |
| 10 | 0.4478 | 0.0754 | 0.0072 |
| 15 | 0.6326 | 0.1173 | 0.0169 |
| 30 | 0.8799 | **0.1243** | 0.0353 |
| 50 | 0.9761 | 0.0948 | **0.0446** |
| 100 | 1.1012 | 0.0589 | 0.0415 |
| 150 | 1.1703 | 0.0390 | 0.0269 |

The mean profile rises monotonically with `y+`, `rms(u')` peaks in the buffer
layer (`y+ ~ 15-30`), and `rms(v')` peaks farther out (`y+ ~ 50-100`) — all as
expected for wall turbulence. **A scrambled point ordering could not reproduce
this**, so it is the strongest end-to-end evidence that the glid regrid is
correct. Time base is uniform (`dt = 0.03`), all values finite, `x` and `z` span
the true `Lx = 2.67` and `Lz = 0.8`.

`tsrs_eval.py` was run against the same data for `--yplus 15/30/100`, the
absent-plane fallback (`--yplus 0` → nearest, `y+ = 5`) and an explicit
`--time`.

## Fixes made during first real-data use (2026-07-20)

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError` aborted the whole case at `env_005` | One unusable environment was fatal | `tsrs_stitch.py` catches per environment, reports, continues, exits `1` if any failed |
| `grid (7,20,20) ... file has 1152 points` gave no clue why | Message reported only a point total | `infer_shape()` reports the file's own layout and `y` planes |
| `tsrs_eval.py: invalid int value: 'oc-mc-dr'` | `case_ids` typed `int`; case ids are directory names | `type=str` (matching the same fix applied to `tsrs_stitch.py`) |
| `u'` colourbar ran `-0.202 -> -0.004`, not centred on zero | `vmin`/`vmax` are ignored when `levels` is an integer — matplotlib re-derives levels from the data range | Explicit symmetric `np.linspace(-lim, lim, 101)`, with a guard for an all-zero plane |

The `env_005`/`env_006` failure in `runs/oc-mc-dr` was **not** a code defect:
those directories hold `pts` files from 2026-07-19 written under the previous
two-plane, `lx1=6` configuration, while `env_001`-`004` were written on
2026-07-20 under the current seven-plane, `lx1=5` configuration. Delete or
re-run the stale environments.

## Known issues / out of scope

- `src/configs.py` does not import under Python 3.12: `Config` uses
  `Simulation()` as a dataclass default, which Python ≥3.11 rejects
  (`mutable default ... use default_factory`). This is **pre-existing** —
  verified identical at `HEAD` without these changes — and the repo's `.pyc`
  files are `cpython-38`. The `y_planes` field and the `write_timeSeries`
  fallback were validated against a standalone replica instead. Fixing it is a
  small, separate change (`field(default_factory=Simulation)`).
- Multiple environments are stitched to separate files and one is plotted
  (`--env`). Ensemble-averaging across environments is not implemented.
- Multi-file (non-`ifmpiio`) tsrs output is rejected rather than supported; no
  such files exist in this project.
- Vorticity (`omega_*`) is read and available via `--fields` but excluded from
  the stitch default to keep file sizes down.
