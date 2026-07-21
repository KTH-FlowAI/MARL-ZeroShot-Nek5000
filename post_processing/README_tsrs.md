# tsrs time-series post-processing

Two scripts turn the solver's `pts*` point time-series into figures:

```
runs/<case>/[eval/]env_XXX/pts<case>0.f#####      raw, one file per output cycle
        │
        │  tsrs_stitch.py     ← reorder + regrid + concatenate in time
        ▼
runs/<case>/[eval/]env_XXX/tsrs_<case>.npz        one array per environment
        │
        ├─ tsrs_eval.py       ← select plane + time, plot fluctuations
        │      ▼
        │  Figs/<HEAD>_tsrs_yp<Y>_t<T>.png
        │
        └─ tsrs_spectra.py    ← spectra, correlations, PSD, convection velocity
               ▼
           Figs/<HEAD>_{spectra,corr,ycorr,freq,Uc}_<field>.png
```

Run both from this directory (`post_processing/`), with the project environment:

```bash
source ~/.bashrc.miniforge
```

Design rationale and the solver-side details live in
`temp/docs/tsrs_glid_ordering_multiplane.md`. This file is usage only.

---

## 1. `tsrs_stitch.py` — raw `pts` → `.npz`

```bash
python tsrs_stitch.py --case_id oc-mc-dr --runs-dir ../runs/
```

Writes `tsrs_<CASENAME>.npz` into **every** `env_*` directory of the case.
Existing files are left alone unless `--overwrite` is given.

| Option | Default | Meaning |
|---|---|---|
| `--case_id` | `302000` | Case directory name. May be numeric or a name (`oc-mc-dr`) |
| `--runs-dir` | `../runs` | Directory holding `<case_id>/` |
| `--fields` | `u,v,w,p` | Subset of `u,v,w,p,omega_x,omega_y,omega_z` |
| `--dtype` | `float32` | `float32` or `float64` storage |
| `--overwrite` | off | Restitch environments that already have a `.npz` |

The case root is resolved as `<runs-dir>/<case_id>/eval/` when that exists,
otherwise `<runs-dir>/<case_id>/`. The sensing grid is **not** guessed from the
data — it is rebuilt from `current_conf.yml` (`retau`, `y_planes` or
`y_sensing`, `Lx`, `Lz`, `Nx`, `Nz`, `lx1`) through the same `channel_grid()`
function that wrote the solver's `int_pos` file.

### Exit codes

`0` if every environment stitched or was deliberately skipped, `1` if any
environment **failed**. Failures do not stop the run — the remaining
environments are still processed.

### Typical output

```
Case oc-mc-dr (phill) in ../runs/oc-mc-dr/eval
  Re_tau = 180.0, planes y+ = [5.0, 10.0, 15.0, 30.0, 50.0, 100.0, 150.0]
  grid (ny, nx, nz) = (7, 20, 20), Lx = 2.67, Lz = 0.8
  fields = ['u', 'v', 'w', 'p'] as float32
--------------------------------
env_001: 25 files
  -> .../env_001/tsrs_phill.npz  fld(7, 20, 20, 2500, 4)  nt = 2500, ...
```

### What the `.npz` contains

| Key | Shape | Meaning |
|---|---|---|
| `fld` | `(ny, nx, nz, nt, nfld)` | The field data |
| `y`, `x`, `z` | `(ny,)`, `(nx,)`, `(nz,)` | Real coordinates |
| `t` | `(nt,)` | Snapshot times |
| `names` | `(nfld,)` | Field names, in `fld`'s last-axis order |
| `yplus` | `(ny,)` | Plane heights in wall units |
| `retau` | scalar | `Re_tau` |

---

## 2. `tsrs_eval.py` — `.npz` → figures

```bash
python tsrs_eval.py oc-mc-dr --runs-dir ../runs/ --yplus 15
```

One **row per case**, one **column per field**, contouring the fluctuation about
the time mean at a single snapshot.

| Option | Default | Meaning |
|---|---|---|
| `case_ids` | *(required)* | One or more cases, positional; one row each |
| `--runs-dir` | `../runs` | Directory holding `<case_id>/` |
| `--env` | `0` | Which `env_*` directory (0 = the first) |
| `--yplus` | — | Select the plane nearest this `y+` |
| `--plane` | first plane above the wall | Select the plane by index instead |
| `--fields` | `v,p` at the wall, else `u,v,w` | Comma-separated fields |
| `--time` | last common time | Physical time to plot |
| `--head` | `TCF` | Figure filename prefix |
| `--outdir` | `Figs` | Where to write the `.png` |

`--yplus` and `--plane` are mutually exclusive.

### Examples

```bash
# compare three cases at the same physical time, at y+ = 15
python tsrs_eval.py 302000 402032 302999 --yplus 15

# the wall plane, on a case whose y_planes include 0; defaults to v and p
python tsrs_eval.py 302000 --yplus 0

# a specific environment, a specific time, custom fields
python tsrs_eval.py oc-mc-dr --runs-dir ../runs/ --env 2 --yplus 100 \
       --time 520.0 --fields u,p --head OCMCDR
```

Pick `--time` inside the environment's own range — each environment covers a
different window (for `oc-mc-dr`: `env_001` 399-474, `env_002` 449-524,
`env_003` 499-574). Ask for a time outside it and you still get a plot, of the
nearest snapshot, with a warning saying how far off it was.

### Two behaviours worth knowing

**Planes are matched by `y+`, not by index.** If case A was stitched with
`[0,15,30,50]` and case B with `[0,30]`, then `--yplus 30` picks index 2 in A and
index 1 in B. Selecting by index would have compared different physical heights
in the two rows. If a case lacks the requested plane you get a note naming the
substitute — e.g. `oc-mc-dr` has no wall plane (`y_planes` starts at `y+ = 5`),
so `--yplus 0` there reports:

```
note: no y+ = 0 plane in this data ([5, 10, 15, 30, 50, 100, 150]);
      using the nearest, y+ = 5
```

**Snapshots are matched by physical time.** `--time` (or the default, the last
time common to all cases) is resolved to the nearest snapshot per case, so rows
are comparable even when cases have different sampling. A warning is printed if
the nearest snapshot is more than one sampling interval away.

### Reading the figures

Each panel title carries `max|·|`, the peak fluctuation amplitude, and colour
levels are symmetric about zero. This makes a physically-empty panel obvious: at
the wall `userbc` sets `ux = uz = 0`, so `u'` and `w'` there are roundoff and
show as flat with `max|·| ~ 1e-16` instead of a vivid turbo field. Only `v` (the
actuation) and `p` carry signal at `y = 0`, which is why they are the wall
defaults.

---

## 3. `tsrs_spectra.py` — spectra and two-point correlations

```bash
python tsrs_spectra.py oc-mc-dr --runs-dir ../runs/ --planes 5,15,30
python tsrs_spectra.py controlled uncontrolled --planes 15   # overlay two cases
```

Cases are given positionally and each becomes a row, so controlled-vs-baseline
comparison is the normal mode of use.

| Option | Default | Meaning |
|---|---|---|
| `case_ids` | *(required)* | One or more cases; the first is the reference |
| `--env` | `0` | Which `env_*` directory |
| `--field` | `u` | Variable for spectra and correlations |
| `--planes` | all | Comma-separated `y+` values to analyse |
| `--t0`, `--t1` | — | Trim the record (drop a transient) |
| `--utau` | from config | Reference `u_τ` for wall units |
| `--lag` | `1` | Time lag for the convection velocity |
| `--nperseg` | `nt//4` | Welch segment length |
| `--what` | `all` | `spectra,map,correlations,ycorr,frequency,convection` |
| `--save` | — | Also write every computed curve to a `.npz` |
| `--map-quantity` | `uu` | `uu`, `uv` or `both` for the contour map |
| `--map-dir` | `z` | Wavelength axis of the map: `z` or `x` |
| `--map-levels` | `21` | Filled contour levels |
| `--map-points` | off | Overlay the actual `(λ, y⁺)` sample locations |

### What it computes

| `--what` | Output |
|---|---|
| `spectra` | Premultiplied `k_zΦ` and `k_xΦ` as line plots, plus the `uv` co-spectrum (which scales carry the Reynolds shear stress) |
| `map` | The same premultiplied spectrum as a **contour map over (λ⁺, y⁺)**, both log — see below |
| `correlations` | `ρ(Δz⁺)` and `ρ(Δx⁺)`, by Wiener–Khinchin from the same spectra |
| `ycorr` | Correlation matrix between wall-normal planes |
| `frequency` | Temporal PSD (Welch) |
| `convection` | Scale-dependent convection velocity `U_c(λ_x⁺)` |

### The spectral map

```bash
# uu map, spanwise wavelength against wall distance
python tsrs_spectra.py mc-noctrl --map-quantity uu --what map

# controlled vs uncontrolled, uu and uv side by side
python tsrs_spectra.py mc-noctrl oc-mc-dr --map-quantity both --what map
```

The standard wall-turbulence presentation: the ridge traces the energetic scale
at each height, so a controller shows up as a shift or a weakening of that
ridge. On the `mc-noctrl` baseline the `uu` peak sits at **y⁺ = 15** and the
`uv` peak at **y⁺ = 30**, both textbook.

Details that matter when reading it:

- **`y⁺ = 0` is dropped**, with a printed note — it cannot go on a log axis, and
  `u` is identically zero at the wall anyway.
- **Rows share colour scale and axes.** Cases with different `y_planes` would
  otherwise be stretched differently and the peak shift would be hard to read.
  A blank band means that case has no planes there, not zero energy.
- `uu` is positive-definite, so it uses a sequential map from 0; `uv` is
  negative and uses a diverging map centred on 0.
- The star marks the peak. With `--map-points` the actual samples are overlaid,
  so the true (coarse) resolution behind the smooth contours is visible.

### Conventions

Both are asserted at runtime, so a normalisation error fails loudly:

```
sum_k E(k) == <f'^2>          R(0) == <f'^2>
```

`Φ = E/dk` is the density (`∫Φ dk` = variance) and `kΦ` is the premultiplied
spectrum, whose area against `d(ln k)` is the variance.

**Periodicity.** The stored grid includes *both* endpoints, and `x=0`/`x=Lx` are
the same physical point in a periodic channel — they are bit-identical in the
data. Passing all N points to an FFT would declare the period to be `L·N/(N-1)`,
wrong by ~5% at N=20. `periodic_view()` drops the duplicate automatically; the
header line reports the resulting shape.

**Mean removal differs by analysis, deliberately.** Spatial statistics subtract
the *instantaneous* plane mean `⟨f⟩_xz(t)`, which makes them immune to a
drifting bulk state. Frequency spectra subtract the *global* mean, because
per-instant removal would delete the low frequencies they exist to measure.

**No windowing in x and z.** Those directions are genuinely periodic, so the
DFT is exact and a window would only smear it. Time is windowed (Welch), since
it is not periodic.

### Reading the diagnostics

```
drift of <u>_xz: -0.07833 (-17.3% of the mean, -1.04 x rms)  <-- NOT stationary
u_tau: reference 0.0650, actual 0.0535 (Re_tau 148, drag reduction +32.2%)
[oc-mc-dr y+=    15] nperseg=625 var_ratio=0.59
[oc-mc-dr y+=    30] Uc=0.8463 (Uc+=13.02)  wrap_risk=0.10
```

- **drift** — stationarity is judged on the drift relative to the *rms*, not to
  the mean, so it stays meaningful for zero-mean quantities like `v`. Above
  0.5×rms the record is flagged; trim it with `--t0`/`--t1`.
- **u_tau** — the stored `y+` labels are *nominal* (fixed physical planes in
  reference wall units). Control changes the real `u_τ`, and this line reports
  the actual value estimated from the viscous sublayer, so the shift is never
  invisible.
- **var_ratio** — the fraction of the variance recovered by `∫psd df`. Below 1
  is normal: Welch cannot represent periods longer than `nperseg*dt` and also
  removes each segment mean. Raise `--nperseg` to capture more, at the cost of
  a noisier estimate. It cannot validate `dt` — the integral is invariant to it.
- **wrap_risk** — the phase method is unambiguous only while `|k·U_c·τ| < π`.
  This is a *predictive* estimate of that ratio; `>= 1` means the answer is
  aliased and `--lag` must be reduced. It is predictive because a wrapped phase
  folds back into `(-π, π]` and cannot be detected after the fact.

### Box-size caveat

Spectra only say something when the box is larger than the structures. For a
minimal channel such as `oc-mc-dr` (`Lx+ = 481`, `Lz+ = 144`) the near-wall
streaks — `λ_z+ ≈ 100`, `λ_x+ ≈ 1000` — do not fit, so the premultiplied spectra
rise monotonically to the fundamental with no peak, and `ρ_uu(Δx)` is still ≈0.3
at half the box. That is the box talking, not the flow. The `lc_*` cases
(`Lx+ ≈ 1758`, `Lz+ ≈ 879`) resolve both scales and are the ones to draw
conclusions from.

---

## Using the data directly

```python
from postlib import tsrs

d = tsrs.load('../runs/oc-mc-dr/eval/env_001/tsrs_phill.npz')
u = tsrs.field(d, 'u', iy=2)        # (nx, nz, nt) at plane index 2
print(d['yplus'], d['t'].shape)

# fluctuation about the time mean, and a per-plane rms profile
for i, yp in enumerate(d['yplus']):
    ui = tsrs.field(d, 'u', iy=i)
    print(f"y+={yp:6.1f}  <u>={ui.mean():.4f}  rms(u')={ui.std():.4f}")
```

`postlib/tsrs.py` also exposes the lower-level pieces: `read_pts` (one raw file),
`sort_by_glid`, `to_grid`, `validate_grid`, `stitch` and `infer_shape`.

---

## Configuring the sensing planes

The planes come from `current_conf.yml` and are written by
`src/lib/writer_int_pos.py` when the case is set up:

```yaml
simulation:
    y_sensing : 15.0                       # DRL sensing plane (unchanged, scalar)
    y_planes  : [0, 15, 30, 50]            # tsrs diagnostic planes
```

- Omit `y_planes` (or leave it empty) and you get the historical two planes,
  `[0, y_sensing]`.
- `y_planes` affects the **tsrs diagnostic only**. The DRL sensing plane reaches
  the solver as a UPARAM, so adding planes does not change the RL state.
- More planes means more points. `lhis` in `SIZE` caps the points **per rank**;
  `write_channel` warns when the point count cannot fit. If the solver aborts
  inside `tsrs`, raise `lhis` and recompile, or use fewer planes.

Changing `y_planes`, `Nx`, `Nz` or `lx1` changes the point set, so `pts` files
written before the change cannot be read with the new config — see below.

---

## Troubleshooting

**`grid mismatch: ... the file's own y planes are [...]`**
The `pts` files in that environment were written under a different
configuration. The message reports the layout and plane heights the file
actually contains, so compare them with the config. Example: `[0.0, 0.083333]`
at `Re_tau = 180` is `y+ = 0, 15` — the old two-plane setup. Delete or re-run
those environments; the others still stitch.

**`no stitched .npz in ... -- run tsrs_stitch.py first`**
That environment was never stitched, often because it was the one that failed
above. Check the `tsrs_stitch.py` output for it.

**`--yplus needs y+ metadata; this .npz predates it`**
The `.npz` was written before `yplus`/`retau` were stored. Restitch with
`--overwrite`, or select by `--plane`.

**`invalid int value` / `no current_conf.yml under ...`**
Case ids are directory names and may be non-numeric. Check `--runs-dir` and that
the case has a `current_conf.yml` (in `eval/` if that subdirectory exists).

**Stitched files are large.**
`fld` is `ny*nx*nz*nt*nfld` values. Trim with `--fields` (vorticity is excluded
by default) and keep `--dtype float32`.
