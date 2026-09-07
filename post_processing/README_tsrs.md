# tsrs time-series post-processing

Three scripts turn the solver's `pts*` point time-series into figures, plus one
that asks whether the record is good enough to draw them from, plus a notebook
that does the whole thing for several cases at once:

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
        ├─ tsrs_spectra.py    ← spectra, correlations, PSD, convection velocity
        │      ▼
        │  Figs/<CASE>_{spectra,corr,ycorr,freq,Uc}_<field>_<uact|uref>.png
        │  Figs/<CASE>_map<z|x>_<uu|uv|uu-uv>[_mesh]_<uact|uref>.png
        │
        ├─ tsrs_diag.py       ← record, scaling, profiles, noise floor, error bars
        │      ▼
        │  stdout only
        │
        └─ tsrs_cases.ipynb   ← several cases overlaid, snapshots + spectra
               ▼
           data/results/<solver_case>/<run_name>/{tsrs,spectra,figs}/
```

`tsrs_spectra.py` draws; `tsrs_diag.py` decides whether the drawing means
anything. Both share `postlib/tsrs_case.py`, which locates the `.npz`, trims it
and resolves the wall units, so the two always speak about the same case in the
same units. `tsrs_cases.ipynb` is the multi-case comparison — see section 5.
Section 6 says where everything any of them writes ends up.

Run both from this directory (`post_processing/`), with the project environment:

```bash
source ~/.bashrc.miniforge
```

Design rationale and the solver-side details live in
`temp/docs/tsrs_glid_ordering_multiplane.md`; the case-comparison pipeline
(sections 5 and 6) is written up in `temp/docs/tsrs_case_postprocessing.md`.
This file is usage only.

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
| `--utau` | from config | **Reference** `u_τ` — the one `y+` and `ν` were built with |
| `--scale` | `actual` | Wall units to draw in: `actual` (the case's own `u_τ`) or `reference` |
| `--lag` | `1` | Time lag for the convection velocity |
| `--nperseg` | `nt//4` | Welch segment length |
| `--what` | `all` | `spectra,map,correlations,ycorr,frequency,convection` |
| `--save` | — | Also write every computed curve to a `.npz` |
| `--map-quantity` | `uu` | `uu`, `uv` or `both` for the contour map |
| `--map-dir` | `z` | Wavelength axis of the map: `z` or `x` |
| `--map-levels` | `21` | Filled contour levels |
| `--map-points` | off | Overlay the actual `(λ, y⁺)` sample locations |
| `--map-style` | `contour` | `contour` interpolates between samples; `mesh` draws one cell per sample |
| `--head` | the case ids | Figure filename prefix |
| `--outdir` | `Figs` | Where to write the `.png` |
| `--results` | off | Also save each spectrum as its own `.npz` in the results tree (section 6) |
| `--results-root` | `<repo>/data/results` | Override the results base |

Figure names are `<head>_<tag>_<scale>.png`, where the tag carries every switch
that changes the picture — `spectra_u`, `mapz_uv`, `mapz_uu-uv_mesh`, `mapx_ww`
— and `<scale>` is `uact` or `uref` per `--scale`. So the `uu` and `uv` maps of
one case coexist instead of the second silently overwriting the first, and maps
of different cases never collide in `Figs/`.

### `--field` and `--map-quantity` are different things

`--field` picks *which velocity component*; `--map-quantity` picks *auto- or
co-spectrum*. Together they resolve to one quantity, and the map tag names the
resolved pair, never the shorthand:

| `--field` | `--map-quantity` | quantity | map tag |
|---|---|---|---|
| `u` | `uu` | `Φ_uu = ⟨û'û'*⟩` | `mapz_uu` |
| `u` | `uv` | `Φ_uv = Re⟨û'v̂'*⟩` | `mapz_uv` |
| `u` | `both` | both, side by side | `mapz_uu-uv` |
| `w` | `both` | `Φ_ww` and `Φ_wv` | `mapz_ww-wv` |

In every case the **input to the transform is the fluctuation `u'`, and the
output is the energy `u'u'`** — `Φ_uu` is `|û'|²`, which is the only way it can
be built. `Σ_k E(k) = ⟨u'u'⟩` is asserted on every call.

### Which `u_τ`? — `--scale`

The `y+` labels and `Re_τ` stored in the `.npz` are **nominal**: they were built
from the *reference* `u_τ` in `current_conf.yml`, which is the uncontrolled
value. A controlled case runs at a different `u_τ`, so those labels are wrong
for it by exactly the drag-reduction factor.

`--scale actual` (default) rescales everything by the `u_τ` estimated from the
viscous sublayer of that case — lengths by `Re_τ,act/Re_τ,ref`, energies by
`(u_τ,ref/u_τ,act)²`. `--scale reference` keeps the nominal scaling, which is
what a *case-to-case energy comparison* wants, since it puts every row on one
common axis. Every figure states in its title which one is in force, and the
filename ends in `_uact` or `_uref`.

For `lc_omega1_SS_oc_solonek` (`u_τ,ref = 0.0630`, actual `0.0524`, +30.7% DR)
this moves the `uu` peak from `λz⁺=130, y⁺=23` to `λz⁺=108, y⁺=19` and raises
the peak of `k_zΦ_uu/u_τ²` from 3.0 to 4.3 — a 1.20× / 1.44× shift that is
purely a labelling choice, not a change in the data.

Both need a reference `u_τ` (from the config or `--utau`); without one the
estimate is impossible and the script says so and falls back to code units.
Note that `--planes` always selects on the **nominal** `y+`, whatever the
scaling — those are the numbers in `y_planes` in the config. The mapping from
nominal to actual is printed per case.

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

# one cell per sample, claiming no resolution the data does not have
python tsrs_spectra.py oc-mc-dr --map-quantity both --map-style mesh --what map
```

The standard wall-turbulence presentation: the ridge traces the energetic scale
at each height, so a controller shows up as a shift or a weakening of that
ridge. On the `mc-noctrl` baseline the `uu` peak sits at **y⁺ = 15** and the
`uv` peak at **y⁺ = 30**, both textbook.

Details that matter when reading it:

- **Both axes are logarithmic**, which is the standard presentation and the
  only one that gives the near-wall planes room. `y⁺ = 0` is therefore dropped,
  with a printed note; `u` is identically zero at the wall anyway.
- **`uv` follows `--field`.** The CLI names are `uu`/`uv`, but the co-spectrum
  is always taken against `v`, so `--field w --map-quantity uv` is labelled
  `k_zΦ_wv`.

`--map-style mesh` draws one cell per sample instead of interpolating. With
sparse `y_planes` the contour version implies a smoothness the data does not
have — the mesh version makes the actual resolution impossible to miss, and is
the honest presentation when comparing against a published DNS map. For error
bars on the map values, see `tsrs_diag.py` below.

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

**Mean removal.** `fluctuation()` offers three conventions and defaults to
`mean='time'`:

| `mean` | Subtracts | Used for |
|---|---|---|
| `'time'` | `⟨f⟩_t(x, z)`, per point | **the default.** The Reynolds decomposition. The only one that removes a *stationary spatial pattern* — the actuators sit at fixed `(x, z)`, so a controlled case carries a steady imprint of the control that the other two leave in the fluctuation and count as turbulent energy |
| `'instant'` | `⟨f⟩_xz(t)`, per instant | immune to a drifting bulk state, and forces `E(k=0) = 0` exactly |
| `'global'` | `⟨f⟩_xzt`, one number | frequency spectra, where per-instant removal would delete the low frequencies they exist to measure |

Under `'time'` the `k=0` bin holds the variance of the instantaneous plane mean
about its own time average, so it is no longer identically zero. Every plot
drops `k=0` and Parseval is checked against whatever field it is given, so
nothing is inconsistent — but the premultiplied spectrum then integrates to
slightly less than the full variance. Pass `mean='instant'` to reproduce
figures made before this became the default.

**No windowing in x and z.** Those directions are genuinely periodic, so the
DFT is exact and a window would only smear it. Time is windowed (Welch), since
it is not periodic.

### Reading what tsrs_spectra prints

```
[oc-mc-dr y+=    15] nperseg=625 var_ratio=0.59
[oc-mc-dr y+=    30] Uc=0.8463 (Uc+=13.02)  wrap_risk=0.10
```

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

## 4. `tsrs_diag.py` — is this record good enough?

```bash
python tsrs_diag.py lc_omega1_SS_oc_solonek --utau 0.0630
python tsrs_diag.py <case> --what error --blocks 6 --quantity uv
```

Prints only; nothing here draws. Five sections, `--what` selects a subset.

| Option | Default | Meaning |
|---|---|---|
| `case_ids` | *(required)* | One or more cases, each reported in turn |
| `--what` | `all` | `record,scaling,profiles,floor,error` |
| `--field`, `--planes`, `--t0`, `--t1`, `--utau`, `--scale` | | as in `tsrs_spectra.py` |
| `--blocks` | `6` | Blocks the record is split into for the error analysis |
| `--quantity` | `uu` | Which map the error analysis is of: `uu` or `uv` |
| `--modes` | `8` | How many low modes to tabulate |

**`record`** — box, sampling, mode count, and the two questions that decide
whether anything else is worth reading: how many eddy turnovers (`h/u_τ`) the
record spans, and whether `⟨u⟩_xz` has stopped drifting. Drift is judged
against the *rms*, not the mean, so it stays meaningful for zero-mean
quantities like `v`; above 0.5×rms the record is flagged and should be trimmed
with `--t0`.

**`scaling`** — reference vs actual `u_τ`, the drag reduction, and the nominal →
actual `y+` mapping. The stored `y+` labels are nominal; control changes the
real `u_τ`, so under `--scale actual` this estimate sets the units of every
figure. It is always made on `u`, whatever `--field` is, since `u = u_τ²y/ν` is
a statement about the streamwise component.

**`profiles`** — `U⁺`, `u_rms⁺`, `v_rms⁺`, `−⟨u'v'⟩⁺`, the viscous stress and
their sum against `1 − y/h`. This is the sharpest test that `u_τ` is right,
because it uses the whole profile rather than the two sublayer planes the
estimate came from. A systematic offset means `u_τ` is off — or that the two
walls are not equivalent, which is the case with one-sided control.

**`floor`** — where the spectrum stops being flow and starts being
discretisation. A resolved spectrum decays monotonically; where it flattens or
turns back up, the sampling has run past what the solver holds and those modes
are the interpolant. Reported as the first mode whose ratio to the previous one
exceeds 0.95 once the spectrum is below 1% of its peak.

```
  26     20.7   0.001522   0.771
  28     19.2   0.001176   0.924
  30     18.0   0.001190   0.998  <-- floor
  spectrum stops decaying at m=29, lam+=18.6 -- trust nothing below that
```

Note what this does *not* justify: refining `int_pos` past the solver's own
Nyquist. With `Nz` elements of order `lx1-1` there are `Nz*(lx1-1)` unique
points per periodic line, so `Nz*(lx1-1)/2` modes. Sampling finer than that
draws the polynomial interpolant, not turbulence.

**`error`** — standard error of every map value, from the scatter of block
means. The map's uncertainty is strongly scale-dependent: only `Lz/λ`
wavelengths of a given `λz` fit across the span, and the large ones decorrelate
slowest, so the outer end of the map is by far the least converged part.

```
--- error (uu, 6 blocks of 40.0 time units = 2.09 turnovers) ---
  peak +4.449 +/- 0.445 (10%) at lam+=108, y+=19.06
    5.3% of the map is above 10% relative error
    0.0% of the map is above 25% relative error
```

Two caveats. It is a **lower bound** — blocks shorter than the integral time
are not independent, so keep each block ≥1 eddy turnover and accept few blocks
rather than many short ones (the tool warns when they are shorter). And it
measures *statistical* scatter only; it says nothing about a record that has
not reached a stationary state, which is what `record` is for.

---

## 5. `tsrs_cases.ipynb` — several cases in one place

The notebook form: a case table at the top, then snapshots and spectra with
every case drawn into the same figure. `postlib/tsrs_post.py` holds the logic;
the notebook is the thin layer, laid out the way `drlrec_cases.ipynb` is.

```python
from postlib import tsrs_post as tp

cases = tp.resolve_all([
    dict(case='mc-noctrl', label='uncontrolled', style=STYLE_A),
    dict(case='oc-mc-dr',  label='opposition',   style=STYLE_B),
], runs_dir='../runs/')

wall_velocity = dict(velocity='u', tau_component='u',
                     planes=(2, 5, 10, 15, 30, 50), nwall=3,
                     max_lag_time=20.0)
tp.process(cases, planes=(0.0, 15.0), field='u', scale='reference',
           wall_velocity=wall_velocity)

tp.plot_snapshot_grid(cases, yplus=15.0)      # rows cases, cols variables
tp.plot_map(cases, quantities=('uu', '-uv'))  # rows quantities, cols cases
tp.plot_correlation(cases, yplus=15.0)        # one subfigure per case
tp.plot_wall_velocity_correlation(cases, yplus=wall_velocity['planes'])
tp.animate(cases['mc-noctrl'], yplus=15.0, field='u')
```

**Two phases, with the results tree as the boundary.** `process()` is the only
expensive call: it reads each stitched record **once** — the `lc_*` ones are
~5 GB — and writes everything derived from it into the case results folder
(section 6). Every plotting call reads *those*, memory-mapped, so re-running a
figure is free and never touches the run directory. Each step skips work whose
output already exists; `overwrite=True` forces it.

The snapshots are one plain `.npy` per `(plane, variable)` rather than a bundle
precisely so they can be memory-mapped: drawing one frame of a 3529-snapshot
record costs one page read instead of 81 MB. The per-point time mean is stored
beside each one, so the raw field is `snap[:, :, it] + mean`.

**What it does differently from the scripts, and why**

- `env_001` by default, addressed **by directory name**. `tsrs_case.find_npz`
  takes environments by position in a sorted list; with `env_005` missing,
  position 4 is `env_006` and a figure captioned env_005 is of something else.
- `scale='reference'` by default, where `tsrs_spectra.py` uses `'actual'`. In
  actual units every case is divided by its own `u_τ` and placed on its own
  `y⁺` axis, which scales the drag reduction out of the picture meant to show
  it. Both sets can live in the archive at once — the units are in the filename.
- One colour scale per column in the snapshot grid, and per row in the map, for
  the same reason. `clim='panel'` restores per-panel scaling.
- A requested plane a case does not hold is **dropped with a note**, never
  replaced by the nearest one: `oc-mc-dr` starts at `y⁺ = 5`, and drawing that
  as "the wall" would be a quietly wrong figure rather than a missing one.
- `-uv` flips the co-spectrum's sign for the picture only; the archived data
  stays sign-true.
- A case whose `current_conf.yml` no longer describes its `pts` files (edited
  `y_planes`, `Nx`, `lx1`) takes `y_planes=[...]` on its case row.
- `wall_velocity` adds the Brown--Thomas long-time correlation
  `R_{tau_xw,u}(T;y)`: signed streamwise wall shear against streamwise velocity
  at each requested wall-normal plane.  The TSRS field does not store `tau_xw`,
  so it is reconstructed as `nu*dudy|wall` with a nonuniform one-sided stencil
  through `y+=0,2,5`. A case without those planes is reported as unavailable;
  it is never replaced by a velocity proxy.  The archived result records the
  derivative weights and both three- and two-point correlation curves.

---

## 6. Results live in one place per case

```
data/results/<solver_case>/<run_name>/
    meta.yml       written by the producers (spectra, drl records)
    collect.yml    written by utils/collect-results
    config/        current_conf.yml exactly as the run used it
    model/         eval_model_<run_name>.zip, or the exported actor_*.pol
    drl/           reward / action / observation records
    spectra/       one .npz per spectrum or correlation
    tsrs/<env>/    snap_<plane>_<field>.npy   (nx, nz, nt) fluctuation
                   mean_<plane>_<field>.npy   (nx, nz) the time mean removed
                   axes.npz, snap_meta.yml
                   tsrs_<solver_case>.npz     the stitched record, archived
                   stitch.yml                 its size and mtime at archive time
    figs/          figures
```

This is the root `utils/mv-data` archives raw solver output to
(`env_<id>/res_N/`), so one case has one home and the two never collide.
`<solver_case>` is `CASENAME` from the config (`tcf`), `<run_name>` the `runs/`
folder. Neither has to be typed — both are read off the data.

Four producers fill it:

```bash
# spectra: one self-contained .npz per spectrum
python tsrs_spectra.py <case> --results

# snapshots + spectra + the archived record: tsrs_cases.ipynb, i.e.
python -c "from postlib import tsrs_post as tp; \
           tp.process(tp.resolve_all([dict(case='<case>')], runs_dir='../runs/'))"

# reward / action / observation: postlib/determine.py, i.e. deterministic.ipynb
python -c "from postlib.determine import read_deterministic_run; \
           read_deterministic_run('../runs/', ['<case>'])"

# config + model + anything the above did not place
../utils/collect-results --run_name <case>
```

`utils/collect-results` is also the catch-up path for runs that finished before
this existed. A fresh evaluation now drops `eval_model_<run_name>.zip` into
`logs/` **and into every `eval/env_XXX/`** by itself
(`src/lib/sb3_utils.py: archive_eval_checkpoint`), so which checkpoint produced
which environment's data stays traceable after new checkpoints overwrite the
generic names. For `OC` and other non-learnt runs there is no `.zip`; the
collector takes `actor_*.pol` and `drl_policy.in` instead, which is what
actually acted.

### What one saved spectrum contains

One file per spectrum — `spec_z_uu`, `spec_x_uu`, `spec_z_uv`, `mapz_uu`,
`mapz_uv`, `corr_z_uu`, `corr_x_uu`, `ycorr_uu`, `psd_u`, `uc_u` — named
`<run>_<spectrum>_<uact|uref>.npz`. Each is self-contained: the arrays as
plotted, the same arrays in code units, and **every scaling scalar as its own
variable**, so it can be replotted in different wall units years later without
the run it came from.

```python
d = np.load('..._mapz_uu_uact.npz')

d['lam_plus'], d['yplus'], d['kPhi_plus']   # exactly what was drawn
d['lam'], d['y'], d['kPhi']                 # in h and code units
d['yplus_nominal']                          # the y_planes as configured

# rescale to any other u_tau without touching the original run
d['kPhi'] / other_utau ** 2
d['lam'] * other_retau
```

| Scalar | Meaning |
|---|---|
| `utau`, `retau` | the units the `_plus` arrays are in |
| `utau_ref`, `retau_ref` | the nominal ones stored with the data |
| `nu`, `drag_reduction_pct` | tie the two together |
| `scale` | `actual` or `reference` |
| `Lx`, `Lz`, `dt`, `t0`, `t1`, `nt`, `nx`, `nz` | the record it came from |
| `case`, `source`, `written` | provenance |
| `kind`, `field`, `quantity`, `direction`, `xlabel`, `ylabel` | what it is |

The layout and the save format live in `postlib/results.py`; `--results-root`
overrides the base directory for a one-off.

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
