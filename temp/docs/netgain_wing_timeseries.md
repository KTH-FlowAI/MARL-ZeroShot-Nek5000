# Net-Energy-Saving on the Wing — Per-Chord-Point Time-Series Logging

**Date:** 2026-06-09
**Branch:** `dev_yw_cost`
**Author:** yuningw (with Claude Code)

## Goal

The net-energy-saving (NETGAIN) reward had already been validated in both
Python and Fortran for the wing. We wanted a **time-series of the three reward
components** written in the spirit of the channel monitor
`envs/cases/lc_n7/drl/drl_reward.f:452` (`write_reward_monitor`).

The channel case writes **one line per control step** with three scalars,
because the flow is homogeneous in **both** x (streamwise) and z (spanwise)
(`rwd_xavg=.TRUE., rwd_zavg=.TRUE.`), so every agent holds the identical value
and a global mean recovers it.

**The wing is different:** it is homogeneous **only in z** (spanwise);
`small_wing_nes/inc_src/DRL` sets `rwd_xavg=.FALSE., rwd_zavg=.TRUE.`. The
chordwise direction is inhomogeneous, and the control band (`ctrlys=-0.1 →
0.6`) spans both suction and pressure sides. So each chord location has its own
value and we need a **profile per timestep**, not a single scalar.

## Key decision: implement in Python, not Fortran

The Fortran→Python coupling already ships everything needed, so no Fortran
change was required:

- **At handshake** (`small_wing_nes/drl/drl_IO.f`): per-agent positions
  (`x`, `y`, `z`) + `NID/GLLID/FACEID/ix/iy/iz`, assembled into
  `self.agent_info` and dumped to `NODE_INFO.csv`. The cross-rank gather and
  ordering problem is already solved here.
- **Every control step** (`i_evolv == ndrl`): per-agent `rwd_tau`, `rwd_pw`,
  `rwd_v3` arrive (MPI tags 80000/81000/82000) and become `tau_dist`,
  `pw_dist`, `v3_dist` keyed by `agent_name`.
- There is already a per-episode save hook in `reset()` (`rewlog_*.npz`).

Doing it in Fortran would mean re-implementing a worse version of this gather
(agents are scattered across MPI ranks with no chordwise ordering).

## What the values mean

`rwd_tau/rwd_pw/rwd_v3` are the **running mean over `i_evolv` within one
control step** (see `compute_netGain_posOnly` in `drl/drl_reward.f`). So each
saved row is one sample **per DRL control step** — same time resolution as the
channel monitor.

- `tau` = wall-shear-stress term `tau_w = mu * |dU/dy|` (rotated to wall frame)
- `pw`  = pumping-power term `|p'_w * v_w|` (negative values filtered to 0)
- `v3`  = kinetic-energy term `0.5 * |v_w^3|`

## Implementation

### 1. `src/nek_marl_meta_netgain.py`

**`_build_chord_map()`** (runs once at init, after `possible_agents` is built)
- z-averaging makes all spanwise-duplicate agents (same `(x, y)`, different
  `z`) hold identical values. Group them by rounded `(x, y)` (8 decimals).
- Each unique chord point stores: representative `x`, `y`, a `side` label, and
  the list of agent names in its group (logged value = group mean = that
  shared value).
- **Side classification (chosen rule):** `suction` if `y >= chord-line y`,
  else `pressure`, where the chord line is the leading→trailing edge segment
  (min-x agent → max-x agent). The saved `x, y` allow any other split offline.
- Output sorted by side then chord position. Prints a summary of point counts.

**Per-step logging in `evolve()`** — where `tau_dist/pw_dist/v3_dist` exist,
appends one profile row per component (group-mean per chord point) to
`self.netgain_log`.

**Fixed-frequency flushing** (added per the wing's reality that episodes
usually never finish and the buffer can get huge). Mirrors the
`MetaPolicy._save_buffer` / `io_iter` pattern:
- `_save_netgain_ts()` writes the buffer to `history/netgain_ts_<ioiter>.npz`
  (monotonic chunk numbering), tags each chunk with its `episode`
  (`restart_index`), then clears the buffer. No-op on empty buffer or
  non-`net_gain` runs.
- `evolve()` flushes once the buffer reaches `netgain_io_freq` rows → memory is
  bounded by `netgain_io_freq × nChordPts × 3`, independent of episode length.
- `reset()` flushes the tail **before** bumping `restart_index` (so the chunk
  is tagged with the episode that just ran).
- `close()` flushes the remainder — the main save path on the wing, since the
  job usually ends mid-episode.

**npz chunk contents** (`history/netgain_ts_<ioiter>.npz`):
| key       | shape             | meaning                                  |
|-----------|-------------------|------------------------------------------|
| `episode` | scalar            | episode (restart) index                  |
| `step`    | `[n_step]`        | control-step index within that episode   |
| `tau`     | `[n_step, nPts]`  | wall-shear-stress term                    |
| `pw`      | `[n_step, nPts]`  | pumping-power term                        |
| `v3`      | `[n_step, nPts]`  | kinetic-energy term                       |
| `x`       | `[nPts]`          | chord x of each column                    |
| `y`       | `[nPts]`          | wall y of each column                     |
| `side`    | `[nPts]`          | `'suction'`/`'pressure'` per column       |

### 2. `src/configs_meta.py`

Added runner field:
```python
netgain_io_freq: int = 200  # Flush per-chord-point NETGAIN time-series every N control steps
```
Tune lower for more frequent on-disk checkpoints, higher for fewer/larger
files. Read via `getattr(self.conf.runner, 'netgain_io_freq', 200)`.

### 3. `post_processing/netgain_ts.py` (new)

Loader + plotter, matching existing post-processing style (argparse, serif rc,
`Figs/` output).
- `load_netgain_ts(path)` — load one chunk into a dict (sides decoded to str).
- `load_run(run_dir, episode=None)` — glob and concatenate all chunk files in
  time order (x/y/side from the first chunk; tau/pw/v3/step stacked), optional
  per-episode filter.
- `plot_spacetime(data, png)` — chord-x vs sample-index contour, rows =
  suction/pressure, cols = tau/pw/v3. Uses a contiguous sample index on the
  y-axis because per-episode `step` resets and is not monotonic across chunks.
- `plot_mean_profile(data, png)` — time-averaged chord profile, suction vs
  pressure overlaid.

**CLI:**
```bash
python netgain_ts.py --run ../runs/302000              # concat all chunks
python netgain_ts.py --run ../runs/302000 --episode 1  # one episode only
python netgain_ts.py --file path/to/netgain_ts_00000.npz
```

## Files touched

- `src/nek_marl_meta_netgain.py` — chord map, per-step logging, frequency flush
- `src/configs_meta.py` — `netgain_io_freq` config field
- `post_processing/netgain_ts.py` — new loader/plotter

**No Fortran change** — the existing coupling already provides per-agent
components and positions.

## Validation done

- Python files syntax-clean (`py_compile`).
- Chord-map dedup + side logic verified on a synthetic airfoil point set (12
  raw `(x,y,z)` agents → 6 unique chord points, correct suction/pressure split,
  sorted output).
- Loader/plotter verified end-to-end on synthetic chunks: 3 chunks (130 steps)
  concatenate correctly; `--episode 0` isolates its 2 chunks (100 steps); both
  figures render.

## Not yet done / to verify on a real run

- Has **not** been run against a live wing simulation (no real
  `netgain_ts_*.npz` exist yet — produced once an episode runs with
  `reward_fn: net_gain`).
- On the first real run, check:
  - `[STB3] CHORD MAP: ... suction / ... pressure` counts look right.
  - `[STB3] SAVE NETGAIN TS chunk=... ep=... (N steps)` cadence matches
    `netgain_io_freq`.
  - Side split is sensible (the LE→TE straight-chord assumption is fine for
    labeling; reclassify offline from saved `x,y` if a camber-based split is
    ever wanted).
