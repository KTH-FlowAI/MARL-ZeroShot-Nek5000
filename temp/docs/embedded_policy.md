# Embedded policy — deterministic control inside Nek5000, without Python

Reference for the "nek-solo" evaluation mode: Nek5000 evaluates the trained
actor itself, so a deterministic evaluation runs as a plain MPI job with no
Python rank, no `MPI_Comm_spawn`, and no per-cycle message exchange.

Companion file: [`embedded_policy_TODO.md`](embedded_policy_TODO.md).

---

## 1. Why this works at all

The actor is an 8-neuron MLP over a 2-component observation — about 40 FLOPs
per agent per control cycle. More importantly it is **pointwise in the agent
index**: every rank already holds `val_obs(1:2, 1:NUMCTRL)` for its own agents
after `sensing_pts_compute`, and already writes its own `act_buffer` into
`ACTIONS`. Python sat between those two points doing no reduction at all (for
`znmf_avg = 1` the ZNMF averaging is done by Nek, not Python).

So the entire round trip collapses into a local function call. Nothing about
the physics changes.

---

## 2. What was measured

| Check | Result |
|---|---|
| Fortran actor vs SB3 `predict`, 20 000 obs | max 8.76e-08, rms 4.41e-09 (action full scale 0.128) |
| Weight ASCII round-trip | 33/33 bit-exact |
| **Channel: drag reduction, 1240 interactions from `init_1`** | **coupled +46.8270 %, embedded +46.8270 %, diff 8e-06 pp** |
| Trajectory divergence over the same run | 7.1e-09 (early) → 5.1e-07 (late) |
| Speed, 200 interactions, 10 ranks | **1.57× wall clock, 1.41× per timestep, ~38 % less CPU** |
| Recorder: 80 000 (obs, action) pairs replayed through SB3 | max 2.05e-08 |
| Recorder vs the independent `GAINMONITOR` writer | 1.3e-08 relative, identical time arrays |
| Same case at 10 vs 12 ranks after canonical ordering | identity keys identical, data agrees to 1e-14 |
| Opposition control vs `lib.AFC.OppoCtrl` (offline) | max 5.0e-11 |
| Opposition control on the wing, on refresh cycles | **0.000e+00 — bitwise** |
| Dedalus observation reversal vs SB3 fed flipped obs | max 5.4e-08 |

The residual ~1e-07 in the learnt-policy cases is **not** a precision choice: it
comes from summation order in the 8-term dot product and from libm `tanh`
differing from torch's kernel. Measured, float32 and float64 accumulation are
equally far from torch (float32 is actually *worse* in the maximum), so the
default is float64 — see §7.

---

## 3. Files

### New

| File | Role |
|---|---|
| `src/lib/pol_export.py` | export library: SB3 → `.pol`, analytic policies, `drl_policy.in` |
| `utils/sb3_to_f77.py` | thin CLI over the above |
| `utils/pol_selftest.f` / `.sh` | offline test: replays a reference table through the Fortran, no Nek, no MPI |
| `envs/cases/<case>/drl/pol_core.f` | **include-free** network reader + forward (f32/f64) |
| `envs/cases/<case>/drl/pol_net.f` | Nek-side: config read, rank-0 load + broadcast, agent→region map |
| `envs/cases/<case>/drl/pol_main.f` | control scheduler and mode entry point |
| `envs/cases/<case>/drl/pol_IO.f` | per-rank binary trajectory recorder |
| `envs/cases/<case>/inc_src/POLICY` | common block (`LPOL=8`, `LNLAY=6`, `LNW=64`) |
| `post_processing/read_drlrec.py` | reader: merges ranks and segments |
| `post_processing/drlrec_analysis.ipynb` | analysis notebook |
| `execs/nek-solo-run.sh` | launcher |

`pol_core.f` has no includes and no common blocks **on purpose**: the offline
test links it alone, so the network arithmetic is proven before any solver is
involved.

### Modified

| File | Change |
|---|---|
| `drl/drl_action.f` | `recv_Actions` split into the MPI receive plus a shared `apply_actions(act_buffer)` — behaviour-preserving |
| `drl/drl_main.f` | branch to `POL_main` on the mode switch, **before** `MPI_INTERCOMM_CREATE` |
| `drl/drl_IO.f` | `drl_reward_out` returns early in embedded mode (also drops the per-step CFL hand-shake) |
| `compile_script`, `makefile_usr.inc` | `--solver raw\|drl`, new objects |
| `utils/compile_case.sh` | `--solver` passthrough, prompt suppression, shebang fix |
| `src/configs.py`, `src/configs_meta.py` | `Embedded` block |
| `src/initial.py`, `src/initial_meta.py` | pass `embedded` and `logging` into `NEK_INIT` |
| `src/lib/nek_utils.py` | honour `exeName`; `UPARAM(10)` (v19) / `p091` (v17); emit `.pol` + `drl_policy.in` + `drlrec/`; meta-aware policy table |

---

## 4. The two solver builds

The DRL solvers (`KTH_DRL_Framework`, `DEC_DRL_Nek5000`) are patched in ways
that make a Python-free run impossible:

* `drive.f` splits `MPI_COMM_WORLD` and **reserves global rank 0 for Python** —
  that rank never enters `nek_init`.
* `nek_solve` ends with an unconditional `goto` back to the top of the time
  loop (the episode-restart mechanism), so it never returns; everything after
  it, including the post-processing branch, is unreachable.
* `prepost.f` carries an output-index reset that only exists because
  `drl_chkpt_reset` sets `ISTEP = 0` mid-run.

The **raw** solvers (`KTH_Framework`, `DEC_Nek5000`) have none of that and are
stock. So embedded mode builds against those:

```bash
./utils/compile_case.sh --path envs/cases/mini_channel --solver raw   # -> nek5000_solo
./utils/compile_case.sh --path envs/cases/mini_channel --solver drl   # -> nek5000
```

Both binaries live in the same case folder from the same sources. `BINNAME`
and `LIBNAME` are overridden through `MAKEFLAGS` (GNU make treats those as
command-line assignments, which beat the ones baked into `makefile.template`),
so **the solver tree is never edited** — important, because it is gitignored
and regenerated from the tarballs by `utils/initialize_solver.sh`.

Switching solvers forces a from-scratch rebuild, which costs nothing extra
because `compile_case.sh` always cleans first. `make clean` only removes the
binary its own `BINNAME` names, so the other build survives — verified
byte-for-byte.

---

## 5. The timing contract

The coupled mode's cadence is driven by the request stream from Python. Traced
through `DRL_main`, it reduces to:

```
i_evolv = mod(ISTEP - 1, ndrl) + 1        update when i_evolv == 1
```

so control updates land on `ISTEP = 1, ndrl+1, 2*ndrl+1, …`, and the reward
moving average covers samples `1..ndrl` of each cycle — **including** the
detail that `drl_reward(1)` fires before the first solve with the new boundary
condition. `pol_main.f` reproduces exactly this. Getting it wrong changes the
drag reduction, which is why it is spelled out in the file header.

Per-region update intervals (`drl_steps` in a meta config → `pol_nupd`) work on
top: region *k* refreshes on the first interaction and then every
`pol_nupd(k)` interactions, holding its previous action in between. This
mirrors `MetaPolicy._partial_predict` and is verified on the wing.

---

## 6. File formats

### `.pol` — one actor

Plain text, always regenerated by the prepare step, never hand-edited.
Comments start with `#`; data is list-directed.

```
2                       format version
1                       number of linear layers
2 1                     layer widths, first entry is the input width
0                       activation after each layer: 0=identity 1=relu 2=tanh
0 -1.0 1.0              squash flag, action-space low, high
0.064 0.064             scl_obs (divide), scl_act (multiply)
1 2                     observation permutation (1-based); 2 1 = Dedalus order
0.0 -1.0                W, row-major
0.0                     b
```

Weights are the **exact float32 values from the checkpoint**, written at 9
significant digits so they round-trip bit-exactly. Because the activation and
the squash flag are data rather than hardcoded, classical laws are expressible
— see §9.

`squash=1` applies SB3's `unscale_action`, `low + 0.5*(a+1)*(high-low)`. That
looks like the identity for `[-1, 1]` but is **not** in float32: `(a+1)` rounds
to a multiple of 2⁻²³, so SB3's actions are quantised at ~1.2e-07. Reproducing
it is required for agreement; skipping it would be a systematic error.

PPO is intentionally unsupported in embedded mode. Its configurable hidden
activation and Box-clipped (rather than squashed) output are not represented
by this export path; run PPO through the coupled Python controller.

### `drl_policy.in` — the run

```
# nb_interactions  rec_freq  rec_bufsize  iprec
200 1 100 8
# reward_mode(0=dudy,1=net_gain)  dudy_ref  alpha beta gamma
1 11.75 1.0 1.0 1.0
# number of policies
4
# xmin xmax iside utau amp nupd 'file.pol'      iside: 0=any 1=SS(y>0) 2=PS(y<0)
0.23 0.40 1 1.0 1.0 4 'actor_000.pol'
...
```

### `drlrec/drlrec_sNNNNN_pNNNNN.bin` — the trajectory

One file per rank per segment, stream access, native endianness (marked in the
header). Header carries the per-agent identity; records carry
`time, istep, icycle, obs(nfld,nagent), act(nagent), rwd(3,nagent)`.

Three design points:

1. **Per-rank, no gather.** No collective on the write path and no rank-0
   serialisation point, so the recorder stays off the critical path.
2. **Partition-independent identity.** Each agent is stamped with
   `(ieg, iface, ix, iy, iz)` — global element, face, GLL indices — which does
   not change with `nproc`. The reader sorts on it, so a run on 12 ranks lines
   up column-for-column with a run on 10. *Verified: keys identical,
   coordinates bit-identical, data agreeing to 1e-14.*
3. **Segments.** Each run claims the next free segment index instead of
   truncating, so a job killed on wall clock and resumed leaves `s00000` intact
   and writes `s00001`. The reader stitches them in time order and **warns on
   overlap** rather than silently splicing two trajectories.

---

## 7. Precision

Weights are stored as float32 (exactly what the checkpoint holds). The
**accumulation** precision is a separate axis, set by `embedded.net_precision`:

* `8` (default) — float64 accumulation. Smaller worst-case error against SB3,
  identical rms, and it is the more faithful evaluation of the same network.
* `4` — float32, matching torch's own precision. Lands on torch's exact
  rounding more often (81 % vs 48 % of samples bit-identical) but has a
  *larger* maximum deviation.

Bitwise agreement with torch is not reachable either way, and the difference is
~1e-07 on an action of full scale 0.128. FMA is not the cause —
`-ffp-contract=off` and `=fast` give byte-identical results.

---

## 8. Configuration

Everything derives from the run YAML; `.pol` and `drl_policy.in` are generated,
never hand-edited.

To continue an interrupted embedded run, use `nek-solo-run.sh --resume`. It
keeps the existing environment folder, creates the next recorder segment, and
for the v19 channel build selects the newest *complete* three-file checkpoint
set by saved time before rewriting `.par`. (Each solver invocation resets its
step count.) The v17 wing uses its existing `<case>.restart` pointer. Do not
combine `--resume` with `--skip-prepare`.

### Single-region (channel, `configs.py`)

```yaml
embedded:
    enabled: True            # all the minimal channel needs
    net_precision: 8         # 8 (default) or 4
    rec_freq: 1              # record every N control cycles
    rec_bufsize: 100         # records between flushes
    resume: False            # set by nek-solo-run.sh --resume
    coupled_recorder: False  # opt-in binary recorder for Python-coupled runs
```

The checkpoint resolves as `runs/<agent_run_name>/logs/<policy>`.

`coupled_recorder` is independent of `enabled`: leave `enabled: false` for a
normal Python-coupled training/evaluation run and set only
`coupled_recorder: true` to get `drlrec/` output. Its records contain the
actions Python actually supplied, and the existing reader handles the resulting
segments. It is off by default and does not require an exported actor.

### Multi-region (wing, `configs_meta.py`)

The region table is read from the Runner fields MetaPolicy already uses —
`agent_ctrl_area`, `agent_ctrl_side`, `u_tau`, `action_bounds`, `drl_steps`,
`source_solvers` — with checkpoints resolved as
`<logging.policy_dir>/<agent_run_name>/logs/<policy>.zip`. So a wing config
also needs only `embedded.enabled: True`. The `embedded.*` list fields exist
only to override that per region.

Configs habitually carry one spare entry (5 `u_tau` values for 4 regions);
extras are trimmed rather than treated as an error.

---

## 9. Classical control (opposition, steady blowing)

Supported with no new code, because the activation and squash are data:

| Law | `.pol` |
|---|---|
| `OppoCtrl`: `action = -amp · v'/u_tau` | 1 layer, identity activation, `W = [0, -1]`, `b = 0`, `squash = 0` |
| `BLCtrl`: `action = amp` | `W = 0`, `b = 1` |

```python
from lib.pol_export import write_analytic_pol
write_analytic_pol('actor_000.pol', 'OC', u_tau, ctrl_max_amp)
```

`lib.AFC.SinWave` is **not** expressible — it is a function of position and
time, not of the observation, and would need actual code.

Because OC has an exactly predictable answer, it is the recommended smoke test
when bringing up a new case.

---

## 10. Transferred (Dedalus) policies

`MetaPolicy._obs_solver_arrange` reverses the observation field order for a
Dedalus-trained policy (`np.flip(obs, axis=1)`), i.e. it feeds `(v', u')` where
Nek presents `(u', v')`. That is carried as an explicit permutation line in the
`.pol`, driven by `source_solvers` in the config.

It is deliberately **not** folded into the weight matrix: doing so would break
the "weights are exactly the checkpoint's" property and the bit-exact
round-trip test that depends on it.

---

## 11. v17 (wing) vs v19 (channel)

The wing DRL has **no `UPARAM` at all** — it reads `PARAM(nn)` straight from the
`.rea`. (`uparam.f` in the wing case is the KTH `.upar` namelist reader used by
the toolbox modules; a different mechanism.)

| | channel (v19) | wing (v17) |
|---|---|---|
| ndrl | `UPARAM(1)` | `PARAM(89)` |
| znmf | `UPARAM(2)` | `PARAM(90)` |
| reward mode | `UPARAM(9)`, runtime | `#ifdef NETGAIN`, compile time |
| **control mode** | `UPARAM(10)` | **`PARAM(91)`** (was a free `.rea` slot) |

`rewrite_REA_v17` writes `p091` explicitly rather than through
`V17_PARAM_MAPPING`, because its value comes from the Embedded config rather
than from `Simulation`.

Production wing builds use `GAINMONITOR` and omit `YWDEBUG`: the former writes
a compact rank-0 reward cross-check each cycle, while the latter creates
per-rank debug files and low-step `outpost` output. The wing smoke recorder
and monitor agree to `1.16e-08` relative.

---

## 12. Running

```bash
# build once
./utils/compile_case.sh --path envs/cases/mini_channel --solver raw

# run
./execs/nek-solo-run.sh --config conf/mini_channel/MC-nes.yml --nenv 4
./execs/nek-solo-run.sh --config conf/mini_channel/MC-nes.yml --dry-run

# continue an existing embedded environment from its latest complete restart
./execs/nek-solo-run.sh --config conf/mini_channel/MC-nes.yml --resume

# generate and submit the corresponding SLURM job (no Python MPI rank)
./execs/sjob-solo --config conf/mini_channel/MC-nes.yml \
    -J solo-nes -t 24:00:00 --submit -- --nb-interactions 20000
```

### Large-channel (lc_n7) deployment

The Reτ=207 omega1 suction-side energy-gain evaluation is configured in
`conf/omega1_BASE/lc-omega1-SS-eng-embedded-lc_n7.yml`. It uses the
`envs/cases/lc_n7` source, links `data/simulations/large_channel/` for the
mesh, and expects the source controller under `runs/300101/logs/best_model.zip`.
On the HPC checkout, build the raw solver once, then generate and submit the
512-rank job:

```bash
./utils/compile_case.sh --path envs/cases/lc_n7 --solver raw
./execs/sjob-solo \
  --config conf/omega1_BASE/lc-omega1-SS-eng-embedded-lc_n7.yml \
  -J lc-omega1-solo -t 24:00:00 --submit
```

The config derives 30,000 steps from 2,500 interactions × `ndrl=12`, records
every tenth control cycle to limit trajectory output, and writes a checkpoint
every 6,000 steps for `--resume`.

The matching analytic opposition-control deployment is
`conf/omega1_BASE/lc-omega1-SS-oc-embedded-lc_n7.yml`. It writes a one-layer
identity actor at preparation time and therefore needs no SB3 checkpoint. Its
`embedded.u_tau: [1.0]` and `embedded.ctrl_max_amp: [1.0]` deliberately make
the law `action = -v'`, exactly matching the legacy
`lc-omega1-SS-oc.yml`/`lib.AFC.OppoCtrl` setup.

Per env rank the launcher runs `python -m nek_MARL initial` for a channel or
`python -m meta_MARL initial` for a wing, then stages the mode switch, `.pol`
files, `drl_policy.in`, and `drlrec/`. It then runs
`mpirun -n <nproc> ./nek5000_solo`. **Every rank runs the solver** — rank 0 is
not reserved.

It refuses to launch if `nek5000_solo` is missing, rather than letting the
coupled binary hang forever in `MPI_INTERCOMM_CREATE` waiting for a Python rank
that will never connect. It also raises the stack limit — see §14.

### Reading the results

```bash
python3 post_processing/read_drlrec.py runs/<name>/eval/env_001
```

```python
from read_drlrec import load_run
rec = load_run('runs/<name>/eval/env_001')
rec.obs      # (nrec, nagent, nfld)
rec.act      # (nrec, nagent)
rec.rwd      # (nrec, nagent, 3)   tau_w, |p'v|, 0.5|v^3|
rec.agents   # DataFrame: nid, ieg, iface, ix, iy, iz, ipol, x, y, z
rec.drag_reduction(dudy_ref=11.75, nu=1/2900, t_start=...)
```

`post_processing/drlrec_analysis.ipynb` covers agent layout, reward components
and drag reduction, action statistics and a spanwise space-time map,
observation distributions, and the learned control law. Point `RUN` at your
run folder.

---

## 13. Offline validation

```bash
./utils/pol_selftest.sh --ckpt runs/mc_nes_nek/logs/best_model.zip \
                        --config conf/mini_channel/MC-nes.yml
```

Exports the checkpoint, then replays a reference table produced by **SB3's own
`predict`** through `pol_core.f` — so the Fortran is checked against the real
evaluation path, not against a re-implementation of it. Reports max/rms
deviation in both precisions and the count of bit-exact rows.

It builds under `~/.cache` because `/tmp` is mounted `noexec` on these nodes.

---

## 14. Gotchas worth knowing

**Stack, not a leak (fixed).** The large dudy/net-gain work arrays are now in
the named `DRL_RWD_DUDY` and `DRL_RWD_NETGAIN` common blocks, hence BSS rather
than the process stack. This removes the wing's former ~6.7 MB per-call stack
frame and its `lelt` sensitivity. The launcher still raises the stack limit as
a harmless safeguard for unrelated solver work.

**F77 column 72.** Fixed-form source is truncated at column 72 without warning
— a long `read` statement silently loses its last variable. Check new sources:
`awk 'length($0)>72 && $0 !~ /^[c!#*C]/' file.f`.

**`# !/bin/bash` is not a shebang.** Interactive bash runs such files itself, so
this hides until the script is invoked under `timeout`/`srun`/`xargs`, where
`execvp` falls back to `/bin/sh` (dash) and `[[ ]]` fails. Fixed in
`utils/compile_case.sh`; worth grepping for elsewhere.

**`OmegaConf` returns `ListConfig`**, which is a Sequence but *not* a `list`
subclass — `isinstance(x, list)` is False and silently skips the branch.

**Missing shared data becomes a bizarre error.** If a file listed in
`get_Case_Files` is absent, Nek creates a 0-byte file on open and dies with
`NONSTD HDR, parse_hdr, abort / ABORT: too many restart files!`, which points
nowhere near the cause. Check the `[IO] WARNING: optional ... not FOUND`
line in the prepare log first.

**`.ipynb` `source` lines keep their trailing newline.** Splitting on `\n`
without `keepends` glues a whole cell into one unparseable line.
