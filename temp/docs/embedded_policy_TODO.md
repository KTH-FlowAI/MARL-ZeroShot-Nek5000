# Embedded policy — outstanding work

Companion to [`embedded_policy.md`](embedded_policy.md). Ordered by priority
within each group. Status as of the end of the July 2026 session.

---

## A. Blocking the wing being usable

### A1. Run the wing with the four trained actors
**Smoke test complete (2026-07-28).** The raw 12-rank wing build loaded all
four trained TD3 actors (`2 → 8 → 1`) and completed two embedded interactions
from `init_1`. It wrote four per-rank recorder files and two monitor records;
the recorder has 1,632 agents and two control cycles. The launcher now selects
`meta_MARL` and its `RUN_PATH_<case_name>.txt` cache automatically for a wing
configuration (the earlier channel-only launcher could not start this case).

The configured 10,000,000-interaction evaluation is a production experiment,
not a smoke test; submit it once an allocation and output-retention plan are
chosen:

```bash
./utils/compile_case.sh --path envs/cases/small_wing_nes --solver raw
# prepare + run via nek-solo-run.sh with conf/small_wing/WING-PATH-TEST.yml
```

Expect ~0.31 s/timestep on 12 ranks, so budget accordingly.

### A2. Decide the `mask_small_wing0.f00002` fix
**Completed.** `shared_data_path` is now an ordered list: the 200k directory
remains first for the mesh/restart data, and the 75k directory is the fallback
for the only missing mask. Preparation confirms a symlink to the 75k mask and
never creates the fatal empty placeholder.

### A3. Wing embedded-vs-coupled comparison
**Initial same-restart check complete.** A two-interaction, 12-rank embedded
smoke run and a coupled smoke run both started from `init_1`; their first
independent `GAINMONITOR` reward vector agrees to `1.0e-15` absolute
(`2.6e-08` maximum relative, on the tiny `v^3` term). A long trajectory/drag
comparison analogous to the channel's 1,240-interaction result remains the
production validation.

### A4. Wing timing
The 1.57× wall-clock speedup is a *minimal-channel* number, where fixed
per-cycle MPI overhead is largest relative to solver work. On the wing each
timestep costs far more, so the ratio will likely shrink — though the exchange
also grows with rank count and control points. Measure rather than assume.

---

## B. Correctness / robustness

### B1. Move `drl_reward.f` work arrays off the stack
**Completed.** The dudy and net-gain work arrays now live in named common
blocks (`DRL_RWD_DUDY` / `DRL_RWD_NETGAIN`) in both cases. The wing raw and
coupled binaries compile with the blocks in BSS (`~3.1 MB` and `~6.3 MB`), so
reward evaluation no longer consumes the default process stack. The embedded
and coupled smoke rewards agree as described in A3.

### B2. Checkpoint chaining on resume
**Completed.** `nek-solo-run.sh --resume` now prepares the existing environment
with `embedded.resume=True` and refuses the incompatible `--skip-prepare`.
For the v19 channel it scans the local `rs?case0.f00001..00006` files, ignores
incomplete three-file sets, and writes the set with the newest saved **time**
to `CHKPFNUMBER` before `.par` is regenerated. (The solver resets `ISTEP` for
each invocation, so it cannot order separate runs.) The v17 wing retains its
own `<case>.restart` pointer and is checked rather than rewritten.

A channel raw-solver smoke produced records at 375.0400 and then, after
`--resume`, 375.0800: two recorder segments, no overlap, and the reader merged
both records successfully.

### B3. Derive `numSteps` from `nb_interactions × ndrl`
**Completed.** Before either a v17 `.rea` or v19 `.par` is written, embedded
preparation sets `numSteps = nb_interactions × ndrl` and reports the override.
The wing smoke, for example, changed 5,000,000 to 6 for two interactions at
`ndrl=3`.

### B4. Validate the SAC export against a real checkpoint
**Completed.** `runs/mc_nes_sac_fast/logs/best_model.zip` passed the offline
20,000-row SB3 comparison (`2 → 16 → 16 → 8 → 1`): f64 maximum action error
was `2.75e-07`, with no solver or MPI involved.

### B5. PPO export
**Intentional limitation.** PPO export deliberately raises
`NotImplementedError` and is not supported by embedded-policy evaluation.
Unlike the supported deterministic actors, PPO's activation comes from
`policy_kwargs` (tanh by default, often ReLU) and its output is clipped to the
action space rather than squashed. Keep using the coupled Python path for PPO.

---

## C. Integration / usability

### C1. `sjob-gen.sh` support for `nek-solo`
**Completed.** `sjob-gen.sh --case nek-solo` wraps `nek-solo-run.sh`; automatic
node sizing uses exactly `nproc` because embedded evaluation has no extra
Python rank.

### C2. Drop `YWDEBUG` from the wing `PPLIST` for production runs
**Completed.** Production builds use `MPIIO DRL UTAU NETGAIN GAINMONITOR`.

### C3. `GAINMONITOR` on the wing
**Completed.** The wing now writes a rank-0 monitor at the end of each control
cycle. Its two smoke-test vectors agree with the binary recorder to
`1.16e-08` relative (the text monitor's `E16.8` formatting is the limit).

### C4. Recording in coupled/training mode
**Completed (opt-in).** Set `embedded.coupled_recorder: true` with
`embedded.enabled: false` to write the existing per-rank `drlrec` binary format
while Python continues to select and send the actions. Preparation writes the
small `drl_record.in` configuration; no `.pol`, F77 actor, or embedded solver
is involved. The writer copies the action actually applied by Python and writes
the reward at each completed control cycle. It opens a new segment across an
episode reset, while empty reset-only segments are safely skipped by the
reader.

A 10-rank channel coupled smoke built the modified channel and wing solvers,
then recovered one 400-agent, three-rank record with the expected observations,
actions, reward and time from `read_drlrec.py`. The option remains false by
default, so existing training I/O is unchanged.

---

## D. Physics questions raised, not answered

### D1. Revisit `drl_steps` now that the MPI cost is gone
`conf/small_wing/WING-PATH-TEST.yml` carries
`drl_steps: [4, 5, 6, 9, 4]` with the comment *"make ndrl=4 therefore mpi time
will be reduced!"*. Those per-region intervals were chosen partly to economise
on MPI — a cost embedded mode removes. Worth re-deriving them on physical
grounds. `pol_nupd` reproduces whatever is chosen either way.

### D2. `drl_reward` cost
`compute_netGain_posOnly` runs **every timestep**, doing a `gradm1`, a `mappr`
and four `z_averaging` calls (the `x_averaging` calls beside them are dead:
`rwd_xavg` is a `parameter(...=.FALSE.)`, folded away at compile time).

**Partly addressed (2026-07-30): `w1` is now cached in `z_averaging`.** Each
call used to `gop` twice — once for the field, once for a quadrature-weight
array `w1` built from `dz2` (=1) and `wzm1`. That array depends only on the
GLL weights and the `lglel -> (ex,ey)` map, so it is constant for the whole
run, yet half of the reward's collective traffic went on re-reducing it. It
is now built on the first call and reused. Both wing cases are patched.

Measured with a standalone MPI replica of the routine at each case's real
sizes on 12 ranks (4 calls per timestep):

| | small wing (`lx1=4`, 0.43 MB/gop) | production (`lx1=12`, 5.2 MB/gop) |
|---|---|---|
| before | 3.82 ms/step | 125.0 ms/step |
| `w1` cached | 1.33 ms/step | 60.2 ms/step |

On the small wing that is ~0.8 % of a 0.44 s timestep — below run-to-run
noise, and a 40-interaction run confirmed no measurable wall-clock change.
It is worth having for the production case, where the message is 12× larger
and the real run is ~3,840 ranks rather than 12.

Validation: a 40-interaction, 12-rank `WING-PATH-TEST.yml` run (1,632 agents,
four policies at `drl_steps` 4/5/6/9, 120 timesteps) reproduced all four
`drlrec` segment files and `reward_monitor00000.dat` **bit-for-bit**.

Still on the table, not done: the Step-3 kinetic-energy term is a function of
`ACTIONS` alone, so it is constant within a control cycle and its
`z_averaging` could be hoisted to `i_evolv.eq.1`. Measured at a further
0.34 ms/step (small wing) / 15 ms/step (production) — a real but much smaller
win than the `w1` cache, and unlike it, only equal to round-off rather than
bit-exact.

**Latent bug, unrelated to the above, left alone:** `drl_action.f:321` calls
`z_averaging(velV,avgVX)` where `avgVX(LX1,LY1,ynel,znel)` holds 48 reals but
`z_averaging` writes `LX1*LY1*xnel*ynel` (56,000 on the small wing). It is
unreachable only because `rwd_xavg` is `.FALSE.` at compile time. Anyone who
switches streamwise averaging on will corrupt memory. Present in both wing
cases.

---

## E. Housekeeping

Scratch artefacts created during this work, safe to delete:

| Path | What |
|---|---|
| `runs/solo_timing_cmp/` | coupled-vs-embedded timing comparison |
| `conf/mini_channel/_timing_cmp.yml` | its config |
| `runs/wing_solo_smoke/` | wing OC smoke test |
| `runs/nek_solo_rec/` | moved here from `~/.cache`; **v1-format** records (pre-segment layout, readable) |
| `~/.cache/nek_solo_*`, `~/.cache/nek_rep_*`, `~/.cache/wing_polgen` | staged test runs |
| `~/.cache/nek_pol_selftest/` | offline test build dir (recreated on demand) |

Note the `.pol` format moved to version 2 (adds a version line and the
observation permutation). `.pol` files are always regenerated by the prepare
step, so no migration is needed — but any hand-kept v1 file will be rejected
with `ierr = 16`.

The **record** format is at v2 as well (adds the segment index). The reader
still accepts v1 files; that branch is covered by a synthetic-file test, since
no real v1 data survives.
