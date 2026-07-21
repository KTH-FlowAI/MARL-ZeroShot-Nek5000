# Job submission refactor — payload scripts + SLURM generator

**Author:** wangyuning
**Branch:** `dev_yw_cost`
**Date:** 2026-07-21
**Scope:** `execs/` (new files only — nothing existing was deleted)

---

## Motivation

Submitting a job meant copying one of the `execs/sjob-*.sh` files and editing it
by hand. Each copy carried a full duplicate of the run logic (environment,
`initial` call, `RUN_PATH` lookup, `mpirun` MPMD line), so:

* **The job name, partition and begin time were buried** in a header that also
  contained the payload — changing `-J` or `--begin` meant editing a 100-line
  file, and the log file names (`--output=log-files/train-%j`) usually stayed
  stale from the previous experiment.
* **Node count was baked into the file name** (`sjob-train-n7.sh`,
  `sjob-train-n11.sh`) although it is fully determined by `nproc` in the config.
* **Run logic drifted between copies.** `sjob-eval-n7.sh` forced
  `reward_fn=net_gain` only on the `evaluate` call, while `run-script` forced it
  on `initial` *and* `evaluate`; the statistics variant hard-coded its case list
  in a `CASE_LIST=(...)` array that was edited in place and committed.
* **`run-script` was not the same code path as the batch scripts**, so a case
  that ran interactively could still fail under `sbatch`.
* The old generator in `utils/sjob_gen/` (python + shell wrapper) still emitted
  the pre-refactor layout: `../conf/$CONFIG`, `RUN_PATH.txt` at the CWD, and the
  shell-side `mv history roundXXX` archiving that now lives in
  `src/initial.py:preserve_and_clean_train`.

## Design

Split **what to run** from **how it is queued**:

```
execs/channel-run.sh ─┐                        ┌─ #SBATCH header (generated)
execs/wing-run.sh   ──┴─ payload, no SLURM ←───┤
                                               └─ execs/sjob-gen.sh
```

* A **payload** script owns the environment and every `mpirun` line. It has no
  `#SBATCH` directive, and picks its environment flavour from `$SLURM_JOB_ID`
  (`--site local|hpc` overrides). The same command line therefore runs
  interactively and inside a batch job — one code path, not two.
* The **generator** writes an sbatch file whose only body is one call to a
  payload. All queue-facing knobs are command-line flags.

Generated scripts are ordinary sbatch files: editable by hand when a one-off
needs something the generator does not expose.

---

## 1. `execs/channel-run.sh` — minimal channel (`python -m nek_MARL`)

Replaces `run-script`, `sjob-train.sh`, `sjob-train-n7.sh`, `sjob-train-n11.sh`,
`sjob-eval.sh`, `sjob-eval-n7.sh` and `sjob-stat-n7.sh`.

```bash
./execs/channel-run.sh --config conf/MC-ng-111.yml --mode train
./execs/channel-run.sh --config conf/MC-ng-111.yml --mode evaluate --nenv 2 \
                       --iostep 10000 --smpstep 12
./execs/channel-run.sh --mode evaluate --nenv 1 --nb-interactions 20000 \
                       --config conf/MC-shapcf.yml --config conf/MC-shapvel.yml
```

| Flag | Meaning | Default |
|---|---|---|
| `--config PATH` | config path; **repeatable / comma-separated** (replaces `CASE_LIST`) | `conf/MC16-TD3.yml` |
| `--mode` | `train`\|`run` (both → `nek_MARL run`) or `evaluate` | `train` |
| `--site` | `local`\|`hpc` | auto from `$SLURM_JOB_ID` |
| `--load-agent` | `True`\|`False` → `runner.load_agent` | config value |
| `--nenv N` | evaluate: loops `runner.rank = 1..N` | `2` |
| `--iostep`, `--write-interval`, `--smpstep` | `simulation.*` | `5000`, follows `--iostep`, `6` |
| `--reward-fn`, `--alpha`, `--beta`, `--gamma` | evaluate reward | `net_gain`, `1.0`×3 |
| `--random-init`, `--nb-interactions` | evaluate overrides | `-1`, config value |
| `--mpi-opts "FLAGS"` | replaces the site default `mpirun` flags | site default |
| `--extra "k=v k=v"` | free-form overrides appended to every python call | — |
| `--dry-run` | print the commands instead of running them | off |

Behaviour preserved from the scripts it replaces:

* `initial` runs first and writes `.caches/RUN_PATH_<agent_run_name>.txt`;
  line 1 = run path, last line = policy. History archiving stays inside
  `initial` (`preserve_and_clean_train`) — no shell-side `mv`.
* **Evaluation passes `reward_fn` identically to `initial` and `evaluate`.**
  It sets `UPARAM(9)`, i.e. the Fortran 1-vs-3 MPI buffer count; forcing only
  one side desyncs the buffer count and deadlocks MPI. See
  [readme_runtime_reward_switch.md](readme_runtime_reward_switch.md).
* Local training keeps the single-communicator, rank-0-branch `mpirun` form that
  oversubscribed workstation runs used; HPC uses the MPMD (`… : -n NTOT …`) form.
  Both proven paths were kept rather than unified.

Site defaults:

| | `local` | `hpc` |
|---|---|---|
| env | unset `OMPI_MCA_pml/osc`, `UCX_*`; `UCX_TLS=sm,self,tcp,…`, `OMPI_MCA_btl=self,vader,tcp` | `UCX_WARN_UNUSED_ENV_VARS=n` |
| `mpirun` (run) | — | `--mca io ompio` |
| `mpirun` (eval) | — | `--mca pml ucx` |

## 2. `execs/wing-run.sh` — NACA4412 wing (`python -m meta_MARL`)

Replaces `wing-script` and `sjob-wing.sh`. Same flag conventions; `evaluate` is
the only supported mode.

```bash
./execs/wing-run.sh --config conf/NACA4412-SHAP-Vel-2540.yml --mv-data yes
```

`--mv-data yes` archives the previous results through `utils/mv-data`
(`--case-name`, `--id` select the target folder).

**Two fixes over `sjob-wing.sh`:**

1. `utils/mv-data` is now run in a child shell (`bash utils/mv-data …`) instead
   of being **sourced**. The helper assigns `CASE_NAME` and `RUN_PATH` itself, so
   sourcing it overwrote the caller's `RUN_PATH` immediately before `mpirun`
   consumed it — that only worked because the clobbered value happened to be an
   equivalent absolute path.
2. `LOGFILE` is exported to the previous evaluation log, so the `cp "$LOGFILE"`
   inside `mv-data` copies something instead of failing on an empty argument.

Its `--source_root` also points at the repo root instead of `..` (which resolved
above the repo after the run-from-root change).

## 3. `execs/sjob-gen.sh` — SLURM job generator

```bash
./execs/sjob-gen.sh --case channel --config conf/MC-ng-111.yml --mode train \
    -J ng-111 -p batch -t 24:00:00 --begin +2h --submit
```

writes `execs/sjobs/<job-name>.sh`, prints a summary and the `sbatch` line.

| Flag | `#SBATCH` | Note |
|---|---|---|
| `-J, --job-name` | `-J` | also names the generated script and the log files |
| `-p, --partition` | `-p` | |
| `-b, --begin` | `--begin` | `+2h`, `+30m`, `+1d`, `16:30`, `now`, `tomorrow`, `2026-06-29T16:23:42` |
| `-t, --time` | `-t` | default `24:00:00` |
| `-N, --nodes` | `-N` | **`auto` (default)**: `ceil((nproc + 1) / ntasks-per-node)` read from the config |
| `--ntasks-per-node`, `--cpus-per-task` | same | on a 1-node job `ntasks-per-node` collapses to the exact task count |
| `-A, --account`, `-e, --email`, `--mail-type` | `-A`, `--mail-user`, `--mail-type` | |
| `--no-exclusive` | drops `--exclusive` | |
| `--slurm-out`, `--slurm-err` | `--output`, `--error` | default `log-files/<job>-%j.out\|.err` |
| `-o, --output` | — | where to write the script (`execs/sjobs/<job>.sh`) |
| `--submit` / `--print` | — | sbatch immediately / dump to stdout |

Everything after a bare `--` is forwarded verbatim to the payload:

```bash
./execs/sjob-gen.sh --case wing --config conf/NACA4412-SHAP-Vel-2540.yml \
    -J shap-wing -N 86 --begin 2026-06-29T16:23:42 -- --mv-data yes
```

`--nodes auto` is what removes the `-n7` / `-n11` script variants: `nproc` is
read from the config, `+1` for the agent rank, divided by `--ntasks-per-node`.

The generated script bakes in the **absolute** repo root
(`ROOT_DIR="${NEK_ROOT_DIR:-/abs/path}"`) rather than using `$SLURM_SUBMIT_DIR`,
so it no longer matters from which directory `sbatch` is called.

Example output:

```bash
#!/bin/bash -l
#SBATCH -A deepwing
#SBATCH -t 24:00:00
#SBATCH -p batch
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH --ntasks-per-node=11
#SBATCH --cpus-per-task=1
#SBATCH -J ng-111
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuninw@umich.edu
#SBATCH --output=log-files/ng-111-%j.out
#SBATCH --error=log-files/ng-111-%j.err
#SBATCH --begin=now+2hours

ROOT_DIR="${NEK_ROOT_DIR:-/home/yuninw/codes/drl/1_Nek/nek_power_saving}"
cd "${ROOT_DIR}" || exit 1
mkdir -p "${ROOT_DIR}/log-files" "${ROOT_DIR}/.caches"

"${ROOT_DIR}"/execs/channel-run.sh --config conf/MC-ng-111.yml --mode train
```

---

## Migration

| Old | New |
|---|---|
| `./execs/run-script --config C --run-mode run` | `./execs/channel-run.sh --config C --mode train` |
| `./execs/run-script --config C --run-mode evaluate` | `./execs/channel-run.sh --config C --mode evaluate --nenv 2` |
| `./execs/wing-script --config C` | `./execs/wing-run.sh --config C` |
| `sbatch execs/sjob-train.sh` | `./execs/sjob-gen.sh --case channel --config C --mode train -J <name> --submit` |
| `sbatch execs/sjob-train-n7.sh` / `-n11` | same, `-N auto` picks the node count |
| `sbatch execs/sjob-eval-n7.sh` | `./execs/sjob-gen.sh --case channel --config C --mode evaluate -J <name> --submit -- --nenv 2 --iostep 10000 --smpstep 12` |
| `sbatch execs/sjob-stat-n7.sh` (edit `CASE_LIST`) | `… --mode evaluate -- --nenv 1 --nb-interactions N --config A --config B` |
| `sbatch execs/sjob-wing.sh` | `./execs/sjob-gen.sh --case wing --config C -N 86 --submit -- --mv-data yes` |
| `utils/sjob_gen/create_job.sh -o job.sh …` | `execs/sjob-gen.sh …` (the old generator targets the pre-refactor layout) |

The old scripts are left in place and still work; nothing was deleted.

## Verification

Syntax-checked (`bash -n`) and exercised with `--dry-run` on both payloads and
both site flavours, plus a full generate → execute chain through a generated
script. No job was submitted to SLURM during the change.

## Not covered

* `--nenv` iterates `runner.rank` sequentially inside a single allocation, as the
  old eval scripts did — there is no job-array support.
* Log names are still `log.{initial,run,eval}.<config>` and are overwritten on
  each run (no job-id suffix), matching the previous behaviour.
* `conf/conf_juwels/` and `execs/sjobs_juwels.tar` were not touched; the site
  defaults in the generator are the deepwing/batch ones.
