# Running jobs

Three scripts, one idea: the **payload** scripts hold every command needed to run
a case, the **generator** wraps a payload into an sbatch file whose header you
control from the command line.

| Script | Role |
| --- | --- |
| [channel-run.sh](channel-run.sh) | all commands for the minimal-channel cases (`python -m nek_MARL`) |
| [wing-run.sh](wing-run.sh) | all commands for the NACA4412 wing cases (`python -m meta_MARL`) |
| [sjob-gen.sh](sjob-gen.sh) | writes (and optionally submits) a SLURM script wrapping either payload |

The payload scripts carry **no** SLURM directives and detect their environment
from `$SLURM_JOB_ID`, so the identical command line works on a workstation and
inside a batch job. Run everything from the repo root.

---

## 1. Running interactively

```bash
# channel — training
./execs/channel-run.sh --config conf/MC-ng-111.yml --mode train

# channel — evaluate a trained policy over two environments
./execs/channel-run.sh --config conf/MC-ng-111.yml --mode evaluate \
    --nenv 2 --iostep 10000 --smpstep 12

# channel — statistics sweep over several cases
./execs/channel-run.sh --mode evaluate --nenv 1 --nb-interactions 20000 \
    --config conf/MC-shapcf.yml --config conf/MC-shapvel.yml

# wing
./execs/wing-run.sh --config conf/NACA4412-SHAP-Vel-2540.yml --mv-data yes
```

`--dry-run` prints the mpirun lines instead of executing them, and `-h` lists
every flag. Useful ones:

- `--config` is repeatable (or comma-separated) — the script loops over the list.
- `--load-agent True|False` overrides `runner.load_agent` without editing the config.
- `--extra "runner.seed=7 runner.nb_episodes=10"` appends arbitrary overrides.
- `--site local|hpc` forces the environment flavour if the auto-detection is wrong.
- Evaluation forces `reward_fn=net_gain` (change with `--reward-fn`) on **both**
  the `initial` and the `evaluate` call — they set `UPARAM(9)`, i.e. the Fortran
  MPI buffer count, and must agree or the run deadlocks.

## 2. Generating a SLURM job

```bash
./execs/sjob-gen.sh --case channel --config conf/MC-ng-111.yml --mode train \
    -J ng-111 -t 24:00:00 -p batch --begin +2h
```

writes `execs/sjobs/ng-111.sh` and prints the `sbatch` line. Add `--submit` to
send it straight away, or `--print` to dump it to stdout without writing a file.

The knobs that used to require editing a template:

| Flag | `#SBATCH` |
| --- | --- |
| `-J, --job-name` | `-J` (also names the script and the log files) |
| `-p, --partition` | `-p` |
| `-b, --begin` | `--begin` — accepts `+2h`, `+30m`, `+1d`, `16:30`, `tomorrow`, `2026-06-29T16:23:42` |
| `-t, --time` | `-t` |
| `-N, --nodes` | `-N`; **`auto` (default)** = `ceil((nproc + 1) / ntasks-per-node)` read from the config |
| `--ntasks-per-node`, `--cpus-per-task` | same |
| `-A, --account`, `-e, --email`, `--mail-type` | `-A`, `--mail-user`, `--mail-type` |
| `--no-exclusive` | drops `--exclusive` |
| `--slurm-out`, `--slurm-err` | `--output`, `--error` (default `log-files/<job>-%j.out/.err`) |

**Reminder — every payload flag is reachable, and it works with `--submit`.**
Anything after a bare `--` is forwarded verbatim to the payload script, so you do
**not** need to mirror payload flags in the generator or hand-edit the job file:
the generator owns only the `#SBATCH` header, the payload owns the run
parameters, and `--` bridges the two. Run `execs/channel-run.sh --help` /
`execs/wing-run.sh --help` for the full list of what you can pass after `--`.

```bash
# fully-parameterized channel evaluate, submitted directly
./execs/sjob-gen.sh --case channel --config conf/MC-ng-111.yml --mode evaluate \
    -J eval-ng111 -N 2 --begin 2026-06-29T16:23:42 --submit -- \
    --nenv 2 --iostep 10000 --write-interval 10000 --smpstep 12 \
    --reward-fn net_gain --alpha 0.5 --nb-interactions 20000

# wing evaluate with archiving, submitted directly
./execs/sjob-gen.sh --case wing --config conf/NACA4412-SHAP-Vel-2540.yml \
    -J shap-wing -N 86 -t 24:00:00 --submit -- \
    --mv-data yes --case-name naca_wing --id 002
```

Generated scripts are ordinary sbatch files, but hand-editing is only a last
resort for something no flag exposes (e.g. an exotic `#SBATCH` directive) — the
`--` passthrough already covers every payload argument.

## 3. Logs

- SLURM stdout/stderr: `log-files/<job-name>-<jobid>.out|.err`
- Solver / agent output: `log-files/log.initial.<config>`, `log.run.<config>`,
  `log.eval.<config>`
- `.caches/RUN_PATH_<agent_run_name>.txt` is written by `initial` and holds the
  run path (first line) and the policy path (last line).

```bash
squeue -u $USER        # queue status
scancel  <jobid>       # cancel
```

---

The previous-generation scripts (`run-script`, `wing-script`, `sjob-*.sh`, and
`../utils/sjob_gen/`) are kept for reference but are superseded by the three
scripts above.
