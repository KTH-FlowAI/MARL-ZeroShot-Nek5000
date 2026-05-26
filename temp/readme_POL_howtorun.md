## Overview

This guide walks through a complete training cycle:

1. Set up the environment on JUWELS
2. Choose and modify parameters
3. Compile the Fortran solver
4. Submit and monitor the job
5. Evaluate the trained policy
6. Plot results

The framework couples a NEK5000 CFD solver (Fortran, runs as MPI workers) with a Python
Stable-Baselines3 RL agent (runs as MPI rank 0). They communicate via an MPI intercommunicator.

---

## 1. Environment Setup

### One-time: verify `.bashrc` sources the right helpers

Your `~/.bashrc` must contain:

```bash
source ~/.bashrc.miniforge     # activates the 'nek' conda env (Python 3.8)
source ~/.bashrc.openmpi_ucx   # loads OpenMPI 4.1.4 + UCX
```

The helper files point to:
- Miniforge: `/p/project1/deepwing/polsm/env_setup/miniforge3`
- OpenMPI+UCX: `/p/project1/deepwing/polsm/env_setup/ucx_mpi/`

Verify on a login node:

```bash
source ~/.bashrc
python --version          # Python 3.8.20
mpirun --version          # Open MPI 4.1.4
which python              # .../miniforge3/envs/nek/bin/python
```

### Every session

All job scripts source these automatically — no manual action needed when submitting via
`sbatch`. For interactive testing on a login node, run:

```bash
source ~/.bashrc.openmpi_ucx
source ~/.bashrc.miniforge
cd /p/project1/deepwing/polsm/11-MARL-ZeroShot-Nek5000/src
```

---

## 2. Choose and Modify Parameters

All run-time parameters live in a single YAML file under `conf/`.

### Existing configs

| File | Algorithm | ndrl | Purpose |
|------|-----------|------|---------|
| `MC16-TD3.yml` | TD3 | 6 | Standard training baseline |
| `MC16-TD3-ng-val.yml` | TD3 | 6 | NETGAIN reward, α=1 β=0 γ=0 (validation) |
| `MC16-TD3-ng-full.yml` | TD3 | 6 | NETGAIN reward, α=β=γ=1 (net energy saving) |
| `MC16-PPO.yml` | PPO | 7 | PPO variant |

### Starting a new experiment

Copy the closest existing config and modify only what changes:

```bash
cp conf/MC16-TD3.yml conf/MC16-TD3-myexp.yml
```

**Key parameters to know:**

```yaml
runner:
  agent_run_name  : 2010        # Unique integer ID — sets the output folder runs/2010/
  seed            : 2010        # Random seed (match to agent_run_name for bookkeeping)
  nb_episodes     : 20          # Total training episodes
  nb_interactions : 2540        # Steps per episode (t+ ≈ 1500 wall units = 2540 * ndrl * dt)
  RL_algorithm    : "TD3"       # TD3 | DDPG | PPO
  action_noise    : 0.1         # Exploration noise amplitude
  learning_rate   : 1e-3
  batch_size      : 256
  gradient_steps  : 64
  evaluation      : False       # True = run inference only, no training

  # Reward (requires NETGAIN compile flag if net_gain):
  reward_fn       : 'net_gain'  # 'dudy' | 'net_gain'
  reward_alpha    : 1.0         # weight on drag-reduction term
  reward_beta     : 0.0         # weight on |p'v| actuator cost
  reward_gamma    : 0.0         # weight on 0.5|v³| kinetic cost

  u_tau           : 0.0638      # Friction velocity of uncontrolled flow
  dUdy            : 12.4271     # dU/dy at the wall, uncontrolled

simulation:
  nproc           : 10          # NEK MPI ranks (must match SIZE file)
  ndrl            : 6           # NEK timesteps per RL action (t+ ≈ 0.6)
  restart_folder  : "../data/restarts/rs6_mini_Channel"
  TOTCTRL         : 220         # Total actuator points (from SIZE)
  viscosity       : -2800       # Negative = ν = 1/|viscosity|
  retau           : 180.0
```

> **Important:** `agent_run_name` must be unique across experiments. Output goes to
> `runs/{agent_run_name}/train/`. Reusing an ID overwrites previous results.

---

## 3. Compile the Fortran Solver

The solver binary must be recompiled whenever you change:
- The reward type (add/remove `NETGAIN` flag)
- Any Fortran source file in `envs/cases/mini_channel/drl/` or `inc_src/`

### Toggle the reward flag

Edit `envs/cases/mini_channel/compile_script`, line ~26:

```bash
# Plain dUdy reward (default):
export PPLIST="MPIIO DRL YWDEBUG TSRS"

# Net-gain reward (requires reward_fn: net_gain in YAML):
export PPLIST="MPIIO DRL YWDEBUG TSRS NETGAIN"
```

> **Critical:** The Fortran flag and the Python `reward_fn` config **must agree**.
> Mismatching them causes an MPI deadlock (Fortran sends 3 buffers, Python expects 1 or vice versa).

### Run the compiler

```bash
source ~/.bashrc.openmpi_ucx
./utils/compile_case.sh --m mini_channel
```

A successful compile prints:

```
#############################################################
#                  Compilation successful!                  #
#############################################################
```

Verify the binary contains the NETGAIN symbols (if using net_gain):

```bash
strings envs/cases/mini_channel/nek5000 | grep -i netgain
# Expected output:
# [REWARD] NETGAIN HANDLE INIT!
# compute_netgain_
```

---

## 4. Create a Slurm Job Script

Use the generator or copy an existing script.

### Option A — copy and edit

```bash
cp execs/sjob-ng-val.sh execs/sjob-myexp.sh
```

Edit the top section:

```bash
#SBATCH -A deepwing           # project account
#SBATCH -t 02:00:00           # wall time (2h gets into queue faster than 24h)
#SBATCH -p batch              # partition
#SBATCH -N 1                  # nodes
#SBATCH --ntasks-per-node=11  # nproc + 1  (10 NEK + 1 Python)
#SBATCH -J myexp              # job name shown in squeue
#SBATCH --mail-user=polsm@kth.se

CONFIG_NAME="MC16-TD3-myexp.yml"
```

> **Rule:** `--ntasks-per-node` = `simulation.nproc + 1`

### Option B — use the generator

```bash
cd execs
python ../utils/sjob_gen/generate_slurm_job.py \
    --config-name MC16-TD3-myexp.yml \
    --job-name myexp \
    --ntasks-per-node 11 \
    --nek5000-mpi-ranks 10 \
    --time-limit 02:00:00 \
    --mail-user polsm@kth.se \
    -o sjob-myexp.sh
```

### Submit

```bash
cd execs
sbatch sjob-myexp.sh
```

Check the queue:

```bash
squeue -u suarezmorales1 -o "%.10i %.9P %.10j %.8T %.10M %R"
```

Common `STATE` values:
- `PENDING` — waiting for resources (`Priority` = normal queue)
- `RUNNING` — active
- `COMPLETING` — wrapping up (normal, brief)
- `FAILED` — check the `.err` and `log.initial.*.yml` files

> **Tip:** Short wall times (2h) get higher priority than 24h jobs.
> Use 2h for initial experiments and early debugging; switch to longer once stable.

---

## 5. Monitor Training

### Sanity check: did `initial` succeed?

```bash
tail -5 execs/log-files/log.initial.MC16-TD3-myexp.yml
# Must end with:  [STB3] INITIALIZATION COMPLETE
```

If you see a Python traceback instead, the training MPI processes will fail immediately.

### Live episode reward (quickest check)

```bash
tail -f runs/2010/train/history/rewards_episodes_*.csv
```

Output columns: `timestamp, episode, total_steps, mean_reward, std_reward, min_reward, max_reward, total_reward, duration_seconds`

`mean_reward × 100` = drag reduction % relative to the uncontrolled flow.
Expect positive values after ~5–10 episodes for a well-initialised run.

### Live step-level reward (with components)

```bash
tail -f runs/2010/train/history/rewards_aggregated_*.csv
```

Flushed every 10 steps. Columns:

```
timestamp, episode, step, mean_reward, std_reward, min_reward, max_reward,
total_reward, num_agents,
mean_R_tau,   ← drag reduction component  (R_tau = 1 − τ_w/τ_ref)
mean_R_pw,    ← pressure-velocity cost    (R_pw  = −|p'v|/τ_ref)
mean_R_v3     ← kinetic energy cost       (R_v3  = −0.5|v³|/τ_ref)
```

`mean_R_tau/pw/v3` are `NaN` for runs using `reward_fn: dudy` (old behaviour).

### Live monitoring plot (all runs + component breakdown)

```bash
python utils/monitor_runs.py          # saves to utils/monitor_runs.png
python utils/monitor_runs.py --out /tmp/my_plot.png   # custom path
```

Plots runs 2001 and 2002 side by side. When component columns are present the layout shows
R_τ, R_pw, and R_v3 panels separately so you can see which term dominates. Typical
magnitudes at Reτ = 180: `|R_pw| ≈ 0.2–0.7`, `|R_v3| ≈ 10⁻⁴`, `|R_tau| ≈ 0.01–0.25`
(see `readme_POL_netgain.md` for the full table).

### Slurm output log

```bash
tail -f execs/log-files/ng-val-<JOBID>.out
```

### TensorBoard (actor/critic loss + SB3 env reward)

On JUWELS (in a separate terminal or screen):

```bash
cd /p/project1/deepwing/polsm/11-MARL-ZeroShot-Nek5000
tensorboard --logdir runs/ --port 6006 --bind_all
```

On your laptop:

```bash
ssh -L 6006:localhost:6006 suarezmorales1@juwels.fz-juelich.de
```

Then open `http://localhost:6006` — you can overlay multiple runs for comparison.

### Reward curve plot (after episodes accumulate)

`utils/read-history.py` reads the `rewlog_*.npz` archives and plots a smoothed
moving-average reward curve. Before using it, set the physical parameters for this case
at the top of the file:

| Variable | Value for MC16-TD3 |
|----------|--------------------|
| `DT` | `1e-2` |
| `utau` | `0.0638` |
| `mu` | `1.0/2800.0` |
| `ndrl` | `6` |

Then run from the project root:

```bash
cd /p/project1/deepwing/polsm/11-MARL-ZeroShot-Nek5000
python utils/read-history.py --id 2010 --mean
# Prints: Max R = X.XX% at Episode N
# Saves:  runs/2010/train/history/figs/moving_avg_reward.jpg
```

---

## 6. Policy Evaluation (Inference)

Once training has produced a checkpoint (saved every `ckpt_int=5` episodes to
`runs/2010/train/logs/`), run deterministic evaluation.

### Find the checkpoint

```bash
ls runs/2010/train/logs/
# Example: 1777414695-rl_model_50800_steps.zip
```

### Run evaluation via Slurm

Copy an existing job script and set `evaluation: True` and `load_agent: True` in the YAML,
or use command-line overrides:

```bash
# In the job script, change the mpirun line to:
mpirun --mca pml ucx --mca io ompio \
    -n 1 python -m nek_MARL evaluate ../conf/MC16-TD3-myexp.yml \
        runner.evaluation=True \
        runner.learnt_policy=True \
        runner.load_agent=True \
        runner.policy=1777414695-rl_model_50800_steps \
        runner.rank=1 :\
    -n 10 bash -c "cd ${RUN_PATH} && ./nek5000"
```

Set `runner.rank` to select which restart condition (1–6) to evaluate from.

The evaluation produces `rewlog_eval_*.npz` files and prints mean drag reduction
to the log file.

---

## 7. Plot Results

### Interactive notebook (recommended)

```bash
cd /p/project1/deepwing/polsm/11-MARL-ZeroShot-Nek5000/post_processing
jupyter notebook exploration.ipynb
```

`exploration.ipynb` uses `postlib/explore.py` which:
- Loads all `round*/rewlog_*.npz` + `history/rewlog_*.npz` automatically across restarts
- Computes smoothed reward curves
- Loads actor/critic loss from `tensorboard/progress.csv`
- Scales reward to drag reduction %
- Supports multi-run overlay for comparison

To compare two runs (e.g. validation vs full net-gain), edit the `case_list` at the top
of the notebook:

```python
case_list = [2001, 2002]   # agent_run_names to compare
run_path  = '../runs'
case_dict = initalize_case(run_path, case_list)
```

### Quick static plot from command line

```bash
python utils/read-history.py --id 2001 --mean   # validation run
python utils/read-history.py --id 2002 --mean   # full net-gain run
```

Output saved to `runs/{id}/train/history/figs/moving_avg_reward.jpg`.

---

## 8. Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: lib.replay_buffer` | `replay_buffer.py` missing from `src/lib/` | Already fixed — `try/except` guard in `sb3_utils.py` and `transfer_learning.py` |
| `FileNotFoundError: data/restarts/rs6_mini_Channel/init_1` | Restart files not present | Already copied to `data/restarts/rs6_mini_Channel/` |
| MPI deadlock (job hangs forever) | Fortran NETGAIN flag ≠ Python `reward_fn` | Recompile with matching flag, or change YAML |
| `RUN_PATH_<ID>.txt: No such file` | `initial` step failed silently | Check `log.initial.*.yml` for the real error |
| `./nek5000: No such file or directory` | Run folder not initialized | `initial` step must complete before `mpirun` launches NEK |
| `bind: warning: line editing not enabled` | `.bashrc` readline binds in non-interactive shell | Harmless, ignore |

---

## 9. Directory Structure Reference

```
11-MARL-ZeroShot-Nek5000/
├── conf/                        ← YAML configs (one per experiment)
├── src/                         ← Python training code
│   ├── nek_MARL/                ← Entry point: python -m nek_MARL
│   ├── nek_marl.py              ← PettingZoo env + MPI communication
│   ├── run.py                   ← SB3 training loop
│   ├── initial.py               ← Folder setup + restart file copy
│   ├── configs.py               ← OmegaConf dataclass schema
│   └── lib/
│       ├── sb3_utils.py         ← SB3 model init, callbacks, logger
│       ├── nek_utils.py         ← Run folder setup, restart management
│       └── reward_logger.py     ← CSV reward logging
├── envs/cases/mini_channel/     ← Fortran case
│   ├── compile_script           ← Toggle NETGAIN here
│   ├── inc_src/DRL              ← Fortran common blocks (reward arrays)
│   └── drl/
│       ├── drl_reward.f         ← compute_dudy / compute_netGain
│       └── drl_IO.f             ← MPI send of reward buffers
├── data/restarts/
│   └── rs6_mini_Channel/        ← Turbulent initial conditions (init_1..6)
├── runs/                        ← Training output (created at runtime)
│   └── {agent_run_name}/train/
│       ├── history/             ← rewlog_*.npz, rewards_*.csv, tensorboard/
│       ├── logs/                ← SB3 model checkpoints (.zip)
│       ├── round001/            ← Archived history from previous job runs
│       └── nek5000              ← Compiled binary (copied here by initial)
├── execs/                       ← Job scripts and run utilities
│   ├── sjob-ng-val.sh           ← NETGAIN validation job (α=1, β=γ=0)
│   ├── sjob-ng-full.sh          ← NETGAIN full job (α=β=γ=1)
│   ├── sjob-template.sh         ← Template for new jobs
│   ├── run-script               ← Local (non-Slurm) launcher
│   └── RUN_PATH_<ID>.txt        ← Points to runs/{ID}/train (auto-created)
├── utils/
│   ├── compile_case.sh          ← Wrapper around NEK makenek
│   ├── initialize_solver.sh     ← Extract solver tarballs (one-time)
│   ├── read-history.py          ← Plot reward curves from rewlog_*.npz
│   └── sjob_gen/
│       └── generate_slurm_job.py ← CLI tool to generate Slurm scripts
└── post_processing/
    ├── exploration.ipynb        ← Interactive reward + loss analysis
    ├── deterministic.ipynb      ← Deterministic policy evaluation
    └── postlib/explore.py       ← Multi-run loader used by notebooks
```
