# MARL Reinforcement Learning using NEK5000 

## Get Started
### Dependencies
+ Build your own MPI: 

        ./utils/install_mpi.sh > log.mpi 2>&1 

+ Setup your dependices of miniforge: 

        ./utils/install_miniforge.sh  > log.miniforge 2>&1 

+ Install the python env

        ./utils/install_auto_env.sh nek  > log.python 2>&1 

+ Automated modification on the package 

        source ~/.bashrc.miniforge && python ./utils/patch_supersuit.py

### Initial SetUp 
+ First set up the Solvers via: 

        ./utils/initialize_solver.sh

+ Complie the code via: 

        source ~/.bashrc.openmpi_ucx

        ./utils/compile_case.sh --m mini_channel

+ For the wing nek: 

        ./utils/compile_case.sh --m small_wing --version v17 --case_name_v17 small_wing

+ For the naca0012 nek: 

        ./utils/compile_case.sh --m naca0012_200k --version v17 --case_name_v17 naca_wing


### Running a minimal-channel drl
Scripts are launched **from the repo root** (not from `execs/`), and `--config`
takes a **path** to the YAML (relative to the root, or absolute). The payload
script is the same interactively and under SLURM — it detects the environment
itself:

        ./execs/channel-run.sh --config conf/MC16-TD3.yml --mode train
        ./execs/channel-run.sh --config conf/MC16-TD3.yml --mode evaluate --nenv 2

Switch policy loading on/off with `--load-agent` (default: use the config's value):

        ./execs/channel-run.sh --config conf/MC16-TD3.yml --load-agent False   # train from scratch
        ./execs/channel-run.sh --config conf/MC16-TD3.yml --load-agent True    # continue from the latest checkpoint

Add `--dry-run` to print the `mpirun` lines without executing them, and `-h` for
the full flag list.

### Submitting to SLURM
[execs/sjob-gen.sh](execs/sjob-gen.sh) generates the batch script — job name,
partition, begin time and wall time are command-line flags, and `-N` defaults to
`auto` (derived from `nproc` in the config):

        ./execs/sjob-gen.sh --case channel --config conf/MC16-TD3.yml --mode train \
            -J mc16-td3 -p batch -t 24:00:00 --begin +2h --submit

Drop `--submit` to only write `execs/sjobs/<job-name>.sh` (then `sbatch` it
yourself), or use `--print` to dump it to stdout. Anything after a bare `--` is
forwarded to the payload script:

        ./execs/sjob-gen.sh --case channel --config conf/MC16-TD3.yml --mode evaluate \
            -J eval-mc16 --begin 2026-06-29T16:23:42 -- --nenv 2 --iostep 10000 --smpstep 12

`--begin` accepts `+2h`, `+30m`, `+1d`, `16:30`, `tomorrow` or a full ISO
timestamp. See [execs/readme.md](execs/readme.md) for every flag, and
[temp/docs/job_submission_refactor.md](temp/docs/job_submission_refactor.md) for
the design and the migration table from the old `sjob-*.sh` files.

### Meta-MARL usage
+ Wing evaluation (initialization + solver launch in one go):

        ./execs/wing-run.sh --config conf/NACA4412-SHAP-Vel-2540.yml --mv-data yes

        ./execs/sjob-gen.sh --case wing --config conf/NACA4412-SHAP-Vel-2540.yml \
            -J shap-wing -N 86 --submit -- --mv-data yes

+ Or drive the module directly:

        meta-marl initial  conf/your_meta_conf.yml
        meta-marl evaluate conf/your_meta_conf.yml

Note: set `simulation.solver_version: "v17"` in the meta config when using NEK5000 v17.

+ To inspect the training status: 
        
        python utils/read-history --id 1998 --mean 

+ Live multi-run monitoring (total reward + R_τ / R_pw / R_v3 components):

        python utils/monitor_runs.py    # saves utils/monitor_runs.png

### Reward Functions

Two reward modes are available, selected via `runner.reward_fn` in the YAML config:

| Mode | Formula | Fortran flag |
|---|---|---|
| `dudy` (default) | `R = 1 − dU/dy / (dU/dy)_ref` | no flag needed |
| `net_gain` | `R = α·(1−τ_w/τ_ref) + β·(−\|p'v\|/τ_ref) + γ·(−0.5\|v³\|/τ_ref)` | compile with `NETGAIN` |

Quick config for `net_gain`:

```yaml
runner:
  reward_fn: net_gain
  reward_alpha: 1.0   # weight on drag-reduction term
  reward_beta:  1.0   # weight on pressure-velocity cost
  reward_gamma: 1.0   # weight on kinetic-energy cost
```

> **Important:** `reward_fn: net_gain` in Python **must** match the `NETGAIN` compile flag
> in Fortran (`compile_script`). Mismatching causes MPI deadlock (3 buffers vs 1).

See [readme_POL_netgain.md](readme_POL_netgain.md) for the full derivation, averaging
pipeline, MPI protocol, and ablation run table.

+ To visualize snapshots via VISIT: 

        visit -o utils/nek_visit.NEK5000 

### Save and accumulate data  

Use command like follows to accumlate the results based on your configuration

        ./utils/mv-data --case_name small_wing --run_name 601001 --id 001 


## **TensorBoard Logging**

Training logs are automatically saved and can be viewed with TensorBoard:
```bash
# live (on-the-fly) run
tensorboard --logdir runs/your_run_name/train/history/tensorboard
# most-recent completed run (after it is archived)
tensorboard --logdir runs/your_run_name/history/current_history/tensorboard
``` 

## **IMPORTANT MODIFICATION - AUTOMATED**
The Supersuit library requires a modification to optimize single environment usage. This can now be done automatically:
### **Manual Patching**
```bash
# Install the package first
pip install -e . --no-deps

# Apply the Supersuit patch
python utils/patch_supersuit.py
```

### **What the patch does:**
The patch modifies the `vec_env_args` function in Supersuit to avoid unnecessary environment copying when `num_envs == 1`:

```python
def vec_env_args(env, num_envs):
    if num_envs == 1:
        def env_fn():
            env_copy = env  # Direct reference for single env
            return env_copy
    else: 
        def env_fn():
            env_copy = cloudpickle.loads(cloudpickle.dumps(env))  # Deep copy for multiple envs
            return env_copy

    return [env_fn] * num_envs, env.observation_space, env.action_space
```


### **Restore Original Supersuit (if needed):**
```bash
python utils/patch_supersuit.py restore
```
def vec_env_args(env, num_envs):
    def env_fn():
        env_copy = cloudpickle.loads(cloudpickle.dumps(env))
        return env_copy

    return [env_fn] * num_envs, env.observation_space, env.action_space


## Run directory layout & workflow updates

Recent I/O and workflow changes (full index in
[temp/docs/session_2026-07_changes.md](temp/docs/session_2026-07_changes.md)):

- **Launch from the repo root** — run `./execs/<script>` / `sbatch execs/<script>`
  from the project root, not from `execs/`. Run logs go to `./log-files/`, and the
  `RUN_PATH_*.txt` handoff files to `./.caches/`.
- **One payload per case, one generator for SLURM** — `execs/channel-run.sh` and
  `execs/wing-run.sh` hold the commands; `execs/sjob-gen.sh` wraps either of them
  into an sbatch script with the header set from the command line. The old
  `run-script` / `wing-script` / `sjob-*.sh` files still work but are superseded.
- **`--config` is a path** — pass `conf/<name>.yml` (root-relative) or an absolute
  path. Path fields inside the YAMLs (`compile_path`, `restart_folder`,
  `policy_file`) are root-relative too.
- **`policy` is the FULL checkpoint name** — no `agent_run_name-` prefix is added
  automatically anymore; give the complete file stem (e.g.
  `mc_nes_nek-rl_model_500_steps`, or `best_model`).
- **`--load-agent True|False`** — switch between continuing from a checkpoint and
  training fresh, without editing the config.

A run folder `runs/<agent_run_name>/` is organised as:

    runs/<agent_run_name>/
      logs/                     # SB3 checkpoints (best_model*, <run>-rl_model_*_steps, eval_model_*)
      train/                    # live training run (wiped & rebuilt each start)
        history/                #   on-the-fly reward logs + tensorboard of the CURRENT run
      history/                  # PERSISTENT training history (survives train/ cleanup)
        current_history/        #   the most-recent completed run
        round001/ round002/ …   #   older runs
      eval/                     # evaluation runs
        env_XXX/                #   solver output + vars_record_*.mat
        history/                #   eval reward logs

The reward-on-the-fly post-processing (`post_processing/postlib/explore.py`)
reads all of `history/round*`, `history/current_history`, and the live
`train/history`, so an in-progress run shows up in the notebook.

## The strcutures of the framework
    nek-drl/
        |-- envs/

            |-- cases # flow cases to used 
                |-- mini-channel
                |-- large-channel 
                |-- wings 

            |-- solvers # Nek5000 solvers with old and new verisons 
                |-- v17_DRL 
                |-- v17_clean 
                |-- v19_DRL
                |-- v19_clean
        
       |-- src # The source codes of the framework 
                |-- __Nek_MARL__
                |-- nek_marl.py # Environment connecting to the nek 
                |-- run.py # run the script 
                |-- evaluate.py # evaluate the run
                |-- init.py # initialization 
                |-- lib/ # Other utilities for Nek and STB3.  

       |-- utils # useful scripts for compliation and job scripts. 

       |-- runs  # The production path for cases

       |-- data  # The production path for cases

       |-- conf # The configurations used for the polices. 

       |-- execs # The folder for submitting jobs or execute the job locally
       
       |-- postprocessing # The jupyter-notebooks for post-processing
