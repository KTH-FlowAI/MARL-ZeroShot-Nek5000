# Workflow & I/O changes — July 2026

Index of the workflow, path, and checkpoint-handling changes, with pointers to
the detailed per-topic docs. All code edits are tagged `#[MOD]` in the source.

---

## 1. Training-data cleanup & history consolidation
**Detail:** [train_cleanup_history_consolidation.md](train_cleanup_history_consolidation.md)

On every non-eval `initial`, `src/initial.py:preserve_and_clean_train` preserves
the training history under `runs/<run>/history/` (`current_history` + `roundXXX`)
and then wipes the bulky, unused `./train/` data. Shell-side history archiving in
`execs/` was removed. Post-processing (`explore.py`, `plot_reward_monitor_terms.py`)
now reads history from `history/` instead of `train/`.

## 2. Launch scripts from the repo root
**Detail:** [execs_run_from_root.md](execs_run_from_root.md)

`./execs/<script>` (interactive) and `sbatch execs/<script>` (SLURM) run from the
repo root. Logs → `./log-files/`; `RUN_PATH_*.txt` → `./.caches/`. Root resolved
via `$BASH_SOURCE` (interactive) or `$SLURM_SUBMIT_DIR` (batch).

## 3. `--config` as an absolute path
**Detail:** [config_as_absolute_path.md](config_as_absolute_path.md)

Scripts take a config **path** (relative to the repo root, or absolute), resolved
once with `realpath -e`; log filenames use the basename. The `../` prefixes inside
all config YAMLs (`compile_path`, `restart_folder`, `policy_file`) and the
`save_dir` / `policy_dir` defaults were made root-relative. Applied to `conf/` and
`conf/conf_juwels/`.

## 4. Evaluation runs under a dedicated `eval/` subfolder
**Detail:** [eval_subfolder.md](eval_subfolder.md)

An evaluation job writes to `runs/<run>/eval/env_XXX/` (solver output +
`vars_record_*.mat`) and `runs/<run>/eval/history/` (reward logs), instead of
scattering `env_XXX` and reward logs at the run root. Post-processing consumers
(`determine.py`, `deterministic.py`, `tsrs_eval.py`, `plot_reward_monitor_terms.py`,
`read-record.py`) prefer `eval/` with a legacy fallback.

## 5. Checkpoint naming: `policy` is the FULL file name (no auto-prefix)

The `agent_run_name-` prefix is **no longer prepended** by the loaders — `policy`
must be the complete checkpoint stem. Changed in every loading path:
`MetaPolicy._load_policy`, `sb3_utils.init_model` (PPO + DDPG/TD3 restart **and**
the replay-buffer path), and `evaluate.py`. `src/initial.py:get_latest_checkpoint`
now returns the full stem (`<run>-rl_model_<N>_steps`) so the auto-reload via
`RUN_PATH` stays consistent.

Configs updated to full names where they relied on the prefix: the 6 Meta configs
(list-valued `policy`, + reminder lines) and 2 scalar configs
(`conf_juwels/lc-omega1-SS-eng-seed2.yml`, `conf_juwels/MC-SHAP-2080-Fede.yml`).
`best_model*` / `eval_model_*` names were already full names (unchanged).

## 6. `MetaPolicy` — prefer the local policy copy

`_load_policy` loads the policy from the current `agent_run_name` logs folder if an
identical copy already exists there; it verifies the copy matches the source
(`filecmp.cmp`) and warns on any discrepancy (source differs or is missing). Avoids
re-copying and the `SameFileError` when source == target.

## 7. `load_agent` — config flip + CLI switch

- All channel training configs (mini-channel `MC-*` and large-channel `lc-*`,
  `evaluation: False` + `learnt_policy: True`) were set to `load_agent: True` so
  they load the policy (OC/eval configs left `False`).
- Training scripts (`run-script`, `sjob-train.sh`, `sjob-train-n7.sh`,
  `sjob-train-n11.sh`) gained a `--load-agent True|False` argument that overrides
  `runner.load_agent` on the training run (empty = use the config value).

## 8. Reward-on-the-fly post-processing

`post_processing/postlib/explore.py` now includes the live `train/history` as the
newest history segment (after `history/current_history`), so an in-progress run's
rewards appear in the notebook. Also fixes first-run inspection (no `history/` yet).

## 9. Job submission: payload scripts + SLURM generator
**Detail:** [job_submission_refactor.md](job_submission_refactor.md)

`execs/channel-run.sh` (nek_MARL) and `execs/wing-run.sh` (meta_MARL) hold every
run command and no `#SBATCH` directive — they pick the `local`/`hpc` environment
from `$SLURM_JOB_ID`, so one code path serves interactive and batch runs.
`execs/sjob-gen.sh` wraps either payload into an sbatch script whose job name,
partition, begin time, wall time and node count are command-line flags (`-N auto`
derives the node count from `nproc`, replacing the `-n7`/`-n11` variants).
Supersedes `run-script`, `wing-script`, `sjob-*.sh` and `utils/sjob_gen/`, all of
which are left in place.

---

## Quick reference

```bash
# train (from the repo root)
./execs/channel-run.sh --config conf/MC16-TD3.yml --mode train
./execs/channel-run.sh --config conf/MC16-TD3.yml --load-agent False   # fresh

# submit to SLURM
./execs/sjob-gen.sh --case channel --config conf/MC16-TD3.yml --mode train \
    -J mc16-td3 -t 24:00:00 --begin +2h --submit

# tensorboard (live vs archived)
tensorboard --logdir runs/<run>/train/history/tensorboard
tensorboard --logdir runs/<run>/history/current_history/tensorboard
```
