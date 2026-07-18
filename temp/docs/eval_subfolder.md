# Evaluation runs under a dedicated `eval/` subfolder

**Date:** 2026-07-17
**Goal:** Keep evaluation output out of the run root. Previously an evaluation
job created `runs/<agent_run_name>/env_XXX` directly in the case folder and wrote
the on-the-fly reward logs into `runs/<agent_run_name>/history` (the same folder
used by training). Now everything for an evaluation lives under a single
`runs/<agent_run_name>/eval/` subfolder.

All code changes are tagged with `#[MOD]`.

## New layout

```
runs/<agent_run_name>/
├── logs/                 # checkpoints (shared; unchanged, NOT under eval/)
├── history/              # TRAINING history (unchanged)
├── train/                # training run dir (unchanged)
└── eval/                 # NEW — everything for an evaluation job
    ├── env_002/          # solver run dir: reward_monitor*.dat, field files,
    │                     #                 vars_record_*.mat
    └── history/          # on-the-fly reward logs: rewlog_*.npz,
                          #   rewards_aggregated_*.csv, rewards_episodes_*.csv,
                          #   eval_rewards_*, NODE_INFO.csv, current_conf.yml
```

`current_conf.yml` is also written at `eval/current_conf.yml` (the env's
`self.folder`), matching the previous behaviour one level down.

## Why it works

In evaluation the solver and the Python env already used *different* folders (the
solver ran in `env_XXX`, the Python env pointed at the case root), connected over
the split MPI intercommunicator (`sub_comm`) — not via the folder. So relocating
the Python-side records into `eval/` does not affect the solver connection; the
`mpi_info` `wdir` is not the connection mechanism.

- `src/initial.py` creates the solver dir at `eval/env_XXX` (`os.makedirs`, so the
  `eval/` parent is created). The `RUN_PATH` cache therefore points the solver at
  `eval/env_XXX`.
- `src/evaluate.py` builds the Python env with `rank_folder = <run>/eval`, so the
  env's `history_path` (all reward logs, NODE_INFO, current_conf) becomes
  `eval/history`. `vars_record_*.mat` is written to `eval/env_XXX`.

## Changes applied

### Source

| File | Change |
|------|--------|
| `src/initial.py` | eval branch: `rank_folder = <run>/eval/env_XXX`; `os.mkdir` → `os.makedirs` |
| `src/evaluate.py` | env `rank_folder = <run>/eval` (reward logs → `eval/history`); `vars_record` → `eval/env_XXX` |

`src/nek_marl.py` needs no change — it just uses the `rank_folder` it is given.

### Post-processing (reads evaluation results)

| File | Change |
|------|--------|
| `post_processing/plot_reward_monitor_terms.py` | Default reward-monitor paths now include `eval/` (`runs/<case>/eval/env_XXX/…`). `find_case_conf` already resolves via `path.parent.parent`, so it finds `eval/current_conf.yml` automatically |
| `post_processing/postlib/determine.py` | `read_deterministic_run` prefers `runs/<case>/eval/` (falls back to the case root for legacy runs) |
| `post_processing/deterministic.py` | Uses the eval-resolved `case_dict[case]['path']` instead of recomputing the case root |
| `post_processing/tsrs_eval.py` | `data_path`/`config_path` prefer `runs/<case>/eval/` (legacy fallback) |
| `src/read-record.py` | Reads env records from `eval/env_XXX`; NODE_INFO from the shared `eval/history` |

`post_processing/netgain_ts.py` needs no change: it globs `run_dir/**/netgain_ts_*.npz`
recursively, so it finds files under `eval/` automatically.

## Backward compatibility

`determine.py` and `tsrs_eval.py` fall back to the case root when no `eval/`
subfolder is present, so **older** evaluation runs (env_XXX directly under the
case) still load. `plot_reward_monitor_terms.py` takes explicit paths, so point
it at the actual location per run.

## Scope boundary (not changed)

The **meta / wing** evaluation pipeline (`src/initial_meta.py`,
`src/eval_meta.py`) still uses `runs/<case>/env_XXX` without the `eval/` level. It
is a separate workflow with its own consumers, so it was left untouched to avoid
breaking wing runs. Extending the `eval/` convention there is a follow-up if
wanted — it would require updating `initial_meta.py` and `eval_meta.py` together.

`src/read-record.py` still loads `vars_record_*.npz`, whereas `evaluate.py` saves
`.mat`; that format mismatch is pre-existing and out of scope — only the folder
paths were updated here.
