# Training-data cleanup & history consolidation

**Date:** 2026-07-17
**Goal:** Stop accumulating unused on-the-fly simulation output in
`runs/<agent_run_name>/train/`. On every (re)start of *training* (i.e. not in
evaluation mode), wipe `./train/` to reclaim space — but first preserve all the
training **history** by consolidating it under a persistent
`runs/<agent_run_name>/history/` folder.

All code changes are tagged with `#[MOD]` in the source.

## TL;DR

- New helper `preserve_and_clean_train(run_folder)` in `src/initial.py` runs at
  the start of every non-eval `initial` call. It:
  1. migrates any legacy `train/roundXXX` → `history/roundXXX`,
  2. archives the previous live `train/history` → `history/current_history`
     (promoting an existing `current_history` to the next `roundXXX` first, so
     nothing is overwritten),
  3. `rmtree`s `./train/` to drop the bulky, unused simulation data.
- The shell-side round archiving (`mv train/history train/roundXXX`) in
  `execs/run-script` and `execs/sjob-train.sh` is now **removed** — `initial`
  owns history archiving.
- Post-processing that read history/rounds from `train/` now reads from
  `history/` (`explore.py`, `plot_reward_monitor_terms.py`).

---

## Folder layout

**Before** (everything inside `train/`, which was never cleaned):

```
runs/<agent_run_name>/
├── logs/                 # SB3 checkpoints (unchanged, outside train/)
└── train/
    ├── history/          # live history of the current run
    ├── round001/         # archived prior runs
    ├── round002/
    └── <lots of NEK sim output — restart files, field dumps, …>
```

**After** (history is persistent; `train/` is disposable):

```
runs/<agent_run_name>/
├── logs/                 # SB3 checkpoints (unchanged)
├── history/              # PERSISTENT — survives train/ cleanup
│   ├── current_history/  # newest run's history (was train/history)
│   ├── round001/         # older runs, kept + accumulated
│   └── round002/
└── train/                # wiped & rebuilt fresh each training start
    └── <NEK sim output, repopulated by NEK_INIT>
```

Each `initial` (+ `run`) cycle contributes one run's history: the previous
`current_history` is promoted to the next `roundXXX`, and the just-finished
`train/history` becomes the new `current_history`.

---

## Changes applied

| File | Change | Why |
|------|--------|-----|
| `src/initial.py` | Added `preserve_and_clean_train(run_folder)` (module-level) | Consolidate history + wipe `train/` |
| `src/initial.py` | Call it in `initial()` when `not conf.runner.evaluation`, right after `run_folder` exists and before `NEK_INIT` | Must run before NEK repopulates `train/` |
| `execs/sjob-train.sh` | Removed `mv "${RUN_PATH}/history" "${RUN_PATH}/round…"` block | Archiving now owned by `initial` |
| `execs/run-script` | Removed the `run`-mode `mv history roundXXX` block | Same |
| `post_processing/postlib/explore.py` | Read rounds + `current_history` from `runs/<case>/history/`; read `current_conf.yml` from `current_history/` | History moved out of `train/` |
| `post_processing/plot_reward_monitor_terms.py` | Added `history/current_history/current_conf.yml` as a config candidate (old paths kept as fallback) | Config now lives in `current_history/` |

---

## Why the ordering matters

`preserve_and_clean_train` must run **before** `NEK_INIT(...).main()`, because
`NEK_INIT` sets up and populates `train/` (copying restart files, writing the
`.rea`/`.par`, etc.). The sequence in `initial()` is:

1. ensure `run_folder` exists →
2. `preserve_and_clean_train(run_folder)` (non-eval only) — moves history up,
   then `rmtree(train/)` →
3. `if not exists(rank_folder=train): mkdir` recreates an empty `train/` →
4. `NEK_INIT.main()` repopulates it.

Checkpoints live in `runs/<agent_run_name>/logs/` (outside `train/`), so wiping
`train/` never touches the model checkpoints used for restart. `NEK_INIT`
re-copies restart data from `simulation.restart_folder`, so a wiped `train/` is
correctly rebuilt on both fresh runs and restarts.

## Safety notes

- Fresh run (no `train/` yet): the helper is a no-op beyond creating an empty
  `history/`.
- Round numbering is computed as `max(existing history/round*) + 1`, so
  migrated legacy rounds and promoted `current_history` folders never collide.
- Only `not conf.runner.evaluation` triggers cleanup; evaluation uses
  `env_XXX` folders and is untouched.

## Follow-ups / things to watch

- `post_processing/postlib/determine.py` reads `runs/<case>/current_conf.yml`
  (deterministic-eval path, not `train/`) and is unaffected.
- Any older analysis notebooks that hard-code `runs/<case>/train/history` or
  `train/roundXXX` will need the same `train/ → history/` path update.
