# Launch execs scripts from the repo root

**Date:** 2026-07-17
**Goal:** Stop `cd`-ing into `./execs/` before running. Launch the executables
from the **main path** (repo root), keep run logs at the repo root, and store
the `RUN_PATH_*.txt` handoff files in a dedicated `./.caches/` directory.

All code changes are tagged with `#[MOD]` in the source.

## New usage

```bash
# from the repo root (the "main path")
./execs/run-script  --config MC-nes-v.yml --run-mode run
./execs/wing-script --config WING.yml     --run-mode evaluate

sbatch execs/sjob-train.sh
sbatch execs/sjob-eval.sh
sbatch execs/sjob-wing.sh
```

- Logs land in `./log-files/` (repo root), not `./execs/log-files/`.
- `RUN_PATH_<name>.txt` files land in `./.caches/` (repo root).

## How it works

The scripts run **from the repo root** (root-relative model): everything is
addressed relative to the root — `conf/<config>.yml`, and `save_dir: runs`,
`compile_path: envs/...`, `restart_folder: data/...`, `policy_file: conf/...`
inside the configs. Each script resolves the root and `cd`s into it.

- **Interactive scripts** (`run-script`, `wing-script`) derive the root from the
  script's own location:
  ```bash
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
  ```
- **SLURM batch scripts** (`sjob-*.sh`) use `$SLURM_SUBMIT_DIR` (the directory
  you ran `sbatch` from — the repo root), because SLURM spools the batch script
  so `$BASH_SOURCE` is unreliable. A `$BASH_SOURCE`-based fallback covers a
  direct (non-SLURM) `bash execs/sjob-*.sh` run:
  ```bash
  ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)}"
  ```

Each script then sets:
```bash
cd "${ROOT_DIR}"                  # run python from the repo root
LOG_DIR="${ROOT_DIR}/log-files"   # logs at repo root
CACHE_DIR="${ROOT_DIR}/.caches"   # RUN_PATH cache at repo root
mkdir -p "${LOG_DIR}" "${CACHE_DIR}"
# ... python -m nek_MARL initial conf/${CONFIG_NAME} ...
```

### Config path conversion

Because Python now runs from the repo root, every cwd-relative path was
converted from `../<x>` to `<x>`:

| Location | Field | Before | After |
|----------|-------|--------|-------|
| all `conf/*.yml` | `policy_file` | `../conf/…` | `conf/…` |
| all `conf/*.yml` | `compile_path` | `../envs/…` | `envs/…` |
| all `conf/*.yml` | `restart_folder` | `../data/…` | `data/…` |
| `src/configs.py` | `save_dir` | `../runs` | `runs` |
| `src/configs_meta.py` | `save_dir`, `policy_dir` | `../runs` | `runs` |

`rank_folder` is derived from `save_dir` (`runs/<name>/train`), so the shell's
`cd ${RUN_PATH}` and the absolute path stored in `SESSION.NAME` both stay
correct with the root as the working directory.

## The RUN_PATH cache (`./.caches/`)

`RUN_PATH_<name>.txt` tells the shell where to `cd` and launch `nek5000`. It is
**written by Python** and **read by the shell**, so both sides must agree on the
location:

- **Python** (`src/initial.py`, `src/initial_meta.py`) writes it to
  `<repo_root>/.caches/`, where the root is derived from the module file itself
  (`os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`) — independent
  of the current working directory.
- **Shell** reads it from `${CACHE_DIR}` = `<repo_root>/.caches/`.

`./.caches/.gitkeep` keeps the directory tracked; the `RUN_PATH_*.txt` files
themselves stay ignored (`*RUN_PATH*` in `.gitignore`).

## Changes applied

| File | Change |
|------|--------|
| `src/initial.py` | Write `RUN_PATH_*.txt` into `<repo_root>/.caches/` (from `__file__`) |
| `src/initial_meta.py` | Same, for the meta/wing pipeline (`case_name`) |
| `execs/run-script` | Resolve root, `cd root`, `conf/…`, `LOG_DIR`/`CACHE_DIR` at root, RUN_PATH from cache |
| `execs/wing-script` | Same |
| `execs/sjob-train.sh` | Root via `SLURM_SUBMIT_DIR`, `cd root`, `conf/…`, `LOG_DIR`/`CACHE_DIR`, RUN_PATH from cache |
| `execs/sjob-eval.sh` | Same |
| `execs/sjob-wing.sh` | Same (also fixes a previously **undefined** `LOG_DIR`; `mv-data --source_root .`) |
| `conf/*.yml` | Strip `../` from `policy_file` / `compile_path` / `restart_folder` (root-relative) |
| `src/configs.py` | `save_dir: '../runs'` → `'runs'` |
| `src/configs_meta.py` | `save_dir` and `policy_dir` `'../runs'` → `'runs'` |
| `.gitignore` | `/execs/log-files/` → `/log-files/`; keep `.caches/` + `.gitkeep` tracked |
| `.caches/.gitkeep` | New — keeps the cache dir in the tree |
| `log-files/` | Created at the repo root |

Existing `execs/RUN_PATH_*.txt` files were moved to `.caches/`.

## Notes / gotchas

- The `#SBATCH --output=log-files/...` directives are resolved by SLURM relative
  to the **submit directory** (the repo root under the new workflow), so they now
  write to `./log-files/`. That directory must exist **before** submission (SLURM
  opens the file before the script body runs `mkdir`); it is created in this repo,
  and on a fresh clone `mkdir -p log-files` once. `log-files/` is git-ignored
  (`*log*`), so it is not tracked.
- The scripts `cd "${ROOT_DIR}"` (the repo root) and address everything
  root-relative (`conf/…`, `runs/…`, `envs/…`, `data/…`). This required stripping
  the `../` prefixes inside the config files (see the conversion table above).
- Submitting from a different directory still works: SLURM uses
  `$SLURM_SUBMIT_DIR`, so `sbatch execs/sjob-train.sh` resolves the root wherever
  you run it from, as long as `execs/` sits under that root.
