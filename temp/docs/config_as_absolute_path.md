# `--config` takes a config PATH, resolved to an absolute path

**Date:** 2026-07-17
**Goal:** Make the exec scripts cleaner and more general by passing a **path** to
the config file (not just its name) and resolving it to an **absolute** path
once, up front. Downstream commands then reference `${CONFIG_NAME}` directly with
no `conf/` / `../conf/` prefix, so the same script works regardless of where the
config lives.

All changes are tagged `#[MOD]`.

## What changed

Right after argument parsing, each script resolves the config to an absolute
path and derives a short tag for log filenames:

```bash
_cfg_in="${CONFIG_NAME}"
CONFIG_NAME="$(realpath -e "${_cfg_in}" 2>/dev/null)" || {
    echo "[ERR] Config file not found: ${_cfg_in}" >&2; exit 1; }
CONFIG_TAG="$(basename "${CONFIG_NAME}")"   # short tag for log filenames
```

- `realpath -e` resolves relative **or** absolute inputs and fails early (clean
  error, non-zero exit) if the file does not exist.
- Every `python -m ... <config>` and `grep ... <config>` now uses
  `${CONFIG_NAME}` (the absolute path) — the `conf/` / `./conf/` / `../conf/`
  prefixes were removed.
- Log filenames use `${CONFIG_TAG}` (the basename) instead of `${CONFIG_NAME}`,
  so an absolute path with `/` does not turn into a nested log path.

## Resolution base

All exec scripts now use the **same** root-relative model: they resolve the repo
root and `cd` into it, so a relative config path is resolved from the repo root
(an absolute path always works too). Default `CONFIG_NAME` is `conf/<name>.yml`.

```bash
# interactive (from the repo root)
./execs/run-script  --config conf/MC-nes-v.yml
./execs/run-script  --config /abs/path/to/MC-nes-v.yml   # absolute also works

# SLURM batch (submit from the repo root)
sbatch execs/sjob-train.sh --config conf/MC16-TD3-ng-111.yml
```

## Files changed

All exec scripts — default → a config path; resolve to absolute via
`realpath -e`; drop the `conf/` prefix on all usages; log filenames →
`${CONFIG_TAG}`:
`execs/run-script`, `execs/wing-script`, `execs/sjob-train.sh`,
`execs/sjob-eval.sh`, `execs/sjob-wing.sh`, `execs/sjob-train-n7.sh`,
`execs/sjob-train-n11.sh`, `execs/sjob-eval-n7.sh`, `execs/sjob-stat-n7.sh`.

For `sjob-stat-n7.sh` (which loops over a `CASE_LIST` array instead of taking
`--config`), the array entries were made root-relative (`conf/<name>.yml`) and
the resolution runs inside the loop.

## Notes

- The batch scripts (`sjob-*.sh`, incl. the `-n7`/`-n11`/`-stat` variants) had
  been reverted to a run-from-execs model; they were reconciled back to the
  root-relative model at the same time (`cd "${ROOT_DIR}"`, `LOG_DIR`/`CACHE_DIR`
  at the repo root, `RUN_PATH` from `.caches/`, and the old in-`train/` history
  archiving dropped since `initial.py` handles it). See
  [execs_run_from_root.md](execs_run_from_root.md).
- `realpath` is GNU coreutils (present here as 8.32); `realpath -e` is the
  existence-checking form.
