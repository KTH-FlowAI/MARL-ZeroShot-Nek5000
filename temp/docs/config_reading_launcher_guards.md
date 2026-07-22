# Config reading in the launchers + `initial` pre-flight guards

**Date:** 2026-07-21
**Goal:** Track down why a config taken from `conf/conf_juwels/` failed while the
same-named config in `conf/` worked, and harden every place where the shell
launchers read a YAML config or the `RUN_PATH_*.txt` cache.

All code changes are tagged `#[MOD]`.

## TL;DR

`initial.py` / `initial_meta.py` were **not** the problem — every config in
`conf/conf_juwels/` parses and initializes cleanly. The failure came from
`cfg_get` in `execs/channel-run.sh`: an unanchored `grep` returned the first line
that merely *mentioned* the key, so a `## [REMINDER] ... agent_run_name ...`
comment sitting above the real key was read as the value. That produced a
non-existent cache file, an empty `RUN_PATH`, and a bare `cd` — i.e. every solver
rank started `./nek5000` in `$HOME`. Nothing checked any of it, so the job kept
going.

Four defects were fixed, plus a pre-flight validator for the meta (wing) path.

---

## Changes applied

| File | Where | Change | Why |
|------|-------|--------|-----|
| `execs/channel-run.sh` | `cfg_get` (l.192) | `grep`+`sed` → anchored `awk` | Comment lines and trailing `# ...` no longer leak into the value |
| `execs/channel-run.sh` | `read_run_path` (l.210) | new helper replacing `head`/`tail` | Line 1 = run folder, line 2 = policy; hard error on a missing/empty cache |
| `execs/channel-run.sh` | main loop | `\|\| exit 1` on `initial` | A failed `initial` would otherwise reuse the PREVIOUS run's cache |
| `execs/channel-run.sh` | main loop | validate `agent_run_name` / `nproc`, reject conflicting `--extra` | Shell reads the yml, python reads yml+overrides — they must agree |
| `execs/channel-run.sh` | training branch (l.299) | `runner.load_agent=False` when no checkpoint | A fresh run with `load_agent: True` crashed in `RLA.load()` |
| `execs/channel-run.sh` | l.36 / l.125 | `set -o pipefail`; default config → `conf/mini_channel/MC-ng-111.yml` | Old default `conf/MC16-TD3.yml` no longer exists |
| `execs/wing-run.sh` | l.33, 138, 154, main loop | same `cfg_get`, `read_run_path`, `\|\| exit 1`, `case_name`/`nproc` validation, `pipefail` | Same defects, same fixes |
| `src/initial.py` | l.263 | `f.write(last_agent + "\n")`, dropped stray `f.close()` | Cache file must always have 2 lines |
| `src/initial_meta.py` | `validate_conf()` | new pre-flight validation | Meta config errors surfaced only after the MPI split |
| `src/initial_meta.py` | folder creation | `os.mkdir` → `os.makedirs(..., exist_ok=True)` | Multi-level `save_dir`; race between env ranks |

---

## Detail per issue

### 1. `cfg_get` matched comment lines (the original bug)

```bash
# before
cfg_get() { grep -ri "$2" "$1" | sed -E "s/^[^:]*:[[:space:]]*//; s/[\"']//g; ..." | head -n 1; }
```

Unanchored, case-insensitive, first match wins — including comments.
Reproduction with `conf/conf_juwels/MC-SHAP-2080-Fede.yml`:

```
[CFG] agent_run_name=    ## [REMINDER] policy = FULL checkpoint file name (stem, no .zip); …
head: cannot open '…/.caches/RUN_PATH_    ## [REMINDER] policy = FULL checkpoint….txt'
[RUN] RUN_PATH=  policy=
[DRY] mpirun -n $((1 + 32)) bash -c '
          … python -m nek_MARL run …MC-SHAP-2080-Fede.yml runner.policy=
      else
          cd  && ./nek5000                  ← cd with no argument == $HOME
```

The `#` in the value also comments out the rest of the `python -m nek_MARL run`
line, because `run_cmd` runs the string through `eval`.

Affected configs: `conf/conf_juwels/MC-SHAP-2080-Fede.yml`,
`conf/conf_juwels/lc-omega1-SS-eng-seed2.yml` (both carry the REMINDER comment
*above* the key). All configs under `conf/mini_channel/` and `conf/small_wing/`
happen to be clean.

**Fix** — key anchored at line start, comment lines skipped, trailing comment and
quotes stripped:

```bash
cfg_get() {
    awk -v k="$2" '
        BEGIN { re = "^[[:space:]]*" k "[[:space:]]*:" }
        /^[[:space:]]*#/ { next }
        $0 ~ re {
            sub(/^[^:]*:[[:space:]]*/, "")   # drop the key
            sub(/[[:space:]]*#.*$/, "")      # drop a trailing comment
            gsub(/["'"'"']/, "")             # drop quotes
            sub(/[[:space:]]+$/, "")         # drop trailing blanks
            print; exit
        }' "$1"
}
```

Side note: `grep -r` on a single file is also unportable — `ugrep` prefixes the
filename even for one explicit file argument, which shifts the `sed` field and
returns `nproc   : 10` instead of `10`. `/usr/bin/grep` (what the scripts get
after sourcing the profiles) does not, but the `awk` version is immune either way.

### 2. `tail -n 1` returned the run path as the policy

`initial.py` wrote the cache as `rank_folder + "\n"` then `last_agent`. With no
checkpoint, `last_agent` is `""`, so the file had **one** line and
`tail -n 1` returned line 1. The launcher then passed
`runner.policy=<rank_folder>` to `run`, and `sb3_utils.py` built
`f"{run_folder}/logs/{policy}"` → `FileNotFoundError` on the first training run
of a new `agent_run_name`.

Fixed on both sides: `initial.py` always writes 2 lines, and the launcher reads
the lines by number (`sed -n 1p` / `sed -n 2p`) inside `read_run_path`.

### 3. Nothing checked that the read succeeded

No `set -e`, no return-code checks. A failed `initial` (e.g. the
`FileNotFoundError` on a missing `data/restarts/...` folder) left the previous
run's `RUN_PATH_*.txt` in place and the launcher happily started the solver in
whatever folder that file pointed at. `read_run_path` now fails loudly on a
missing cache file or an empty first line, `initial` failures `exit 1`, and
`--dry-run` prints a placeholder instead so dry runs stay usable.

### 4. Shell reads the yml, python reads yml + overrides

`AGENT_RUN_NAME` / `CASE_NAME` / `NTOT` come from the file text while `initial`
uses the merged config. `--extra "runner.agent_run_name=foo"` writes
`RUN_PATH_foo.txt` while the shell looks for the old name; `--extra
"simulation.nproc=64"` puts 64 in the `.par` but launches `-n <yml value>` — the
same rank/buffer desync the `reward_fn` comment in the eval branch warns about.
Overrides of `agent_run_name` / `case_name` / `nproc` / `save_dir` are now
rejected with an explanation.

### 5. Meta pre-flight (`initial_meta.py`)

`MetaPolicy._initialize_config` indexes nine per-region lists with `[il]` and
loads `<policy_dir>/<agent_run_name>/logs/<policy>.zip`. None of that was
checked in `initial`, so a short list or a typo in a policy name only raised
**inside `evaluate`**, after `MPI_COMM_WORLD` was split and the solver ranks were
already running: rank 0 dies, the nek ranks wait, the job hangs until the wall
clock kills it.

`validate_conf(conf)` now runs before any folder is touched and checks:

- `case_name` non-empty (it names `runs/<case>` and `.caches/RUN_PATH_<case>.txt`)
- `agent_ctrl_area` non-empty
- every per-region list — **error** if shorter than the region count,
  **warning** if longer (extra entries are silently ignored by `MetaPolicy`)
- every policy checkpoint exists, accepting the local `<run_folder>/logs/` copy
  that `MetaPolicy._load_policy` prefers, skipping `OC`/`BL` (analytic, no file)

---

## Verification

| check | result |
|---|---|
| `bash -n` both launchers | OK |
| `conf/conf_juwels/MC-SHAP-2080-Fede.yml` dry run | `agent_run_name=801001  nproc=32` |
| every `conf/**/*.yml` through the new `cfg_get` | clean key + numeric `nproc` for all |
| fresh training run (no checkpoint) | `runner.policy=` empty, `runner.load_agent=False` |
| run with 2 checkpoints in `logs/` | picks `…rl_model_1200_steps` (latest of 900/1200) |
| cache file missing / empty, non-dry | `[ERR] …` + exit 1 |
| `--extra "simulation.nproc=64"` | rejected |
| all 9 `conf/mini_channel/*.yml` through `initial` | initialize OK |
| `conf/small_wing/WING-PATH-TEST.yml` | validates (6 length warnings), initializes, all 4 policies load with `TD3.load()` |
| `conf/WING-SMALL.yml` | `VALIDATED: 5 control region(s), all policies present` |
| `conf/conf_juwels/WING-SMALL.yml` | now errors: `source_solvers` empty vs 5 regions (was an `IndexError` after the MPI split) |
| `conf/NACA4412-SHAP-Vel-2540.yml` | now errors: `runs/naca4412_ref_SHAP_vel_2540/logs/best_model_SHAP_Vel.zip` missing |

---

## Config-level findings (not code bugs)

- **`conf/mini_channel/MC-ng-100.yml` and `MC-ng-101.yml` share
  `agent_run_name: 2002`** → same run folder, same cache file, same checkpoint
  prefix; whichever starts second resumes from the other's policy. Given the
  111→2001, 110→2004 pattern, `MC-ng-101` was probably meant to be `2003`.
- **`conf/conf_juwels/*` describe the JUWELS build**, not the local one:
  `nproc: 32`/`512`/`4096` against a `mini_channel/SIZE` compiled with `lp=10`,
  `lx1: 6` vs the local `conf/` variants, and restart folders
  (`rs6_lc_noresize_*`, `rs8_naca4412_200k_ref`) that do not exist locally —
  `initial` aborts with `FileNotFoundError` in `init_restart`.
- **`conf/small_wing/WING-PATH-TEST.yml`**: 4 control regions but 5 entries in
  six per-region lists. Verified harmless — the dropped 5th entries are exactly
  the `PS` region that `WING-SMALL` has and this file doesn't. `ndrl: 3` gives
  periods `12/15/18/27`, while the comment on `drl_steps` targets `36/45/54/81`
  (that needs `ndrl: 9`); same pair in `WING-SMALL`, so pre-existing.
  `policy_dir` points into `temp/archived/...`; byte-identical copies live in
  `data/policies_meta_wing_dr/`.
- **`write_timeSeries` writes `int_pos` into the shared `compile_path`**, not the
  run folder (`lib/nek_utils.py`). Sequential runs are fine (each `initial`
  regenerates it), but two jobs initialized concurrently from the same repo race
  on `envs/cases/<case>/int_pos`, and the point count differs per config
  (e.g. `MC-noctrl` writes 12 planes / 4800 points, the others 2 / 1152).

## Open items

- `src/eval_meta.py:83` — `if conf.runner.agent_run_name != 0:` compares a
  **list** to `0`, so it is always true and raises a misleading "The folder
  containing the trained agent does not exist". The clean fix is to drop it and
  call `validate_conf(conf)` there instead, so `evaluate` validates the same way
  when run without `initial`.
- `CONFIG_TAG="$(basename …)"` in both launchers — there are now 9 duplicate
  basenames across `conf/`, `conf/mini_channel/` and `conf/conf_juwels/`
  (`MC16-OC.yml`, `MC-ng-1*.yml`, `WING-SMALL.yml`,
  `NACA4412-SHAP-Vel-2540.yml`), so `log.initial.*` / `log.run.*` overwrite each
  other. Deriving the tag from the path relative to `conf/` would fix it.
