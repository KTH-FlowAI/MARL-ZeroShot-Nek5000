# `nek5000_solo` aborts in `MPI_INTERCOMM_CREATE`: `embedded.enabled` never reaches the `.par`/`.rea` writer

## Summary

Running an embedded (Python-free) evaluation via `nek-solo-run.sh` /
`sjob-gen.sh --case nek-solo` fails immediately: every MPI rank of
`nek5000_solo` aborts inside `MPI_INTERCOMM_CREATE`, even though the run
config sets `embedded.enabled: true` and the prepare step's own debug dump
confirms it parsed that correctly. The Fortran side never sees it — it takes
the coupled-mode branch and tries to build an intercommunicator with a Python
rank that doesn't exist in a solo run.

## Environment

- Repo: `KTH-FlowAI/MARL-ZeroShot-Nek5000`, branch `dev_yw_cost`
- Case: `envs/cases/lc_n7` (large-channel, omega1, v19), raw solver build
  (`nek5000_solo`)
- Config: `conf/omega1_BASE/lc-omega1-SS-eng-embedded.yml`
- Launch: `./execs/sjob-gen.sh --case nek-solo --config conf/omega1_BASE/lc-omega1-SS-eng-embedded.yml -J eval_O1_embedded --submit`
- SLURM job: `14163262` (also reproduced on `14163243`, `14163248`)
- Docs: `temp/docs/embedded_policy.md`

## Steps to reproduce

```bash
./utils/compile_case.sh --path envs/cases/lc_n7 --solver raw
./execs/sjob-gen.sh --case nek-solo \
  --config conf/omega1_BASE/lc-omega1-SS-eng-embedded.yml \
  -J eval_O1_embedded --submit
```

## Expected behavior

Per `temp/docs/embedded_policy.md` §5/§12, `drl/drl_main.f` should read
`UPARAM(10)==1`, branch to `POL_main`, and `return` — never reaching
`MPI_INTERCOMM_CREATE` — because a solo run has no Python rank on the other
side of that communicator.

## Actual behavior

All 512 ranks abort in `PMPI_Intercomm_create` on startup:

```
[jwc01n251:850962] ... PMPI_Intercomm_create+0x1eb ...
[jwc01n250.juwels:896963] 511 more processes have sent help message
  help-mpi-errors.txt / mpi_errors_are_fatal
```
(`log-files/log.solo.lc-omega1-SS-eng-embedded.yml`)

```
[ERR] solver failed, see log-files/log.solo.lc-omega1-SS-eng-embedded.yml
```
(`log-files/eval_O1_embedded-14163262.err`)

## Root cause

`envs/cases/lc_n7/drl/drl_main.f:37-42` gates the embedded branch on
`UPARAM(10)`:

```fortran
ictrl_mode = nint(UPARAM(10))
pol_ifsolo = ictrl_mode.eq.1
if (pol_ifsolo) then
   call POL_main
   return
endif
... ! falls through to MPI_INTERCOMM_CREATE
```

`UPARAM(10)` is written by `NEK_INIT.rewrite_REA_v19`
(`src/lib/nek_utils.py:459-460`):

```python
fpar.write('userParam%02d = %s \n' % (userp, 1 if self._embedded_on() else 0))
```

and `_embedded_on()` (`src/lib/nek_utils.py:63-65`) is:

```python
def _embedded_on(self) -> bool:
    return self.emb is not None and bool(getattr(self.emb, "enabled", False))
```

`self.emb` is set from the `emb=` constructor argument
(`src/lib/nek_utils.py:44-58`), **but neither caller passes it**:

- `src/initial.py:324`
  ```python
  initializer = NEK_INIT(nek=conf.simulation, drl=conf.runner, rank_folder=rank_folder)
  ```
- `src/initial_meta.py:165`
  ```python
  initializer = NEK_INIT(nek=conf.simulation, drl=conf.runner, rank_folder=rank_folder)
  ```

Neither passes `emb=conf.embedded`, so `self.emb` is always `None`,
`_embedded_on()` is always `False`, and `userParam10` is always written as
`0` — regardless of the YAML. Confirmed in the actual written file:

`runs/lc_omega1_SS_nes_solonek/eval/env_001/tcf.par`:
```
userParam10 = 0
```

while the same prepare run's own debug print shows the config was parsed
correctly:

`log-files/log.solo-initial.lc-omega1-SS-eng-embedded.yml`:
```
'embedded': {'enabled': True, 'net_precision': 8, ...}
```

So the break is specifically at the `NEK_INIT(...)` call sites — the
`embedded.*` block from the YAML never reaches the object that decides
`userParam10` (and, for v17 wing cases, `p091` — same code path via
`rewrite_REA_v17`, `src/lib/nek_utils.py:367`).

This also means `temp/docs/embedded_policy.md`'s own changelog
(§3, "Modified" table) is stale: it lists `src/initial.py`, `src/initial_meta.py`
as changed to "pass `embedded` and `logging` into `NEK_INIT`", but the
current checkout doesn't do that.

## Suggested fix

At both call sites, pass the embedded config (and, per the doc, logging)
through:

```python
initializer = NEK_INIT(nek=conf.simulation, drl=conf.runner,
                        rank_folder=rank_folder, emb=conf.embedded,
                        log=conf.logging)
```

(`src/initial.py:324`, `src/initial_meta.py:165`)

## Impact

- Blocks every embedded/`nek-solo` evaluation for both the channel
  (`nek_MARL`/`initial.py`) and wing (`meta_MARL`/`initial_meta.py`) paths —
  `userParam10`/`p091` is unconditionally `0`.
- Silent: the prepare step logs `embedded.enabled: True` correctly, so
  nothing before the solver launch hints that the flag was dropped.
- `nek-solo-run.sh`'s own guard (refusing to launch when `nek5000_solo` is
  missing, to avoid the coupled binary hanging forever in the same call)
  doesn't catch this case, because the binary *is* present and *is* the raw
  build — it's just being told (via `.par`) to behave as if it were coupled.

## Suggested regression check

After the fix, assert on the written `.par`/`.rea` in an embedded run:
`userParam10 == 1` (v19) / `p091 == 1.0` (v17), e.g. as a smoke check in
`nek-solo-run.sh` or a unit test around `NEK_INIT.rewrite_REA_v19`.
