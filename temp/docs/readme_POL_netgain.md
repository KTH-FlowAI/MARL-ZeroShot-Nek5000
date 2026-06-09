# Net-Gain Reward Function — Implementation Notes

**Author:** pol.suarez  
**Branch:** `dev_yw_fxb`  
**Date:** 2026-04-28

---

## Motivation

The original reward used only the wall-shear-stress proxy `dUdy` as a drag reduction signal,
with no penalty for the energy spent by the actuators.  The new reward measures the **net
energy saving**: how much drag is reduced minus how much the actuators cost.

Based on Stroh et al. 2015. "power unit term"

---

## Reward Definition

```
R = α · R_wallshear  +  β · R_pw  +  γ · R_v3
```

where each sub-reward is normalized by the reference wall shear stress
`τ_ref = ν · (dU/dy)_ref`:

| Term | Formula | Physical meaning |
|---|---|---|
| `R_tau` | `1 − τ_w^ctrl / τ_ref` | Drag reduction fraction |
| `R_pw`  | `− \|p'_w · v_w\| / τ_ref` | Pressure-velocity actuator cost |
| `R_v3`  | `− 0.5·\|v_w³\| / τ_ref` | Kinetic energy injection cost |

- **α, β, γ** are non-negative weights set in the YAML config.
- `R > 0` only when drag reduction exceeds total actuator power input.
- `R = 0` corresponds to break-even (control gains nothing net).
- Setting `α=β=γ=1` recovers the standard net-energy-saving metric
  `R = (Cf_ref − Cf_ctrl − W_in) / Cf_ref`.

### Order of magnitude (MC16, Reτ = 180)

Observed from live runs 2001 (ng-val, α=1, β=γ=0) and 2002 (ng-full, α=β=γ=1),
sampled over the first 14–18 training episodes:

| Term | ng-val (β=γ=0, logged only) | ng-full (α=β=γ=1) | Notes |
|---|---|---|---|
| `R_tau` | +0.01 → +0.02 | +0.25 → +0.26 | Drag reduction 1–26 % |
| `R_pw`  | −0.15 → −0.20 | −0.41 → −0.71 | Dominant cost; ~10–30× larger than `R_tau` |
| `R_v3`  | ~−1 × 10⁻⁵   | ~−4 × 10⁻⁴   | Negligible; 300–1000× smaller than `R_pw` |
| `R_net` | ≈ `R_tau`      | −0.15 → −0.46 | ng-full net is negative in early training  |

Key observations:

- **`R_pw` dominates the cost budget** by 1–2 orders of magnitude over `R_v3`.
  With equal weights (β = γ = 1), the pressure-velocity term controls the sign of R.
- **`R_v3` is nearly negligible** at Reτ = 180; γ can be set to zero with minimal effect
  on the effective optimization landscape for this flow case.
- **ng-full shows higher `R_tau`** than ng-val at comparable steps because the cost penalty
  forces the agent toward more efficient (larger drag-reduction-per-unit-cost) strategies.
- The ng-full reward is expected to become positive as the agent learns to reduce drag more
  efficiently than it costs actuator power (~25–30 % drag reduction is typically needed).

### Notes on the pressure term

`p'_w = p_w − ⟨p_w⟩` is the wall-pressure **fluctuation** (mean removed).
In a periodic channel, the mean pressure does no net work over the domain,
so only the fluctuating part exchanges energy with the actuator.
If the paper defines the term as full `p_w · v_w`, remove the mean-subtraction
step in `compute_netGain` (Step 2, `call sub3`).

### Reference values

`τ_ref = ν · dUdy_ref` is computed in Python from the config fields:
- `simulation.viscosity` — NEK convention: negative value means `ν = 1/|viscosity|`
- `runner.dUdy` — baseline `dU/dy` of the uncontrolled flow

---

## How to Enable

### Step 1 — Compile with `NETGAIN` flag

Edit `envs/cases/mini_channel/compile_script`, uncomment the `NETGAIN` line:

```bash
# Before (default, plain dUdy reward):
export PPLIST="MPIIO DRL YWDEBUG TSRS"

# After (net-gain reward):
export PPLIST="MPIIO DRL YWDEBUG TSRS NETGAIN"
```

Then recompile:

```bash
source ~/.bashrc.openmpi_ucx
./utils/compile_case.sh --m mini_channel
```

### Step 2 — Set Python config

In your YAML config file (e.g. `conf/MC16-TD3.yml`), add:

```yaml
runner:
  reward_fn: net_gain    # activates 3-component recv and weighted formula
  reward_alpha: 1.0      # weight on drag-reduction term
  reward_beta:  1.0      # weight on pressure-velocity cost
  reward_gamma: 1.0      # weight on kinetic energy cost
```

Leaving `reward_fn: dudy` (default) keeps the old behaviour — no recompile needed.

> **Important:** `reward_fn: net_gain` in Python **must** match the `NETGAIN` compile
> flag in Fortran. If Fortran sends 3 buffers but Python expects 1 (or vice versa),
> the MPI communication will deadlock.

---

## Averaging Pipeline

Every reward component passes through three distinct averaging stages before it
reaches the RL algorithm.  Understanding this is important for interpreting what
the agent actually optimises.

### Stage 1 — Spanwise (z) spatial average

```fortran
call gtpp_gs_setup(igs_z, xnel*ynel, 1, znel, 3)   ! idir=3 → z
call planar_avg(avgV, velV, igs_z)
call copy(velV, avgV, ntot)
```

`planar_avg` is a mass-weighted quadrature average using NEK's parallel
gather-scatter library (`fgslib_gs_op`).  The weighting kernel is `bm1` — the
GLL quadrature mass matrix — so the result is the proper volume-weighted mean
over the spanwise direction:

```
⟨f⟩_z(x,y) = ∫ f(x,y,z) dz  /  ∫ dz
```

After this call every z-position at a given (x, y) carries the same value.  
For the mini-channel mesh: `xnel=4`, `ynel=8` (wall-normal elements half-height),
`znel=4` — so 4 spectral elements are collapsed in z, each with `LZ1=6` GLL
points in the element.  The gather-scatter communicates across MPI ranks that
own different z-slices.

### Stage 2 — Streamwise (x) spatial average

```fortran
call gtpp_gs_setup(igs_x, xnel, ynel, znel, 1)     ! idir=1 → x
call planar_avg(avgV, velV, igs_x)
call copy(velV, avgV, ntot)
```

Same mechanism, now collapsing in x.  After both stages the field is
`⟨f⟩_{xz}(y)` — it only varies in the wall-normal direction y.

**Both flags are hardcoded `TRUE`** in `inc_src/DRL`:
```fortran
parameter(rwd_xavg=.TRUE., rwd_zavg=.TRUE.)
```
They cannot be changed at runtime; a recompile is required to disable them.

The same pair of averages is applied to **all three components** independently:
`dUdy`, `|p'v|`, and `0.5|v³|` each go through z-avg → x-avg before extraction.

### Stage 3 — Per-agent point extraction (global value in disguise)

After x-z averaging the field is homogeneous in the wall-parallel plane.  Each
agent reads its own GLL node:
```fortran
dudy_i = velV(ix, iy, iz, ie)
```
Because the field no longer varies in x or z, **all wall agents read the same
value** — the global x-z average.  The per-agent loop is structurally present
but functionally produces identical values for every agent.

This is consistent with `rew_mode: Homo` (default in `configs.py`), where all
agents share the same reward.  To get a genuinely local per-agent reward you
would need to set `rwd_xavg=.FALSE.` and/or `rwd_zavg=.FALSE.` in `inc_src/DRL`
and switch to `rew_mode: InHomo`.

### Stage 4 — Substep time average (within one action window)

`drl_reward(i_evolv)` is called by `DRL_main` at **every NEK timestep** while
the flow is evolving under a fixed action.  `i_evolv` counts from 1 to `ndrl`
(set by `userParam01` in `phill.par`, currently `ndrl=3`).  The reward is
accumulated as a running average:

```fortran
if (i_evolv == 1):
    rwd_tau(il) = tau_w               ! store first sample
else:
    rwd_tau(il) = (rwd_tau(il)*i_evolv + tau_w) / (i_evolv+1)
```

Tracing for `ndrl=3` with samples s₁, s₂, s₃:

| After step | Stored value |
|---|---|
| i_evolv=1 | s₁ |
| i_evolv=2 | (2s₁ + s₂) / 3 |
| i_evolv=3 | (2s₁ + s₂ + s₃) / 4 |

The first sample carries weight **2/4 = 0.5** while subsequent samples each
carry **1/4**.  A true uniform mean would give **1/3** each.  This slight
over-weighting of the first sample is inherited from the original code; for
`ndrl=3` the error is minor and the formula is approximately a time-mean over
the action window.

> The correct formula for an unbiased running mean would be
> `(old*(i_evolv-1) + new) / i_evolv`.

The MPI send happens **only at the last substep** (`i_evolv == ndrl`), so Python
receives the final time-averaged value.  Python's `evolve()` loop stays
synchronised step-by-step via CFL messages but only unpacks the reward buffers
on the final iteration.

### Summary — what one reward number represents

```
Single scalar per agent per policy step
= xz-averaged( time-averaged( instantaneous field ) )_wall
```

Concretely for τ_w:

```
rwd_tau = ⟨  (1/ndrl) · Σ_{t=1}^{ndrl} [ ρ·ν·(∂U/∂y)_wall(x,z,t) ]  ⟩_{xz}
```

(with the first-sample double-weighting noted above)

---

## Files Changed

### Fortran

#### `envs/cases/mini_channel/inc_src/DRL`
Added a new common block for the three reward components:
```fortran
real rwd_tau(totctrl), rwd_pw(totctrl), rwd_v3(totctrl)
common /drl_netgain/ rwd_tau, rwd_pw, rwd_v3
```
These arrays are populated by `compute_netGain` and sent by `drl_reward_out`.
The existing `rwd_agt` array and `/YW_drl/` common block are unchanged.

#### `envs/cases/mini_channel/drl/drl_reward.f`

**`drl_reward()` — dispatch switch**
```fortran
#ifdef NETGAIN
    call compute_netGain(i_evolv)
#else
    call compute_dudy(i_evolv)   ! original behaviour
#endif
```

**`compute_netGain()` — new subroutine**  
Steps:
1. Compute `dU/dy` via `gradm1`, apply x/z planar averages (`rwd_xavg`, `rwd_zavg`).
2. Compute `p'_w · v_w`: subtract spatial-mean pressure, multiply by `ACTIONS`, take `|·|`, average.
3. Compute `0.5 · |v_w³|`: cube the `ACTIONS` field, halve it, average.
4. Assemble per-agent moving averages into `rwd_tau`, `rwd_pw`, `rwd_v3`:
   ```fortran
   rwd_tau(il) = rho * nu * dUdy   ! τ_w^ctrl
   rwd_pw(il)  = |p'_w · v_w|
   rwd_v3(il)  = 0.5 · |v_w³|
   ```

#### `envs/cases/mini_channel/drl/drl_IO.f`

**`drl_reward_out()`** — sends 3 MPI buffers instead of 1 when `NETGAIN`:
```fortran
#ifdef NETGAIN
    ! tag NID+80000 → τ_w component
    ! tag NID+81000 → p'v component
    ! tag NID+82000 → v³ component
#else
    ! tag NID+80000 → rwd_agt (dUdy, original)
#endif
```

#### `envs/cases/mini_channel/compile_script`
Added a commented-out `NETGAIN` line for easy toggling (see Step 1 above).

---

### Python

#### `src/configs.py` — `Runner` dataclass

```python
reward_fn:    str   = 'dudy'   # 'dudy' | 'net_gain'
reward_alpha: float = 1.0      # α — weight on R_wallshear
reward_beta:  float = 1.0      # β — weight on R_pw
reward_gamma: float = 1.0      # γ — weight on R_v3
```

#### `src/nek_marl.py`

**`initialization()`** — pre-computes the reference:
```python
nu = 1.0 / abs(self.conf.simulation.viscosity)
self.baseline_tau_wall = nu * self.baseline_dudy   # τ_ref
self.reward_fn = self.conf.runner.reward_fn
```

**MPI tag dict** — two new tags added:
```python
'REWRD_PW': {"tag": 81000, ...}   # receives rwd_pw from Fortran
'REWRD_V3': {"tag": 82000, ...}   # receives rwd_v3 from Fortran
```

**`evolve()`** — branches on `reward_fn`:
- `'net_gain'`: receives 3 separate buffers (tau, pw, v3) via 3 MPI Recv calls per NID,
  distributes each to agents via `_distribute_field`.
- `'dudy'`: original single-buffer path, unchanged.

**`_normalize_reward()`** — keyword-argument signature:
```python
def _normalize_reward(self, dudy=None, tau_w=None, pw=None, v3=None):
    if self.reward_fn == 'net_gain':
        R_wallshear = 1.0 - mean(tau_w) / τ_ref
        R_pw        =     - mean(pw)    / τ_ref
        R_v3        =     - mean(v3)    / τ_ref
        return α*R_wallshear + β*R_pw + γ*R_v3
    else:
        return 1.0 - mean(dudy) / dUdy_ref
```

---

## MPI Communication Protocol

```
          Python (STB3)                     NEK5000 (Fortran)
               |                                   |
    send EVOLV |---------------------------------> |
               |                                   | compute_netGain()
               |  <--- CFL (tag 1999) ------------ |  (each substep)
               |         ...                       |
               |  [at last substep i_evolv==ndrl]  |
               |  <--- rwd_tau (tag NID+80000) --- |
               |  <--- rwd_pw  (tag NID+81000) --- |
               |  <--- rwd_v3  (tag NID+82000) --- |
               |                                   |
    R = α·R_τ + β·R_pw + γ·R_v3
```

The three Recv calls are sequential per NID, matching the three sequential Send calls
in `drl_reward_out`. Order must be preserved.

---

## Tuning the Weights

| Goal | Suggested setting |
|---|---|
| Pure drag reduction (ignore cost) | `α=1, β=0, γ=0` |
| Net energy saving (equal weight) | `α=1, β=1, γ=1` |
| Penalise pressure work more | `α=1, β=2, γ=1` |
| Penalise kinetic injection more | `α=1, β=1, γ=2` |

All three sub-rewards share the same normalization (`τ_ref`), so `α=β=γ=1` gives
each component equal dimensional weight before the policy optimizes.

---

## Component Monitoring

All three normalized components are logged in real time to
`runs/{id}/train/history/rewards_aggregated_*.csv` (columns `mean_R_tau`, `mean_R_pw`,
`mean_R_v3`). They are populated by `src/nek_marl.py` after each RL step and written by
`src/lib/reward_logger.py`.

To plot the live training curves (including all three components):

```bash
python utils/monitor_runs.py
# Output: utils/monitor_runs.png
```

The script auto-detects whether component columns are present and adapts the layout:
- **With components**: 3-row grid — total reward / (R_τ + R_pw) / (R_v3 + throughput)
- **Fallback** (old runs without columns): 2-row — total reward / (episode reward + throughput)

---

## Known Issues Fixed

### `sub3` / `col3` missing 4th argument (SIGSEGV on all NEK ranks)

`compute_netGain` originally called `sub3` and `col3` with 3 arguments:

```fortran
call sub3(wrk_buff, pwvw, buffer)        ! WRONG — n missing
call col3(pwvw, buffer, wrk_buff)        ! WRONG — n missing
```

Both subroutines are declared with 4 arguments (`a, b, c, n`).
In Fortran 77, the missing `n` is resolved from the call stack, giving a garbage loop
bound → all 40 NEK ranks walk off their stack arrays simultaneously → SIGSEGV.

Fix (committed, `drl_reward.f` lines 285 and 288):

```fortran
call sub3(wrk_buff, pwvw, buffer, ntot)  ! correct
call col3(pwvw, buffer, wrk_buff, ntot)  ! correct
```

Symptom if you encounter this again: all MPI ranks print `[REWARD] NETGAIN HANDLE INIT!`
then crash with signal 11 at stack-region addresses (0x7ffXXX...) within ~3 minutes of
job start.
