# lc_n7_WallPressure

Standalone channel case derived from `../lc_n7_Bewley`. Mesh/order, sensing,
actuation, restart handling and forcing retain that setup. The new reward is
selected by `runner.reward_fn: wall_pressure` (`UPARAM(9)=2`):

```text
R = reward_alpha * (1 - tau_w / tau_w_ref)
  + reward_beta  * (1 - p_rms / pressure_rms_ref)

tau_w     = mu * <dU/dy>_space,cycle
tau_w_ref = mu * runner.dUdy
pbar(t)   = integral_control_wall(p(t) dS) / integral_control_wall(dS)
p_rms     = sqrt(<(p(t) - pbar(t))**2>_space,cycle)
```

`mu` is Nek's dynamic viscosity, `param(2)`. The existing dU/dy calculation
remains. Pressure is mechanical pressure in solver units. Each instantaneous
wall-area mean uses Nek surface quadrature and an MPI sum, with each lower-wall
control face counted once. Entire faces contribute, including edge nodes omitted
from agent registration by `gll_unique=1`. Squared fluctuations use the parent's
x/z planar averages, then actual-timestep cycle weights. The square root is taken
**after** these averages. Both cycle means reset at `i_evolv=1`.

This is fluctuation relative to the instantaneous spatial wall mean, not a
pointwise temporal mean. A uniform pressure gauge shift cancels. Pressure RMS
does not depend on the actuation mask or multiply by jet velocity: it remains
measurable with zero actuation. There is no pressure–velocity or kinetic-energy
reward contribution, and `reward_gamma` is ignored. Increases above either
reference give negative contributions; reductions give positive contributions.

## Configuration

The complete training template is
[`conf/omega2_BASE/lc-omega2-SS-wall-pressure.yml`](../../../conf/omega2_BASE/lc-omega2-SS-wall-pressure.yml).
It uses a separate run name and starts training without loading an old policy.
Set `runner.pressure_rms_ref` to your measured uncontrolled RMS before running.
Its default `0.0` is deliberately invalid for this mode; preparation rejects
zero, negative, infinite and NaN references. No physical baseline is assumed.
Use the same pressure units, wall region, mean-removal definition and averaging
settings when determining the reference. Existing `runner.dUdy` supplies the
uncontrolled shear reference; `reward_alpha` and `reward_beta` set the weights.

```yaml
runner:
  reward_fn: wall_pressure
  reward_alpha: 1.0
  reward_beta: 1.0
  pressure_rms_ref: 0.0  # REPLACE with your positive uncontrolled wall RMS
simulation:
  compile_path: envs/cases/lc_n7_WallPressure
```

This case supports `wall_pressure` and the original shear-only `dudy` mode.
For `net_gain`, select a Bewley case. Preparation rejects `wall_pressure` with
older case sources to prevent a mismatched MPI receive protocol. The new mode
is implemented for this v19 channel case and the channel Python environment.
When using `execs/channel-run.sh evaluate`, also pass `--reward-fn wall_pressure`:
that launcher otherwise overrides the YAML reward with `net_gain`.

## Records and statistics

The MPI and binary recorder layout stays at three slots: `rwd_tau = tau_w`,
`rwd_pw = p_rms`, and `rwd_v3 = 0` (reserved). The text monitor labels these
`rwd_tau`, `p_rms`, and `reserved_zero`. Python CSV component columns `R_tau`
and `R_pw` contain the corresponding relative reductions; `R_v3` is zero.
Embedded runs record these raw quantities; normalization uses the run YAML
references and weights during analysis, just as existing embedded recordings
do. Legacy energy-cost analysis must not interpret slot 2 as pumping power.

Raw STS fields 1–44 retain the original KTH moments. Field **45** contains the
span/time average of `(p-pbar_wall)**2`; there is no appended kinetic-cost field.
To obtain an RMS from this field, finish the desired wall-area and time averaging
before taking the square root. STS sampling uses `AVSTEP`, so matching a reward
cycle requires matching sampling and averaging windows. These files have 45
fields; keep them separate from the Bewley case's 46-field statistics.

## Build and validation

From the repository root, with the project's compiler/Python environment:

```bash
bash utils/compile_case.sh --path envs/cases/lc_n7_WallPressure --solver raw
bash utils/compile_case.sh --path envs/cases/lc_n7_WallPressure --solver drl
python utils/test_wall_pressure_reward.py
```

Both `nek5000_solo` (embedded) and `nek5000` (Python-coupled) use these sources.
Relevant source changes are marked `[WALL_PRESSURE]`. The regression compiles
production reward and statistics routines and tests 1/2/3 MPI ranks, including
an empty rank, unequal quadrature weights, face deduplication, omitted edge
agents, pressure gauge invariance, zero actuation, timestep weighting, resets,
statistics preservation, normalization, reference validation and input writers.
It uses synthetic fields and geometry operators; no production DNS is launched.
