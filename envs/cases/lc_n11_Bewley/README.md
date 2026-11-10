# lc_n11_Bewley

Derived from `../lc_n11`. Mesh/order, rank sizing, sensing, actuation, policy
sources, restarts and forcing retain the parent setup. Both `nek5000_solo`
(embedded) and `nek5000` (coupled) are built from this case. No DNS is launched
by building it.

## Actuator-cost definition

The net-gain reward uses the conservative actuator-cost expression from
Bewley, Moin & Temam (2001), DOI 10.1017/S0022112001005821:

```
phi = ACTIONS * msk_act
pbar = integral_control_wall(p dS) / integral_control_wall(dS)
cost_p = abs((p - pbar) * phi)
cost_k = 0.5 * rho * abs(phi)**3
```

`phi` matches the velocity returned by `ACTUATE_JET` and imposed by `userbc`.
There is no additional uniform-suction velocity in these channel boundary
conditions. In particular, do not add the wing case's -0.002 offset. Some
channel configurations use US02 in their name but emulate effects through
body-force damping; the name does not change the wall boundary condition.

`pbar` is the instantaneous area-weighted mean on the registered lower-wall
control faces, using Nek surface quadrature. Each face is integrated once;
all ranks participate, including ranks without control points. Spatially
uniform pressure shifts leave the costs unchanged. The current setups control
only the lower wall; separate wall references would be needed for a future
independently actuated upper wall.

Both absolute values are taken at individual GLL nodes, before x/z averaging.
Blowing and suction both incur kinetic cost. The actual mask is applied to the
velocity before multiplication/cubing. Costs are zero at inactive nodes.
The pressure convention is mechanical pressure, and the kinetic term includes
`rho = param(1)`.

## Records and statistics

`compute_netGain_Bewley` preserves the three recorder slots: `rwd_tau`,
`rwd_pw`, and `rwd_v3`. Cycle averages are weighted by the actual timestep and
reset at `i_evolv=1`. Lower-wall shear uses `param(2) * mean(dU/dy)`; Nek's
`param(2)` is dynamic viscosity, so no additional density factor is applied.
This gives the parent's shear result at the configured density of one.
The existing Python normalization/baseline and gain monitor remain in use.

The local copy `stats/stat.f` retains all original 44 KTH statistics and adds
one hook to `stats/statistics_bewley.f`. It evaluates the same instantaneous
cost helper as the reward, then uses the existing statistics reshuffle,
spanwise quadrature and time weights:

| Raw STS field | Meaning |
| --- | --- |
| 1–44 | Original moments, including signed pressure/velocity and cubic terms |
| 45 | Span/time average of `cost_p` (`act_cost_p`) |
| 46 | Span/time average of `cost_k` (`act_cost_k`) |

`STATD` now allocates 46 fields. Both backends build the case-local `stat.f`;
the shared solver/Toolbox sources and original cases are unchanged. Statistics
are sampled before the next control update, at the parent's `AVSTEP` cadence.
A comparison with cycle records must use matching windows and sampling.

These channel STS files have **46 fields**, unlike the 68-field wing schema.
Do not mix old 44-field and new 46-field statistics within one averaging run.
A channel post-processor must carry raw fields 45/46 through its own reordering
and export; they must not be interpreted as wing fields 67/68. This case change
does not change the separate `nekstat` channel post-processing dependencies.
Only wall values represent actuator power; integrate them over wall area or
use consistent area averages when normalizing energy saving.

## Build and use

From the `nek_power_saving` repository root:

```bash
bash utils/compile_case.sh --path envs/cases/lc_n11_Bewley --solver raw
bash utils/compile_case.sh --path envs/cases/lc_n11_Bewley --solver drl
```

In a copy of the corresponding parent run configuration, select:

```yaml
runner:
  reward_fn: net_gain
  reward_alpha: 1.0
  reward_beta: 1.0
  reward_gamma: 1.0
simulation:
  compile_path: envs/cases/lc_n11_Bewley
```

Retain the parent run's remaining simulation and policy settings. The local
`.par` also defaults `userParam09` to 1 (`net_gain`); the launcher regenerates
this from `runner.reward_fn`. Selecting `dudy` intentionally retains the old
shear-only reward mode. No policy is retrained or copied by creating this case.
Use a separate results location for the new energy-accounting definition.

## Validation

```bash
python3 utils/test_bewley_channel.py
```

The test compiles every channel variant's production reward, cost and statistics
routines with its real `DRL` and `STATD` declarations, and exercises 1, 2 and 3
MPI ranks. It covers local absolute values before spatial averaging, pressure
gauge invariance, area weighting, face deduplication, an empty rank, fractional
actuation masks, both velocity signs, density scaling, variable-timestep cycle
means/reset, statistics reshuffling/time weighting/reset and preservation of
original moments. Geometry/planar-average solver operators use synthetic data;
these checks are not a production channel DNS.
