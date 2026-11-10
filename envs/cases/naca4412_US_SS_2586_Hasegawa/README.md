# naca4412_US_SS_2586_Hasegawa

Created from `../naca4412_US_SS_2586_Bewley`, including its local spanwise
pressure reference, with the positive-only switches from Eq. (4.1) in
[Sonoda, Liu, Itoh & Hasegawa (2023), JFM 960, A30](https://doi.org/10.1017/jfm.2023.147).
The underlying simulation setup follows `../naca4412_US_SS_2586`: mesh,
suction-side control band (`x/c = 0.25..0.86`), sensing, local friction-velocity
scaling, zero-net-mass-flux treatment of DRL actions, and policy interfaces.

The matching configuration is
`conf/naca4412_200k/NACA4412-US-SS-2586-Hasegawa.yml`. It evaluates the four **US02 NES-trained TD3 policies** from
`data/policies_meta_wing_nes` with `reward_fn: net_gain`, retaining the
original restart source. It does not create or retrain policies.

## OC and NES-DRL configurations

- `conf/naca4412_200k/NACA4412-US-SS-2586-Hasegawa.yml`: NES-DRL,
  output under `runs/naca4412_US_SS_2586_Hasegawa_NES/env_001`.
- `conf/naca4412_200k/NACA4412-US-SS-2586-Hasegawa-OC.yml`: analytic OC,
  output under `runs/naca4412_US_SS_2586_Hasegawa_OC/env_001`.

The NES configuration uses `archived_lc_omega{1,2,3,4}_SS_US02_nes` and
checkpoint stems `eval_model_omega{1,2,3,4}_SS_US02_nes` (no `lc_` in the
checkpoint stem). These are the existing NES-trained policies; changing the
cost accounting does not retrain them against the new definition.

OC uses an analytic linear actor with `action = -v'`, before the existing
spanwise zero-flux correction and the -0.002 uniform suction. The solver
normalizes observations and restores actions with local friction velocity;
the config's unit observation/action scales avoid double scaling. OC needs
no checkpoint, uses `embedded.analytic_policy: OC`, and updates every solver
step (`ndrl=1`, all regional `drl_steps=1`). NES retains `ndrl=9` and regional
update intervals `[4,5,6,9]`, i.e. `[36,45,54,81]` solver steps.

Both definitions use the same controllers and spatial settings. All four
configurations retain the previous effective budget of 90,000,000 solver
steps: 10,000,000 NES cycles or 90,000,000 OC cycles. Embedded records have
the same 9,000-step stride. Equal solver-step budgets need not imply equal
physical duration because timesteps adapt to the CFL target; compare matched
physical-time statistics windows. Automatic checkpoint resume is disabled;
pass `--resume` to continue an existing run.

For a shorter 900-step smoke run (still using the production 4096-rank mesh):

```bash
bash execs/nek-solo-run.sh --config conf/naca4412_200k/NACA4412-US-SS-2586-Hasegawa.yml --nb-interactions 100 --rec-freq 10
bash execs/nek-solo-run.sh --config conf/naca4412_200k/NACA4412-US-SS-2586-Hasegawa-OC.yml --nb-interactions 900 --rec-freq 90
```

Both commands record every 90 solver steps. Use the same overrides for the
other cost definition. The inherited restart directory
`data/restarts/rs8_naca_wing200k_02US/init_1` must be present on the execution
machine; it is absent from this local checkout. All four configs passed the meta schema/checkpoint checks, policy export
and launcher dry runs. Exported OC weights implement the linear opposition
law; controller files match across the two cost definitions apart from
generation timestamps. These checks do not require flow restart files.
No simulation was launched during preparation.

## Actuator power

Eq. (4.1) switches the pressure-work and kinetic-flux terms independently:
`max(p_w*v_w, 0) + 0.5*max(v_w**3, 0)` in the paper's nondimensional form.
Here the same switches are applied with the wing pressure reference chosen
for the Bewley case. This local reference is a wing adaptation; Eq. (4.1)
itself writes wall pressure `p_w` without this chord-dependent subtraction.

At every solver step `drl/drl_hasegawa.f` computes:

```
phi        = ACTIONS + hg_suction          # total applied normal amplitude
hg_suction = -0.002                        # existing uniform suction
pbar(s,t)  = spanwise_mean(p(s,z,t))        # separate at every surface point
pw_local   = max((p - pbar) * phi, 0)
v3_local   = 0.5 * rho * max(phi, 0)**3
```

Nek uses mechanical pressure; density is one in the supplied configuration.
The shared `inc_src/HASEGAWA` parameter defines the suction offset for both
`userbc` and cost accounting. The offset applies only at active `msk_act`
nodes, where `ACTIONS` is the normal amplitude from `ACTUATE_JET`, before
projection onto Cartesian components. Costs are zero off the actuator mask.

Each term is clipped **pointwise before any spatial or time averaging**.
There is no clipping of the combined pressure-plus-kinetic sum. Negative
pressure work is discarded; suction has zero kinetic contribution. Suction
can still have positive pressure cost when `p - pbar` is negative. Zero DRL
action retains the uniform suction; a positive DRL action below 0.002 still
produces net suction, so its kinetic contribution is zero.

The instantaneous reference `pbar(s,t)` is computed independently at every
cross-section GLL location. `hasegawa_spanwise_pressure` uses `z_averaging`
and `z_avg_reshape` to combine matching cross-section points across all MPI
ranks and replicate the result along the span. There is no chordwise
pressure averaging. GLL weights assume the existing uniform-span mesh.
Unused allocated element slots are excluded (`nelv`, not `lelt`). Adding
any spanwise-constant pressure offset at a given chord location leaves its
cost unchanged. Accounting uses the local spanwise reference and does not
model external pump efficiency or a suction reservoir pressure.

The local costs are spanwise averaged (`rwd_zavg=True`), retaining the
chordwise profiles (`rwd_xavg=False`). DRL cycle means use `abs(dt)` weights;
the cycle clock resets at `i_evolv=1`, including after restart.

## Recorded quantities

The existing three-slot interface is retained:

1. `rwd_tau`: inherited shear proxy `mu * abs(dUdy)`, dt-weighted.
2. `rwd_pw`: spanwise/cycle mean of the positive pressure work.
3. `rwd_v3`: spanwise/cycle mean of the positive kinetic term.

The original shear projection, baseline and regional normalization remain
in place. This is an actuator-cost update within the existing NES estimate,
not a new whole-wing drag-power calculation. Coupled MPI transfer and the
embedded recorder both consume these buffers. The embedded recorder must
use mode 1 (`net_gain`), as in the YAML. Coupled Python output keeps the
three profiles in `history/netgain_ts_*.npz`.

## Statistics and post-processing

`stats/statistics_hasegawa.f` appends fields 67 and 68 after the original 66
statistics (including the signed `pv` and `vvv` terms):

| Field | Name in nekstat | Quantity |
| --- | --- | --- |
| 67 | `act_cost_p` | Span/time mean of `max((p-pbar)*phi, 0)` |
| 68 | `act_cost_k` | Span/time mean of `0.5*rho*max(phi, 0)**3` |

Both DRL and statistics call `hasegawa_cost_fields`. Statistics use their
own instantaneous samples, reshuffle/span operators and `alpha`, `beta`
time weights, rather than DRL cycle averages. `STAT_LVAR=72` accommodates
the 68 output fields. Statistics are sampled at `stat_comp` before
`DRL_main`, using the current action mask and amplitudes.

Use `post_process/nekstat/configs/naca4412_hasegawa_post.yml`, setting its
`source.path` to this case's results. It uses separate run/output names and
reuses `data/dependencies/naca4412_bewley`: the interpolation mesh and
68-field layout are identical, and that interpolator passes through the
cost scalars without recomputing them. Python/MAT exports preserve both
costs without rotation or another absolute value. Use wall values and
integrate along controlled-wall arc length for power per unit span.

Keep Hasegawa and Bewley statistics in separate runs/averaging windows:
their field counts match but their cost definitions differ. Existing
averaged `pv`, `vvv`, or Bewley costs cannot reconstruct the positive-only
terms without the instantaneous fields or additional signed moments.

## Build and run

From the simulation repository root:

```bash
bash utils/compile_case.sh --path envs/cases/naca4412_US_SS_2586_Hasegawa --solver raw
bash utils/compile_case.sh --path envs/cases/naca4412_US_SS_2586_Hasegawa --solver drl
bash execs/nek-solo-run.sh --config conf/naca4412_200k/NACA4412-US-SS-2586-Hasegawa.yml
```

The inherited production configuration uses 4096 MPI ranks. Normal launch
preparation stages the shared mesh/restart and exports the inherited
policies. Canonical embedded-policy sources and `POLICY` declarations are
shared from `../naca4412_ref_SS_2570`, as in the parent case.

## Validation

```bash
python3 utils/test_hasegawa_reward.py
python3 utils/test_bewley_reward.py
```

The new MPI test compiles the production cost, pressure-reference,
statistics and span operators against synthetic solver fields on 1, 2 and
3 ranks. Hand-calculated results check independent local clipping versus
absolute values, clipping after averaging, and clipping the sum. It also
checks reversed jets, total velocity across the suction offset, zero
velocity, inactive nodes, density, GLL weights, distinct means along the
chord, pressure-offset invariance, empty ranks, variable timesteps, cycle
resets and statistics time weights. The original 66 statistics are preserved.
These fixtures do not replace a production DNS run.

Both Hasegawa and Bewley tests passed on all three rank counts. The new
`nek5000_solo` (embedded) and `nek5000` (coupled) executables built successfully
with the production SIZE and GNU Fortran/Open MPI. No production simulation
was launched.
