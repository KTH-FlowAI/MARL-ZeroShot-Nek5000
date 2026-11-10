# naca4412_US_SS_2586_Bewley

Derived from `../naca4412_US_SS_2586`. The mesh dimensions, suction-side
control band (`x/c = 0.25..0.86`), sensing, local friction-velocity scaling,
zero-net-mass-flux treatment of the DRL actions and policy/recorder plumbing follow
the parent case. The matching configuration is
`conf/naca4412_200k/NACA4412-US-SS-2586-Bewley.yml`.
It retains the parent's four **DR-trained policies** and restart source,
sets a distinct run name, and evaluates them with `reward_fn: net_gain`.
It does not create or retrain policies.

## Actuator power

`drl/drl_bewley.f` implements the conservative estimate on p. 218 of
[Bewley, Moin & Temam (2001), JFM 447, 179–225](https://doi.org/10.1017/S0022112001005821).
At every solver step, before averaging, it computes

```
phi      = ACTIONS + bw_suction             # total applied normal amplitude
bw_suction = -0.002                         # existing uniform suction
pbar     = integral_patch(p dS) / integral_patch(dS)
pw_local = abs((p - pbar) * phi)
v3_local = 0.5 * rho * abs(phi)**3
```

Nek's mechanical-pressure convention is used; density is one in the
provided configuration. `ACTIONS` is the amplitude returned by `ACTUATE_JET`;
the boundary condition adds uniform suction before projecting onto Cartesian
components. The shared `inc_src/BEWLEY` parameter fixes that offset in both
`userbc` and power accounting. The offset is applied only where `msk_act` is
active; actuator costs are zero elsewhere. Zero DRL action still incurs the
uniform-suction cost. Multiplication by `body_sin` would give only the y component.

The pressure reference is one instantaneous, surface-area-weighted mean
over **all registered control faces on the suction-side patch**. This is
the explicit adaptation of the paper's channel-wall mean to this wing.
Nek's `surface_int` supplies surface Jacobian and GLL quadrature weights;
each face is counted once, and all ranks participate in the global sum.
There is no independent mean subtraction at each chord location. Adding
a spatially uniform pressure offset leaves the costs unchanged.

Both blowing and suction contribute to the absolute costs. Negative local
pressure work is not clipped and cannot cancel positive work. The local
costs are spanwise averaged (`rwd_zavg=True`), retaining chordwise variation
(`rwd_xavg=False`), then averaged within each DRL cycle with weights `abs(dt)`.
The cycle clock resets at `i_evolv=1`, including after a restart.

## Saved quantities and interpretation

The existing three-slot interface is retained:

1. `rwd_tau`: the parent's shear proxy `mu * abs(dUdy)`, now dt-weighted.
2. `rwd_pw`: spanwise/cycle mean of `pw_local`.
3. `rwd_v3`: spanwise/cycle mean of `v3_local`.

The parent shear-gradient/projection code and baseline convention are
preserved; this change addresses actuator-power accounting. The normalized
reward still uses the existing shear reference and regional policy scales.
It is not a new whole-wing force/drag-power calculation.

Both coupled MPI transfer and embedded `drlrec` output consume these same
buffers. The embedded recorder must use reward mode 1 (`net_gain`), as the
provided YAML specifies. In the coupled Python workflow, the separate
`history/netgain_ts_*.npz` files retain the three chordwise profiles.
Existing positive-only/averaged records cannot be converted to this
definition without the original instantaneous wall fields.

## On-the-fly statistics and post-processing

The original 66 statistics are retained, including signed `pv` and `vvv`.
`stats/statistics_bewley.f` appends two fields through `user_stat_compute`:

| Field | Name in nekstat | Quantity |
| --- | --- | --- |
| 67 | `act_cost_p` | Span/time mean of `pw_local` |
| 68 | `act_cost_k` | Span/time mean of `v3_local` |

Both statistics and DRL rewards call `bewley_cost_fields` for the instantaneous
native-order costs. Statistics then reshuffle and apply the existing spanwise
quadrature and time weights (`alpha`, `beta`); no DRL cycle averages are reused.
The existing allocation `STAT_LVAR=72` accommodates the 68 output fields.
Statistics are sampled at `stat_comp` before `DRL_main`, using the current
action mask/amplitudes. Their cadence/window can differ from the DRL recorder.

Use `post_process/nekstat/configs/naca4412_bewley_post.yml`, which selects
`data/dependencies/naca4412_bewley` and the separate 68-field interpolator.
Its copied NACA4412 interpolation mesh is unchanged. Python/MAT exports retain
both costs as scalars without rotation or further absolute values. Use wall
values only and integrate over controlled-wall arc length for power per span.
Old 66-field statistics cannot be mixed into this post-processing run.

The pressure reference remains the chosen controlled-patch mean. Accounting
now includes the imposed suction velocity; it does not model an external
suction-system reservoir pressure or pump efficiency.

## Build and run

From the repository root, with the usual MPI environment loaded:

```bash
# Python-free evaluation executable
bash utils/compile_case.sh --path envs/cases/naca4412_US_SS_2586_Bewley --solver raw

# Optional coupled Python/Nek executable
bash utils/compile_case.sh --path envs/cases/naca4412_US_SS_2586_Bewley --solver drl

# Launch using the inherited 4096-rank production setup
bash execs/nek-solo-run.sh --config conf/naca4412_200k/NACA4412-US-SS-2586-Bewley.yml
```

The normal launch preparation stages the shared mesh and restart files and
exports the inherited policies. No simulation is started by creating or
building this case. The canonical embedded-policy sources and `POLICY`
declarations are shared from `../naca4412_ref_SS_2570`, as in the parent.

## Validation

```bash
python utils/test_bewley_reward.py
```

This compiles the production reward routine against synthetic solver data
and exercises it on 1, 2, and 3 MPI ranks. Checks cover surface quadrature,
face deduplication, an empty control rank, pressure-offset invariance,
both signs of actuation/work, absolute values before averaging, total normal
amplitude, variable-timestep cycle means, cycle reset, density scaling,
uniform-suction cost with zero DRL action, and zero cost off the actuated wall.
The production statistics reshuffle and averaging operators are also checked,
including time weights/reset and preservation of the original statistics.
The fixture supplies the solver operators; it is not a full wing DNS test.

Both `nek5000_solo` (raw backend) and `nek5000` (coupled backend) have been
built successfully with the production SIZE and GNU Fortran/Open MPI.
The synthetic tests passed on all three rank counts. The production
4096-rank wing simulation has not been launched.
