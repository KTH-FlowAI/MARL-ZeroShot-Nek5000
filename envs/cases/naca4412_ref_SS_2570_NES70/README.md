# naca4412_ref_SS_2570_NES70

Derived from `../naca4412_ref_SS_2570`. The parent's SIZE, control band, sensing,
actuation, mesh/restart conventions, and policy plumbing are preserved.
Both cost definitions are now sampled from the same instantaneous pressure
and total prescribed wall velocity. Original cases and 68-field variants
remain separate.

## Statistics contract

The original 66 statistics are retained in their original order. Four
scalar costs are appended, making the STS header report **70** fields:

| STS field | Python / MAT name | Instantaneous quantity before averaging |
| --- | --- | --- |
| 67 | `act_cost_p_bewley` | `abs((p - pbar) * phi)` |
| 68 | `act_cost_k_bewley` | `0.5 * rho * abs(phi)**3` |
| 69 | `act_cost_p_hasegawa` | `max((p - pbar) * phi, 0)` |
| 70 | `act_cost_k_hasegawa` | `0.5 * rho * max(phi**3, 0)` |

`pbar(s,t)` is the instantaneous spanwise mean at each cross-section GLL
location, with no chordwise averaging. This retains the wing adaptation
used in the tested Bewley and Hasegawa cases. Both definitions share one
pressure mapping and one spanwise pressure reduction per call. Absolute
values/positive-part switches are applied independently at each point
before the statistics spanwise quadrature and time weighting. Nek pressure
is mechanical pressure; density enters the kinetic term.

The 72-field statistics allocation is sufficient. `nes_stat_compute`
checks that it receives 66 original fields, appends the four costs and
returns `lnvar=70`. No original signed `pv` or `vvv` field is replaced.

## Total applied velocity

DRL normal amplitude with no uniform offset.

`nes_wall_velocity` mirrors the boundary-condition precedence and gives
zero outside the active boundary. Background blowing is restricted to
velocity BC faces, never assigned to interior nodes. The US/blowing
constants are shared with userbc through `inc_src/NES`. Positive `phi`
means blowing into the fluid. Both costs include the prescribed background
actuation where present; external pump/reservoir losses are not modeled.
The spanwise operators use actual local elements (`nelv`), excluding unused
allocated element slots, and retain the existing uniform-span GLL weights.

## DRL recorder

The existing three slots remain `rwd_tau`, `rwd_pw`, `rwd_v3`; the binary
recorder/MPI format is unchanged. `nes_reward_model=1` in `inc_src/NES`
selects **Bewley** costs by default; set it to 2 and rebuild to record
Hasegawa costs instead. Both definitions are always saved in STS, regardless
of this choice. Use `reward_fn: net_gain` for the three-component recorder.
The reward calculation follows the tested Bewley wing shear proxy and
uses `abs(dt)`-weighted cycle means. This is not a whole-wing drag-power
integral. The recorder samples DRL agent nodes; separate background blowing
outside those nodes is included in STS and must be included when integrating
total actuator power.

## Build and use

From the simulation repository root:

```bash
bash utils/compile_case.sh --path 'envs/cases/naca4412_ref_SS_2570_NES70' --solver raw
bash utils/compile_case.sh --path 'envs/cases/naca4412_ref_SS_2570_NES70' --solver drl
```

In a copy of the parent run YAML, set `simulation.compile_path` to
`envs/cases/naca4412_ref_SS_2570_NES70`, choose a distinct `runner.case_name`, and keep the
parent's geometry, restart, controller and policy settings. Select
`simulation.exeName: nek5000_solo` for embedded evaluation or `nek5000` for
the coupled backend. The US02 NES and OC configurations are provided as
`conf/naca4412_200k/NACA4412-US-SS-2586-NES70.yml` and `...-NES70-OC.yml`.
They use the existing NES-trained policies or analytic OC, respectively.

## Post-processing

Use `post_process/nekstat/configs/naca4412_nes70_post.yml` and set its source
path and sampling window. It selects a separate 70-field interpolator and
copies of the matching NACA4412 geometry/interpolation mesh. The new
cost scalars require no mesh regeneration, rotation, clipping or derivatives.
The Python reader/MAT exporter preserve all four explicit field names.
The original derivative block remains 68 fields, distinct from STS 67–70.

Do not combine 66-, 68-, and 70-field STS files within one window. The new
interpolators reject incompatible headers before reading scalars. Integrate
the wall cost values along the full actuated surface, including any separate
blowing patch, to obtain actuator power per unit span.

## Verification

`python3 utils/test_nes70.py` compiles each case's production cost, statistics,
spanwise and reshape routines against the established independent golden
fixtures on 1, 2 and 3 MPI ranks, including an empty rank. Both definitions,
mean-pressure locality, clipping order, time weights/reset, density,
background offsets, and blowing/DRL precedence are checked. All 24 MPI
checks passed. Both production executables built successfully.
No production simulation was launched.
