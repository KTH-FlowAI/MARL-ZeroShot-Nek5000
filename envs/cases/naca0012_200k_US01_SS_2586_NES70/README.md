# naca0012_200k_US01_SS_2586_NES70

Standalone case derived from `../naca0012_200k_NES70`, with uniform suction
at 0.1% of Uinf and a fixed action scale. Case-specific Fortran, statistics,
policy sources and headers are local copies; no sibling case is required
to compile. The shared solver trees and NACA0012 mesh/restart inputs are
still required.

## Actuation

With Uinf = 1, `inc_src/NES` defines:

- `nes_suction = -0.001`: suction into the wing.
- `nes_action_scale = 3.0 * 0.0624 = 0.1872`: the constant `uti` used in
  `drl/drl_action.f::znmf_avg` for both embedded and coupled controllers.
- `nes_blow = 0`: no separate blowing patch.

At active suction-side wall nodes with `0.25 <= x/c <= 0.86`, the applied
normal velocity is

```text
phi = ACTIONS - 0.001
ACTIONS = 0.1872 * (a - <a>_z)
```

Here `a` is the controller output arriving at the Fortran action routine.
The existing GLL-weighted spanwise zero-mean correction acts on the scaled
controller output. The suction offset is added afterward in `userbc`, so
it is neither scaled by 0.1872 nor removed by that correction. The velocity
is projected onto Cartesian components using the existing wing normal.
Zero controller output therefore still produces `phi = -0.001`.

The parent selects complete wall-element faces using their midpoint.
The suction offset additionally checks each GLL node's actual coordinates,
including both x endpoints, so it cannot extend past the specified band.
The side guard is `0 < y/c <= 0.6`, matching the parent suction-side region.
The parent controller face selection itself is retained.

`X2Utau` remains in the sensing-height calculation (`y+ = 15`) and
observation normalization. It is not called to scale the control action.
Keep the `UTAU` build flag enabled. Avoid applying another action scale in
the Python/embedded configuration (`runner.rescale_actions: false`).
Use `simulation.znmf_avg: 1` to enable the parent action-processing path.

## Statistics and recorder

The original 66 statistics are retained, followed by both cost definitions:

| STS field | Name | Instantaneous quantity |
| --- | --- | --- |
| 67 | `act_cost_p_bewley` | `abs((p - pbar) * phi)` |
| 68 | `act_cost_k_bewley` | `0.5 * rho * abs(phi)**3` |
| 69 | `act_cost_p_hasegawa` | `max((p - pbar) * phi, 0)` |
| 70 | `act_cost_k_hasegawa` | `0.5 * rho * max(phi**3, 0)` |

`nes_wall_velocity` uses the same total applied velocity and suction-region
guards as `userbc`. `pbar(s,t)` is evaluated separately at every cross-section
GLL position, averaging only the homogeneous span with GLL weights. The
pressure-reference operator retains the parent's uniform-span assumption.
Absolute values and positive parts are applied pointwise before statistics
averaging, which uses GLL weights and the geometric line Jacobian, followed
by time averaging. The allocated statistics capacity is 72, with exactly
70 output variables.

The three-component DRL recorder defaults to Bewley (`nes_reward_model=1`);
set it to 2 in `inc_src/NES` and rebuild for Hasegawa recording. STS always
contains all four costs. Use `runner.reward_fn: net_gain`. The recorder's
shear-based reward retains the parent definition; a whole-surface NES
requires surface integration of the saved wall quantities.

## Build and use

From the simulation repository root:

```bash
bash utils/compile_case.sh --path envs/cases/naca0012_200k_US01_SS_2586_NES70 --solver raw
bash utils/compile_case.sh --path envs/cases/naca0012_200k_US01_SS_2586_NES70 --solver drl
```

Both executables have been built: `nek5000_solo` (embedded controller) and
`nek5000` (coupled controller). The legacy `makenek` and `compile-script`
entry points delegate to `compile_script` with the same NES70 build flags.

In your NACA0012 run configuration, use:

```yaml
runner:
    case_name: naca0012_200k_US01_SS_2586_NES70
    reward_fn: net_gain
    rescale_actions: false
simulation:
    CASENAME: naca_wing
    compile_path: envs/cases/naca0012_200k_US01_SS_2586_NES70
    shared_data_path: data/simulations/naca0012_200k
    exeName: nek5000_solo  # use nek5000 for the coupled backend
    znmf_avg: 1
```

Keep the NACA0012 restart, policy selection, controller cadence and remaining
run settings from your intended experiment. The fragment above is not a full
run configuration. No production simulation has been launched.

The existing `post_process/nekstat/configs/naca0012_nes70_post.yml` and
`data/dependencies/naca0012_nes70` handle these fields; no new interpolation
mesh is needed. Set the post-processing source path to the new run.

## Verification

- `python3 utils/test_nes70.py`: all five NES70 cases, both recorder
  definitions, on 1/2/3 MPI ranks (30 checks passed).
- `python3 utils/test_naca0012_us01.py`: production action processing,
  boundary projection and NES routines on 1/2/3 ranks (3 checks passed).
  Checks the fixed scale, unequal quadrature weights, nonzero and zero
  controller output, suction after zero-mean correction, both band endpoints,
  points outside the band/side, inactive nodes, and all four cost terms.

The tests compile the production routines with synthetic mesh/flow fixtures;
they are not a production DNS validation.
