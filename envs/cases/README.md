# Case Description 

## `cluster_mini`: Clustering multiple nodes as one agent 

## `pw_mini_channel`: The reward consider the pressure-velocity correlation and cubic of energy input.

## `lc_n7`: large channel setup of modernate Re 

## Bewley channel variants

Each copy preserves its parent's setup and corrects actuator power by taking
local absolute values before spatial/time averaging. Raw statistics retain
fields 1–44 and append pressure/kinetic costs as fields 45/46.

- [lc_n7_Bewley](lc_n7_Bewley/README.md)
- [lc_n11_Bewley](lc_n11_Bewley/README.md)
- [lc_extra_n11_Bewley](lc_extra_n11_Bewley/README.md)
- [lc_n7_onlyU_Bewley](lc_n7_onlyU_Bewley/README.md)
- [lc_n7_onlyV_Bewley](lc_n7_onlyV_Bewley/README.md)
- [lc_n7_UcombV_Bewley](lc_n7_UcombV_Bewley/README.md)
- [mini_channel_Bewley](mini_channel_Bewley/README.md)

## Wall-pressure RMS channel reward

[lc_n7_WallPressure](lc_n7_WallPressure/README.md) is a standalone copy of
`lc_n7_Bewley` with `reward_fn: wall_pressure`: alpha times relative shear
reduction plus beta times relative wall-pressure RMS reduction. Set the new
`runner.pressure_rms_ref` to a positive uncontrolled reference. It retains
dU/dy and replaces both actuator-power terms; STS field 45 stores pressure
variance, following the original 44 moments.

## NACA0012 with US01 and both NES definitions

[naca0012_200k_US01_SS_2586_NES70](naca0012_200k_US01_SS_2586_NES70/README.md)
is a standalone copy of `naca0012_200k_NES70`. It adds suction at
`0.001 Uinf` on the suction side within `0.25 <= x/c <= 0.86`, after the
controller's zero-mean correction, and fixes the action scale to
`uti = 3 * 0.0624 = 0.1872`. STS retains 70 fields, including both Bewley
and Hasegawa costs based on the total applied wall velocity.
