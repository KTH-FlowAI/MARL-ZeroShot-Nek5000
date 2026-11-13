"""Runtime reward modes shared by input writers and the channel environment."""

import math


def reward_mode(runner):
    """Map the config to UPARAM(9), validating the pressure RMS reference."""
    modes = {'dudy': 0, 'net_gain': 1, 'wall_pressure': 2}
    name = getattr(runner, 'reward_fn', 'dudy')
    if name not in modes:
        raise ValueError(f"Unknown reward_fn {name!r}; expected {tuple(modes)}")
    if name == 'wall_pressure':
        # [WALL_PRESSURE] No measured baseline is implied by a default value.
        ref = float(getattr(runner, 'pressure_rms_ref', 0.0))
        if not math.isfinite(ref) or ref <= 0.0:
            raise ValueError('wall_pressure requires a finite, positive '
                             'runner.pressure_rms_ref from an uncontrolled run')
    return modes[name]
