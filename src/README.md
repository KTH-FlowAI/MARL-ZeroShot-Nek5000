# Source code

This folder contains the code to train and evaluate the MARL agents:
- **configs.py**: default configuration schema (`Runner` and `Simulation` dataclasses). Reward
  parameters (`reward_fn`, `reward_alpha/beta/gamma`) live here.
- **run.py** and **evaluate.py**: training and deterministic evaluation loops.
- **nek_marl.py**: PettingZoo environment + MPI communication with NEK5000.
  Implements `_normalize_reward()` for both `dudy` and `net_gain` reward functions.
- **lib/**:
  - `sb3_utils.py` — SB3 model init, callbacks, MPI split
  - `nek_utils.py` — run folder setup, restart file management
  - `reward_logger.py` — real-time CSV logging of per-step rewards and reward components
    (`mean_R_tau`, `mean_R_pw`, `mean_R_v3`) to `history/rewards_aggregated_*.csv`

A script for evaluation is also provided:
- **evaluate-script.sh**: runs several evaluations at once; results saved in `runs/[timestamp]/`.

