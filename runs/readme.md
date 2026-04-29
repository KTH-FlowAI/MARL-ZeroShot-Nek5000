# Production Data

Output folder for all training and evaluation runs. Each subdirectory is named by
`agent_run_name` from the YAML config (e.g. `runs/2001/`).

## Current experiments

| ID   | Config                  | Reward                       | Description |
|------|-------------------------|------------------------------|-------------|
| 2001 | MC16-TD3-ng-val.yml     | NETGAIN α=1 β=0 γ=0          | Validation: pure drag reduction via net-gain code path |
| 2002 | MC16-TD3-ng-full.yml    | NETGAIN α=1 β=1 γ=1          | Full net-gain: drag reduction minus actuator costs |

## Directory layout

```
runs/{id}/train/
  history/
    rewards_aggregated_*.csv   ← per-step reward + R_tau/R_pw/R_v3 components
    rewards_episodes_*.csv     ← per-episode summary
    rewlog_*.npz               ← numpy archives (used by read-history.py)
    tensorboard/               ← SB3 actor/critic loss logs
    current_conf.yml           ← exact config used for this run
  logs/                        ← SB3 model checkpoints (.zip)
  round001/, round002/, ...    ← archived history from previous job submissions
  nek5000                      ← compiled binary (copied by initial step)
```
