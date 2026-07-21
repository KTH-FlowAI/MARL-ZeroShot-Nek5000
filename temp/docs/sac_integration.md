# SAC (Soft Actor-Critic) Integration — Implementation Notes

**Author:** wangyuning
**Branch:** `dev_yw_cost`
**Date:** 2026-07-21
**SB3 version targeted:** 1.7.0 (`stable_baselines3.SAC`)

---

## Motivation

The framework supported `PPO` (on-policy) and `DDPG`/`TD3` (off-policy
deterministic), plus the classical AFC baselines (`OC`, `BL`, `SIN`).
SAC adds a **maximum-entropy off-policy** learner: a stochastic, tanh-squashed
Gaussian actor with twin critics and an automatically tuned temperature.

For this application the interesting properties are:

* **Sample efficiency of TD3** (replay buffer, twin critics) — each transition
  costs a Nek5000 solver step, so on-policy PPO is expensive here.
* **Exploration is intrinsic.** TD3 needs a hand-tuned `action_noise` (currently
  `0.1` in the `MC-*` configs). SAC's entropy term adapts the exploration level
  itself, which removes one of the more sensitive hyper-parameters.
* **No extra tuning burden.** Every SAC-specific option added below defaults to
  `auto`/off, so switching `RL_algorithm: "SAC"` is enough.

---

## Design: SAC is a member of the existing off-policy family

SAC reuses, **unchanged**, the whole machinery already built for DDPG/TD3:

| Shared with DDPG/TD3 | Where |
|---|---|
| replay buffer, `train_freq=(train_steps,"step")`, `gradient_steps` | `init_model()` |
| checkpoint + pickled-replay-buffer restart | `init_model()` restart branch |
| checkpoint callback / `EvalCallback` / logger | `callback_checkpoint()`, `callback_evalenv()` |
| SuperSuit MARL vectorisation (`nAgents` parallel agents) | `init_env()` |
| checkpoint pruning on restart | `initial.py:get_latest_checkpoint()` |

**No environment change was needed.** With `rescale_actions: True`,
`nek_marl.parallel_env` already exposes a symmetric `Box(-1, 1)` action space
(`src/nek_marl.py:124-131`), which is exactly the domain of SAC's tanh squashing;
the physical amplitude rescaling by `ctrl_max_amp` happens inside the env
(`src/nek_marl.py:358-363`) and is algorithm-agnostic.

---

## Files modified

All edits are marked `#[MOD]` in the source.

### 1. `src/configs.py` — new `Runner` fields

```python
#[MOD] ---- only for SAC ----
sac_ent_coef:Any        = 'auto'   # 'auto' | 'auto_0.1' | float
sac_target_entropy:Any  = 'auto'   # 'auto' -> -dim(A); or an explicit float
target_update_interval:int = 1
sac_action_noise:bool   = False
use_sde:bool            = False
sde_sample_freq:int     = -1
```

Everything else is **reused**: `learning_rate`, `batch_size`, `buffer_size`,
`tau`, `gamma`, `learning_starts`, `train_steps`, `gradient_steps`, `seed`,
`policy_file`, `custom_buffer`/`keep_frac`/`buffer_mode`.

Two naming decisions worth recording:

* **`sac_ent_coef`, not the existing `ent_coef`.** `Runner.ent_coef:float = 0.0`
  (line 37) is currently **dead code** — the PPO branch never passes it to SB3.
  It could not be reused anyway: it is typed `float`, and OmegaConf's structured
  merge would reject the string `'auto'`. The new fields are typed `Any` so both
  `'auto'` and a numeric value validate. (Verified: `sac_ent_coef=0.05` and
  `sac_target_entropy=-0.5` both parse.)
* **`sac_target_entropy`, not `target_entropy`,** and note that the unrelated
  reward weight `reward_gamma` already exists — the `sac_` prefix keeps the
  entropy knobs unambiguous.

### 2. `src/lib/sb3_utils.py` — `init_model()` *(the main change)*

The branch condition became a membership test:

```python
elif conf.runner.RL_algorithm in ('DDPG', 'TD3', 'SAC'):
```

Three SAC-specific behaviours inside it:

1. **`action_noise = None` for SAC** unless `runner.sac_action_noise=True`.
   Adding external Gaussian noise on top of an already-stochastic policy fights
   the entropy objective; DDPG/TD3 keep their `NormalActionNoise` exactly as before.
2. **`algo_kwargs`** — the SAC-only constructor arguments, built **once** so the
   fresh-start call and the restart `custom_objects` dict can never drift apart.
   For DDPG/TD3 this dict is `{}`, so **their behaviour is bit-for-bit unchanged**.
   Note this is also where `gamma` is passed for the first time in the off-policy
   branch; it is behaviour-neutral today, since no config overrides the discount
   factor and both `Runner.gamma` and the SB3 default are `0.99`.
3. **`algo_reload_kwargs`** — the subset that may be overridden when resuming.
   `use_sde`/`sde_sample_freq` are **deliberately excluded**: toggling gSDE
   changes the actor architecture, so passing it through `custom_objects` on a
   restart would break parameter loading.

**Entropy state survives a restart.** SB3 stores `log_ent_coef` and
`ent_coef_optimizer` inside the checkpoint zip, so `'auto'` temperature tuning
resumes where it stopped rather than restarting from `exp(0)=1`. Verified
bit-exact in the smoke test (`0.9841901063919067` before save == after load).

### 3. `src/lib/sb3_utils.py` — `callback_checkpoint()` *(pre-existing bug fixed)*

```python
# before
save_replay_buffer=(True if conf.runner.RL_algorithm == 'DDPG' or 'TD3' else False)
# after
is_off_policy = conf.runner.RL_algorithm in ('DDPG', 'TD3', 'SAC')
save_replay_buffer=is_off_policy
```

The old expression parses as `(algo == 'DDPG') or ('TD3')` and a non-empty string
is truthy, so it evaluated to `True` **for every algorithm**, PPO included. It was
harmless in practice (SB3 skips the dump when the model has no `replay_buffer`),
but it had to be correct before SAC relied on it: losing the buffer on restart
throws away transitions that each cost a solver step.

### 4. `src/evaluate.py`

Added the `SAC` import branch. `predict(deterministic=True)` on SAC returns
`tanh(mu)` — the mode of the squashed Gaussian — so evaluation is deterministic,
consistent with how TD3 is evaluated.

### 5. `src/MetaPolicy.py` — `_load_policy()`

`"SAC"` added to the SB3-backed algorithms (both the outer guard and the import
chain), so a meta run (e.g. the wing, one policy per chord region) can mix a
SAC-trained region with TD3/DDPG/PPO ones. The outer condition was also collapsed
from a chained `or` into an `in (...)` test.

### 6. `src/transfer_learning.py` — SAC explicitly **not** supported

```python
if conf.runner.RL_algorithm == 'SAC':
    raise NotImplementedError(...)
```

**Rationale.** `transfer()` warms up in two stages: freeze the actor, train the
critic, then unfreeze. `freeze_actor_critic()` only touches
`model.policy.actor` / `model.policy.critic`. SAC's temperature lives in a
*third* optimizer (`ent_coef_optimizer` over `log_ent_coef`) which would keep
being tuned against a frozen policy — the entropy target drifts during warm-up
and stage 2 restarts from a badly scaled temperature. Rather than silently train
something meaningless, the run fails immediately with a message pointing to
`run.py`. A docstring note was added to `freeze_actor_critic()` too.

Supporting it later means freezing `log_ent_coef.requires_grad` (or zeroing
`ent_coef_optimizer`'s LR) alongside the actor.

### 7. `conf/default_sac_policy.yml` *(new)*

```yaml
net_arch:
    qf: [16,64,64]
    pi: [16,16,8]
log_std_init : -2.3
n_critics : 2
```

Kept separate from `default_ddpg_policy.yml` because (a) that file's `pi: [8,]`
is too small for a stochastic actor that must also carry the `log_std` head, and
(b) `log_std_init`/`n_critics` are accepted by `SACPolicy` but **not** by DDPG's
policy. `log_std_init: -2.3` (std ≈ 0.1 on the normalized `[-1,1]` action)
mirrors the PPO setup and keeps the policy away from bang-bang control early on.

---

## How to run

Only the algorithm block of an existing config changes. Starting from
`conf/MC-ng-111.yml`:

```yaml
runner:
    RL_algorithm    : "SAC"                        # was "TD3"
    policy_file     : 'conf/default_sac_policy.yml'  # was default_ddpg_policy.yml

    # unchanged, shared with TD3:
    rescale_actions : True      # REQUIRED: gives SAC a symmetric [-1,1] action space
    learning_rate   : 1e-3
    batch_size      : 256
    buffer_size     : 5_000_000
    learning_starts : 100
    tau             : 0.005
    train_steps     : 300
    gradient_steps  : 64

    # SAC-specific: all optional, these ARE the defaults
    # sac_ent_coef           : 'auto'
    # sac_target_entropy     : 'auto'   # -> -1.0, since dim(A)=1
    # target_update_interval : 1
    # sac_action_noise       : False
    # use_sde                : False
```

`action_noise` and `policy_delay` are simply ignored under SAC.

Launch exactly as before (`execs/sjob-train*.sh` → `src/run.py`); evaluation via
`src/evaluate.py` with `learnt_policy: True`.

---

## Verification performed

A smoke test drove `lib.sb3_utils.init_model()` with a dummy vec-env whose spaces
match the real ones (`obs Box(2,1,1) float32`, `act Box(-1,1,(1,))`):

| Check | Result |
|---|---|
| SAC built via `init_model` with `default_sac_policy.yml` | OK — `n_critics=2`, actor `[16,16,8]` |
| `target_entropy` resolved from `'auto'` | `-1.0` (= −dim(A)) |
| `action_noise` suppressed for SAC | `None` |
| `train_freq` / `gradient_steps` plumbed through | `TrainFreq(10,'step')` / `4` |
| `model.learn()` runs, temperature updates | OK |
| save → reload via the restart branch | OK, `ent_coef` restored bit-exact |
| pickled replay buffer restored (real `*-rl_model_replay_buffer_N_steps.pkl` name) | OK, 40 transitions |
| `predict(deterministic=True)` | OK, action inside `Box(-1,1)` |
| **TD3 regression** — noise + gamma unchanged | OK, `NormalActionNoise(sigma=0.00638)` |
| `save_replay_buffer` flag: SAC/TD3 → True, PPO → False | OK (bug fix confirmed) |
| `transfer()` rejects SAC | OK, `NotImplementedError` |
| All 59 `conf/**.yml` still parse against the new `Runner` | OK (the 6 wing/meta files that "fail" use `configs_meta.Config`, unrelated and pre-existing) |

**Not tested:** an actual coupled Nek5000 run. The MPI/solver path is untouched
by this change, but a short `MC-*` run is still the sensible next step before
committing to a long training.

---

## Open points / follow-ups

1. **`learning_starts: 100`** is inherited from the TD3 configs. SAC's entropy
   term is most useful when the buffer already has some diversity; if early
   training looks unstable, this is the first knob to raise — not the entropy
   settings.
2. **`gradient_steps: 64` with `train_steps: 300`** is the TD3 ratio. SAC updates
   two critics plus the temperature per gradient step, so wall-clock per training
   phase will be somewhat higher (still negligible against the solver cost).
3. **Transfer learning** — see §6 for what a proper SAC implementation needs.
4. `lib/replay_buffer.py` (`SelectiveReplayBuffer`) does **not** exist in this
   branch; `custom_buffer` must stay `False` for SAC just as for TD3, otherwise
   `init_model` passes `replay_buffer_class=None` to SB3. Pre-existing, unchanged.
