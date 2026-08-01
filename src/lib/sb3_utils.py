"""
Collection of SB3 utilities
Sep 23, 2025 
@yuningw 
"""

from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common import env_checker
# MPI4PY
from mpi4py import MPI
# Tensorboard
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.logger import configure
try:
    from lib.replay_buffer import SubsampleOnInsertBuffer, SelectiveReplayBuffer
except ModuleNotFoundError:
    SubsampleOnInsertBuffer = None
    SelectiveReplayBuffer = None
import yaml
import glob
import os
import shutil
import pickle
import numpy as np
import nek_marl_cluster
import nek_marl
import supersuit as ss


def mpi_split(comm_world):
    mpi_rank = comm_world.Get_rank()
    mpi_size = comm_world.Get_size()
    if mpi_size < 2:
        raise RuntimeError("Requires at least 2 processes (1 Master + 1 Worker)")
    # Step 1: Split communicators
    if mpi_rank == 0:
        color = 0  # Master
    else:
        color = 1  # Workers
    
    local_comm = comm_world.Split(color, mpi_rank)
    print(f"[PY] SPLIT the color comm!",flush=True)
    sub_comm = local_comm.Create_intercomm(local_leader=0, peer_comm=comm_world, 
                                            remote_leader=1, tag=99)
    return sub_comm


def duplicate_comm(comm):
    """
    Create an independent MPI communicator handle.

    This is useful when two environment instances need to talk to the same
    solver/worker group without sharing communicator state or lifecycle.
    """
    if comm == MPI.COMM_NULL:
        return MPI.COMM_NULL
    return comm.Dup()

def io_path(conf):
    # Identify if it is a pre-trained model
    if conf.runner.agent_run_name != "":  # "" == fresh run (was != 0)
        conf.logging.run_name = conf.runner.agent_run_name
        # conf.runner.load_agent = True
    else:
        conf.runner.rewrite_input_files = True
    # Create run folder
    run_folder = conf.logging.save_dir+f'/{conf.logging.run_name}'
    if not os.path.exists(run_folder):
        if conf.runner.agent_run_name != "":  # "" == fresh run (was != 0)
            raise ValueError("The folder containing the trained agent "+\
                            "does not exist")
        os.mkdir(run_folder)
        print(f"[IO] MAKE RUN FOLDER:\n{run_folder}",flush=True)
    return run_folder


def archive_eval_checkpoint(conf, ckpt_path, dst_dir=None):
    """
    Make a copy of the evaluated checkpoint and rename it as
    ``eval_model_{agent_run_name}`` whenever an evaluation job is launched.

    This keeps a stable, run-named snapshot of exactly which checkpoint was
    evaluated, next to the original logs so it is not lost when new
    checkpoints overwrite the generic policy file names.

    Parameters
    ----------
    conf : OmegaConf
        Parsed configuration; ``conf.runner.agent_run_name`` names the copy.
    ckpt_path : str
        Path to the evaluated checkpoint as passed to ``RL_algorithm.load``.
        SB3 stores checkpoints as ``.zip``; the extension is optional here.
    dst_dir : str, optional
        Directory where the copy is written. Defaults to the directory of
        ``ckpt_path``.

    Returns
    -------
    str or None
        Path to the copied checkpoint, or ``None`` if the source is missing.
    """
    # SB3 checkpoints are stored as .zip; normalise the source path
    src = ckpt_path if ckpt_path.endswith('.zip') else ckpt_path + '.zip'
    if not os.path.exists(src):
        print(f"[EVAL] WARN: checkpoint not found, skip copy:\n{src}",
              flush=True)
        return None

    if dst_dir is None:
        dst_dir = os.path.dirname(src)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

    # Name the copy after the run folder, which is what the results tree and
    # every post-processing script key on.  io_path() sets logging.run_name
    # from runner.agent_run_name, so the fallback is only for hand-built confs.
    run_name = getattr(getattr(conf, 'logging', None), 'run_name', None) \
        or conf.runner.agent_run_name
    dst = os.path.join(dst_dir, f"eval_model_{run_name}.zip")
    shutil.copy2(src, dst)
    print(f"[EVAL] COPIED EVALUATED CHECKPOINT:\n{src}\n-> {dst}", flush=True)

    # Drop a copy in every evaluation environment as well.  The env folders are
    # what gets archived and post-processed, and a run whose checkpoint is only
    # in logs/ becomes untraceable as soon as new checkpoints overwrite it.
    for env_dir in sorted(glob.glob(os.path.join(
            conf.logging.save_dir, str(run_name), 'eval', 'env_[0-9][0-9][0-9]'))):
        env_dst = os.path.join(env_dir, os.path.basename(dst))
        if not os.path.exists(env_dst):
            shutil.copy2(dst, env_dst)
            print(f"[EVAL] tracked checkpoint in {env_dir}", flush=True)
    return dst


def init_env(conf, run_folder, sub_comm):
    """
    Initialize the environment
    """
    env = nek_marl.parallel_env(conf=conf, rank_folder=run_folder,sub_comm=sub_comm)
    #env = nek_marl_cluster.parallel_env(conf=conf, rank_folder=run_folder,sub_comm=sub_comm)
    nAgents = env.nAgents
    action_space_sample = env.action_space(env.possible_agents[0])
    observation_space_sample = env.observation_space(env.possible_agents[0])
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 
                                num_vec_envs = 1, 
                                num_cpus=0, 
                                base_class="stable_baselines3")
    print(f"[STB3] RAW ENV INIT, AGENT={nAgents}",flush=True)
    return env, nAgents, action_space_sample, observation_space_sample

def init_model(conf, env, run_folder,nAgents, 
          action_space_sample,observation_space_sample,device):
    # Definition of the agent
    if conf.runner.custom_policy:
        with open(conf.runner.policy_file) as file:
            policy_kwargs = yaml.load(file, Loader=yaml.FullLoader)
        print(f"[STB3] POLICY KWARGS,{policy_kwargs}",flush=True)
    else:
        policy_kwargs = {}

    if conf.runner.seed != -5000:
        seed_ = conf.runner.seed 
    else: 
        seed_ = None 
    if conf.runner.RL_algorithm=='PPO':
        from stable_baselines3 import PPO as RLA
        
        
        if conf.runner.load_agent == False:
            model = RLA('MlpPolicy', env, verbose=3,
                                    gamma=conf.runner.gamma,
                                    target_kl=conf.runner.target_kl,
                                    clip_range=conf.runner.likhood_clipping,
                                    policy_kwargs=policy_kwargs,
                                    learning_rate=conf.runner.learning_rate,
                                    n_steps=conf.runner.train_steps,
                                    batch_size=conf.runner.batch_size,
                                    n_epochs=conf.runner.n_epochs,
                                    seed = seed_,
                                    # use_sde=True,
                                    # sde_sample_freq=nAgents,
                                    device=device,
                                    )
            print('[IO] TRAIN FROM SCRATCH',flush=True)
        else: 
            print('[IO] RESTART TRAINING',flush=True)
            
            # [MOD] Prefix removed for consistency with MetaPolicy._load_policy:
            # `policy` is now the FULL checkpoint file name (no agent_run_name prefix).
            ckpt_file = f"{run_folder}/logs/{conf.runner.policy}"

            model= RLA.load(path=ckpt_file,
                    env=env,
                    custom_objects={'action_space':action_space_sample,
                                    "observation_spce":observation_space_sample
                                    },
                    # device=device
                    )
            print('[IO] RESTART W&B LOADED'+ckpt_file,flush=True)
            
    #[MOD] SAC joins the off-policy family: it reuses the very same replay
    #[MOD] buffer / train_freq / gradient_steps / checkpoint-restart machinery
    #[MOD] as DDPG and TD3, and only differs by its entropy-related kwargs.
    elif conf.runner.RL_algorithm in ('DDPG', 'TD3', 'SAC'):
        if conf.runner.RL_algorithm=='DDPG':
            from stable_baselines3 import DDPG as RLA
        elif conf.runner.RL_algorithm=='SAC':   #[MOD]
            from stable_baselines3 import SAC as RLA
        else:
            from stable_baselines3 import TD3 as RLA
        from stable_baselines3.common.noise import NormalActionNoise

        # n_actions = env.action_space.shape[-1]
        n_actions = action_space_sample.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                        sigma=conf.runner.action_noise*np.ones(n_actions))

        #[MOD] SAC explores through its own tanh-squashed Gaussian policy, so the
        #[MOD] external Gaussian noise used by DDPG/TD3 is disabled by default
        #[MOD] (opt back in with runner.sac_action_noise=True).
        if conf.runner.RL_algorithm == 'SAC' and not conf.runner.sac_action_noise:
            action_noise = None
            print("[STB3] SAC: NO EXTERNAL ACTION NOISE (entropy-driven exploration)",
                  flush=True)

        if conf.runner.gradient_steps == 0:
            conf.runner.gradient_steps = nAgents*\
                                        conf.runner.train_steps

        #[MOD] Algorithm-specific kwargs, defined once so that the fresh-start
        #[MOD] call and the restart (custom_objects) path can never drift apart.
        #[MOD] Empty for DDPG/TD3 => their behaviour is bit-for-bit unchanged.
        if conf.runner.RL_algorithm == 'SAC':
            algo_kwargs = dict(
                gamma=conf.runner.gamma,
                ent_coef=conf.runner.sac_ent_coef,
                target_entropy=conf.runner.sac_target_entropy,
                target_update_interval=conf.runner.target_update_interval,
                use_sde=conf.runner.use_sde,
                sde_sample_freq=conf.runner.sde_sample_freq,
            )
            print(f"[STB3] SAC KWARGS: {algo_kwargs}", flush=True)
        else:
            algo_kwargs = {}

        #[MOD] Subset that may be overridden when resuming from a checkpoint.
        #[MOD] use_sde/sde_sample_freq are excluded on purpose: toggling gSDE
        #[MOD] changes the actor architecture and would break parameter loading.
        algo_reload_kwargs = {k: v for k, v in algo_kwargs.items()
                              if k not in ('use_sde', 'sde_sample_freq')}

        if conf.runner.load_agent == False:
            model = RLA('MlpPolicy', env, 
                                verbose=1,
                                learning_starts=conf.runner.learning_starts,
                                tau=conf.runner.tau,
                                learning_rate=conf.runner.learning_rate,
                                buffer_size=conf.runner.buffer_size,
                                batch_size=conf.runner.batch_size,
                                policy_kwargs=policy_kwargs,
                                action_noise=action_noise,
                                train_freq=(conf.runner.train_steps, "step"),
                                gradient_steps=conf.runner.gradient_steps,
                                seed=seed_,
                                device=device,
                                **algo_kwargs,   #[MOD] SAC-only extras (empty for DDPG/TD3)
                                **({'replay_buffer_class': SelectiveReplayBuffer,
                                    'replay_buffer_kwargs': {'keep_frac':conf.runner.keep_frac,'mode':conf.runner.buffer_mode,}}
                                   if getattr(conf.runner, 'custom_buffer', True) else {}),
                                )

            print('[IO] TRAIN FROM SCRATCH',flush=True)
        
        else: 
            print('[IO] RESTART TRAINING',flush=True)
            
            ckpt_file = f"{run_folder}/logs/"

            # [MOD] Prefix removed for consistency with MetaPolicy._load_policy:
            # `policy` is now the FULL checkpoint file name (no agent_run_name prefix).
            ckpt_file += f"{conf.runner.policy}"
            
            model= RLA.load(path=ckpt_file,
                    env=env,
                    custom_objects={'action_space':action_space_sample,
                                    'learning_rate':conf.runner.learning_rate,
                                    'batch_size':conf.runner.batch_size,
                                    'buffer_size':conf.runner.buffer_size,
                                    'gradient_steps':conf.runner.gradient_steps,
                                    'tau':conf.runner.tau,
                                    'learning_starts':conf.runner.learning_starts,
                                    'train_freq':(conf.runner.train_steps, "step"),
                                    'action_noise':action_noise,
                                    'seed':seed_,
                                    #[MOD] SAC-only extras (empty for DDPG/TD3).
                                    #[MOD] NOTE: the saved log_ent_coef and its
                                    #[MOD] optimizer are restored from the zip, so
                                    #[MOD] 'auto' entropy tuning resumes where it
                                    #[MOD] stopped instead of restarting from scratch.
                                    **algo_reload_kwargs,
                    },
                    print_arguments=True,
                    )
        
            print(f'[IO] RESTART W&B LOADED:{ckpt_file}',flush=True)
            
            
            # [MOD] Prefix removed for consistency: `policy` (hence buffer_name)
            # is now the FULL name, so no agent_run_name prefix is added here
            # (otherwise the buffer name would be doubly prefixed).
            buffer_name=conf.runner.policy.replace('rl_model_','rl_model_replay_buffer_')
            buffer_file = f"{run_folder}/logs/{buffer_name}.pkl"
            print(f"[IO] Buffer File:{buffer_file}",flush=True)
            is_exist = os.path.exists(buffer_file)
            if is_exist:
                with open(buffer_file,'rb') as f:
                    replay_buffer = pickle.load(f)

                model.replay_buffer = replay_buffer
                print(f'[IO] REPLAY BUFFER LOADED! {buffer_file}',flush=True)
            else:
                print('[IO] NO REPLAY BUFFER!',flush=True)
        
    return model


def init_logger(run_folder,env,model):
    log_dir = run_folder+'/history/tensorboard'
    os.makedirs(log_dir, exist_ok=True)
    # Monitor the environment
    env=VecMonitor(env, os.path.join(log_dir, 'env.csv'))
    # Configure the logger
    logger = configure(log_dir, ["tensorboard", "csv"])
    model.set_logger(logger)
    return 


def callback_checkpoint(conf, run_folder):
    callbacks = []
    #[MOD] Explicit membership test. The previous expression
    #[MOD] `(True if conf.runner.RL_algorithm == 'DDPG' or 'TD3' else False)`
    #[MOD] was ALWAYS True (the bare string 'TD3' is truthy), so on-policy runs
    #[MOD] also requested a replay-buffer dump. SAC is off-policy: its buffer
    #[MOD] must be checkpointed, otherwise a restart throws away every collected
    #[MOD] transition -- expensive here, since each one costs a solver step.
    is_off_policy = conf.runner.RL_algorithm in ('DDPG', 'TD3', 'SAC')
    checkpoint_callback = CheckpointCallback(
                                save_freq=conf.runner.nb_interactions*conf.runner.ckpt_int,
                                save_path=run_folder+'/logs/',
                                name_prefix=f'{conf.logging.run_name}-rl_model',
                                save_replay_buffer=is_off_policy,
                                save_vecnormalize=is_off_policy,
                                )
    callbacks.append(checkpoint_callback)
    return callbacks

def get_base_env(env):
    """
    Traverse SuperSuit + SB3 wrapper chain to reach the base env.
    Handles: concat_vec_envs_v1, pettingzoo_env_to_vec_env_v1, SB3 VecEnvWrapper.
    """
    while True:
        if hasattr(env, 'vec_envs'):      # SuperSuit ConcatVecEnv
            env = env.vec_envs[0]
        elif hasattr(env, 'par_env'):     # SuperSuit MarkovVectorEnv
            env = env.par_env
        elif hasattr(env, 'venv'):        # SB3 VecEnvWrapper
            env = env.venv
        elif hasattr(env, 'envs'):        # SB3 DummyVecEnv
            env = env.envs[0]
        elif hasattr(env, 'env'):         # Standard Gym wrapper
            env = env.env
        else:
            break                         # reached the base
    return env

def callback_evalenv(conf, env, nAgents, eval_freq):
    """
    Evaluation callback for stable-baselines3
    conf: configuration object
    eval_freq: frequency of evaluation (in timesteps)
    """ 
    from stable_baselines3.common.callbacks import EvalCallback
    from copy import deepcopy
    import os

    # class NekEvalCallback(EvalCallback):
    #     """EvalCallback that switches the shared Nek env into eval mode."""
    #     def _on_step(self) -> bool:
    #         # Switch to deterministic eval mode
    #         if hasattr(self.eval_env, 'eval_mode'):
    #             print(f"[STB3] SWITCH ENV TO EVAL MODE",flush=True)
    #             self.eval_env.eval_mode = True
    #             result = super()._on_step()   # runs evaluate_policy internally
    #             # Switch back to training mode
    #             self.eval_env.eval_mode = False
    #             return result
    #         else:
    #             print(f"[STB3] WARNING: EvalEnv does not have 'eval_mode' attribute. EvalCallback will run without switching modes.", flush=True)

    class NekEvalCallback(EvalCallback):
        def _on_step(self) -> bool:
            base_env = get_base_env(self.eval_env)
            if hasattr(base_env, 'eval_mode'):
                print(f"[SB3] SWITCH ENV TO EVAL MODE — base: {type(base_env).__name__}", flush=True)
                base_env.eval_mode = True
                result = super()._on_step()
                base_env.eval_mode = False
                return result
            else:
                print(f"[SB3] WARNING: {type(base_env).__name__} has no 'eval_mode'. "
                    f"Running eval without mode switch.", flush=True)
                return super()._on_step()
    rank_folder = io_path(conf)
    eval_callback = NekEvalCallback(
        eval_env=env,
        eval_freq=eval_freq*conf.runner.nb_interactions, # Evaluate every eval_freq episodes, in terms of timesteps, it is eval_freq*nb_interactions
        n_eval_episodes=nAgents*1, # Evaluate each agent for 1 episode, total n_eval_episodes = nAgents*1
        best_model_save_path=rank_folder+"/logs/",
        log_path=rank_folder+"/logs/",
        deterministic=True,
    )
    evaluate_npz = os.path.join(rank_folder,'logs/evaluations.npz')
    if os.path.exists(evaluate_npz):
        print(f"[STB3] Resumed EvalCallback with history from {evaluate_npz}",flush=True)
        eval_callback = resume_eval_callback(eval_callback, 
                                            npz_path=evaluate_npz)
    return eval_callback


def resume_eval_callback(callback, npz_path: str):
    """
    Inject prior evaluation history into an EvalCallback before resuming training.

    This handles the fact that EvalCallback._init_callback() resets internal
    evaluation lists at the start of learn(), so history must be re-injected
    after that reset via a patched _init_callback.

    Args:
        callback:  The EvalCallback instance (not yet passed to learn()).
        npz_path:  Path to the previous run's evaluations.npz file.

    Returns:
        The same callback, modified in-place.
    """
    from stable_baselines3.common.callbacks import EvalCallback
    import numpy as np
    import os
    old_eval = np.load(npz_path)

    old_timesteps = old_eval["timesteps"]          # shape: (n_evals,)
    old_results   = old_eval["results"]            # shape: (n_evals, n_episodes)
    old_lengths   = old_eval["ep_lengths"]         # shape: (n_evals, n_episodes)

    old_mean_rewards = old_results.mean(axis=1)

    # --- 1. best_mean_reward: set now, it is NOT reset by _init_callback ---
    callback.best_mean_reward = float(old_mean_rewards.max())
    callback.last_mean_reward = float(old_mean_rewards[-1])

    # --- 2. Stash history so we can inject it AFTER _init_callback resets lists ---
    callback._resume_timesteps = old_timesteps.tolist()
    callback._resume_results   = old_results.tolist()
    callback._resume_lengths   = old_lengths.tolist()

    # --- 3. Patch _init_callback to inject history after the reset ---
    _original_init = callback._init_callback

    def _patched_init():
        _original_init()                                          # runs the reset
        callback.evaluations_timesteps = list(callback._resume_timesteps)
        callback.evaluations_results   = list(callback._resume_results)
        callback.evaluations_length    = list(callback._resume_lengths)

    callback._init_callback = _patched_init

    print(f"[resume_eval_callback] Loaded {len(old_timesteps)} prior evaluations.")
    print(f"  best_mean_reward  : {callback.best_mean_reward:.4f}")
    print(f"  last_mean_reward  : {callback.last_mean_reward:.4f}")
    print(f"  last timestep     : {old_timesteps[-1]}")

    return callback
#--------------------------------
# Transfer Learning
#--------------------------------

def freeze_actor_critic(conf,model,net='actor'):
    """
    Freeze the model for warm-up

    #[MOD] NOT valid for SAC: freezing the actor here leaves SAC's entropy
    #[MOD] coefficient (log_ent_coef, its own optimizer) still being tuned
    #[MOD] against a frozen policy. transfer_learning.transfer() rejects SAC
    #[MOD] up front for that reason.
    """

    if net == 'actor':
        # --- Freeze actor for warm-up ---
        for p in model.policy.actor.parameters():
            p.requires_grad = False
        # In SB3 TD3, actor/critic have separate optimizers; set actor LR to 0 (if accessible), or simply rely on requires_grad=False
        model.policy.actor.optimizer.param_groups[0]['lr'] = 0.0
        print(f"[STB3] FREEZE ACTOR",flush=True)
    elif net == 'critic':
    # --- Freeze critic for warm-up ---
        for p in model.policy.critic.parameters():
            p.requires_grad = False
        model.policy.critic.optimizer.param_groups[0]['lr'] = 0.0
        print(f"[STB3] FREEZE CRITIC",flush=True)
    else:
        raise ValueError(f"Invalid network: {net}")

    return model

def unfreeze_actor_critic(conf,model,net='actor'):
    """
    Unfreeze the model for training
    """
    if net == 'actor':
        for p in model.policy.actor.parameters():
            p.requires_grad = True
        model.policy.actor.optimizer.param_groups[0]['lr'] = conf.runner.learning_rate
        print(f"[STB3] UNFREEZE ACTOR",flush=True)
    elif net == 'critic':
        for p in model.policy.critic.parameters():
            p.requires_grad = True
        model.policy.critic.optimizer.param_groups[0]['lr'] = conf.runner.learning_rate
        print(f"[STB3] UNFREEZE CRITIC",flush=True)
    else:
        raise ValueError(f"Invalid network: {net}")

    return model    
