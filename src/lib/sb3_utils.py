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
import os
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
    if conf.runner.agent_run_name != 0:
        conf.logging.run_name = conf.runner.agent_run_name
        # conf.runner.load_agent = True
    else:
        conf.runner.rewrite_input_files = True
    # Create run folder
    run_folder = conf.logging.save_dir+f'/{conf.logging.run_name}'
    if not os.path.exists(run_folder):
        if conf.runner.agent_run_name != 0:
            raise ValueError("The folder containing the trained agent "+\
                            "does not exist")
        os.mkdir(run_folder)
        print(f"[IO] MAKE RUN FOLDER:\n{run_folder}",flush=True)
    return run_folder


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
            
            ckpt_file = f"{run_folder}/"+\
                f"logs/{conf.runner.agent_run_name}-"+\
                f"{conf.runner.policy}"
            
            model= RLA.load(path=ckpt_file,
                    env=env,
                    custom_objects={'action_space':action_space_sample,
                                    "observation_spce":observation_space_sample
                                    },
                    # device=device
                    )
            print('[IO] RESTART W&B LOADED'+ckpt_file,flush=True)
            
    elif conf.runner.RL_algorithm=='DDPG' or conf.runner.RL_algorithm=='TD3':
        if conf.runner.RL_algorithm=='DDPG':
            from stable_baselines3 import DDPG as RLA
        else:
            from stable_baselines3 import TD3 as RLA
        from stable_baselines3.common.noise import NormalActionNoise
        
        # n_actions = env.action_space.shape[-1]
        n_actions = action_space_sample.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), 
                                        sigma=conf.runner.action_noise*np.ones(n_actions))
        
        if conf.runner.gradient_steps == 0:
            conf.runner.gradient_steps = nAgents*\
                                        conf.runner.train_steps   

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
                                **({'replay_buffer_class': SelectiveReplayBuffer,
                                    'replay_buffer_kwargs': {'keep_frac':conf.runner.keep_frac,'mode':conf.runner.buffer_mode,}}
                                   if getattr(conf.runner, 'custom_buffer', True) else {}),
                                )

            print('[IO] TRAIN FROM SCRATCH',flush=True)
        
        else: 
            print('[IO] RESTART TRAINING',flush=True)
            
            ckpt_file = f"{run_folder}/logs/"

            # [YW] If the policy name already contains 'best_model' (e.g., from EvalCallback), use it directly; 
            if 'best_model' in conf.runner.policy:
                ckpt_file += f"{conf.runner.policy}"
            # otherwise, construct the checkpoint filename based on agent_run_name and policy
            else:
                ckpt_file +=f"{conf.runner.agent_run_name}-"+\
                            f"{conf.runner.policy}"
            
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
                    },
                    print_arguments=True,
                    )
        
            print(f'[IO] RESTART W&B LOADED:{ckpt_file}',flush=True)
            
            
            buffer_name=conf.runner.policy.replace('rl_model_','rl_model_replay_buffer_')
            buffer_file = f"{run_folder}/"+\
                f"logs/{conf.runner.agent_run_name}-"+\
                f"{buffer_name}" + ".pkl"
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
    checkpoint_callback = CheckpointCallback(
                                save_freq=conf.runner.nb_interactions*conf.runner.ckpt_int, 
                                save_path=run_folder+'/logs/',
                                name_prefix=f'{conf.logging.run_name}-rl_model',
                                save_replay_buffer=(True if conf.runner.RL_algorithm == 'DDPG' or 'TD3' else False),
                                save_vecnormalize=(True if conf.runner.RL_algorithm == 'DDPG' or 'TD3' else False),
                                )
    callbacks.append(checkpoint_callback)
    return callbacks

def callback_evalenv(conf, env, nAgents, eval_freq):
    """
    Evaluation callback for stable-baselines3
    conf: configuration object
    eval_freq: frequency of evaluation (in timesteps)
    """ 
    from stable_baselines3.common.callbacks import EvalCallback
    from copy import deepcopy
    import os

    class ClosingEvalCallback(EvalCallback):
        """
        EvalCallback that closes its evaluation environment explicitly.

        The evaluation env owns a duplicated MPI communicator, so closing it
        is what releases the communicator handle.
        """

        def _on_training_end(self) -> None:
            if hasattr(self, "eval_env") and self.eval_env is not None:
                self.eval_env.close()
            super()._on_training_end()

    eval_conf = deepcopy(conf)
    eval_conf.runner.random_init = -1 # No shuffle for evaluation
    rank_folder = io_path(eval_conf)
    eval_folder = os.path.join(rank_folder,'eval')
    eval_callback = EvalCallback(
        eval_env=env,
        eval_freq=eval_freq*conf.runner.nb_interactions, # Evaluate every eval_freq episodes, in terms of timesteps, it is eval_freq*nb_interactions
        n_eval_episodes=nAgents*1, # Evaluate each agent for 1 episode, total n_eval_episodes = nAgents*1
        best_model_save_path=rank_folder+"/logs/",
        log_path=rank_folder+"/logs/",
    )
    evaluate_npz = os.path.join(rank_folder,'eval/evaluations.npz')
    if os.path.exists(evaluate_npz):
        print(f"[STB3] Resumed EvalCallback with history from {evaluate_npz}",flush=True)
        eval_callback = resume_eval_callback(eval_callback, 
                                            npz_path=evaluate_npz))
    return eval_callback


def resume_eval_callback(callback: EvalCallback, npz_path: str) -> EvalCallback:
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
