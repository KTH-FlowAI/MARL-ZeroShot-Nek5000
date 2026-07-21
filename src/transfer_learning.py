"""
Transfer Learning
Sep 23, 2025 
@yuningw 
"""
# The dependencies
from cgi import print_arguments
import os, subprocess
os.environ['TQDM_DISABLE'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]=""
# Reading config 
from omegaconf import OmegaConf 
import yaml 
import numpy as np 
import argparse 
from pathlib import Path 
from typing import List
import pickle

import torch as th 
from configs import Config
import supersuit as ss
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn as  parallel_to_aec 
# import nek_marl_elem as nek_marl
import nek_marl
from lib.nek_utils import *
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
from lib.sb3_utils import *


device = ('cpu' if not th.cuda.is_available() else "cuda")
print(f"[SYS] DEVICES: {device}")

# Accepting the argument by argparse 
def add_subparser(parser: argparse.ArgumentParser):
    subparser = parser.add_parser("transfer", help="Transfer Learning")
    subparser.add_argument("conf_file", type=Path, help="YAML configuration")
    subparser.add_argument(
        "overrides",
        type=str,
        nargs="*",
        help="Config overrides, e.g. `other.gpus=4`",
    )
    subparser.set_defaults(cmd=transfer)

def parse_omegaconf(conf_file: str, overrides: List[str]):
    conf = OmegaConf.merge(
        OmegaConf.structured(Config()),
        OmegaConf.load(conf_file),
        OmegaConf.from_dotlist(overrides),
    )
    return conf


def transfer(conf_file, overrides, **ignored_kwargs):
    """
    Main subroutine for transfer learning
    """
    print(f"[SYS] TRANSFER LEARNING",flush=True)
    ##### Obtain the MPI #########
    #==================================
    comm_world = MPI.COMM_WORLD
    sub_comm = mpi_split(comm_world)
    #==================================
    
    #--------------------------------
    # Parse the configuration
    #--------------------------------
    conf = parse_omegaconf(conf_file,overrides)
    print(f"[DEBUG] CONFIG: {conf}")

    #[MOD] SAC is deliberately NOT supported for transfer learning yet.
    #[MOD] The warm-up below freezes the actor while the critic catches up, but
    #[MOD] SAC's entropy coefficient keeps being tuned against that frozen actor
    #[MOD] (the log_ent_coef optimizer is independent of actor/critic), so the
    #[MOD] temperature drifts and the unfrozen stage restarts from a badly scaled
    #[MOD] entropy target. Fail early instead of training something meaningless.
    if conf.runner.RL_algorithm == 'SAC':
        raise NotImplementedError(
            "[STB3] SAC is not supported for transfer learning: the actor-freeze "
            "warm-up does not freeze the entropy coefficient. Use TD3/DDPG here, "
            "or train SAC from scratch via `run` (src/run.py).")

    #--------------------------------
    # Create the run folder
    #--------------------------------
    run_folder = io_path(conf)
    #--------------------------------
    env, nAgents, act_sp, obs_sp = init_env(conf, run_folder, sub_comm)
    show_title()
    
    #--------------------------------
    # Initialize the model
    #--------------------------------
    model = init_model(conf, env, run_folder, nAgents, 
                    act_sp,obs_sp,device,)
    print(f"[STB3] MODEL INIT",flush=True)
   
    #--------------------------------
    # Definition of the learning callbacks
    callbacks = callback_checkpoint(conf, run_folder)
    ##--------------------------------
    # Configure the logger
    init_logger(run_folder,env,model)
    ##--------------------------------

    #--------------------------------
    # Start Training the model
    # Actual training
    ### Stage 1: Freeze the actor to warm up the critic
    model = freeze_actor_critic(conf,model,'actor')
    print(model.policy.actor.optimizer)
    print(model.policy.critic.optimizer)


    base_steps = nAgents* conf.runner.nb_interactions
    #--------------------------------
    model.learn(total_timesteps=conf.runner.nb_warmup_episodes*base_steps,
                callback=callbacks,
                log_interval=1,
                reset_num_timesteps=False,
                )
    #--------------------------------
    model = unfreeze_actor_critic(conf,model,'actor')
    #--------------------------------
    
    ### Stage 2: Unfreeze the actor to train the model
    #--------------------------------
    model.learn(total_timesteps=conf.runner.nb_episodes*base_steps,
                callback=callbacks,
                log_interval=1,
                reset_num_timesteps=True,
                )
    #--------------------------------


    print(f"[STB3] LEARNING FINISH!")
    model.env.close()
    print(f"[STB3] CLOSE ENV!")
    save_final_model= run_folder+'/logs/'+f"{conf.runner.agent_run_name}"+"-rl_model_final_steps"
    model.save(save_final_model)
    print(f"[STB3] SAVE POLICY!\n{save_final_model}")
    show_end()
