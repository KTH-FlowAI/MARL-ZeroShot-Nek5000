"""
Script for evaluating the learned policy in deterministic
@yuningw
"""
import os, subprocess,time
os.environ['TQDM_DISABLE'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]=""
# Reading config 
from omegaconf import OmegaConf 
import yaml 
import numpy as np 
import scipy.io as sio
import argparse 
from pathlib import Path 
from typing import List

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
from lib.AFC import * 
from mpi4py import MPI
from lib.sb3_utils import *

def add_subparser(parser: argparse.ArgumentParser):
    subparser = parser.add_parser("evaluate", help="Run NEK5000")
    subparser.add_argument("conf_file", type=Path, help="YAML configuration")
    subparser.add_argument(
        "overrides",
        type=str,
        nargs="*",
        help="Config overrides, e.g. `other.gpus=4`",
    )
    subparser.set_defaults(cmd=evaluate)

def parse_omegaconf(conf_file: str, overrides: List[str]):
    conf = OmegaConf.merge(
        OmegaConf.structured(Config()),
        OmegaConf.load(conf_file),
        OmegaConf.from_dotlist(overrides),
    )
    return conf

def solver_dep_obs(conf, obs):
    """
    Adjust observations based on solver-specific requirements
    """
    # This is required because the obs from env is flipped in y direction 
    # due to the way we read the data from Fortran. 
    if conf.runner.source_solver == 'dedalus':
        for agent in obs.keys():# Flip the observation order for Dedalus
            obs[agent] = np.flip(obs[agent], axis=0)  
        return obs_flipped
        print(f"[DEDALUS] FLIPPED OBSERVATIONS {obs_flipped}",flush=True)
    else:
        return obs

def evaluate(conf_file,overrides,**ignored_kwargs):
    """
    Main subroutine for evaluation
    """
    conf = parse_omegaconf(conf_file,overrides)
    print(f"[DEBUG] CONFIG: {conf}")

    ##### Obtain the MPI #########
    #==================================
    comm_world = MPI.COMM_WORLD
    sub_comm = mpi_split(comm_world)
    #==================================
    conf = parse_omegaconf(conf_file,overrides)
    print(f"[DEBUG] CONFIG: {conf}")

    #--------------------------------
    # Create the run folder
    #--------------------------------
    run_folder = io_path(conf)
    #--------------------------------
    
    #--------------------------------
    # Initialize the ENV 
    #--------------------------------
    env = nek_marl.parallel_env(conf=conf, rank_folder=run_folder,sub_comm=sub_comm)
    nAgents = env.nAgents
    agents_list = env.possible_agents  
    ## Create a dictionary for collecting the data. 
    agent_dict  = { agent : {"obs_rec":[],
                            "act_rec":[],
                            "rew_rec":[],
                            } for agent in agents_list }
    #---------------------------------------------
    
    # It is possible to evaluate a trained policy or a functional policy
    if conf.runner.learnt_policy==True:
        # Importing the required RL algorithm
        if conf.runner.RL_algorithm=='PPO':
            from stable_baselines3 import PPO as RL_algorithm
        elif conf.runner.RL_algorithm=='DDPG':
            from stable_baselines3 import DDPG as RL_algorithm
        elif conf.runner.RL_algorithm=='TD3':
            from stable_baselines3 import TD3 as RL_algorithm

        # Definition of the agent
        if conf.runner.custom_policy:
            with open(conf.runner.policy_file) as file:
                policy_kwargs = yaml.load(file, Loader=yaml.FullLoader)
        else:
            policy_kwargs = {}
        
        # Load model from path
        # custom_objects is required because the action_space
        # is not correctly deserialized when loading from file 
        ckpt_path=f"{run_folder}/logs/"
        if 'best' in conf.runner.policy: 
            ckpt_path+=f"{conf.runner.policy}"
        else:
            ckpt_path+=f"{conf.runner.agent_run_name}-"+\
               f"{conf.runner.policy}"
            
        loaded_model = RL_algorithm.load(ckpt_path,
            custom_objects={'action_space':env.action_space(env.possible_agents[0]),
                            "observation_space":env.observation_space(env.possible_agents[0]),
                            },
            print_system_info=False,)

    ## Classical AFC 
    else:
        if conf.runner.RL_algorithm == 'OC':
            from lib.AFC import OppoCtrl as RL_algorithm 
            loaded_model = RL_algorithm(agent_list=agents_list,
                                    ctrl_max_amp=conf.runner.ctrl_max_amp,
                                    )
            
        elif conf.runner.RL_algorithm == 'BL':
            from lib.AFC import BLCtrl as RL_algorithm
            loaded_model = RL_algorithm(agent_list=agents_list,
                                    ctrl_max_amp=conf.runner.ctrl_max_amp,
                                    )
        elif conf.runner.RL_algorithm == 'SIN':
            from lib.AFC import SinWave as RL_algorithm
            loaded_model = RL_algorithm(agent_list=agents_list,
                                    ctrl_max_amp=conf.runner.ctrl_max_amp,
                                    Kx = 1, Kz=1,Lx=conf.simulation.Lx,Lz=conf.simulation.Lz
                                    )
            loaded_model.load_node_info(env.agent_info)
        else: 
            raise NotImplementedError("[ERROR] Please use a Trained/Known Policy!")

    # Vectorizing the environment
    if conf.runner.learnt_policy:
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, 1, num_cpus=0, base_class="stable_baselines3")
        episode_starts = np.ones((env.num_envs,), dtype=bool)
    else:
        episode_starts = np.ones((1,), dtype=bool)
    # Vectorizing the environment

    ## Update the run folder
    run_folder = conf.logging.save_dir+f'/{conf.logging.run_name}'+f'/env_{conf.runner.rank:03d}'

    ### Start Main Evaluations ###
    show_title()
    # Evaluate policy
    states = None
    observations = env.reset()
    # Evaluate policy
    # Main Loop 

    for i in range(conf.runner.nb_interactions-1):
        # print(f"[DEBUG] OBSERVATIONS {observations}")
        observations = solver_dep_obs(conf, observations)
        # Agent Actutaion
        actions, states = loaded_model.predict(observations, state=states, 
                episode_start=episode_starts, deterministic=True)
        
        # print(f"[DEBUG] ACTIONS {actions}")
    
        # States from Env, here observations are redistributed 
        observations, rewards, dones, infos = env.step(actions)
        ## IF Write record
        if conf.runner.vars_record:
            if (i % conf.runner.vars_record_freq) ==0:
                if not conf.runner.learnt_policy:
                    for agent in agent_dict.keys():
                        agent_dict[agent]['obs_rec'].append((observations[agent]).reshape(1,-1))
                        agent_dict[agent]['act_rec'].append((actions[agent]).reshape(1,-1))
                        agent_dict[agent]['rew_rec'].append((rewards[agent]).reshape(1,-1))
                else:
                    for il, agent in enumerate(agent_dict.keys()):
                        agent_dict[agent]['obs_rec'].append((observations[il]).reshape(1,-1))
                        agent_dict[agent]['act_rec'].append((actions[il]).reshape(1,-1))
                        agent_dict[agent]['rew_rec'].append((rewards[il]).reshape(1,-1))

                print(f"[STB3] AT {i}/{conf.runner.nb_interactions} SAVE Trajectories",flush=True)    
    
    # Finish the Loop
    env.close()
    print(f"[ENV] DONE: EVLUATION",flush=True)
    
    # Save the observation recorded
    if conf.runner.vars_record:
        for agent in agent_dict.keys():
            for items in agent_dict[agent].keys():
                agent_dict[agent][items] = np.concatenate(agent_dict[agent][items])
                #print(f"[IO] AGENT {agent}: the {items}={agent_dict[agent][items].shape}")

    sio.savemat(run_folder +f'/vars_record_{int(time.time())}.mat',
                agent_dict,)
    print(f"[IO] SAVED RECORD",flush=True)
    show_end()
