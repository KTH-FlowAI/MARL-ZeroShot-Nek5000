"""
Using the system excution for running the programme
Oct 4, 2024 
@yuningw 
"""
# The dependencies
import os
os.environ['TQDM_DISABLE'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]=""
# Reading config 
from omegaconf import OmegaConf 
import argparse 
from pathlib import Path 
from typing import List
import torch as th 
from configs import Config
from lib.nek_utils import *
from mpi4py import MPI
from lib.sb3_utils import *
from stable_baselines3.common.callbacks import CallbackList


device = ('cpu' if not th.cuda.is_available() else "cuda")
print(f"[SYS] DEVICES: {device}")

# Accepting the argument by argparse 
def add_subparser(parser: argparse.ArgumentParser):
    subparser = parser.add_parser("run", help="Run NEK")
    subparser.add_argument("conf_file", type=Path, help="YAML configuration")
    subparser.add_argument(
        "overrides",
        type=str,
        nargs="*",
        help="Config overrides, e.g. `other.gpus=4`",
    )
    subparser.set_defaults(cmd=run)

def parse_omegaconf(conf_file: str, overrides: List[str]):
    conf = OmegaConf.merge(
        OmegaConf.structured(Config()),
        OmegaConf.load(conf_file),
        OmegaConf.from_dotlist(overrides),
    )
    return conf


def run(conf_file, overrides, **ignored_kwargs):
    
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

    #--------------------------------
    # Create the run folder
    #--------------------------------
    rank_folder = io_path(conf)
    run_folder = os.path.join(rank_folder,'train')
    if not os.path.exists(run_folder): 
        os.mkdir(run_folder)
        print(f"Make Folder: {run_folder}",flush=True)
    #--------------------------------
    env, nAgents, act_sp, obs_sp = init_env(conf, run_folder, sub_comm)
    show_title()

    #--------------------------------
    # Initialize the model
    #--------------------------------
    model = init_model(conf, env, rank_folder, nAgents, 
                    act_sp,obs_sp,device,)
    print(f"[STB3] MODEL INIT",flush=True)
    
    #--------------------------------
    # Definition of the learning callbacks
    callbacks = callback_checkpoint(conf, rank_folder)
    if conf.runner.if_eval:
        callbacks.append(callback_evalenv(conf, env, nAgents, conf.runner.eval_freq))
        print(f"[STB3] EVAL CALLBACK",flush=True)
    callbacks = CallbackList(callbacks) # YW: Combine the callbacks into a CallbackList
    ##--------------------------------
    # Configure the logger
    init_logger(run_folder,env,model)
    ##--------------------------------

    #--------------------------------
    # Start Training the model
    # Actual training
    base_steps = nAgents* conf.runner.nb_interactions
    #--------------------------------
    model.learn(total_timesteps=conf.runner.nb_episodes*base_steps,
                callback=callbacks,
                log_interval=nAgents,
                reset_num_timesteps=False,
                )

    print(f"[STB3] LEARNING FINISH!")
    model.env.close()
    print(f"[STB3] CLOSE ENV!")
    save_final_model= rank_folder+'/logs/'+f"{conf.runner.agent_run_name}"+"-rl_model_final_steps"
    model.save(save_final_model)
    print(f"[STB3] SAVE POLICY!\n{save_final_model}")
    show_end()
