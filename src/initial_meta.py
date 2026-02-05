"""
Script for initialization (meta)
@yuningw
"""
import os, subprocess, time
os.environ['TQDM_DISABLE'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Reading config
from omegaconf import OmegaConf
import yaml
import numpy as np
import scipy.io as sio
import argparse
from pathlib import Path
from typing import List

from configs_meta import Config
import supersuit as ss
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn as parallel_to_aec
import nek_marl_meta as nek_marl
from lib.nek_utils import *
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common import env_checker
from lib.AFC import *


def add_subparser(parser: argparse.ArgumentParser):
    subparser = parser.add_parser("initial", help="Run SIMSON")
    subparser.add_argument("conf_file", type=Path, help="YAML configuration")
    subparser.add_argument(
        "overrides",
        type=str,
        nargs="*",
        help="Config overrides, e.g. `other.gpus=4`",
    )
    subparser.set_defaults(cmd=initial)


def parse_omegaconf(conf_file: str, overrides: List[str]):
    conf = OmegaConf.merge(
        OmegaConf.structured(Config()),
        OmegaConf.load(conf_file),
        OmegaConf.from_dotlist(overrides),
    )
    return conf


def initial(conf_file, overrides, **ignored_kwargs):
    """
    Initialization of the program
    """

    print("="*30, flush=True)
    print(f"INITIALIZATION START", flush=True)
    print("="*30, flush=True)

    conf = parse_omegaconf(conf_file, overrides)
    print(f"[DEBUG] CONFIG: {conf}")

    # Create run folder
    conf.logging.run_name = conf.runner.case_name
    run_folder = conf.logging.save_dir + f'/{conf.logging.run_name}'
    print(f'[IO] RUN Folder=:{run_folder}', flush=True)

    if not os.path.exists(run_folder):
        if conf.runner.case_name != 0:
            print("The folder containing the trained agent " +
                  "does not exist", flush=True)
            os.mkdir(run_folder)
        print(f"[IO] MAKE RUN FOLDER:\n{run_folder}", flush=True)

    if not conf.runner.evaluation:
        rank_folder = run_folder
    else:
        rank_folder = run_folder + f'/env_{conf.runner.rank:03d}'
        # make the env folder and copy all the necessary files
        if not os.path.exists(rank_folder):
            os.mkdir(rank_folder)

    print(f"[IO] Folder: {conf.runner.rank}:\n{rank_folder}", flush=True)

    # Preparation for NEK
    remove_sch(rank_folder)
    initializer = NEK_INIT(nek=conf.simulation, drl=conf.runner, rank_folder=rank_folder)
    initializer.main()

    # Re-direct the running path
    with open('RUN_PATH.txt', "w") as f:
        f.write(rank_folder + "\n")

    print("="*30, flush=True)
    print(f"[STB3] INITIALIZATION COMPLETE", flush=True)
    print("="*30, flush=True)
