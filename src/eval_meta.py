"""
Script for evaluating the learned policy in deterministic (meta)
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
from lib.AFC import *
from MetaPolicy import MetaPolicyRunner
from mpi4py import MPI


def add_subparser(parser: argparse.ArgumentParser):
    subparser = parser.add_parser("evaluate", help="Run SIMSON")
    subparser.add_argument("conf_file", type=Path, help="YAML configuration")
    subparser.add_argument(
        "overrides",
        type=str,
        nargs="*",
        help="Config overrides, e.g. `other.gpus=4`",
    )
    subparser.set_defaults(cmd=meta_eval)


def parse_omegaconf(conf_file: str, overrides: List[str]):
    conf = OmegaConf.merge(
        OmegaConf.structured(Config()),
        OmegaConf.load(conf_file),
        OmegaConf.from_dotlist(overrides),
    )
    return conf


def meta_eval(conf_file, overrides, **ignored_kwargs):
    """
    Main subroutine for evaluation
    """
    conf = parse_omegaconf(conf_file, overrides)
    print(f"[DEBUG] CONFIG: {conf}")

    # Obtain the MPI
    comm_world = MPI.COMM_WORLD
    mpi_rank = comm_world.Get_rank()
    mpi_size = comm_world.Get_size()
    if mpi_size < 2:
        raise RuntimeError("Requires at least 2 processes (1 Master + 1 Worker)")

    # Split communicators
    if mpi_rank == 0:
        color = 0  # Master
    else:
        color = 1  # Workers

    local_comm = comm_world.Split(color, mpi_rank)
    print(f"[PY] SPLIT the color comm!", flush=True)
    sub_comm = local_comm.Create_intercomm(local_leader=0, peer_comm=MPI.COMM_WORLD,
                                           remote_leader=1, tag=99)

    # Set configuration
    conf = parse_omegaconf(conf_file, overrides)

    # Create run folder
    conf.logging.run_name = conf.runner.case_name
    run_folder = conf.logging.save_dir + f'/{conf.logging.run_name}'

    if not os.path.exists(run_folder):
        if conf.runner.agent_run_name != 0:
            raise ValueError("The folder containing the trained agent " +
                             "does not exist")
        os.mkdir(run_folder)
        print(f"[IO] MAKE RUN FOLDER:\n{run_folder}", flush=True)

    # Initialize the ENV
    env = nek_marl.parallel_env(conf=conf, rank_folder=run_folder, sub_comm=sub_comm)
    # Setup the Policy Runner to coordinate the polices
    metaRunner = MetaPolicyRunner(conf=conf, env=env, run_folder=run_folder)

    # Start Main Evaluations
    show_title()
    metaRunner.run()
    show_end()
