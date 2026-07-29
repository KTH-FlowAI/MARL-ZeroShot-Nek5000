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


#[MOD] Pre-flight checks for the meta configuration.
#[MOD] Everything verified here is otherwise only discovered by MetaPolicy
#[MOD] inside `evaluate`, i.e. AFTER MPI_COMM_WORLD has been split and the
#[MOD] solver ranks are already running: rank 0 dies on an IndexError /
#[MOD] FileNotFoundError while the nek5000 ranks keep waiting, so the job hangs
#[MOD] until the wall clock kills it. Checking in `initial` costs nothing and
#[MOD] fails while the allocation is still empty.
# Algorithms that read a checkpoint from disk; OC/BL are analytic controllers.
_SB3_ALGOS = ('PPO', 'DDPG', 'TD3', 'SAC')
# The per-region lists MetaPolicy._initialize_config indexes with [il].
_PER_REGION_KEYS = ('agent_ctrl_side', 'agent_run_name', 'policy',
                    'RL_algorithm', 'source_solvers', 'u_tau', 'dUdy',
                    'action_bounds', 'drl_steps')


def validate_conf(conf):
    """
    Validate a meta configuration before any folder or file is touched.

    Raises
    ------
    ValueError
        Empty ``case_name``/``agent_ctrl_area``, or a per-region list that is
        shorter than the number of control regions.
    FileNotFoundError
        A policy checkpoint named by the config does not exist.
    """
    runner = conf.runner

    if not str(runner.case_name):
        raise ValueError(
            "[CFG] runner.case_name is empty: it names runs/<case_name> and "
            ".caches/RUN_PATH_<case_name>.txt, so the launcher cannot find the run")

    n_region = len(runner.agent_ctrl_area)
    if n_region == 0:
        raise ValueError("[CFG] runner.agent_ctrl_area is empty: no control region defined")

    too_short = {k: len(runner[k]) for k in _PER_REGION_KEYS if len(runner[k]) < n_region}
    if too_short:
        raise ValueError(
            f"[CFG] {n_region} control regions in runner.agent_ctrl_area but "
            f"shorter per-region list(s) {too_short}: MetaPolicy indexes all of "
            f"them with the region number and would raise IndexError")

    for key in _PER_REGION_KEYS:
        n_key = len(runner[key])
        if n_key > n_region:
            print(f"[CFG][WARN] runner.{key} has {n_key} entries for {n_region} "
                  f"control regions; the last {n_key - n_region} are IGNORED",
                  flush=True)

    # MetaPolicy._load_policy reads <policy_dir>/<agent_run_name>/logs/<policy>.zip,
    # and prefers a copy already sitting in <run_folder>/logs (from a previous run).
    run_folder = os.path.join(conf.logging.save_dir, str(runner.case_name))
    missing = []
    for il in range(n_region):
        if runner.RL_algorithm[il] not in _SB3_ALGOS:
            continue
        source = os.path.join(conf.logging.policy_dir, str(runner.agent_run_name[il]),
                              'logs', f'{runner.policy[il]}.zip')
        local = os.path.join(run_folder, 'logs', f'{runner.policy[il]}.zip')
        if not os.path.isfile(source) and not os.path.isfile(local):
            missing.append(f'region {il} ({runner.RL_algorithm[il]}): {source}')
    if missing:
        raise FileNotFoundError(
            "[CFG] policy checkpoint(s) not found:\n  " + "\n  ".join(missing))

    print(f"[CFG] VALIDATED: {n_region} control region(s), all policies present",
          flush=True)
    return True


def initial(conf_file, overrides, **ignored_kwargs):
    """
    Initialization of the program
    """

    print(f"[IO] METAMARL INITIALIZATION", flush=True)
    print("="*30, flush=True)
    print(f"INITIALIZATION START", flush=True)
    print("="*30, flush=True)

    conf = parse_omegaconf(conf_file, overrides)
    print(f"[DEBUG] CONFIG: {conf}")

    #[MOD] Fail here rather than half-way through the evaluation run.
    validate_conf(conf)

    # Create run folder
    conf.logging.run_name = conf.runner.case_name
    run_folder = conf.logging.save_dir + f'/{conf.logging.run_name}'
    print(f'[IO] RUN Folder=:{run_folder}', flush=True)

    if not os.path.exists(run_folder):
        #[MOD] makedirs so a missing save_dir (fresh clone, or a save_dir with
        #[MOD] several levels) is created too; exist_ok covers the race between
        #[MOD] concurrently launched env ranks.
        os.makedirs(run_folder, exist_ok=True)
        print(f"[IO] MAKE RUN FOLDER:\n{run_folder}", flush=True)

    if not conf.runner.evaluation:
        rank_folder = run_folder
    else:
        rank_folder = run_folder + f'/env_{conf.runner.rank:03d}'
        # make the env folder and copy all the necessary files

    if not os.path.exists(rank_folder):
        os.makedirs(rank_folder, exist_ok=True)
        print(f"[IO] MAKE ENV FOLDER:\n{rank_folder}", flush=True)

    print(f"[IO] Folder: {conf.runner.rank}:\n{rank_folder}", flush=True)

    # Preparation for NEK
    remove_sch(rank_folder)
    #[MOD] embedded + logging are passed so the prepare step can emit the
    #[MOD] .pol networks and drl_policy.in; the meta stack resolves its
    #[MOD] checkpoints through logging.policy_dir.
    initializer = NEK_INIT(nek=conf.simulation, drl=conf.runner,
                           rank_folder=rank_folder,
                           emb=conf.get('embedded', None),
                           log=conf.get('logging', None))
    initializer.main()

    # Re-direct the running path
    #[MOD] RUN_PATH_*.txt now live in <repo_root>/.caches (derived from this
    #[MOD] file's location, so it is independent of the current working dir).
    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".caches")
    os.makedirs(cache_dir, exist_ok=True)
    run_path_file = os.path.join(
        cache_dir, f'RUN_PATH_{conf.runner.case_name}.txt')
    with open(run_path_file, "w") as f:
        f.write(rank_folder + "\n")

    print("="*30, flush=True)
    print(f"[STB3] INITIALIZATION COMPLETE", flush=True)
    print("="*30, flush=True)
