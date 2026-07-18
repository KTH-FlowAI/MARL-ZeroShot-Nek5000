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


def preserve_and_clean_train(run_folder):
    #[MOD] Preserve the training history, then wipe the bulky, throw-away
    #[MOD] simulation data in ./train/ before a (re)start of training.
    """
    Consolidate every history artefact under ``runs/<agent_run_name>/history/``
    so it survives the cleanup of ``./train/``:

        history/current_history   <- the previous run's ./train/history
        history/roundXXX          <- archived rounds (migrated + accumulated)

    Steps (all guarded so a fresh run with no ./train/ is a no-op):
      1. Migrate any legacy ``./train/roundXXX`` -> ``history/roundXXX``.
      2. Promote an existing ``history/current_history`` to the next
         ``history/roundXXX`` (so it is never overwritten), then move
         ``./train/history`` -> ``history/current_history``.
      3. Delete everything left in ``./train/`` (large, unused sim output).

    Parameters
    ----------
    run_folder : str
        The run directory ``runs/<agent_run_name>``.
    """
    import shutil, re

    train_folder = os.path.join(run_folder, "train")
    history_root = os.path.join(run_folder, "history")
    os.makedirs(history_root, exist_ok=True)

    def _next_round_name():
        # Next round index = max existing round in history/ + 1 (zero-padded)
        idx = 0
        for f in os.listdir(history_root):
            m = re.fullmatch(r'round(\d+)', f)
            if m:
                idx = max(idx, int(m.group(1)))
        return f"round{idx + 1:03d}"

    if not os.path.isdir(train_folder):
        print(f"[IO] NO ./train TO CLEAN: {train_folder}", flush=True)
        return

    # 1) Migrate legacy rounds that were archived inside ./train/
    for f in sorted(os.listdir(train_folder)):
        if re.fullmatch(r'round\d+', f):
            src = os.path.join(train_folder, f)
            dst = os.path.join(history_root, f)
            if not os.path.exists(dst):
                shutil.move(src, dst)
                print(f"[IO] MIGRATE ROUND: train/{f} -> history/{f}", flush=True)

    # 2) Archive the previous live history as history/current_history
    prev_hist = os.path.join(train_folder, "history")
    if os.path.isdir(prev_hist):
        cur = os.path.join(history_root, "current_history")
        if os.path.exists(cur):
            promoted = os.path.join(history_root, _next_round_name())
            shutil.move(cur, promoted)
            print(f"[IO] PROMOTE current_history -> history/"
                  f"{os.path.basename(promoted)}", flush=True)
        shutil.move(prev_hist, cur)
        print(f"[IO] ARCHIVE train/history -> history/current_history",
              flush=True)

    # 3) Remove the bulky simulation data that is not used after the fact
    shutil.rmtree(train_folder)
    print(f"[IO] CLEAN UP ./train: {train_folder}", flush=True)


def initial(conf_file,overrides,**ignored_kwargs):
    """
    Initialization of the program
    """

    def get_latest_checkpoint(agent_run_name,log_dir: str,cleanup_buffers: bool = True) -> str:
        """
        Finds the checkpoint file with the highest step number in the log_dir.

        Optionally deletes the replay buffers that do not belong to the latest
        checkpoint, so their (large) storage is released.

        Args:
            agent_run_name (int): The ID of the job
            log_dir (str): Path to the log directory.
            cleanup_buffers (bool): If True, remove every replay buffer whose
                step does not match the latest checkpoint.

        Returns:
            str: Name the latest checkpoint file, or empty string if none found.
        """
        import os
        import re

        # re.escape: agent_run_name is now a string and may contain regex metachars
        esc = re.escape(str(agent_run_name))
        pattern = re.compile(rf'{esc}-(rl_model_(\d+)_steps)\.zip')
        # Replay buffers saved alongside DDPG/TD3 checkpoints (see sb3_utils.py)
        buffer_pattern = re.compile(rf'{esc}-rl_model_replay_buffer_(\d+)_steps\.pkl')
        latest_step = -1
        latest_file = ""
        buffer_files = {}  # step -> filename

        for filename in os.listdir(log_dir):
            match = pattern.match(filename)
            if match:
                step = int(match.group(2))
                if step > latest_step:
                    latest_step = step
                    latest_file = match.group(1)
                continue
            bmatch = buffer_pattern.match(filename)
            if bmatch:
                buffer_files[int(bmatch.group(1))] = filename

        # Free storage: keep only the replay buffer of the latest checkpoint
        if cleanup_buffers and latest_step >= 0:
            for step, filename in buffer_files.items():
                if step != latest_step:
                    buffer_path = os.path.join(log_dir, filename)
                    # Remove only if the file actually exists, so we never
                    # fall into an OSError on a missing/already-removed file.
                    if os.path.isfile(buffer_path):
                        try:
                            os.remove(buffer_path)
                            print(f"[IO] REMOVED OLD REPLAY BUFFER: {filename}",flush=True)
                        except OSError as e:
                            print(f"[IO] FAILED TO REMOVE {filename}: {e}",flush=True)
                    else:
                        print(f"[IO] SKIP (NOT FOUND): {filename}",flush=True)

        # return os.path.join(log_dir, latest_file) if latest_file else ""
        return latest_file if latest_file else ""

    print("="*30,flush=True)
    print(f"INITIALIZATION START",flush=True)
    print("="*30,flush=True)

    conf = parse_omegaconf(conf_file,overrides)
    print(f"[DEBUG] CONFIG: {conf}")

    # Identify if it is a pre-trained model
    if conf.runner.agent_run_name != "":  # "" == fresh run (was != 0)
        conf.logging.run_name = conf.runner.agent_run_name
        conf.runner.load_agent = True
    else:
        conf.runner.rewrite_input_files = True
    

    # Create run folder
    run_folder = conf.logging.save_dir+f'/{conf.logging.run_name}'
    print(f'[IO] RUN Folder=:{run_folder}',flush=True)
    
    if not os.path.exists(run_folder):
        os.mkdir(run_folder)
        print(f"[IO] MAKE RUN FOLDER:\n{run_folder}",flush=True)

    #[MOD] Before a training (re)start, preserve the history under
    #[MOD] runs/<agent_run_name>/history/ and wipe the unused ./train/ data.
    #[MOD] Skipped in evaluation mode (env_XXX folders are handled below).
    if not conf.runner.evaluation:
        preserve_and_clean_train(run_folder)

    if not conf.runner.evaluation:
        rank_folder = run_folder + "/train" # Folder for training
    else:
        #[MOD] Evaluation runs live under a dedicated eval/ subfolder:
        #[MOD] runs/<agent_run_name>/eval/env_XXX
        rank_folder = run_folder+f'/eval/env_{conf.runner.rank:03d}'
        # make the env folder and copy all the necessary files
    if not os.path.exists(rank_folder):
            os.makedirs(rank_folder)  #[MOD] makedirs so the eval/ parent is created too
    
    print(f"[IO] Folder: {rank_folder}",flush=True)            
    
    
    # Preparation for NEK
    #-----------------------------------
    initializer = NEK_INIT(nek=conf.simulation,drl=conf.runner,rank_folder=rank_folder)
    initializer.main()
    #-----------------------------------
    
    # Check the checkpoints 
    if not conf.runner.evaluation:
        if conf.runner.load_agent: 
            ckpt_path = os.path.join(run_folder,'logs')
            if os.path.exists(ckpt_path):
                last_agent = get_latest_checkpoint(conf.runner.agent_run_name,ckpt_path)
            else:
                last_agent = ""
                print(f"[IO] NO CKPT!",flush=True)
            print(f"[IO] The LAST CKPT={last_agent} in {ckpt_path}" ,flush=True)
        else:
            last_agent = ""
            print(f"[IO] NO CKPT!",flush=True)
    else:
        last_agent = ""

    # Re-redict the running path
    #[MOD] RUN_PATH_*.txt now live in <repo_root>/.caches (derived from this
    #[MOD] file's location, so it is independent of the current working dir).
    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".caches")
    os.makedirs(cache_dir, exist_ok=True)
    run_path_file = os.path.join(
        cache_dir, f"RUN_PATH_{conf.runner.agent_run_name}.txt")
    with open(run_path_file, "w") as f:
        f.write(rank_folder + "\n")
        f.write(last_agent)
    f.close()
    
    print("="*30,flush=True)
    print(f"[STB3] INITIALIZATION COMPLETE",flush=True)
    print("="*30,flush=True)

    
