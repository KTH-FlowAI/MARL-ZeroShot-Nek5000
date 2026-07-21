from omegaconf import OmegaConf
from dataclasses import dataclass
from configs_meta import Config
import yaml
import numpy as np
import scipy.io as sio
import argparse
from copy import deepcopy
from pathlib import Path
from typing import List
import os, shutil, filecmp
import supersuit as ss


class MetaPolicyRunner():
  def __init__(self, conf: Config, env, run_folder) -> None:
    """
    Class for managing multiple polices governing various control area.
    """

    self.conf = conf
    print(f"[Meta] GET Overall Config", flush=True)
    self._initialize_env(env)
    print(f"[Meta] Environment Handled", flush=True)
    self.policy_dict = {}
    self.run_folder = run_folder
    self._initialize_config()


  # --------------------------------------------
  def _initialize_config(self):
    """
    Assigning the policy dictionary for
    """

    # Get the overall control region
    self.ctrl_areas = np.array(self.conf.runner.agent_ctrl_area)
    self.ref_num_agents = len(self.Agents_List)
    # Initialize
    self.io_iter = 0

    actual_num_agents = 0
    # Groupping the
    for il, ctrl_region in enumerate(self.conf.runner.agent_ctrl_area):
      _name = f"CTRL{il:03d}"
      self.policy_dict[_name] = {}

      # Define the control range
      x_min, x_max = ctrl_region
      self.policy_dict[_name]["x_min"] = x_min
      self.policy_dict[_name]["x_max"] = x_max
      # Define the control side
      self.policy_dict[_name]["side"] = self.conf.runner.agent_ctrl_side[il]

      # Distribute the agents to the current policy
      self.policy_dict[_name] = self._distribute_agents(self.policy_dict[_name])

      # Define the agent name
      self.policy_dict[_name]['rL_algorithm'] = self.conf.runner.RL_algorithm[il]
      self.policy_dict[_name]['agent_run_name'] = self.conf.runner.agent_run_name[il]
      self.policy_dict[_name]['policy'] = self.conf.runner.policy[il]
      #[YW-MOD] The source solver for transfer learning
      self.policy_dict[_name]['source_solver'] = self.conf.runner.source_solvers[il]

      # Update time interval
      self.policy_dict[_name]['drl_step'] = self.conf.runner.drl_steps[il]
      # A step counter, can be useful for calculating the averaged rewards
      self.policy_dict[_name]['i_step'] = 0

      # Scaling factor
      self.policy_dict[_name]['u_tau'] = self.conf.runner.u_tau[il]
      self.policy_dict[_name]['baseline_dUdy'] = self.conf.runner.dUdy[il]
      ctrl_min_amp, ctrl_max_amp = self.conf.runner.action_bounds[il]
      self.policy_dict[_name]['ctrl_min_amp'] = ctrl_min_amp
      self.policy_dict[_name]['ctrl_max_amp'] = ctrl_max_amp

      # Reading Models
      case_dict = self._load_policy(case_dict=self.policy_dict[_name],
                                    policy_folder=os.path.join(self.conf.logging.policy_dir,
                                                              self.policy_dict[_name]['agent_run_name']))
      # Case dict is updated with the loaded model and rescale factors, we assign it back to the policy dict
      self.policy_dict[_name] = case_dict

      # Indication of start
      self.policy_dict[_name]['episode_starts'] = np.ones((self.policy_dict[_name]['num_env'],), dtype=bool)

      actual_num_agents += len(self.policy_dict[_name]['agent_idx'])
      print(f"[META] POLICY {il} : {_name} SET UP!")

    # Sanity Check of Assignment
    # assert actual_num_agents == self.ref_num_agents, print(f"No.Agents NOT MATCH! {actual_num_agents} v.s {self.ref_num_agents}")
    return

  # -------------------------------------------------
  def _initialize_env(self, env):
    """
    Initialize the Nek Environment
    """
    self.CTRL_MAP = env.agent_info
    print(f"[Meta] GET CTRL MAP", flush=True)

    self.Agents_List = np.array(env.possible_agents, dtype=np.string_)
    self.Agents_List_str = env.possible_agents
    print(f"[Meta] GET Agent List", flush=True)

    # [YW] We assume the observation space and action space are the same across agents
    agent0 = env.possible_agents[0]
    self.obs_space = env.observation_space(agent0)
    self.act_space = env.action_space(agent0)

    # Vectorizing the environment
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=0, base_class="stable_baselines3")
    self.env = env
    print(f"[Meta] Env Vectorized!", flush=True)

    return

  # -------------------------------------------------
  def _initialize_buffer(self):
    """
    Initialize the Nek Environment
    """
    # Create a dictionary for collecting the data.
    self.agent_buffer = {agent: {"obs_rec": [], "act_rec": [], "rew_rec": []} for agent in self.Agents_List_str}
    print(f"[Meta] BUFFER INIT!", flush=True)
    return

  # -------------------------------------------------
  def _save_buffer(self):
    """
    Initialize the Nek Environment
    """
    # Create a dictionary for collecting the data.
    for agent in self.agent_buffer.keys():
      for items in self.agent_buffer[agent].keys():
        self.agent_buffer[agent][items] = np.concatenate(self.agent_buffer[agent][items])
    sio.savemat(self.run_folder+f'/vars_record_{self.conf.runner.case_name}_{self.io_iter:05d}.mat',
                self.agent_buffer,)
    print(f"[IO] SAVED BUFFER IOSTEP={self.io_iter}", flush=True)
    return

  # --------------------------------------------
  def _distribute_agents(self, case_dict):
    """
    Distribute the agents, assigning to the current policy for MARL
    """
    x_max, x_min = case_dict['x_max'], case_dict['x_min']
    side = case_dict['side']
    print(f"[Meta] Current CTRL REGION: {x_min} -- {x_max}, Side: {side}", flush=True)
    glob_max, glob_min = np.max(self.CTRL_MAP['x']), np.min(self.CTRL_MAP['x'])

    # Safety Check
    if (glob_max <= x_max):
      x_max = glob_max
      print(f"[Meta] Adapt to a valid value: MAX {glob_max}:{x_max}")
    if (glob_min >= x_min):
      x_min = glob_min
      print(f"[Meta] Adapt to a valid value: MIN {glob_min}:{x_min}")

    # Get the agent indices based on the control region and side
    if case_dict['side'] == 'SS': # Suction side, y > 0:
      agent_idx = np.where((self.CTRL_MAP['x'] >= x_min) & (self.CTRL_MAP['x'] <= x_max) & (self.CTRL_MAP['y'] > 0))[0]
    elif case_dict['side'] == 'PS': # Pressure side, y < 0:
      agent_idx = np.where((self.CTRL_MAP['x'] >= x_min) & (self.CTRL_MAP['x'] <= x_max) & (self.CTRL_MAP['y'] < 0))[0]
    else:
      raise ValueError(f"[Meta] Invalid control side: {case_dict['side']}")

    case_dict['x_min'] = x_min
    case_dict['x_max'] = x_max
    case_dict['agent_idx'] = agent_idx
    case_dict['num_env'] = len(agent_idx)
    case_dict['agent_name'] = self.Agents_List[case_dict['agent_idx']]
    print(f"[Meta] Loc IDX Obtained: Range = {case_dict['x_min']:.2e} ~ {case_dict['x_max']:.2e}, Assigned Agents = {case_dict['num_env']}", flush=True)
    return case_dict

  # -------------------------------------------
  def run(self):
    """
    Main Programme for meta-policy exploitation
    """

    # Initialize the observations
    observations = self.env.reset()
    states = None
    self._initialize_buffer()
    self.io_iter = 0
    self.iter = 0

    # We start counting from 1, so that il%1=0
    for il in range(1, self.conf.runner.nb_interactions):
      self.iter = il

      # Examine which policy requires update
      if_updates = [True if (il == 1) or ((il % self.policy_dict[k]['drl_step']) == 0) else False
                    for k in self.policy_dict.keys()]

      # Actions will be update based on this list:
      actions, states = self.predict(observations,
                                     state=states,
                                     if_updates=if_updates,
                                     deterministic=True)
      # Get Observation and Reward buffer
      observations, rewards, dones, infos = self.env.step(actions)

      # Process the rewards
      rewards = self.local_reward(rewards)

      # IF Write record
      if self.conf.runner.vars_record:
        if (il % self.conf.runner.vars_record_freq) == 0:
          for jl, agent in enumerate(self.agent_buffer.keys()):
            self.agent_buffer[agent]['obs_rec'].append((observations[jl]).reshape(1, -1))
            self.agent_buffer[agent]['act_rec'].append((actions[jl]).reshape(1, -1))
            self.agent_buffer[agent]['rew_rec'].append((rewards[jl]).reshape(1, -1))
        print(f"[STB3] AT {il}/{self.conf.runner.nb_interactions} Collecting Trajectories", flush=True)

        if (il % self.conf.runner.vars_io_freq) == 0:
          self._save_buffer()
          self._initialize_buffer()

        self.io_iter += 1

    self._save_buffer()
    print(f"[IO] SAVED Final RECORD", flush=True)

    # Finish the Loop
    self.env.close()
    print(f"[ENV] DONE: EVLUATION", flush=True)

  # --------------------------------------------
  def predict(self, observation: np.ndarray, state,
              if_updates: list,
              deterministic=True
              ):
    """
    Model Inference based on assigned observation
    """

    # Create an empty buffer for action
    actions = np.zeros(shape=(self.ref_num_agents, self.conf.runner.ctrl_array_size,))

    # We traverse all of the policy, and let the policy update according to the flag in `if_update`.
    for il, case in enumerate(self.policy_dict.keys()):
      partial_act, state = self._partial_predict(case, observation, state, if_update=if_updates[il])
      actions[self.policy_dict[case]['agent_idx'], :] = partial_act[:, :]
    return actions, state

  # --------------------------------------------
  def _partial_predict(self, case_name: str, observation, state,
                      if_update: bool, deterministic=True):
    """
    Apply a partial update for a single policy
    """

    self.policy_dict[case_name]['i_step'] += 1
    if if_update:
      # If requires update, we renew the buffer
      # Get the observation, 
      # [YW] rearrange it based on the source solver, and normalize it based on the local u_tau
      partial_obs = self._obs_solver_arrange(observation[self.policy_dict[case_name]['agent_idx'], :, :, :], 
                                              self.policy_dict[case_name]['source_solver'])

      partial_obs = self._normalize_state(partial_obs, self.policy_dict[case_name]['u_tau'])

      # React to the partial observation
      partial_act, state = self.policy_dict[case_name]['loaded_model'].predict(partial_obs,
                                                                               state=state,
                                                                               episode_start=self.policy_dict[case_name]['episode_starts'],
                                                                               deterministic=deterministic,
                                                                               )
      if isinstance(partial_act, dict):
        partial_act = self._dict_actions_to_array(case_name, partial_act)
      # Rescale the action
      partial_act = self._rescale_actions(partial_act,
                                          self.policy_dict[case_name]['rescale_factors'])

      self.policy_dict[case_name]['partial_act'] = partial_act
      # update the counter
      c_s = self.policy_dict[case_name]['i_step']
      d_s = self.policy_dict[case_name]['drl_step']
      print(f"[Meta] Step={self.iter}: {case_name}: [{c_s}/{d_s}] Update Action", flush=True)

    else:
      # If there is no need for update, we use the action buffer
      partial_act = deepcopy(self.policy_dict[case_name]['partial_act'])

    return partial_act, state

  # --------------------------------------------
  def _dict_actions_to_array(self, case_name: str, actions: dict) -> np.ndarray:
    """
    Convert dict actions keyed by agent name into ordered numpy array.
    """
    agent_names = self.policy_dict[case_name]['agent_name']
    ctrl_array_size = self.conf.runner.ctrl_array_size
    action_array = np.zeros((len(agent_names), ctrl_array_size))

    for i, agent in enumerate(agent_names):
      if agent not in actions:
        raise KeyError(f"[Meta] Missing action for agent: {agent}")
      act = np.asarray(actions[agent]).reshape(-1)
      if act.size == 1:
        action_array[i, :] = act[0]
      elif act.size == ctrl_array_size:
        action_array[i, :] = act
      else:
        raise ValueError(
            f"[Meta] Action size mismatch for agent {agent}: "
            f"got {act.size}, expected {ctrl_array_size}"
        )
    return action_array

  # --------------------------------------------
  def local_reward(self, raw_rewards: np.ndarray):
    """
    Localizing the Rewards
    """

    rewards = np.zeros_like(raw_rewards)

    # Formatting the Logger to print
    text = "\n" + "-" * 40 + "\n"
    text += "|\t\t Policy Rewards \t\t|\n"
    text += "-" * 40 + "\n"

    for il, case in enumerate(self.policy_dict.keys()):

      # Extract the reward buffer
      partial_rew = raw_rewards[self.policy_dict[case]['agent_idx']]

      # Query the current local DRL step
      c_step = self.policy_dict[case]['i_step']
      d_step = self.policy_dict[case]['drl_step']
      # Taking the moving average here
      if c_step == 1:
        self.policy_dict[case]['partial_rew'] = partial_rew
      else:
        old_partial_rew = self.policy_dict[case]['partial_rew']
        self.policy_dict[case]['partial_rew'] = (old_partial_rew + partial_rew*c_step) / (c_step+1)

      # Update the counter:
      if c_step == self.policy_dict[case]['drl_step']:
        self.policy_dict[case]['i_step'] = 0

      # Scale the Rewards by local baseline dUdy
      partial_rew = np.ones_like(partial_rew) * self._scale_reward(self.policy_dict[case]["partial_rew"],
                                                                   self.policy_dict[case]['baseline_dUdy'])
      # Assemble the buffer
      rewards[self.policy_dict[case]['agent_idx']] = partial_rew

      # Write Logger
      r = np.unique(partial_rew).squeeze()
      text += f"| Step={self.iter} [{c_step}/{d_step}] | {case:<16} | R= {r:>7.3f} |\n"

    text += "-" * 40 + "\n"
    print(text, flush=True)

    return rewards

  # --------------------------------------------
  def _load_policy(self,case_dict, policy_folder):
    """
    Load W&B Based on the Algorithm Type
    Args:
      case_dict:[dict] Dictionary contains the policy information
      policy_folder:[str]
    Return:
      case_dict   : case_dict updated with Object of policy and rescale factors
    """
    log_path = os.path.join(self.run_folder,"logs")
    if os.path.exists(log_path) == False:
      os.makedirs(log_path,exist_ok=True)

    #[MOD] "SAC" added to the SB3-backed policies, so a meta run (e.g. the wing)
    #[MOD] can mix a SAC-trained region with TD3/DDPG/PPO ones.
    if case_dict["rL_algorithm"] in ("PPO", "DDPG", "TD3", "SAC"):
      if case_dict["rL_algorithm"] == 'PPO':
        from stable_baselines3 import PPO as RL_algorithm
      elif case_dict["rL_algorithm"] == 'DDPG':
        from stable_baselines3 import DDPG as RL_algorithm
      elif case_dict["rL_algorithm"] == 'TD3':
        from stable_baselines3 import TD3 as RL_algorithm
      elif case_dict["rL_algorithm"] == 'SAC':   #[MOD]
        from stable_baselines3 import SAC as RL_algorithm

      # [MOD] Prefix Removed: 
      # Just specifying the policy file path, and let the load function handle the rest
      policy_file = f"{policy_folder}/" + \
                        f"logs/{case_dict['policy']}.zip"

      # [MOD] If the identical policy file already exists in the current
      # agent_run_name logs folder (from a previous copy), load it from there
      # instead of the external policy_folder. This avoids depending on / re-
      # copying the source (and the SameFileError if source == target).
      # Verify the local copy matches the source; warn on any discrepancy
      # (source differs from the local copy, or source is missing).
      local_policy_file = os.path.join(log_path, os.path.basename(policy_file))
      if os.path.exists(local_policy_file):
        load_path = local_policy_file
        already_local = True
        if not os.path.exists(policy_file):
          print(f"[SB3][WARN] Source policy NOT found: {policy_file}; "
                f"loading local copy (unverified): {local_policy_file}",flush=True)
        elif not filecmp.cmp(local_policy_file, policy_file, shallow=False):
          print(f"[SB3][WARN] Local policy DIFFERS from source!\n"
                f"           local : {local_policy_file}\n"
                f"           source: {policy_file}\n"
                f"           Loading the local copy anyway.",flush=True)
        else:
          print(f"[SB3] Local policy matches source, loading from {load_path}",flush=True)
      else:
        load_path = policy_file
        already_local = False

      try:
        loaded_model = RL_algorithm.load(load_path,
                                       # custom_objects is required because the action_space
                                       custom_objects={'action_space':self.act_space,
                                        "observation_space": self.obs_space,}
                                       )
        # Now make a copy to the current logs (only when loaded externally):
        if not already_local:
          shutil.copy(src=policy_file,dst=log_path)
          print(f"[SB3] Copy Policy to {log_path}",flush=True)

      except:
        raise FileNotFoundError("[SB3] ERROR: Target Model NOT Found!")

      is_low_equal = (loaded_model.action_space.low[0] == case_dict['ctrl_min_amp'])
      is_high_equal = (loaded_model.action_space.high[0] == case_dict['ctrl_max_amp'])
      print(f"[STB3] Action Space: Lower bound: {loaded_model.action_space.low[0]},  {case_dict['ctrl_min_amp']}")
      print(f"[STB3] Action Space: Upper bound: {loaded_model.action_space.high[0]}, {case_dict['ctrl_max_amp']}")

      if (case_dict['rL_algorithm'] == 'DDPG' or case_dict['rL_algorithm'] == 'TD3'):
        if (is_low_equal == False) or ((is_high_equal == False)):
          print(f'[STB3] Rescaling Actions', flush=True)
          rescale_factors = [
              case_dict['ctrl_min_amp'] / loaded_model.action_space.low[0],
              case_dict['ctrl_max_amp'] / loaded_model.action_space.high[0],
          ]
          print(f"[STB3] Recale Factors {rescale_factors}", flush=True)
        else:
          rescale_factors = [1, 1]
      elif (case_dict['rL_algorithm'] == 'PPO'):
        rescale_factors = [
            case_dict['ctrl_min_amp'],
            case_dict['ctrl_max_amp'],
        ]
        print(f"[STB3] Recale Factors {rescale_factors}", flush=True)

    elif (case_dict["rL_algorithm"] == "OC") or (case_dict["rL_algorithm"] == "BL"):
      if case_dict["rL_algorithm"] == 'OC':
        from lib.AFC import OppoCtrl as RL_algorithm
        loaded_model = RL_algorithm(agent_list=case_dict["agent_name"],
                                    ctrl_max_amp=case_dict['ctrl_max_amp'],
                                    )
        rescale_factors = [
            case_dict['ctrl_min_amp'],
            case_dict['ctrl_max_amp'],
        ]
      elif case_dict["rL_algorithm"] == 'BL':
        from lib.AFC import BLCtrl as RL_algorithm
        loaded_model = RL_algorithm(agent_list=case_dict["agent_name"],
                                    ctrl_max_amp=case_dict['ctrl_max_amp'],
                                    )
        rescale_factors = [
            case_dict['ctrl_min_amp'],
            case_dict['ctrl_max_amp'],
        ]
    else:
      raise NotImplementedError("POLICY NOT AVAILABLE")

    # Assign the rescale factors & model to the case_dict
    case_dict['rescale_factors'] = rescale_factors
    case_dict['loaded_model'] = loaded_model
    print(f"[META] W&B LOADED: {case_dict['rL_algorithm']}\t{case_dict['policy']}\t{case_dict['rescale_factors']}", flush=True)

    return case_dict

  # --------------------------------------------
  @staticmethod
  def _scale_reward(reward: np.ndarray, baseline_dUdy):
    """
    Scale the observation based on the friction velocity
    """

    return 1 - (np.mean(reward) / baseline_dUdy)

  # --------------------------------------------
  @staticmethod
  def _normalize_state(observation: np.ndarray, u_tau):
    """
    Scale the observation based on the friction velocity
    """
    return observation / u_tau

  # --------------------------------------------
  @staticmethod
  def _rescale_actions(actions: np.ndarray, rescale_factors):
    """
    Rescale the action based on the rescale factors
    """

    return actions * rescale_factors[-1]

  # --------------------------------------------
  @staticmethod
  def _obs_solver_arrange(observation: np.ndarray, source_solver: str):
    """
    Rearrange the observation based on the source solver
    """
    if source_solver == "dedalus":
      # For Dedalus, we need to flip the observation order
      print(f"[Meta] Rearranging Observation for Dedalus, {observation.shape}", flush=True)
      return np.flip(observation, axis=1)
    else: # if nek
      return observation  # Placeholder - implement actual rearrangement logic based on source_solver
