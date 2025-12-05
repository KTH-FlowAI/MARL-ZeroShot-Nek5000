"""
Utility functions for determining the deterministic run. 
@yuningw
"""

import numpy as np 
import scipy.io as sio 
import pandas as pd 
import os, shutil
import matplotlib.pyplot as plt 
import yaml 

def name_case(conf): 
  """ Name cases based on the algorithm and run name""" 
  run_name = f"{conf['runner']['RL_algorithm']}_Retau{int(conf['simulation']['retau'])}_{conf['runner']['agent_run_name']}"
  return run_name

def rename_file(run_name,env_id,date_str,format_str):
  """ Rename the vars_record files """ 
  file_name = f"{run_name}_env{env_id:03d}_{date_str}"+format_str
  return file_name
  
def read_deterministic_run(run_path,case_list,verbose=True):
    """Read the deterministic run"""
    case_dict = {}
    for il, case in enumerate(case_list):
        if verbose:
            print(f"---- Start Case {il+1} ----") 
        #--- Initialize Case ---
        case_dict[case] = {}
        case_path = os.path.join(run_path, str(case))
        if os.path.exists(case_path):
            case_dict[case]['path'] = case_path
        else:
            if verbose:
                print(f"Case {case} not found")
            raise FileNotFoundError(f"Case {case} not found")
        # Load the config
        with open(os.path.join(case_path, "current_conf.yml"), "r") as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
            case_dict[case]['conf'] = conf
        f.close()

        case_dict[case]['run_name']=name_case(case_dict[case]['conf'])
        save_path = os.path.join(case_path, case_dict[case]["run_name"])
        if not os.path.exists(save_path):
          if verbose:
            print(f"Create new folder: {save_path}")
          os.mkdir(save_path)

        # --- Copy the config file ---  
        shutil.copy(os.path.join(case_path,'current_conf.yml'),os.path.join(save_path,'current_conf.yml'))

        # --- Get the test run list ---  
        env_list = os.listdir(case_path)
        env_list = [os.path.join(case_path, f) for f in env_list if "env" in f]
        env_list.sort()
        if verbose:
            print(f"[IO] {len(env_list)} Environment folders found: {env_list}")
        for il, env_path in enumerate(env_list):
            # Load the record
            file_list = os.listdir(env_path)
            file_list = [os.path.join(env_path, f) for f in file_list if "vars_record" in f]
            file_list.sort()
            for file_path in file_list:
              loc = file_path.find(".mat") 
              date_str = file_path[loc-10:loc]
              format_str = file_path[loc:]
              # Rename files: 
              env_id = il + 1 
              new_file_name = rename_file(case_dict[case]['run_name'],env_id, date_str, format_str)
              new_file_path = os.path.join(save_path,new_file_name)
              # Copy data to new folders
              if not os.path.exists(new_file_path):
                shutil.copy(file_path, new_file_path)
                if verbose:
                    print(f"[IO] Copy New Data {file_path}")
        
        # list datas
        save_path_list = os.listdir(save_path)
        save_path_list = [os.path.join(save_path, f) for f in save_path_list]
        num_files = len(save_path_list) 
        if verbose:
            print(f"[IO] {num_files} Data sorted to new folder: {save_path}: {save_path_list}")

        # Start reading datas: 
        case_dict[case]['reward'] = []
        case_dict[case]['action'] = []
        case_dict[case]['states'] = []

        # We need to match the format: 
        rec_file_list = [f for f in save_path_list if ".mat" in f]
        rec_file_list.sort()
        if verbose:
            print(f"[IO] {len(rec_file_list)} Record files found: {rec_file_list}")
        for rec_data_file in rec_file_list: 
          if verbose:
              print(f"[IO] Reading {rec_data_file}")
          mat_data = sio.loadmat(rec_data_file)
          possible_agents = [k for k in mat_data.keys() if 'jet' in k]
          rew_sub, act_sub, obs_sub = [], [], [] 
          for agent in possible_agents: 
            rew_ = mat_data[agent]['rew_rec'][0][0][:,0] 
            act_ = mat_data[agent]['act_rec'][0][0][:,0] 
            obs_ = mat_data[agent]['obs_rec'][0][0][:,:] 

            # Append results             
            rew_sub.append(np.expand_dims(rew_,0))
            act_sub.append(np.expand_dims(act_,0))
            obs_sub.append(np.expand_dims(obs_,0))
          
          # Concatenate the results
          rew_sub = np.concatenate(rew_sub,0)
          act_sub = np.concatenate(act_sub,0)
          obs_sub = np.concatenate(obs_sub,0)

          # Append the results to the case dictionary
          case_dict[case]['reward'].append(np.expand_dims(rew_sub,0))
          case_dict[case]['action'].append(np.expand_dims(act_sub,0))
          case_dict[case]['states'].append(np.expand_dims(obs_sub,0))

        case_dict[case]['reward'] = np.concatenate(case_dict[case]['reward'],0)
        case_dict[case]['action'] = np.concatenate(case_dict[case]['action'],0)
        case_dict[case]['states'] = np.concatenate(case_dict[case]['states'],0)

        if verbose:
            print(f"[IO] Reward: {case_dict[case]['reward'].shape}")
            print(f"[IO] Action: {case_dict[case]['action'].shape}")
            print(f"[IO] States: {case_dict[case]['states'].shape}")

    return case_dict

              
def process_reward(reward:np.ndarray,agent_idx:list,
  DT:float,N_DRL:int,sample_freq:float,
  Re:float,utau:float,
  trans_time:float=500,eval_time:float=1500):
  """ 
  Process the reward and expressed as a function of t+
  Input:
    - reward: (N_Env, N_agent, N_t)
    - DT: time step
    - N_DRL: number of DRL updates
    - sample_freq: sample frequency
    - Re: Reynolds number
    - utau: friction velocity
    - trans_time: transition time
    - eval_time: evaluation time
  Output:
    - tplus(N_t): time in t+ units
    - mean_reward(N_agent): mean reward evolution
    - std_reward(N_agent): standard deviation of reward evolution
    - scalar_reward(N_agent): mean reward during evaluation
  """
 
  # --- Define size variables ---
  N_Env, N_agent, N_t = reward.shape
  # --- Time in t+ units ---
  t_star = (1./Re) / utau**2
  tplus = np.linspace(0,N_t*DT*N_DRL*sample_freq,N_t) / t_star
  # --- Indices for evaluation ---
  ind_eval = np.where((tplus > trans_time)&(tplus < eval_time))
  ind_trans = np.where(tplus <= trans_time)

  mean_reward = [];std_reward = [];scalar_reward = []
  for il, agent in enumerate(agent_idx):
    reward_ = reward[:,agent,:]
    mean_reward.append(np.mean(reward_,axis=0))
    std_reward.append(np.std(reward_,axis=0))
    scalar_reward.append(np.mean(np.mean(reward_,axis=0)[ind_eval],keepdims=False))
  
  return tplus, mean_reward, std_reward, scalar_reward, ind_eval, ind_trans


def process_action(action:np.ndarray,agent_idx:list,
  DT:float,N_DRL:int,sample_freq:float,
  Re:float,utau:float,
  trans_time:float=500,eval_time:float=1500):
  """ 
  Process the reward and expressed as a function of t+
  Input:
    - reward: (N_Env, N_agent, N_t)
    - DT: time step
    - N_DRL: number of DRL updates
    - sample_freq: sample frequency
    - Re: Reynolds number
    - utau: friction velocity
    - trans_time: transition time
    - eval_time: evaluation time
  Output:
    - tplus(N_t): time in t+ units
    - mean_reward(N_agent): mean reward evolution
    - std_reward(N_agent): standard deviation of reward evolution
    - scalar_reward(N_agent): mean reward during evaluation
  """
 
  # --- Define size variables ---
  N_Env, N_agent, N_t = action.shape
  # --- Time in t+ units ---
  t_star = (1./Re) / utau**2
  tplus = np.linspace(0,N_t*DT*N_DRL*sample_freq,N_t) / t_star
  # --- Indices for evaluation ---
  ind_eval = np.where((tplus > trans_time)&(tplus < eval_time))
  ind_trans = np.where(tplus <= trans_time)

  actions = []
  for il, agent in enumerate(agent_idx):
    action_ = action[:,agent,:]
    actions.append(np.expand_dims(action_,0))
  actions = np.concatenate(actions,0)
  return tplus, actions, ind_eval, ind_trans

def process_observation(observation:np.ndarray,
  DT:float,N_DRL:int,sample_freq:float,
  Re:float,utau:float,
  agent_idx=None,
  trans_time:float=500,eval_time:float=1500):
  """ 
  Process the observation and expressed as a function of t+
  Input:
    - observation: (N_Env, N_t, N_state)
    - DT: time step
    - N_DRL: number of DRL updates
    - sample_freq: sample frequency
    - Re: Reynolds number
    - utau: friction velocity
    - trans_time: transition time
    - eval_time: evaluation time
    - agent_idx: list of agent indices to process
    Output:
    - tplus(N_t): time in t+ units
    - observations(N_agent, N_t, N_state): observations
    - ind_eval: indices for evaluation
    - ind_trans: indices for transition
  """
  # --- Define size variables ---
  N_Env, N_agent, N_t, N_state = observation.shape
  # --- Time in t+ units ---
  t_star = (1./Re) / utau**2
  tplus = np.linspace(0,N_t*DT*N_DRL*sample_freq,N_t) / t_star
  # --- Indices for evaluation ---
  ind_eval = np.where((tplus > trans_time)&(tplus < eval_time))
  ind_trans = np.where(tplus <= trans_time)

  if agent_idx != None:
    observations = []
    for il, agent in enumerate(agent_idx):
      observation_ = observation[:,agent,:]
      observations.append(np.expand_dims(observation_,0))
    observations = np.concatenate(observations,0)
  else:
    observations = observation
  return tplus, observations, ind_eval, ind_trans

def PDF(InterSecX,InterSecY,
        xmin = -1,xmax = 1,x_grid = 50,
        ymin = -1,ymax = 1,y_grid = 50):

    """
    Compute the joint PDF of X and Y 
    Args:
        InterSecX   : numpy array of data 1
        InterSecY   : numpy array of data 2

        xmin, xmax, x_grid  :   The limitation of InterSecX and number of grid to be plot for contour 
        ymin, ymax, y_grid  :   The limitation of InterSecY and number of grid to be plot for contour 

    Returns:
        xx, yy: The meshgrid of InterSecX and InterSecY according to the limitation and number of grids
        pdf   : The joint pdf of InterSecX and InterSecY 
    """
    import numpy as np 
    import scipy.stats as st 
    # Create meshgrid acorrding 
    xx, yy = np.mgrid[xmin:xmax:1j*x_grid, ymin:ymax:1j*y_grid]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values    = np.vstack([InterSecX, InterSecY])
    kernel    = st.gaussian_kde(values)
    pdf       = np.reshape(kernel(positions).T, xx.shape)

    return xx,yy,pdf


