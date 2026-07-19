"""
Utility functions for exploring the training state. 
@yuningw
"""

import numpy as np 
import scipy.io as sio 
import os 
import matplotlib.pyplot as plt 
from postlib.plot import plt_setUp, colorplate as cc
import pandas as pd 
import argparse
import yaml # for loading the config

def smooth_Value(raw_data,window_size):
    """Smooth the value by a convolution"""
    kernel = np.ones(shape=(window_size,))
    data = np.convolve(raw_data,kernel,mode='valid')/window_size
    return data


def initalize_case(run_path,case_list):
    """Initialize the case"""
    case_dict = {}
    for il, case in enumerate(case_list):
        print(f"---- Start Case {il+1} ----") 
        #--- Initialize Case ---
        case_dict[case] = {}
        case_path = os.path.join(run_path, str(case))
        #[MOD] History segments, oldest -> newest (see
        #[MOD] src/initial.py:preserve_and_clean_train):
        #[MOD]   history/round001..N     : archived completed runs
        #[MOD]   history/current_history : the previous completed run
        #[MOD]   train/history           : the LIVE, on-the-fly run (newest)
        #[MOD] The live train/history is included so reward-on-the-fly shows up
        #[MOD] while a job is still running (and so first runs, which have no
        #[MOD] history/ yet, still work).
        history_root = os.path.join(case_path, "history")
        current_hist = os.path.join(history_root, "current_history")
        live_hist    = os.path.join(case_path, "train", "history")

        # roundXXX (archived), oldest first
        round_list = []
        if os.path.exists(history_root):
            round_list = sorted(
                os.path.join(history_root, f)
                for f in os.listdir(history_root) if "round" in f
            )
        # then the previous completed run, then the live run (newest last)
        if os.path.exists(current_hist):
            round_list.append(current_hist)
        if os.path.exists(live_hist):
            round_list.append(live_hist)

        if not round_list:
            print(f"History for case {case} not found")
            raise FileNotFoundError(f"History for case {case} not found")

        # Newest segment is the last one; expose it as the case 'history'
        history_path = round_list[-1]
        case_dict[case]['path']    = history_root if os.path.exists(history_root) else case_path
        case_dict[case]['history'] = history_path

        #[MOD] Read the config from the newest segment that has current_conf.yml
        #[MOD] (prefer the live run, then current_history, then any round).
        conf = None
        for seg in reversed(round_list):
            cfg = os.path.join(seg, "current_conf.yml")
            if os.path.exists(cfg):
                with open(cfg, "r") as f:
                    conf = yaml.load(f, Loader=yaml.FullLoader)
                break
        if conf is None:
            raise FileNotFoundError(f"current_conf.yml not found for case {case}")
        case_dict[case]['conf'] = conf

        # Load the reward history
        reward_list = []
        # Load actor and critic loss history 
        case_dict[case]['actor_loss'] = []
        case_dict[case]['critic_loss'] = []
        for round_path in round_list:
            #---------- Reward history ----------
            file_list = os.listdir(round_path)
            file_list = [os.path.join(round_path, f) for f in file_list if "rewlog" in f]
            file_list.sort()
            reward_list += file_list
            #---------- Actor and critic loss history ----------
            sb3_csv_path = os.path.join(round_path, "tensorboard/progress.csv")
            #
            if os.path.exists(sb3_csv_path):
                try:
                  sb3_csv = pd.read_csv(sb3_csv_path)
                  if not sb3_csv.empty:
                    case_dict[case]['actor_loss'].append(sb3_csv['train/actor_loss'].values.reshape(-1,))
                    case_dict[case]['critic_loss'].append(sb3_csv['train/critic_loss'].values.reshape(-1,))
                except:
                  #print(f"Failed to read SB3 CSV file {sb3_csv_path}")
                  pass
            else:
                #print(f"SB3 CSV file {sb3_csv_path} not found")
                pass

        try:
          case_dict[case]['actor_loss'] = np.concatenate(case_dict[case]['actor_loss']).reshape(-1,)
          case_dict[case]['critic_loss'] = np.concatenate(case_dict[case]['critic_loss']).reshape(-1,)
        except:
          print(f"Failed to concatenate actor and critic loss")
          pass

        print(f"Reward list: {len(reward_list)} files")

        
        #--- Read the data --- 
        case_dict[case]['reward'] = []
        case_dict[case]['reward_mean'] = []
        window_size = conf['runner']['nb_interactions'] 
        # Load the reward data
        actual_file = 0 
        for il, reward_file in enumerate(reward_list):
            data = np.load(reward_file, allow_pickle=True)
            if data['rew'].shape[0] == window_size:
              case_dict[case]['reward'].append(np.reshape(data['rew'][:]*100,-1))
              case_dict[case]['reward_mean'].append(np.mean(data['rew'][:]*100))
              actual_file += 1 
            else:
              #print(f"Reward file {reward_file} has {data['rew'].shape[0]} interactions, expected {window_size}")
              pass

        # Get the mean reward smoothed by a convolution 
        mean_rew = np.concatenate(case_dict[case]['reward']).reshape(-1,)
        case_dict[case]['reward_smooth'] = smooth_Value(mean_rew,
                                                        window_size=window_size)
        print(f"Reward smooth: {case_dict[case]['reward_smooth'].shape}")
        ep_len = mean_rew.shape[0]
        # Smooth func results 
        case_dict[case]['episode'] = np.arange(window_size,ep_len+1)/(window_size)
        case_dict[case]['reward_smooth_max'] = np.max(case_dict[case]['reward_smooth'])
        case_dict[case]['reward_smooth_max_index'] = case_dict[case]['episode'][np.argmax(case_dict[case]['reward_smooth'])]
        case_dict[case]['reward_smooth_max_index'] = np.ceil(case_dict[case]['reward_smooth_max_index']).astype(int)

        # Mean results 
        len_eps = actual_file - len(round_list) + 1 # Remove the zero state and recover the history data 
        case_dict[case]['episode_idx'] = np.linspace(1,len_eps+1,actual_file)
        case_dict[case]['reward_mean_max']   = np.max(case_dict[case]['reward_mean']) 
        case_dict[case]['reward_max_index'] = case_dict[case]['episode_idx'][np.argmax(case_dict[case]['reward_mean'])]
        case_dict[case]['reward_max_index'] = np.ceil(case_dict[case]['reward_max_index']).astype(int)
        
        print(f"Reward mean max R: {case_dict[case]['reward_mean_max']:.2f} at episode {case_dict[case]['reward_max_index']}")

        # Load reward components (mean_R_tau, mean_R_pw, mean_R_v3) from aggregated CSVs
        comp_raw = {'mean_R_tau': [], 'mean_R_pw': [], 'mean_R_v3': []}
        for round_path in round_list:
            try:
                flist = os.listdir(round_path)
            except Exception:
                continue
            csv_files = sorted([
                os.path.join(round_path, f) for f in flist
                if f.startswith('rewards_aggregated_') and f.endswith('.csv')
            ])
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file, usecols=['mean_R_tau', 'mean_R_pw', 'mean_R_v3'])
                    df = df.dropna()
                    if not df.empty:
                        for k in comp_raw:
                            comp_raw[k].append(df[k].values)
                except Exception:
                    pass

        if any(len(v) > 0 for v in comp_raw.values()):
            for k in comp_raw:
                vals = np.concatenate(comp_raw[k]) * 100.0   # → %
                case_dict[case][k] = vals
                case_dict[case][f'{k}_smooth'] = smooth_Value(vals, window_size=window_size)
            n_comp = len(case_dict[case]['mean_R_tau'])
            case_dict[case]['comp_episode'] = np.arange(window_size, n_comp + 1) / window_size
            n_eps_comp = n_comp // window_size
            for k in comp_raw:
                vals = case_dict[case][k]
                case_dict[case][f'{k}_eps'] = np.array(
                    [np.mean(vals[i * window_size:(i + 1) * window_size])
                     for i in range(n_eps_comp)]
                )
            case_dict[case]['comp_episode_idx'] = np.arange(1, n_eps_comp + 1)
            print(f"Components loaded: {n_comp} steps, {n_eps_comp} episodes")
        else:
            print(f"No component data found (non net_gain run?)")

        print(f"---- End ----\n")
    return case_dict