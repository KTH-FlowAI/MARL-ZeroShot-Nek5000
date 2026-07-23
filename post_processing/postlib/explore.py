"""
Utility functions for exploring the training state.
@yuningw
"""

import numpy as np
import scipy.io as sio
import os
import re
import warnings
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


def _round_sort_key(name):
    """Numeric sort key for a history/roundXXX directory name.

    Sorts by the integer XXX so that round2 comes before round10
    (lexicographic sorting would order round10 before round2).
    """
    m = re.match(r'round0*(\d+)', os.path.basename(os.path.normpath(name)))
    return int(m.group(1)) if m else -1


def _rewlog_sort_key(path):
    """Numeric sort key for a rewlog_<number>.npz file path."""
    m = re.search(r'rewlog_0*(\d+)\.npz$', os.path.basename(path))
    return int(m.group(1)) if m else -1


def _discover_segments(case_path):
    """Discover the ordered list of history segments for a case.

    Returns a list of ``(label, dir_path)`` tuples in chronological order
    (oldest -> newest):

    1. Direct files in ``runs/<case>/history/`` (only added when that
       directory *directly* contains ``rewlog_*.npz`` or
       ``rewards_aggregated_*.csv``; child directories are not recursed).
    2. ``history/roundXXX`` directories, numerically sorted by XXX.
    3. ``history/current_history``.
    4. ``train/history`` (the live, on-the-fly run -- newest last).
    """
    history_root = os.path.join(case_path, "history")
    current_hist = os.path.join(history_root, "current_history")
    live_hist    = os.path.join(case_path, "train", "history")

    segments = []

    if os.path.isdir(history_root):
        try:
            root_entries = os.listdir(history_root)
        except OSError:
            root_entries = []

        # (1) Root history segment -- only when it directly holds reward files.
        #     We deliberately do NOT recurse into child round dirs here; those
        #     are added as their own segments below.
        has_direct = any(
            (f.startswith('rewlog_') and f.endswith('.npz')) or
            (f.startswith('rewards_aggregated_') and f.endswith('.csv'))
            for f in root_entries
            if os.path.isfile(os.path.join(history_root, f))
        )
        if has_direct:
            segments.append(('history', history_root))

        # (2) roundXXX directories, numerically sorted by XXX.
        round_dirs = [
            f for f in root_entries
            if re.match(r'round\d+', f)
            and os.path.isdir(os.path.join(history_root, f))
        ]
        for name in sorted(round_dirs, key=_round_sort_key):
            segments.append((name, os.path.join(history_root, name)))

    # (3) previous completed run.
    if os.path.isdir(current_hist):
        segments.append(('current_history', current_hist))

    # (4) live run (newest).
    if os.path.isdir(live_hist):
        segments.append(('train/history', live_hist))

    return segments, history_root


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
        #[MOD]   history/ (direct files) : reward files dropped straight into
        #[MOD]                             the history root (e.g. first runs)
        #[MOD]   history/round001..N     : archived completed runs
        #[MOD]   history/current_history : the previous completed run
        #[MOD]   train/history           : the LIVE, on-the-fly run (newest)
        #[MOD] The live train/history is included so reward-on-the-fly shows up
        #[MOD] while a job is still running (and so first runs, which have no
        #[MOD] history/ yet, still work).
        segments, history_root = _discover_segments(case_path)

        if not segments:
            print(f"History for case {case} not found")
            raise FileNotFoundError(f"History for case {case} not found")

        # Newest segment is the last one; expose it as the case 'history'.
        history_path = segments[-1][1]
        case_dict[case]['path']    = history_root if os.path.isdir(history_root) else case_path
        case_dict[case]['history'] = history_path
        # Optional diagnostic metadata (unused by the notebooks).
        case_dict[case]['history_segments'] = [
            {'label': lbl, 'path': p} for lbl, p in segments
        ]

        #[MOD] Read the config from the newest segment that has current_conf.yml
        #[MOD] (prefer the live run, then current_history, then any round).
        conf = None
        for _, seg in reversed(segments):
            cfg = os.path.join(seg, "current_conf.yml")
            if os.path.exists(cfg):
                with open(cfg, "r") as f:
                    conf = yaml.load(f, Loader=yaml.FullLoader)
                break
        if conf is None:
            raise FileNotFoundError(f"current_conf.yml not found for case {case}")
        case_dict[case]['conf'] = conf
        window_size = conf['runner']['nb_interactions']

        # Load actor and critic loss history
        case_dict[case]['actor_loss'] = []
        case_dict[case]['critic_loss'] = []
        for _, round_path in segments:
            sb3_csv_path = os.path.join(round_path, "tensorboard/progress.csv")
            if os.path.exists(sb3_csv_path):
                try:
                  sb3_csv = pd.read_csv(sb3_csv_path)
                  if not sb3_csv.empty:
                    case_dict[case]['actor_loss'].append(sb3_csv['train/actor_loss'].values.reshape(-1,))
                    case_dict[case]['critic_loss'].append(sb3_csv['train/critic_loss'].values.reshape(-1,))
                except Exception:
                  #print(f"Failed to read SB3 CSV file {sb3_csv_path}")
                  pass

        try:
          case_dict[case]['actor_loss'] = np.concatenate(case_dict[case]['actor_loss']).reshape(-1,)
          case_dict[case]['critic_loss'] = np.concatenate(case_dict[case]['critic_loss']).reshape(-1,)
        except Exception:
          print(f"Failed to concatenate actor and critic loss")
          pass

        #---------------------------------------------------------------
        # Reward history (rewlog_*.npz)
        #---------------------------------------------------------------
        #[MOD] Each rewlog_*.npz stores one *attempted* episode. Empty (0-len)
        #[MOD] and partial (< nb_interactions) arrays are the initial/aborted
        #[MOD] states and are dropped. Every retained file is exactly one
        #[MOD] complete episode, so the global index is simply 1..N -- no
        #[MOD] len(round_list)/linspace correction is needed.
        case_dict[case]['reward'] = []       # list of per-episode arrays (× 100 %)
        case_dict[case]['reward_mean'] = []  # per-episode mean reward (× 100 %)
        reward_provenance = []               # one record per complete episode
        n_rewlog_seen = 0
        for seg_label, seg_path in segments:
            try:
                entries = os.listdir(seg_path)
            except OSError:
                continue
            rewlogs = [
                os.path.join(seg_path, f) for f in entries
                if f.startswith('rewlog_') and f.endswith('.npz')
            ]
            rewlogs.sort(key=_rewlog_sort_key)
            for reward_file in rewlogs:
                n_rewlog_seen += 1
                try:
                    data = np.load(reward_file, allow_pickle=True)
                    rew = data['rew']
                except Exception:
                    continue
                # Keep only complete episodes; skip empty/partial arrays.
                if rew.shape[0] != window_size:
                    continue
                rew = np.reshape(rew * 100, -1)  # preserve percentage scaling
                case_dict[case]['reward'].append(rew)
                case_dict[case]['reward_mean'].append(np.mean(rew))
                reward_provenance.append({
                    'segment': seg_label,
                    'file': os.path.basename(reward_file),
                })

        n_reward_eps = len(case_dict[case]['reward'])
        print(f"Reward files seen: {n_rewlog_seen}, complete episodes: {n_reward_eps}")

        if n_reward_eps == 0:
            print(f"No complete reward episodes found for case {case}")
            case_dict[case]['reward_smooth'] = np.array([])
            case_dict[case]['episode'] = np.array([])
            case_dict[case]['episode_idx'] = np.array([])
            case_dict[case]['reward_smooth_max'] = np.nan
            case_dict[case]['reward_smooth_max_index'] = 0
            case_dict[case]['reward_mean_max'] = np.nan
            case_dict[case]['reward_max_index'] = 0
        else:
            # Get the mean reward smoothed by a convolution
            mean_rew = np.concatenate(case_dict[case]['reward']).reshape(-1,)
            case_dict[case]['reward_smooth'] = smooth_Value(mean_rew,
                                                            window_size=window_size)
            print(f"Reward smooth: {case_dict[case]['reward_smooth'].shape}")
            ep_len = mean_rew.shape[0]
            # Smooth func results (fractional-episode x axis)
            case_dict[case]['episode'] = np.arange(window_size, ep_len + 1) / (window_size)
            case_dict[case]['reward_smooth_max'] = np.max(case_dict[case]['reward_smooth'])
            case_dict[case]['reward_smooth_max_index'] = case_dict[case]['episode'][np.argmax(case_dict[case]['reward_smooth'])]
            case_dict[case]['reward_smooth_max_index'] = np.ceil(case_dict[case]['reward_smooth_max_index']).astype(int)

            # Mean results -- one global episode index per complete episode.
            case_dict[case]['episode_idx'] = np.arange(1, n_reward_eps + 1)
            case_dict[case]['reward_mean_max']   = np.max(case_dict[case]['reward_mean'])
            case_dict[case]['reward_max_index'] = case_dict[case]['episode_idx'][np.argmax(case_dict[case]['reward_mean'])]
            case_dict[case]['reward_max_index'] = np.ceil(case_dict[case]['reward_max_index']).astype(int)

            print(f"Reward mean max R: {case_dict[case]['reward_mean_max']:.2f} at episode {case_dict[case]['reward_max_index']}")

        #---------------------------------------------------------------
        # Reward components (mean_R_tau, mean_R_pw, mean_R_v3)
        #---------------------------------------------------------------
        #[MOD] Component rows live in rewards_aggregated_*.csv, one row per
        #[MOD] interaction step. We MUST split into episodes *within* each CSV
        #[MOD] (by the local 'episode' field) before checking completeness --
        #[MOD] concatenating across CSVs/segments first would splice a partial
        #[MOD] tail onto the next segment and corrupt the alignment.
        comp_keys = ['mean_R_tau', 'mean_R_pw', 'mean_R_v3']
        comp_cols = ['episode', 'step'] + comp_keys
        comp_steps = {k: [] for k in comp_keys}  # per group: full step arrays
        comp_eps   = {k: [] for k in comp_keys}  # per group: episode-mean
        comp_provenance = []
        for seg_label, seg_path in segments:
            try:
                entries = os.listdir(seg_path)
            except OSError:
                continue
            csv_files = sorted(
                os.path.join(seg_path, f) for f in entries
                if f.startswith('rewards_aggregated_') and f.endswith('.csv')
            )
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file, usecols=comp_cols)
                except Exception:
                    # Not a component CSV (e.g. missing columns) -- skip.
                    continue
                df = df.dropna(subset=comp_cols)
                if df.empty:
                    continue
                # Group by the *local* episode field, sort each group by step,
                # keep only complete episodes, and append chronologically.
                for ep_val, grp in df.groupby('episode', sort=True):
                    grp = grp.sort_values('step')
                    if len(grp) != window_size:
                        continue  # discard partial group
                    for k in comp_keys:
                        vals_k = grp[k].values * 100.0  # → %
                        comp_steps[k].append(vals_k)
                        comp_eps[k].append(float(np.mean(vals_k)))
                    comp_provenance.append({
                        'segment': seg_label,
                        'file': os.path.basename(csv_file),
                        'local_episode': int(ep_val),
                    })

        n_comp_eps = len(comp_provenance)
        if n_comp_eps > 0:
            for k in comp_keys:
                vals = np.concatenate(comp_steps[k])
                case_dict[case][k] = vals
                case_dict[case][f'{k}_smooth'] = smooth_Value(vals, window_size=window_size)
                case_dict[case][f'{k}_eps'] = np.asarray(comp_eps[k])
            n_comp_steps = n_comp_eps * window_size
            case_dict[case]['comp_episode'] = np.arange(window_size, n_comp_steps + 1) / window_size
            case_dict[case]['comp_episode_idx'] = np.arange(1, n_comp_eps + 1)
            print(f"Components loaded: {n_comp_steps} steps, {n_comp_eps} episodes")

            # The component and reward-NPZ episode counts should agree.
            if n_comp_eps != n_reward_eps:
                warnings.warn(
                    f"[initalize_case] case '{case}': component episode count "
                    f"({n_comp_eps}) != reward NPZ episode count "
                    f"({n_reward_eps}). rewlog_*.npz and "
                    f"rewards_aggregated_*.csv are out of sync across the "
                    f"history segments {[lbl for lbl, _ in segments]}. "
                    f"Component curves (mean_R_*) may be misaligned with the "
                    f"reward curves; check for missing/partial files.",
                    stacklevel=2,
                )
        else:
            print(f"No component data found (non net_gain run?)")

        # Optional diagnostic metadata (unused by the notebooks).
        case_dict[case]['episode_provenance'] = {
            'reward': reward_provenance,
            'component': comp_provenance,
        }

        print(f"---- End ----\n")
    return case_dict
