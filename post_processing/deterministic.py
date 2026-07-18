import os, copy
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from postlib.plot import plt_setUp, make_style, colorplate as cc
from postlib.determine import (
    read_deterministic_run,
    process_reward,
    process_action,
    process_observation,
    load_reward_monitor_case,
    PDF,
)
parser = argparse.ArgumentParser()
parser.add_argument("--run_path", default="../runs/")
parser.add_argument("--fig_path", default="Figs/")
parser.add_argument("--table_path", default="Tables/")
parser.add_argument("--save_fig",action='store_true')
parser.add_argument("--show_fig",action='store_true')
args = parser.parse_args()

plt_setUp()
run_path = args.run_path
fig_path = args.fig_path
table_path = args.table_path
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
if not os.path.exists(table_path):
    os.makedirs(table_path)

# --- Style ---
STYLE_OC_RED_A06  = make_style(cc.red,       0.6, 'X', 'OC')
STYLE_DDPG_GR_A06 = make_style(cc.deepgreen, 0.6, 'D', 'DDPG')
STYLE_PPO_BL_A06  = make_style(cc.blue,      0.6, 'o', 'PPO')

# --- Case Configuration ---
SAVE_IMG=False; SHOW_IMG=False
if args.save_fig:
    SAVE_IMG = args.save_fig
if args.show_fig:
    SHOW_IMG = args.show_fig

Head = 'Reth365'
case_tuple = [
    [2001, 'TD3',   STYLE_OC_RED_A06],
    [2001, 'TD3',   STYLE_OC_RED_A06],
]
case_list  = [c[0] for c in case_tuple]
case_name  = [c[1] for c in case_tuple]
case_style = [c[2] for c in case_tuple]

case_dict = read_deterministic_run(run_path, case_list, verbose=True)

# ---------------------------------------------------------------------------
# Reward Assessment
# ---------------------------------------------------------------------------
agent_idx = [0]
fig, axs = plt.subplots(1, 1, figsize=(8, 6))

v_r_last = None  # kept for re-use in the Action section below

for il, case in enumerate(case_dict.keys()):
    case_name_ = case_name[il]
    style = copy.deepcopy(case_style[il])

    reward = case_dict[case]['reward']
    tp, m_r, s_r, v_r, i_eval, i_trans = process_reward(
        reward, agent_idx,
        DT=np.abs(case_dict[case]['conf']['simulation']['dt']),
        N_DRL=case_dict[case]['conf']['simulation']['ndrl'],
        sample_freq=case_dict[case]['conf']['runner']['vars_record_freq'],
        Re=np.abs(case_dict[case]['conf']['simulation']['viscosity']),
        utau=case_dict[case]['conf']['runner']['u_tau'],
    )
    v_r_last = v_r  # save for reuse

    shad_style = {'alpha': 0.4, 'color': style['c']}
    axs.fill_between(tp, (m_r[0] - s_r[0]) * 100, (m_r[0] + s_r[0]) * 100, **shad_style)

    style['label']  = case_name_ + f": {v_r[0] * 100:.2f}" + r"$\%$"
    style['marker'] = 'None'
    axs.plot(tp, m_r[0] * 100, **style)

axs.axvspan(xmin=0, xmax=500, color='gray', alpha=0.2)
axs.text(0.1, 0.9, "Transition:\n" + r"$\ t^+ \leq 500$", transform=axs.transAxes, fontsize=12)
axs.text(0.5, 0.9, "Reward Evaluation:\n" + r"$t^+ > 500$",  transform=axs.transAxes, fontsize=12)
axs.set(xlabel=r"$t^+$", xlim=[-1, 1501], ylim=[-1, 60], ylabel=r"$R [\%]$")
axs.legend(prop={"size": 12}, loc='lower right')

if SAVE_IMG:
    out = fig_path + f'{Head}_Reward_Deterministic.jpg'
    plt.savefig(out, dpi=300)
    print(f'Saved {out}')
if SHOW_IMG:
    plt.show()

# ---------------------------------------------------------------------------
# Reward Monitor (per-component reward time series, all envs)
# ---------------------------------------------------------------------------
### Start a panda database to store the avg. results for t+ > 500
df = {"case_name":[],
      "rwd_tau_mean":[],"rwd_pw_mean":[],"rwd_v3_mean":[], # individial terms
      "R_mean":[],"NP_mean":[], # pure DR & Net-pow saving
      "rwd_tau_std":[],"rwd_pw_std":[],"rwd_v3_std":[], # individial terms
      "R_std":[],"NP_std":[], # pure DR & Net-pow saving
      }

COLORS_COMP = ['#2E59A7', '#D23918', '#2CA02C', '#9467BD', '#8C564B']
label_and_scale= [(r'$\tau_w$', 'linear',[0.001,0.005]),
                  (r"$|p'_w v_w|$", "log",[1e-6,1e-2]), (r'$|\rho v^3_w|$', "log",[1e-8,1e-2])]

collected = []  # stores (time, mean_r, std_r, run_name, K) for cross-case plots

for il, case in enumerate(case_dict.keys()):
    #[MOD] Use the eval/-resolved path from read_deterministic_run so env_XXX
    #[MOD] folders (now under runs/<case>/eval/) are found consistently.
    case_path  = case_dict[case]['path']
    save_path  = os.path.join(case_path, case_dict[case]['run_name'])
    run_name   = case_dict[case]['run_name']

    time, reward_array, reward_names = load_reward_monitor_case(
        case_path, save_path, run_name=run_name, verbose=True
    )
    Re=np.abs(float(case_dict[case]['conf']['simulation']['viscosity']))
    utau=float(case_dict[case]['conf']['runner']['u_tau'])
    dUdy=float(case_dict[case]['conf']['runner']['dUdy'])
    tau_w_ref = dUdy/Re
    t_star = (1/Re)/ utau**2
    time -= time[0]  # align the start time to 0
    time = time / t_star
    t500 = np.where(time >= 500)[0]

    K, N, M = reward_array.shape
    mean_r = reward_array.mean(axis=0)   # [N, M]
    std_r  = reward_array.std(axis=0)    # [N, M]

    fig, axes = plt.subplots(N, 1, figsize=(10, max(4.0, 3.0 * N)), sharex=True)
    if N == 1:
        axes = [axes]

    for jl, (ax, name) in enumerate(zip(axes, reward_names)):
        color = COLORS_COMP[jl % len(COLORS_COMP)]
        label_and_scale_ = label_and_scale[jl] if jl < len(label_and_scale) else (name, 'linear')
        ax.fill_between(time, mean_r[jl] - std_r[jl], mean_r[jl] + std_r[jl],
                        alpha=0.25, color=color)
        ax.plot(time, mean_r[jl], lw=1.6, color=color,
                label=f'{name}  (mean and std, {K} envs)')
        ax.set_ylabel(label_and_scale_[0],fontsize=18)
        ax.set_ylim(label_and_scale_[2])
        if label_and_scale_[1] == 'log':
            ax.set_yscale('log',)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')

    axes[-1].set_xlabel('t')
    axes[0].set_title(f'{run_name} — reward components')
    fig.tight_layout()

    if SAVE_IMG:
        out = os.path.join(fig_path, f'{run_name}_reward_monitor.jpg')
        fig.savefig(out, dpi=200)
        print(f'Saved {out}')

    collected.append((time.copy(), mean_r.copy(), std_r.copy(), run_name, K))

    # Calculate R and NP
    df['case_name'].append(run_name)
    ### tau_w
    df['rwd_tau_mean'].append(mean_r[0][t500].mean())
    df['rwd_tau_std'].append(std_r[0][t500].mean())
    ### pv_w
    df['rwd_pw_mean'].append(mean_r[1][t500].mean())
    df['rwd_pw_std'].append(std_r[1][t500].mean())
    ### v3
    df['rwd_v3_mean'].append(mean_r[2][t500].mean())
    df['rwd_v3_std'].append(std_r[2][t500].mean())
    ### R = 1 - tau_w/tau_w
    df['R_mean'].append(1 - df['rwd_tau_mean'][-1] / tau_w_ref)
    df['R_std'].append(df['rwd_tau_std'][-1] / tau_w_ref * df['R_mean'][-1])
    ### NP = 1 - (tau_w + pv + v3)/tau_w_ref
    df['NP_mean'].append(1 - (df['rwd_tau_mean'][-1] + df['rwd_pw_mean'][-1] + df['rwd_v3_mean'][-1]) / tau_w_ref)
    df['NP_std'].append((df['rwd_tau_std'][-1] + df['rwd_pw_std'][-1] + df['rwd_v3_std'][-1]) / tau_w_ref * df['NP_mean'][-1])

# ---------------------------------------------------------------------------
# Per-component figures with all cases overlaid
# ---------------------------------------------------------------------------
if collected:
    N_comp = collected[0][1].shape[0]
    for jl in range(N_comp):
        label_and_scale_ = label_and_scale[jl] if jl < len(label_and_scale) else (reward_names[jl], 'linear', None)
        fig_c, ax_c = plt.subplots(1, 1, figsize=(10, 4))
        for il, (time_c, mean_c, std_c, rname, K_c) in enumerate(collected):
            color = case_style[il]['c']
            label = case_name[il]
            ax_c.fill_between(time_c, mean_c[jl] - std_c[jl], mean_c[jl] + std_c[jl],
                              alpha=0.20, color=color)
            ax_c.plot(time_c, mean_c[jl], lw=1.6, color=color,
                      label=f'{label}  ({K_c} envs)')
        ax_c.set_xlabel(r'$t^+$', fontsize=14)
        ax_c.set_ylabel(label_and_scale_[0], fontsize=18)
        if label_and_scale_[2] is not None:
            ax_c.set_ylim(label_and_scale_[2])
        if label_and_scale_[1] == 'log':
            ax_c.set_yscale('log')
        ax_c.grid(True, alpha=0.3)
        ax_c.legend(fontsize=10, loc='upper right')
        ax_c.set_title(f'Reward component {jl+1}: {label_and_scale_[0]} — all cases')
        fig_c.tight_layout()
        if SAVE_IMG:
            out = os.path.join(fig_path, f'{Head}_reward_comp{jl+1}_all_cases.jpg')
            fig_c.savefig(out, dpi=200)
            print(f'Saved {out}')

# Convert into Np array then output
for key in df.keys():
    if 'case_name' not in key:
        df[key] = np.array(df[key])
df = pd.DataFrame(df)
df.to_csv(os.path.join(table_path, f'{Head}_NP_Summary.csv'), index=False, float_format='%.2e')
print(df.head(10))
# ---------------------------------------------------------------------------
# Action Inspect
# ---------------------------------------------------------------------------
agent_idx = [0, 100, 200]
env_idx   = 0
colors    = ['r', 'g', 'b', 'y', 'c', 'm']
max_amp   = -1

fig, axs = plt.subplots(len(case_name), 1, figsize=(8, 6), sharex=True)

for il, case in enumerate(case_dict.keys()):
    case_name_ = case_name[il]
    style = case_style[il]

    action = case_dict[case]['action']
    tp, a_t, i_eval, i_trans = process_action(
        action, agent_idx,
        DT=np.abs(case_dict[case]['conf']['simulation']['dt']),
        N_DRL=case_dict[case]['conf']['simulation']['ndrl'],
        sample_freq=case_dict[case]['conf']['runner']['vars_record_freq'],
        Re=np.abs(case_dict[case]['conf']['simulation']['viscosity']),
        utau=case_dict[case]['conf']['runner']['u_tau'],
    )

    if "OC" not in case_name_:
        a_t *= case_dict[case]['conf']['runner']['ctrl_max_amp']

    max_amp = max(max_amp, np.max(np.abs(a_t)))

    style['label'] = case_name_ + f": {v_r_last[0] * 100:.2f}" + r"$\%$"
    for jl, agent in enumerate(agent_idx):
        style['c']     = colors[jl]
        style['label'] = f'Agent {jl + 1}'
        style['alpha'] = 0.2
        axs[il].plot(tp, a_t[jl, env_idx, :], **style)

    axs[il].set(
        title=case_name_,
        xlabel=r"$t^+$" if il == len(case_name) - 1 else '',
        xlim=[-1, 1501],
        ylim=[-1.1 * max_amp, 1.1 * max_amp],
        ylabel=r"$a_i$",
    )
    axs[il].legend(prop={"size": 12}, loc='lower right')

if SAVE_IMG:
    out = fig_path + f'{Head}_Action_Deterministic.jpg'
    plt.savefig(out, dpi=300)
    print(f'Saved {out}')
if SHOW_IMG:
    plt.show()
quit()

# ---------------------------------------------------------------------------
# Observation Inspect
# ---------------------------------------------------------------------------
agent_idx = None  # None => all agents
env_idx   = 0
cmap = plt.get_cmap('jet').copy()
cmap.set_under('white')
max_amp = -100
N_mesh  = 100

fig, axs = plt.subplots(len(case_name), 1, figsize=(8, 6), sharex=True)

for il, case in enumerate(case_dict.keys()):
    case_name_ = case_name[il]
    style = case_style[il]

    observation = case_dict[case]['states']
    tp, obs, i_eval, i_trans = process_observation(
        observation, agent_idx=agent_idx,
        DT=np.abs(case_dict[case]['conf']['simulation']['dt']),
        N_DRL=case_dict[case]['conf']['simulation']['ndrl'],
        sample_freq=case_dict[case]['conf']['runner']['vars_record_freq'],
        Re=np.abs(case_dict[case]['conf']['simulation']['viscosity']),
        utau=case_dict[case]['conf']['runner']['u_tau'],
    )

    obs = obs[env_idx]
    obs = np.reshape(obs, (-1, obs.shape[-1]))

    if case_dict[case]['conf']['runner']['normalize_input'] != 'utau':
        obs /= case_dict[case]['conf']['runner']['u_tau']

    max_amp = np.round(np.max(np.abs(obs), axis=0), 1)

    xx, yy, pdf = PDF(
        obs[:, 0], obs[:, 1],
        xmin=-max_amp[0], xmax=max_amp[0],
        ymin=-max_amp[1], ymax=max_amp[1],
        x_grid=N_mesh, y_grid=N_mesh,
    )
    levels = np.linspace(np.nanmin(pdf), np.nanmax(pdf), 20)
    axs[il].contourf(xx, yy, pdf, levels=levels[2:], cmap=cmap, extend='both')

    axs[il].set(
        title=case_name_,
        xlabel=r"$u^+_t |_{y^+ = 15}$" if il == len(case_name) - 1 else '',
        ylabel=r"$v^+_n |_{y^+ = 15}$",
    )
    print(f"Case {case_name_} done")

if SAVE_IMG:
    out = fig_path + f'{Head}_Observation_Deterministic.jpg'
    plt.savefig(out, dpi=300)
    print(f'Saved {out}')
if SHOW_IMG:
    plt.show()
