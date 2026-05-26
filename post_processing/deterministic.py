import os
import copy
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

plt_setUp()

run_path = "../runs/"
fig_path = "Figs/"
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

# --- Style ---
STYLE_OC_RED_A06  = make_style(cc.red,       0.6, 'X', 'OC')
STYLE_DDPG_GR_A06 = make_style(cc.deepgreen, 0.6, 'D', 'DDPG')
STYLE_PPO_BL_A06  = make_style(cc.blue,      0.6, 'o', 'PPO')

# --- Case Configuration ---
SAVE_IMG = True
Head = 'Reth365'
case_tuple = [
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
axs.set(xlabel=r"$t^+$", xlim=[-1, 1501], ylim=[-1, 40], ylabel=r"$R [\%]$")
axs.legend(prop={"size": 12}, loc='lower right')

if SAVE_IMG:
    out = fig_path + f'{Head}_Reward_Deterministic.jpg'
    plt.savefig(out, dpi=300)
    print(f'Saved {out}')
plt.show()

# ---------------------------------------------------------------------------
# Reward Monitor (per-component reward time series, all envs)
# ---------------------------------------------------------------------------
COLORS_COMP = ['#2E59A7', '#D23918', '#2CA02C', '#9467BD', '#8C564B']
label_and_scale= [(r'$\tau_w$', 'linear'), (r"$|p'_w v_w|$", "log"), (r'$|\rho v^3_w|$', "log"),]

for il, case in enumerate(case_dict.keys()):
    case_path  = os.path.join(run_path, str(case))
    save_path  = os.path.join(case_path, case_dict[case]['run_name'])
    run_name   = case_dict[case]['run_name']

    time, reward_array, reward_names = load_reward_monitor_case(
        case_path, save_path, run_name=run_name, verbose=True
    )
    Re=np.abs(float(case_dict[case]['conf']['simulation']['viscosity']))
    utau=float(case_dict[case]['conf']['runner']['u_tau'])
    t_star = (1/Re)/ utau**2
    time -= time[0]  # align the start time to 0
    time = time / t_star

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
plt.show()

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
plt.show()
