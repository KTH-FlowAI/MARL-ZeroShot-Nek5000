#!/usr/bin/env python
"""Live monitor for ng-val (2001) and ng-full (2002) training runs.
Plots total reward + three components (R_tau, R_pw, R_v3) when available.
Run:  python utils/monitor_runs.py
      python utils/monitor_runs.py --out /tmp/my_plot.png
"""
import argparse
import glob
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from omegaconf import OmegaConf
import shutil
import os

BASE = "/p/project1/deepwing/polsm/11-MARL-ZeroShot-Nek5000/runs"
OUT  = "/p/project1/deepwing/polsm/11-MARL-ZeroShot-Nek5000/utils/monitor_runs.png"

RUNS = {
    "2001 ONLY WSE  (α=1 β=0 γ=0)": f"{BASE}/2001/train/history",
    "2002 WSE+pv+v^3 (α=1 β=1 γ=1)": f"{BASE}/2002/train/history",
}
# Professional, colorblind-friendly palette
PALETTE = ["#ff0000", "#2d7600", "#3d00cd", "#d28c00", "#e41a1c"]
COLORS = PALETTE
COMP_STYLE = {
    'mean_R_tau': dict(color="tab:green",  ls="-",  label="R_τ  wallshear"),
    'mean_R_pw':  dict(color="tab:red",    ls="--", label="R_pw  press·vel"),
    'mean_R_v3':  dict(color="tab:purple", ls=":",  label="R_v3  kin. energy"),
}

def load_latest(history_dir, pattern):
    files = sorted(glob.glob(f"{history_dir}/{pattern}"))
    if not files:
        return None
    return pd.read_csv(files[-1], parse_dates=["timestamp"])

def smooth(x, w=1):
    return pd.Series(x).rolling(w, min_periods=1, center=True).mean().values

def has_components(df):
    return 'mean_R_tau' in df.columns and df['mean_R_tau'].notna().any()

def main(out_path):
    # Configure LaTeX-like rendering for paper figures, fall back if unavailable
    if shutil.which("latex") is not None:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        })
    else:
        # LaTeX not available on this system; use serif fonts and similar sizes
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        })
    # ---- decide layout ----------------------------------------
    datasets = {}
    for label, hdir in RUNS.items():
        df = load_latest(hdir, "rewards_aggregated_*.csv")
        if df is not None and len(df) > 1:
            datasets[label] = df

    # If a user config is provided, try to read episode length (nb_interactions)
    conf = None
    global_nb_interactions = None
    if hasattr(main, "user_conf") and main.user_conf:
        try:
            conf = OmegaConf.load(main.user_conf)
            if 'runner' in conf and 'nb_interactions' in conf.runner:
                global_nb_interactions = int(conf.runner.nb_interactions)
        except Exception:
            global_nb_interactions = None

    with_comp = any(has_components(df) for df in datasets.values())

    # Single-column layout for paper (stacked panels)
    if with_comp:
        nrows = 5  # total reward + 3 components + throughput
        # 4:3 aspect ratio for paper figures (width x height)
        fig = plt.figure(figsize=(10,15))
        gs = gridspec.GridSpec(nrows, 1, figure=fig, hspace=0.45)
        ax_rwd = fig.add_subplot(gs[0, 0])
        ax_comp = {
            'mean_R_tau': fig.add_subplot(gs[1, 0]),
            'mean_R_pw':  fig.add_subplot(gs[2, 0]),
            'mean_R_v3':  fig.add_subplot(gs[3, 0]),
        }
        ax_ep = None
        ax_rate = fig.add_subplot(gs[4, 0])
    else:
        nrows = 3  # total reward + episode-level + throughput
        # 4:3 aspect ratio for paper figures (width x height)
        fig = plt.figure(figsize=(7, 10))
        gs = gridspec.GridSpec(nrows, 1, figure=fig, hspace=0.45)
        ax_rwd = fig.add_subplot(gs[0, 0])
        ax_comp = {}
        ax_ep = fig.add_subplot(gs[1, 0])
        ax_rate = fig.add_subplot(gs[2, 0])

    if not datasets:
        fig.text(0.5, 0.5, "No data yet — jobs still initialising",
                 ha="center", va="center", fontsize=14)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved: {out_path}  (no data yet)")
        return


    for (label, hdir), color in zip(RUNS.items(), COLORS):
        df = datasets.get(label)
        if df is None:
            continue

        steps = np.arange(len(df))
        rwd   = df["mean_reward"].values

        # total reward
        ax_rwd.plot(steps, rwd, alpha=0.2, color=color, lw=0.8)
        ax_rwd.plot(steps, smooth(rwd), color=color, lw=2.0, label=label)

        # component panels (only when columns present)
        if has_components(df):
            for col, style in COMP_STYLE.items():
                ax = ax_comp.get(col)
                if ax is None:
                    continue
                vals = df[col].values
                # Use the dataset color consistently across all panels
                ax.plot(steps, vals, alpha=0.2, color=color, lw=0.8)
                ax.plot(steps, smooth(vals), color=color,
                        lw=1.8, ls=style['ls'], label=label)

        # episode-level (only in fallback layout)
        if ax_ep is not None:
            eps = load_latest(hdir, "rewards_episodes_*.csv")
            if eps is not None and len(eps) > 1:
                ax_ep.plot(eps["total_steps"].values, eps["mean_reward"].values,
                           "o-", color=color, lw=1.5, markersize=4, label=label)

        # throughput
        dt   = df["timestamp"].diff().dt.total_seconds().dropna()
        tput = (1.0 / dt.replace(0, np.nan)).rolling(100, min_periods=1).mean()
        ax_rate.plot(np.arange(len(tput)), tput.values,
                     color=color, lw=1.2, alpha=0.8, label=label)

        # Per-run: try to read the run's saved config (current_conf.yml) and draw
        # vertical lines at episode starts using the run's color.
        run_conf_path = os.path.join(hdir, 'current_conf.yml')
        run_nb_interactions = None
        if os.path.exists(run_conf_path):
            try:
                rc = OmegaConf.load(run_conf_path)
                if 'runner' in rc and 'nb_interactions' in rc.runner:
                    run_nb_interactions = int(rc.runner.nb_interactions)
            except Exception:
                run_nb_interactions = None
        # Fallback to global --conf if run-specific not found
        if run_nb_interactions is None:
            run_nb_interactions = global_nb_interactions

        if run_nb_interactions is not None:
            steps_len = len(df)
            starts = np.arange(0, steps_len, run_nb_interactions)
            # skip the initial 0 to avoid clutter; draw lines at subsequent episode starts
            for s in starts[1:]:
                ax_rwd.axvline(s, color=color, ls='--', lw=0.9, alpha=0.9)
                # also draw on component panels if present
                if has_components(df):
                    for ax in ax_comp.values():
                        ax.axvline(s, color=color, ls='--', lw=0.6, alpha=0.7)

    # (Per-run episode boundaries are drawn inside the loop above)

    # ---- decorations ------------------------------------------
    ax_rwd.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
    ax_rwd.set_xlabel("RL step (across all episodes)")
    ax_rwd.set_ylabel("Mean total reward")
    ax_rwd.set_title("Combined reward R = α·R_τ + β·R_pw + γ·R_v3")
    ax_rwd.legend(fontsize=9); ax_rwd.grid(True, alpha=0.3)

    if with_comp:
        titles = {
            'mean_R_tau': r"R_τ = 1 − τ_w / τ_ref   (drag reduction)",
            'mean_R_pw':  r"R_{pw} = −⟨p′v⟩ / τ_ref   (press.·vel. cost)",
            'mean_R_v3':  r"R_{v3} = −½⟨|v|³⟩ / τ_ref  (kin. energy cost)",
        }
        for col, ax in ax_comp.items():
            ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
            ax.set_xlabel("RL step"); ax.set_ylabel("Component value")
            ax.set_title(titles[col], fontsize=9)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    elif ax_ep is not None:
        ax_ep.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
        ax_ep.set_xlabel("Total steps"); ax_ep.set_ylabel("Episode mean reward")
        ax_ep.set_title("Episode-level reward")
        ax_ep.legend(fontsize=9); ax_ep.grid(True, alpha=0.3)

    ax_rate.set_xlabel("Step index"); ax_rate.set_ylabel("RL steps / sec")
    ax_rate.set_title("Throughput (rolling 100-step avg)")
    ax_rate.set_ylim(bottom=4.5)  # start y-axis at 0 for better readability 
    ax_rate.legend(fontsize=9); ax_rate.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT)
    p.add_argument("--conf", default=None,
                   help="Path to user config YAML (used to read nb_interactions/ndrl)")
    args = p.parse_args()
    # Attach config path to main for later reading
    main.user_conf = args.conf
    main(args.out)

# Define default attribute so editors/static-checkers don't flag accesses
main.user_conf = None
