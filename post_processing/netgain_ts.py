"""
Load and plot the per-chord-point NETGAIN time-series written by
nek_marl_meta_netgain.py (history/netgain_ts_<episode>.npz).

The env flushes the buffer every `netgain_io_freq` control steps (and at
reset/close), so a run produces a sequence of chunk files
netgain_ts_00000.npz, netgain_ts_00001.npz, ... each holding:
    episode : scalar           episode (restart) index the chunk belongs to
    step    : [n_step]         control-step index within that episode
    tau     : [n_step, nChordPts] wall-shear-stress term (tau_w)
    pw      : [n_step, nChordPts] pumping-power term      (|p'_w v_w|, >=0)
    v3      : [n_step, nChordPts] kinetic-energy term     (0.5 |v_w^3|)
    x       : [nChordPts]      chord x-coordinate of each column
    y       : [nChordPts]      wall y-coordinate of each column
    side    : [nChordPts]      'suction' / 'pressure' label per column

Usage:
    python netgain_ts.py --run ../runs/302000            # concat all chunks
    python netgain_ts.py --run ../runs/302000 --episode 1  # one episode only
    python netgain_ts.py --file path/to/netgain_ts_00000.npz
"""
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

COMPONENTS = ('tau', 'pw', 'v3')
LABELS = {'tau': r'$\tau_w$', 'pw': r"$|p'_w v_w|$", 'v3': r'$0.5|v_w^3|$'}


def load_netgain_ts(path):
    """Load one netgain_ts_*.npz chunk into a plain dict (sides decoded)."""
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out['side'] = np.array([str(s) for s in out['side']])
    return out


def load_run(run_dir, episode=None):
    """Concatenate all netgain_ts chunk files under <run_dir> in time order.

    The chord-point layout (x/y/side) is static across chunks, so we take it
    from the first chunk and stack tau/pw/v3/step along the time axis. Pass
    `episode` to restrict to a single episode's chunks.
    """
    pat = os.path.join(run_dir, '**', 'netgain_ts_*.npz')
    files = sorted(glob.glob(pat, recursive=True))  # filename io_iter is monotonic
    if not files:
        raise FileNotFoundError(f'No netgain_ts_*.npz under {run_dir}')

    chunks = [load_netgain_ts(f) for f in files]
    if episode is not None:
        chunks = [c for c in chunks if int(c.get('episode', -1)) == episode]
        if not chunks:
            raise FileNotFoundError(f'No chunks for episode {episode} under {run_dir}')

    base = chunks[0]
    data = {k: base[k] for k in ('x', 'y', 'side')}
    for k in ('step', 'tau', 'pw', 'v3'):
        data[k] = np.concatenate([c[k] for c in chunks], axis=0)
    data['episode'] = np.array([int(c.get('episode', -1)) for c in chunks])
    print(f"[netgain_ts] loaded {len(chunks)} chunk(s) from {run_dir}")
    return data


def _set_rc():
    plt.rc("font", family="serif")
    plt.rc("font", size=12)
    plt.rc("axes", labelsize=12, linewidth=1)
    plt.rc("legend", fontsize=12, handletextpad=0.1)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)


def plot_spacetime(data, out_png):
    """Space-time (chord x vs control step) contour for each component & side.

    Rows = suction/pressure sides, columns = tau/pw/v3. Columns are sorted by
    chord x within each side so the x-axis is monotonic.
    """
    _set_rc()
    sides = [s for s in ('suction', 'pressure') if np.any(data['side'] == s)]
    # Use a contiguous sample index for the y-axis: across concatenated chunks
    # the per-episode `step` can reset and is not monotonic.
    step = np.arange(data['tau'].shape[0])
    fig, axs = plt.subplots(len(sides), len(COMPONENTS),
                            figsize=(4 * len(COMPONENTS), 3.2 * len(sides)),
                            squeeze=False)
    for i, side in enumerate(sides):
        col = np.where(data['side'] == side)[0]
        order = col[np.argsort(data['x'][col])]
        x = data['x'][order]
        for j, comp in enumerate(COMPONENTS):
            ax = axs[i, j]
            fld = data[comp][:, order]          # [n_step, n_x]
            cf = ax.contourf(x, step, fld, levels=60, cmap='turbo')
            fig.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, shrink=1)
            if i == 0:
                ax.set_title(LABELS[comp])
            if j == 0:
                ax.set_ylabel(f'{side}\nsample index')
            if i == len(sides) - 1:
                ax.set_xlabel('x/c')
    fig.subplots_adjust(hspace=0.3, wspace=0.35)
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'[netgain_ts] wrote {out_png}')


def plot_mean_profile(data, out_png):
    """Time-averaged chord profile of each component, suction vs pressure."""
    _set_rc()
    sides = [s for s in ('suction', 'pressure') if np.any(data['side'] == s)]
    fig, axs = plt.subplots(1, len(COMPONENTS),
                            figsize=(4 * len(COMPONENTS), 3.5), squeeze=False)
    for j, comp in enumerate(COMPONENTS):
        ax = axs[0, j]
        for side in sides:
            col = np.where(data['side'] == side)[0]
            order = col[np.argsort(data['x'][col])]
            mean = data[comp][:, order].mean(axis=0)
            ax.plot(data['x'][order], mean, marker='.', ms=3, label=side)
        ax.set_title(LABELS[comp])
        ax.set_xlabel('x/c')
        ax.grid(True, alpha=0.3)
    axs[0, 0].set_ylabel('time-mean')
    axs[0, 0].legend()
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'[netgain_ts] wrote {out_png}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None,
                        help="run directory; concatenates all chunk files")
    parser.add_argument("--episode", type=int, default=None,
                        help="with --run, restrict to one episode index")
    parser.add_argument("--file", type=str, default=None,
                        help="explicit single netgain_ts_*.npz chunk to plot")
    parser.add_argument("--outdir", type=str, default="Figs")
    args = parser.parse_args()

    if args.file is not None:
        data = load_netgain_ts(args.file)
        tag = os.path.splitext(os.path.basename(args.file))[0]
    elif args.run is not None:
        data = load_run(args.run, episode=args.episode)
        tag = "netgain_ts_" + os.path.basename(os.path.normpath(args.run))
        if args.episode is not None:
            tag += f"_ep{args.episode:05d}"
    else:
        parser.error("give --file or --run")

    n_step, n_pts = data['tau'].shape
    print(f"[netgain_ts] {n_step} steps, {n_pts} chord points "
          f"({int(np.sum(data['side'] == 'suction'))} suction, "
          f"{int(np.sum(data['side'] == 'pressure'))} pressure)")

    os.makedirs(args.outdir, exist_ok=True)
    plot_spacetime(data, os.path.join(args.outdir, f"{tag}_spacetime.png"))
    plot_mean_profile(data, os.path.join(args.outdir, f"{tag}_meanprofile.png"))
