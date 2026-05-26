#!/usr/bin/env python3
"""Plot reward monitor time series from a Nek-style text file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_reward_monitor(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8") as handle:
        header_line = handle.readline().strip()

    if not header_line.startswith("#"):
        raise ValueError(f"Expected a commented header line in {path}")

    columns = header_line.lstrip("#").split()
    data = np.loadtxt(path, comments="#")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != len(columns):
        raise ValueError(
            f"Column count mismatch in {path}: header has {len(columns)} columns, "
            f"data has {data.shape[1]}"
        )

    series = {name: data[:, idx] for idx, name in enumerate(columns)}

    if "time" not in series:
        raise ValueError(f"No 'time' column found in {path}")

    return series["time"], series


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot reward monitor values as a function of time."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=(
            "/home/yuninw/codes/drl/1_Nek/nek_power_saving/"
            "runs/199835/env_001/reward_monitor.dat"
        ),
        help="Path to reward_monitor.dat",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/home/yuninw/codes/drl/1_Nek/nek_power_saving/post_processing/Figs"
        ),
        help="Directory where the figure will be saved",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output filename (default: derived from the input file name)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_file = Path(args.input_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    time, series = read_reward_monitor(input_file)
    component_names = [name for name in series if name.startswith("rwd_")]

    if "rwd" in series:
        total_reward = series["rwd"]
        total_label = "rwd"
    elif component_names:
        total_reward = np.sum([series[name] for name in component_names], axis=0)
        total_label = "rwd (sum of components)"
    else:
        raise ValueError(f"No reward columns found in {input_file}")

    output_name = args.output_name
    if output_name is None:
        output_name = f"{input_file.stem}_rwd_components.png"

    output_path = output_dir / output_name

    nrows = 1 + len(component_names)
    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(10, max(4.0, 2.6 * nrows)),
        sharex=True,
    )

    if nrows == 1:
        axes = [axes]

    axes[0].plot(time, total_reward, lw=1.8, color="#2E59A7")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(total_label)
    axes[0].set_title(f"{input_file.name} reward components")
    axes[0].grid(True, alpha=0.3)

    for axis, name in zip(axes[1:], component_names):
        axis.plot(time, series[name], lw=1.6, color="#D23918")
        axis.set_ylabel(name)
        axis.set_title(f"{name} component")
        axis.set_yscale("log")
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)

    if args.show:
        plt.show()
    else:
        plt.close(figure)

    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
