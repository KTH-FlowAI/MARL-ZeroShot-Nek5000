#!/usr/bin/env python3
"""Compare reward_monitor*.dat terms across selected runs.

The reward monitor files are written with wrapped numeric records: one logical
row can span multiple text lines. This script reads all numeric tokens after
the header and reshapes them by the number of header columns.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

#[MOD] Evaluation runs now live under runs/<case>/eval/env_XXX/ (see
#[MOD] src/evaluate.py). Reward-monitor paths carry the eval/ level.
DEFAULT_CASES = [
    (
        "mc_nes_nek",
        ("runs/mc_nes_nek/eval/env_002/reward_monitor00000.dat",),
    ),
    (
        "mc_shapvel",
        (
            # "runs/mc_shapvel/eval/env_002/reward_monitor00000.dat",
            "runs/mc_shap_vel/eval/env_002/reward_monitor00000.dat",
        ),
    ),
    (
        "oc_mc",
        (
            # "runs/oc_mc/eval/env_002/reward_monitor00000.dat",
            "runs/oc-mc/eval/env_001/reward_monitor00000.dat",
        ),
    ),
]


@dataclass
class CaseData:
    label: str
    path: Path
    header: list[str]
    data: np.ndarray
    t_star: float
    conf_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot reward terms from reward_monitor*.dat files. By default, "
            "each rwd_* column gets its own subplot and cases are overlaid."
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help=(
            "Case label and reward_monitor path. Relative paths are resolved "
            "from the repository root. Repeat to override the default cases."
        ),
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="Columns to plot. Default: all columns beginning with 'rwd_'.",
    )
    parser.add_argument(
        "--x",
        choices=("tplus", "time", "elapsed", "sample", "i_evolv"),
        default="tplus",
        help="X-axis coordinate. Default: tplus, computed as (time - time[0]) / t_star.",
    )
    parser.add_argument(
        "--trim-mode",
        choices=("time-span", "sample-count", "none"),
        default="time-span",
        help=(
            "How to match curve lengths. Default: time-span trims every case "
            "to the shortest final x value after scaling."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "post_processing" / "Figs",
        help="Directory for output figure. Default: post_processing/Figs.",
    )
    parser.add_argument(
        "--outfile",
        default="reward_monitor_terms_compare.png",
        help="Output image filename. Default: reward_monitor_terms_compare.png.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO_ROOT / "post_processing" / "Figs" / "reward_monitor_terms.npz",
        help="Path to the processed-data cache (.npz). Default: post_processing/Figs.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load processed data from --cache instead of re-reading the .dat files.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively after saving.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def resolve_first_existing(candidates: tuple[str, ...]) -> Path:
    paths = [resolve_path(candidate) for candidate in candidates]
    for path in paths:
        if path.exists():
            return path

    joined = "\n  ".join(str(path) for path in paths)
    raise FileNotFoundError(f"None of these candidate paths exists:\n  {joined}")


def parse_case_specs(case_specs: list[str] | None) -> list[tuple[str, Path]]:
    if case_specs is None:
        return [
            (label, resolve_first_existing(candidates))
            for label, candidates in DEFAULT_CASES
        ]

    cases: list[tuple[str, Path]] = []
    for spec in case_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --case {spec!r}; expected LABEL=PATH")
        label, path_text = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Invalid --case {spec!r}; empty label")
        path = resolve_path(path_text.strip())
        if not path.exists():
            raise FileNotFoundError(path)
        cases.append((label, path))

    if not cases:
        raise ValueError("No cases were specified")
    return cases


def read_reward_monitor(path: Path) -> tuple[list[str], np.ndarray]:
    header: list[str] | None = None
    values: list[float] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                tokens = stripped.lstrip("#").split()
                if tokens:
                    header = tokens
                continue

            for token in stripped.split():
                values.append(float(token.replace("D", "E").replace("d", "e")))

    if header is None:
        raise ValueError(f"{path} does not contain a commented header line")
    if not values:
        raise ValueError(f"{path} does not contain numeric data")

    ncol = len(header)
    remainder = len(values) % ncol
    if remainder != 0:
        raise ValueError(
            f"{path} has {len(values)} numeric values, which is not divisible "
            f"by the {ncol} header columns"
        )

    data = np.asarray(values, dtype=float).reshape((-1, ncol))
    return header, data


def find_case_conf(path: Path) -> Path:
    run_dir = path.parent.parent
    candidates = [
        run_dir / "current_conf.yml",
        #[MOD] History was consolidated under history/current_history
        #[MOD] (see src/initial.py:preserve_and_clean_train). Check it first,
        #[MOD] keeping the old locations as fallbacks for legacy runs.
        run_dir / "history" / "current_history" / "current_conf.yml",
        run_dir / "history" / "current_conf.yml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    joined = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find a case config for {path}:\n  {joined}")


def read_t_star(path: Path) -> tuple[float, Path]:
    conf_path = find_case_conf(path)
    with conf_path.open("r", encoding="utf-8") as handle:
        conf = yaml.load(handle, Loader=yaml.FullLoader)

    re_value = abs(float(conf["simulation"]["viscosity"]))
    u_tau = float(conf["runner"]["u_tau"])
    if re_value <= 0.0:
        raise ValueError(f"{conf_path} has non-positive viscosity scale: {re_value}")
    if u_tau <= 0.0:
        raise ValueError(f"{conf_path} has non-positive u_tau: {u_tau}")

    t_star = (1.0 / re_value) / (u_tau**2)
    return t_star, conf_path


def save_processed_data(loaded_cases: list[CaseData], cache_path: Path) -> Path:
    """Cache the processed CaseData list to a single .npz file.

    Numeric arrays are stored natively; metadata (labels, paths, headers,
    t_star) rides along in an object array so the cache is self-contained.
    """
    cache_path = Path(cache_path)
    if cache_path.suffix != ".npz":
        cache_path = cache_path.with_suffix(".npz")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    payload = [
        {
            "label": case.label,
            "path": str(case.path),
            "header": list(case.header),
            "data": case.data,
            "t_star": float(case.t_star),
            "conf_path": str(case.conf_path),
        }
        for case in loaded_cases
    ]
    np.savez(cache_path, cases=np.array(payload, dtype=object))
    print(f"Saved processed data for {len(loaded_cases)} cases to {cache_path}")
    return cache_path


def load_processed_data(cache_path: Path) -> list[CaseData]:
    """Reconstruct the CaseData list written by save_processed_data."""
    cache_path = Path(cache_path)
    if cache_path.suffix != ".npz":
        cache_path = cache_path.with_suffix(".npz")

    with np.load(cache_path, allow_pickle=True) as bundle:
        payload = bundle["cases"]

    loaded_cases = [
        CaseData(
            label=str(entry["label"]),
            path=Path(entry["path"]),
            header=list(entry["header"]),
            data=np.asarray(entry["data"], dtype=float),
            t_star=float(entry["t_star"]),
            conf_path=Path(entry["conf_path"]),
        )
        for entry in payload
    ]
    print(f"Loaded processed data for {len(loaded_cases)} cases from {cache_path}")
    return loaded_cases


def select_columns(
    header: list[str],
    requested: list[str] | None,
    common_columns: set[str],
) -> list[str]:
    if requested is not None:
        missing = [name for name in requested if name not in common_columns]
        if missing:
            raise ValueError(
                "Requested columns are not present in every case: "
                + ", ".join(missing)
            )
        return requested

    columns = [name for name in header if name.startswith("rwd_")]
    if not columns:
        columns = [
            name
            for name in header
            if name not in {"time", "i_evolv"} and name in common_columns
        ]
    return [name for name in columns if name in common_columns]


def x_values(case: CaseData, x_name: str) -> tuple[np.ndarray, str]:
    column_index = {name: idx for idx, name in enumerate(case.header)}

    if x_name == "sample":
        return np.arange(case.data.shape[0]), "sample"
    if x_name == "elapsed":
        if "time" not in column_index:
            raise ValueError("Cannot use --x elapsed because column 'time' is missing")
        time = case.data[:, column_index["time"]]
        return time - time[0], "time - time[0]"
    if x_name == "tplus":
        if "time" not in column_index:
            raise ValueError("Cannot use --x tplus because column 'time' is missing")
        time = case.data[:, column_index["time"]]
        return (time - time[0]) / case.t_star, r"$t^+$"
    if x_name not in column_index:
        raise ValueError(f"Cannot use --x {x_name!r}; column is missing")
    return case.data[:, column_index[x_name]], x_name


def plot_slice(x: np.ndarray, trim_limit: float | int | None) -> slice:
    if trim_limit is None:
        return slice(None)

    stop = int(np.searchsorted(x, trim_limit, side="right"))
    return slice(0, max(stop, 1))


def trim_limit_for_cases(
    loaded_cases: list[CaseData],
    x_name: str,
    trim_mode: str,
) -> float | int | None:
    if trim_mode == "none":
        return None
    if trim_mode == "sample-count":
        return min(case.data.shape[0] for case in loaded_cases)

    x_end = []
    for case in loaded_cases:
        x, _ = x_values(case, x_name)
        x_end.append(float(np.nanmax(x)))
    return min(x_end)


def prettify_column(name: str) -> str:
    labels = {
        # "rwd_tau": r"$rwd\_tau$",
        "rwd_tau": r"$dU/dy$",
        "rwd_dvdx": r"$d v_w / dx$",
        # "rwd_pw": r"$rwd\_pw$",
        # "rwd_v3": r"$rwd\_v3$",
        # "rwd_uv": r"$rwd\_uv$",
        # "i_evolv": r"$i\_evolv$",
    }
    return labels.get(name, name)


def plot_cases(
    loaded_cases: list[CaseData],
    columns: list[str],
    x_name: str,
    trim_mode: str,
    outpath: Path,
    show: bool,
) -> None:
    ncols = 2 if len(columns) > 4 else 1
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.3 * ncols, 2.8 * nrows),
        sharex=False,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    trim_limit = trim_limit_for_cases(loaded_cases, x_name, trim_mode)

    for ax, column in zip(flat_axes, columns):
        xlabel = x_name
        for case in loaded_cases:
            index = {name: idx for idx, name in enumerate(case.header)}
            x, xlabel = x_values(case, x_name)
            y = case.data[:, index[column]]
            trim = (
                slice(0, int(trim_limit))
                if trim_mode == "sample-count" and trim_limit is not None
                else plot_slice(x, trim_limit)
            )
            if column == "rwd_tau":
                ax.plot(x[trim], y[trim] * 2900, linewidth=1.2, label=case.label)
            else:
                ax.plot(x[trim], y[trim], linewidth=1.2, label=case.label)


        ax.set_title(prettify_column(column))
        # ax.set_yscale("symlog", linthresh=1e-6, base=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(prettify_column(column))
        ax.grid(True, alpha=0.3)
        if ax.get_yscale() == "linear":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        ax.legend(loc="best", fontsize="small", framealpha=0.5)

    for ax in flat_axes[len(columns) :]:
        ax.remove()

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.suptitle(f"reward_monitor00000.dat terms vs {xlabel}", y=0.99)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=min(len(labels), 3),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=250, bbox_inches="tight")
    print(f"Wrote {outpath}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    args = parse_args()

    if args.use_cache:
        loaded_cases = load_processed_data(args.cache)
    else:
        cases = parse_case_specs(args.case)
        loaded_cases = []

        for label, path in cases:
            header, data = read_reward_monitor(path)
            t_star, conf_path = read_t_star(path)
            loaded_cases.append(CaseData(label, path, header, data, t_star, conf_path))
            print(
                f"Loaded {label}: {path} ({data.shape[0]} rows, "
                f"t_star={t_star:.6g} from {conf_path})"
            )

        save_processed_data(loaded_cases, args.cache)

    common_columns = set(loaded_cases[0].header)
    for case in loaded_cases[1:]:
        common_columns &= set(case.header)

    # columns = select_columns(loaded_cases[0].header, args.columns, common_columns)
    columns = ['rwd_tau', 'rwd_dvdx']  # Hardcoded columns to plot
    if not columns:
        raise ValueError("No columns selected for plotting")

    outpath = args.outdir / args.outfile
    plot_cases(loaded_cases, columns, args.x, args.trim_mode, outpath, args.show)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
