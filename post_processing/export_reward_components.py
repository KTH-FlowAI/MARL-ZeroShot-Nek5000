#!/usr/bin/env python3
"""Export training reward components for the selected cases.

Edit ``CASE_TUPLES`` below, then run from anywhere with::

    python post_processing/export_reward_components.py

By default this writes ``reward_components.npz``.  Use ``--format mat`` for
a MATLAB file, or ``--format both`` to write both formats.  The bundle has
arrays named ``episode_*`` (one mean per episode), ``rolling_*`` (the same
one-episode moving mean used in the notebook), and optional ``step_*`` data.

The loader follows archived history rounds and an active ``train/history``
directory.  It ignores incomplete episodes, so the exported learning curves
are safe to use while training is still running.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from postlib.explore import initalize_case


# Customize this list: (directory name below runs/, legend/report label).
CASE_TUPLES = [
    ("mc_nes_v_nek", "mc-nes-vonly"),
    # ("mc_nes_nek", "MC-NES"),
    # ("mc_shap_vel", "MC-shaped velocity"),
    # ("oc-mc", "OC-MC"),
]

COMPONENTS = {
    "R_tau_pct": "mean_R_tau",  # drag-reduction term
    "R_pw_pct": "mean_R_pw",    # pressure-velocity cost
    "R_v3_pct": "mean_R_v3",    # kinetic-energy cost
}


def parse_case_spec(spec: str) -> tuple[str, str]:
    """Parse a command-line CASE=LABEL override."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Invalid case {spec!r}; expected CASE=LABEL."
        )
    case, label = (part.strip() for part in spec.split("=", 1))
    if not case or not label:
        raise argparse.ArgumentTypeError(
            f"Invalid case {spec!r}; both CASE and LABEL are required."
        )
    return case, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export reward-component learning curves to CSV files."
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case_spec,
        metavar="CASE=LABEL",
        help=(
            "Case to export; repeat for multiple cases. Supplying this option "
            "overrides CASE_TUPLES."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=REPO_ROOT / "runs",
        help="Directory containing case directories (default: <repo>/runs).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=SCRIPT_DIR / "Tables" / "reward_components",
        help="Directory for output files (default: post_processing/Tables/reward_components).",
    )
    parser.add_argument(
        "--format",
        choices=("npz", "mat", "both"),
        default="npz",
        help="Output format (default: npz).",
    )
    parser.add_argument(
        "--output-name",
        default="reward_components",
        help="Output basename without .npz/.mat (default: reward_components).",
    )
    parser.add_argument(
        "--include-step-data",
        action="store_true",
        help="Also write every per-interaction component value (can be large).",
    )
    return parser.parse_args()


def values_for(case_data: dict, suffix: str = "") -> dict[str, np.ndarray]:
    """Return component arrays, failing clearly for incomplete component logs."""
    missing = [key for key in COMPONENTS.values() if f"{key}{suffix}" not in case_data]
    if missing:
        raise KeyError(
            "no complete reward-component data found "
            f"(missing: {', '.join(missing)})"
        )
    return {
        output_name: np.asarray(case_data[f"{source_name}{suffix}"], dtype=float)
        for output_name, source_name in COMPONENTS.items()
    }


def component_frame(
    case: str,
    label: str,
    episode: np.ndarray,
    component_values: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build one wide learning-curve table for a case."""
    lengths = {len(episode), *(len(values) for values in component_values.values())}
    if len(lengths) != 1:
        raise ValueError(f"{case}: component arrays have inconsistent lengths: {lengths}")

    return pd.DataFrame(
        {
            "case": case,
            "label": label,
            "episode": episode,
            **component_values,
        }
    )


def export_case(
    case: str,
    label: str,
    case_data: dict,
    include_step_data: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Create episode-level and rolling exports for one loaded case."""
    episode_values = values_for(case_data, "_eps")
    rolling_values = values_for(case_data, "_smooth")

    tables = {
        "episode": component_frame(
            case, label, np.asarray(case_data["comp_episode_idx"]), episode_values
        ),
        "rolling": component_frame(
            case, label, np.asarray(case_data["comp_episode"]), rolling_values
        ),
    }

    if include_step_data:
        step_values = values_for(case_data)
        n_steps = len(next(iter(step_values.values())))
        window_size = int(case_data["conf"]["runner"]["nb_interactions"])
        tables["step"] = component_frame(
            case,
            label,
            np.arange(n_steps) // window_size + 1,
            {
                "interaction": np.arange(n_steps) % window_size + 1,
                **step_values,
            },
        )

    summary = {
        "case": case,
        "label": label,
        "component_episodes": len(tables["episode"]),
        "reward_episodes": len(case_data["reward_mean"]),
        "last_R_tau_pct": tables["episode"]["R_tau_pct"].iloc[-1],
        "last_R_pw_pct": tables["episode"]["R_pw_pct"].iloc[-1],
        "last_R_v3_pct": tables["episode"]["R_v3_pct"].iloc[-1],
    }
    return tables, summary


def array_bundle(
    exported: dict[str, list[pd.DataFrame]], summaries: list[dict[str, object]]
) -> dict[str, np.ndarray]:
    """Flatten tables to named arrays that are convenient in NumPy and MATLAB."""
    bundle: dict[str, np.ndarray] = {}
    for table_name, frames in exported.items():
        if not frames:
            continue
        table = pd.concat(frames, ignore_index=True)
        for column in table.columns:
            bundle[f"{table_name}_{column}"] = table[column].to_numpy()

    summary = pd.DataFrame(summaries)
    for column in summary.columns:
        bundle[f"summary_{column}"] = summary[column].to_numpy()
    return bundle


def write_outputs(
    bundle: dict[str, np.ndarray], output_dir: Path, output_name: str, output_format: str
) -> None:
    """Write a single self-contained bundle in the requested portable format."""
    stem = Path(output_name).stem
    if not stem:
        raise ValueError("--output-name must contain at least one non-dot character.")

    if output_format in {"npz", "both"}:
        npz_path = output_dir / f"{stem}.npz"
        np.savez_compressed(npz_path, **bundle)
        print(f"Wrote {npz_path}")

    if output_format in {"mat", "both"}:
        mat_path = output_dir / f"{stem}.mat"
        savemat(mat_path, bundle, do_compression=True, long_field_names=True)
        print(f"Wrote {mat_path}")


def main() -> None:
    args = parse_args()
    cases = args.case if args.case is not None else CASE_TUPLES
    if not cases:
        raise ValueError("No cases selected. Add entries to CASE_TUPLES or use --case.")

    runs_dir = args.runs_dir.expanduser().resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    output_dir = args.outdir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = initalize_case(str(runs_dir), [case for case, _ in cases])

    exported = {"episode": [], "rolling": [], "step": []}
    summaries: list[dict[str, object]] = []
    for case, label in cases:
        try:
            tables, summary = export_case(
                case, label, loaded[case], args.include_step_data
            )
        except KeyError as exc:
            print(f"Skipping {case}: {exc}")
            continue

        for name, frame in tables.items():
            exported[name].append(frame)
        summaries.append(summary)

    if not summaries:
        raise RuntimeError("None of the selected cases contained complete component data.")

    bundle = array_bundle(exported, summaries)
    write_outputs(bundle, output_dir, args.output_name, args.format)
    print(
        "Arrays available: "
        + ", ".join(sorted(bundle))
    )


if __name__ == "__main__":
    main()
