#!/usr/bin/env python3
"""Read, plot, and export a post-hoc GLL-zero-mean view of ``drlrec`` actions.

``drlrec`` stores the raw policy action before the solver's ZNMF operation.
This program leaves that array unchanged and creates a second array with its
global, element-local GLL-quadrature weighted spatial mean removed at every
record.  It is intended for analysis and visualisation; it is not a bitwise
replay of the solver's applied ``ACTIONS`` field.

Examples
--------
Use the default ``runs/`` directory and correct only controlled (``ipol > 0``)
wall nodes::

    python post_processing/znmf_actuation.py solo_timing_cmp small_wing_nes

Specify a different support and output locations::

    python post_processing/znmf_actuation.py runs/small_wing_nes \
        --support all --gll-nodes 4 --outdir Figs/znmf --export-dir data/znmf

Optional case-specific choices are read from a YAML mapping, for example::

    small_wing_nes:
      support: active
      gll_nodes: 4
    solo_timing_cmp:
      support: all
      gll_nodes: 5

Pass that file with ``--case-options``.  The MATLAB export always contains both
``act_raw`` and ``act_gll_zero_mean``, along with the weights, support mask,
and the mean removed at each record.  It also saves the plot-ready products
behind the observation-component distributions and the action maps (the
conditional mean action on the first two recorded observation components),
plus one-input sensitivity sweeps with the other component near its median.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep Matplotlib's cache out of a read-only home directory on batch nodes.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for import_dir in (SCRIPT_DIR, REPO_ROOT / "src"):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from read_drlrec import load_run  # noqa: E402
from postlib.drlrec import find_env_dirs  # noqa: E402
from lib.lglnodes import lglnodes  # noqa: E402


# The two tangential GLL directions of each Nek face.  The GLL weights must
# follow these directions; treating every face as (ix, iz) is wrong for faces
# 2/4 and 5/6.
FACE_TANGENTS = {
    1: ("ix", "iz"),
    2: ("iy", "iz"),
    3: ("ix", "iz"),
    4: ("iy", "iz"),
    5: ("ix", "iy"),
    6: ("ix", "iy"),
}
AXES = ("ix", "iy", "iz")
# Unlike ``drlrec_cases.ipynb``, this standalone tool does not resolve a
# case's friction velocity, so these axes intentionally remain in recorded
# solver units rather than being labelled with ``+`` superscripts.
OBSERVATION_LABELS = ("u_t (raw)", "v_n (raw)")
DEFAULT_OBSERVATION_CLIP = (0.5, 99.5)


@dataclass
class ZeroMeanResult:
    """The correction and enough metadata to reproduce it."""

    act_raw: np.ndarray
    act_zero_mean: np.ndarray
    removed_mean: np.ndarray
    residual_mean: np.ndarray
    weights: np.ndarray
    support: np.ndarray
    gll_nodes: dict[str, int]


@dataclass
class ObservationProducts:
    """Plot-ready action-map, distribution, and sensitivity arrays."""

    action_map: dict[str, Any]
    component_distribution: dict[str, Any]
    sensitivity: dict[str, Any]


def _as_node_count(value: Any) -> int | str:
    """Normalise an ``auto``/integer GLL-node option."""
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    try:
        node_count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("gll_nodes must be 'auto' or an integer >= 2") from exc
    if node_count < 2:
        raise ValueError("gll_nodes must be at least 2")
    return node_count


def _agent_column(agents: Any, name: str) -> np.ndarray:
    """Return an integer metadata column with a useful error on old records."""
    if name not in agents:
        raise ValueError(f"drlrec agent metadata has no '{name}' column")
    return np.asarray(agents[name], dtype=np.int64)


def _infer_gll_nodes(agents: Any, requested: int | str = "auto") -> dict[str, int]:
    """Infer GLL node counts from recorded metadata, or validate an override.

    The inference uses all recorded nodes, not only the selected correction
    support: a controlled subregion need not contain both endpoints of every
    tangential GLL line.
    """
    requested = _as_node_count(requested)
    iface = _agent_column(agents, "iface")
    bad_faces = np.setdiff1d(np.unique(iface), np.fromiter(FACE_TANGENTS, int))
    if bad_faces.size:
        raise ValueError(f"unsupported Nek face id(s): {bad_faces.tolist()}")

    result: dict[str, int] = {}
    for axis in AXES:
        values = _agent_column(agents, axis)
        if values.size == 0 or values.min() < 1:
            raise ValueError(f"{axis} indices must be positive 1-based GLL indices")
        nodes = int(values.max()) if requested == "auto" else requested
        if values.max() > nodes:
            raise ValueError(
                f"{axis} contains index {values.max()}, larger than --gll-nodes {nodes}"
            )
        result[axis] = nodes
    return result


def face_gll_weights(agents: Any, gll_nodes: int | str = "auto") -> tuple[np.ndarray, dict[str, int]]:
    """Return element-local, face-aware two-dimensional GLL weights.

    No physical-coordinate de-duplication is done.  ``drlrec`` records nodes
    on element faces, so repeated coordinates at element interfaces represent
    distinct element-local quadrature contributions.
    """
    nodes = _infer_gll_nodes(agents, gll_nodes)
    iface = _agent_column(agents, "iface")
    used_axes = {axis for face in np.unique(iface)
                 for axis in FACE_TANGENTS[int(face)]}
    underspecified = [axis for axis in used_axes if nodes[axis] < 2]
    if underspecified:
        raise ValueError(
            "cannot infer a GLL order from a single recorded tangential index "
            f"in {underspecified}; pass --gll-nodes N"
        )
    # A normal direction may have one recorded index (for example iy=1 on a
    # y-normal wall).  It is not a quadrature direction and needs no GLL rule.
    gll_weights = {axis: lglnodes(nodes[axis] - 1)[1] for axis in used_axes}
    indices = {axis: _agent_column(agents, axis) for axis in AXES}
    weights = np.empty(iface.size, dtype=np.float64)

    for face, (axis0, axis1) in FACE_TANGENTS.items():
        selected = iface == face
        if not np.any(selected):
            continue
        weights[selected] = (
            gll_weights[axis0][indices[axis0][selected] - 1]
            * gll_weights[axis1][indices[axis1][selected] - 1]
        )

    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("could not construct positive finite GLL weights")
    return weights, nodes


def zero_mean_action(rec: Any, support: str = "active",
                     gll_nodes: int | str = "auto") -> ZeroMeanResult:
    """Subtract the GLL-weighted mean from each action record on ``support``.

    ``support='active'`` means nodes with ``ipol > 0`` and mirrors the normal
    policy-region selection in the existing DRL-record plots.  ``support='all'``
    includes every node in the recording.  Unselected values are preserved in
    the corrected array so the export remains aligned with the original
    ``drlrec`` agent axis.
    """
    if support not in {"active", "all"}:
        raise ValueError("support must be 'active' or 'all'")

    raw = np.asarray(rec.act, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"rec.act must be 2D (record, agent), got {raw.shape}")
    weights, nodes = face_gll_weights(rec.agents, gll_nodes)
    if raw.shape[1] != weights.size:
        raise ValueError("rec.act agent dimension does not match agent metadata")

    selected = np.ones(weights.size, dtype=bool)
    if support == "active":
        selected = _agent_column(rec.agents, "ipol") > 0
    if not np.any(selected):
        raise ValueError(f"the '{support}' support contains no agents")

    selected_weights = weights[selected]
    total_weight = selected_weights.sum()
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("selected GLL weights do not have a positive finite sum")

    removed = raw[:, selected] @ selected_weights / total_weight
    corrected = raw.copy()
    corrected[:, selected] -= removed[:, None]
    residual = corrected[:, selected] @ selected_weights / total_weight
    return ZeroMeanResult(
        act_raw=raw,
        act_zero_mean=corrected,
        removed_mean=removed,
        residual_mean=residual,
        weights=weights,
        support=selected,
        gll_nodes=nodes,
    )


def _selected_observation_components(rec: Any, result: ZeroMeanResult) -> np.ndarray:
    """Return the first two observation components at the correction support.

    The action map and component distributions use exactly the nodes used for
    the GLL correction.  Keeping that selection common is important on cases
    with idle wall nodes: otherwise the observation distribution would include
    flow states which never contributed an action to the map.
    """
    obs = np.asarray(rec.obs, dtype=np.float64)
    if obs.ndim != 3 or obs.shape[2] < 2:
        raise ValueError(
            "rec.obs must have shape (record, agent, >=2) for an action map; "
            f"got {obs.shape}"
        )
    if obs.shape[:2] != result.act_raw.shape:
        raise ValueError(
            "rec.obs record/agent dimensions do not match the action array"
        )
    if result.support.shape != (obs.shape[1],):
        raise ValueError("correction support does not match the observation agent axis")
    return obs[:, result.support, :2].reshape(-1, 2)


def _observation_range(samples: np.ndarray, clip: tuple[float, float]) -> np.ndarray:
    """Percentile-clipped, non-degenerate limits for a two-component plane."""
    low, high = (float(value) for value in clip)
    if not 0.0 <= low < high <= 100.0:
        raise ValueError("observation clip must satisfy 0 <= low < high <= 100")
    limits = np.percentile(samples, (low, high), axis=0).T
    for component, (minimum, maximum) in enumerate(limits):
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError(f"observation component {component + 1} has no finite samples")
        if minimum == maximum:
            delta = max(abs(float(minimum)), 1.0) * 1.0e-6
            limits[component] = (minimum - delta, maximum + delta)
    return limits


def process_action_map(rec: Any, result: ZeroMeanResult, bins: int = 50,
                       min_count: int = 1,
                       clip: tuple[float, float] = DEFAULT_OBSERVATION_CLIP) -> dict[str, Any]:
    """Bin raw and GLL-zero-mean action against the first two observations.

    This is the per-environment counterpart to ``control_law`` in
    ``drlrec_cases.ipynb``.  ``act_raw_mean`` and
    ``act_gll_zero_mean_mean`` are both retained so post-processing does not
    hide how the zero-mean correction changes the learned action map.  The
    returned ``X``, ``Y``, and each action mean have matching shapes and can
    be passed directly to ``pcolormesh``/``pcolor`` after loading the MAT file.
    """
    if bins < 2:
        raise ValueError("action-map bins must be at least 2")
    if min_count < 1:
        raise ValueError("action-map min_count must be at least 1")

    obs = _selected_observation_components(rec, result)
    raw = result.act_raw[:, result.support].reshape(-1)
    corrected = result.act_zero_mean[:, result.support].reshape(-1)
    valid = np.isfinite(obs).all(axis=1) & np.isfinite(raw) & np.isfinite(corrected)
    obs, raw, corrected = obs[valid], raw[valid], corrected[valid]
    if obs.size == 0:
        raise ValueError("action map has no finite observation/action samples")

    limits = _observation_range(obs, clip)
    histogram_range = limits.tolist()
    counts, xedges, yedges = np.histogram2d(
        obs[:, 0], obs[:, 1], bins=bins, range=histogram_range
    )
    raw_sum, _, _ = np.histogram2d(
        obs[:, 0], obs[:, 1], bins=(xedges, yedges), weights=raw
    )
    corrected_sum, _, _ = np.histogram2d(
        obs[:, 0], obs[:, 1], bins=(xedges, yedges), weights=corrected
    )
    enough = counts >= min_count
    raw_mean = np.full_like(counts, np.nan, dtype=np.float64)
    corrected_mean = np.full_like(counts, np.nan, dtype=np.float64)
    np.divide(raw_sum, counts, out=raw_mean, where=enough)
    np.divide(corrected_sum, counts, out=corrected_mean, where=enough)
    xcentres, ycentres = 0.5 * (xedges[:-1] + xedges[1:]), 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xcentres, ycentres, indexing="ij")

    return {
        "X": X,
        "Y": Y,
        "act_raw_mean": raw_mean,
        "act_gll_zero_mean_mean": corrected_mean,
        "counts": counts.astype(np.int64),
        "xedges": xedges,
        "yedges": yedges,
        "xcentres": xcentres,
        "ycentres": ycentres,
        "range": limits,
        "clip": np.asarray(clip, dtype=np.float64),
        "bins": int(bins),
        "min_count": int(min_count),
        "nsample": int(obs.shape[0]),
        "labels": np.asarray(OBSERVATION_LABELS, dtype=object),
        "note": (
            "Mean raw and post-hoc GLL-zero-mean action in each bin of the "
            "first two raw recorded observation components."
        ),
    }


def process_component_distribution(rec: Any, result: ZeroMeanResult,
                                   bins: int = 90) -> dict[str, Any]:
    """Return marginal PDFs of the two observation components on the support.

    The sample selection follows ``process_action_map`` but each component is
    histogrammed independently, matching the marginal-distribution panels in
    ``drlrec_cases.ipynb``.  Counts are retained beside densities because a
    probability density alone cannot show the sampling level.
    """
    if bins < 2:
        raise ValueError("component-distribution bins must be at least 2")
    obs = _selected_observation_components(rec, result)
    output: dict[str, Any] = {
        "bins": int(bins),
        "labels": np.asarray(OBSERVATION_LABELS, dtype=object),
        "support_nagent": int(result.support.sum()),
    }
    sample_counts = []
    for component in range(2):
        values = obs[:, component]
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(
                f"observation component {component + 1} has no finite samples"
            )
        density, edges = np.histogram(values, bins=bins, density=True)
        counts, _ = np.histogram(values, bins=edges, density=False)
        index = component + 1
        output[f"pdf{index}"] = density
        output[f"counts{index}"] = counts.astype(np.int64)
        output[f"edges{index}"] = edges
        output[f"centres{index}"] = 0.5 * (edges[:-1] + edges[1:])
        sample_counts.append(values.size)
    output["nsample"] = np.asarray(sample_counts, dtype=np.int64)
    output["note"] = (
        "Marginal densities of the first two raw recorded observation "
        "components at nodes selected for the GLL zero-mean correction."
    )
    return output


def process_sensitivity(rec: Any, result: ZeroMeanResult, bins: int = 25,
                        band: float = 0.1, min_samples: int = 50) -> dict[str, Any]:
    """Sweep each observation input while the other remains near its median.

    This is the direct-record equivalent of ``sensitivity_sweep`` in
    ``drlrec_cases.ipynb``.  For input ``k``, samples are retained only when
    the other component lies within ``band`` standard deviations of its
    median.  The returned raw and zero-mean curves make the ZNMF impact
    visible without recomputing the figure after loading a MAT file.
    """
    if bins < 2:
        raise ValueError("sensitivity bins must be at least 2")
    if band <= 0.0:
        raise ValueError("sensitivity band must be positive")
    if min_samples < 1:
        raise ValueError("sensitivity min_samples must be at least 1")

    obs = _selected_observation_components(rec, result)
    raw = result.act_raw[:, result.support].reshape(-1)
    corrected = result.act_zero_mean[:, result.support].reshape(-1)
    valid = np.isfinite(obs).all(axis=1) & np.isfinite(raw) & np.isfinite(corrected)
    obs, raw, corrected = obs[valid], raw[valid], corrected[valid]
    if obs.size == 0:
        raise ValueError("sensitivity analysis has no finite observation/action samples")

    output: dict[str, Any] = {
        "bins": int(bins),
        "band": float(band),
        "min_samples": int(min_samples),
        "nsample": int(obs.shape[0]),
        "labels": np.asarray(OBSERVATION_LABELS, dtype=object),
        "note": (
            "For each input, the other observation component is held within "
            "band standard deviations of its median.  Curves show the mean "
            "raw and post-hoc GLL-zero-mean action."
        ),
    }
    for component in range(2):
        other = 1 - component
        other_values = obs[:, other]
        spread = float(np.std(other_values))
        median = float(np.median(other_values))
        # A constant held component is already at its median for every sample.
        near = (np.abs(other_values - median) <= np.finfo(float).eps
                if spread == 0.0 else np.abs(other_values - median) < band * spread)
        x_values = obs[near, component]
        raw_values = raw[near]
        corrected_values = corrected[near]
        index = component + 1
        output[f"n{index}"] = int(x_values.size)
        if x_values.size < min_samples:
            output[f"x{index}"] = np.array([])
            output[f"act_raw{index}"] = np.array([])
            output[f"act_gll_zero_mean{index}"] = np.array([])
            output[f"counts{index}"] = np.array([], dtype=np.int64)
            continue

        edges = np.percentile(x_values, np.linspace(1.0, 99.0, bins))
        # Repeated percentile edges give zero-width bins.  Remove them so the
        # sensitivity curve stays meaningful for a nearly constant input.
        edges = np.unique(edges)
        if edges.size < 2:
            output[f"x{index}"] = np.array([])
            output[f"act_raw{index}"] = np.array([])
            output[f"act_gll_zero_mean{index}"] = np.array([])
            output[f"counts{index}"] = np.array([], dtype=np.int64)
            continue
        bin_index = np.digitize(x_values, edges)
        occupied = [bin_id for bin_id in range(1, len(edges))
                    if np.any(bin_index == bin_id)]
        output[f"x{index}"] = np.asarray(
            [x_values[bin_index == bin_id].mean() for bin_id in occupied]
        )
        output[f"act_raw{index}"] = np.asarray(
            [raw_values[bin_index == bin_id].mean() for bin_id in occupied]
        )
        output[f"act_gll_zero_mean{index}"] = np.asarray(
            [corrected_values[bin_index == bin_id].mean() for bin_id in occupied]
        )
        output[f"counts{index}"] = np.asarray(
            [(bin_index == bin_id).sum() for bin_id in occupied], dtype=np.int64
        )
    return output


def process_observation_products(rec: Any, result: ZeroMeanResult,
                                 action_map_bins: int = 50,
                                 component_bins: int = 90,
                                 action_map_min_count: int = 1,
                                 observation_clip: tuple[float, float] = DEFAULT_OBSERVATION_CLIP,
                                 sensitivity_bins: int = 25,
                                 sensitivity_band: float = 0.1,
                                 sensitivity_min_samples: int = 50,
                                 ) -> ObservationProducts:
    """Build once the observation products that are both plotted and exported."""
    return ObservationProducts(
        action_map=process_action_map(
            rec, result, bins=action_map_bins, min_count=action_map_min_count,
            clip=observation_clip
        ),
        component_distribution=process_component_distribution(
            rec, result, bins=component_bins
        ),
        sensitivity=process_sensitivity(
            rec, result, bins=sensitivity_bins, band=sensitivity_band,
            min_samples=sensitivity_min_samples
        ),
    )


def _load_case_options(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    with path.open() as stream:
        options = yaml.safe_load(stream) or {}
    if not isinstance(options, dict):
        raise ValueError(f"{path}: expected a YAML mapping keyed by case name")
    for name, value in options.items():
        if not isinstance(value, dict):
            raise ValueError(f"{path}: options for '{name}' must be a mapping")
        unknown = set(value) - {"support", "gll_nodes"}
        if unknown:
            raise ValueError(f"{path}: '{name}' has unknown option(s): {sorted(unknown)}")
    return options


def _case_options(options: dict[str, dict[str, Any]], case_arg: str,
                  case_path: Path, default_support: str,
                  default_gll_nodes: int | str) -> tuple[str, int | str]:
    # Permit either the user spelling (e.g. wing/eval_case) or the directory
    # name, while keeping a supplied spelling as the more-specific choice.
    specific = options.get(case_arg, options.get(case_path.name, {}))
    support = specific.get("support", default_support)
    nodes = specific.get("gll_nodes", default_gll_nodes)
    if support not in {"active", "all"}:
        raise ValueError(f"{case_path}: support must be 'active' or 'all'")
    return support, _as_node_count(nodes)


def _resolve_case(case_arg: str, runs_dir: Path) -> Path:
    path = Path(case_arg).expanduser()
    if not path.is_absolute():
        path = runs_dir / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"case directory does not exist: {path}")
    return path


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.") or "case"


def export_mat(case_name: str, env_id: int, env_path: Path, rec: Any,
               result: ZeroMeanResult, products: ObservationProducts,
               support_name: str, export_dir: Path) -> Path:
    """Export one environment and the exact plot-ready products it used."""
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / f"{_safe_name(case_name)}_env{env_id:03d}_gll_zero_mean_action.mat"
    agent = {
        name: np.asarray(rec.agents[name])
        for name in ("nid", "ieg", "iface", "ix", "iy", "iz", "ipol", "x", "y", "z")
        if name in rec.agents
    }
    metadata = {
        "method": "posthoc_global_element_local_gll_zero_mean",
        "support": support_name,
        "source_drlrec_directory": str(env_path),
        "gll_nodes_ix": result.gll_nodes["ix"],
        "gll_nodes_iy": result.gll_nodes["iy"],
        "gll_nodes_iz": result.gll_nodes["iz"],
        "face_1_3_tangents": "ix,iz",
        "face_2_4_tangents": "iy,iz",
        "face_5_6_tangents": "ix,iy",
        "note": (
            "Post-hoc analysis view of raw drlrec policy action; it is not "
            "a bitwise replay of the solver-applied ACTIONS field."
        ),
    }
    sio.savemat(
        path,
        {
            "time": rec.time,
            "istep": rec.istep,
            "icycle": rec.icycle,
            "act_raw": result.act_raw,
            "act_gll_zero_mean": result.act_zero_mean,
            "removed_gll_mean": result.removed_mean,
            "gll_mean_residual": result.residual_mean,
            "gll_weight": result.weights,
            "correction_mask": result.support.astype(np.uint8),
            "action_map": products.action_map,
            "component_distribution": products.component_distribution,
            "sensitivity": products.sensitivity,
            "agent": agent,
            "metadata": metadata,
        },
        do_compression=True,
        long_field_names=True,
    )
    return path


def _pooled_values(results: list[tuple[int, Any, ZeroMeanResult, ObservationProducts]]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.concatenate([res.act_raw[:, res.support].ravel() for _, _, res, _ in results])
    corrected = np.concatenate(
        [res.act_zero_mean[:, res.support].ravel() for _, _, res, _ in results]
    )
    return raw, corrected


def _trace_environment(
        environments: list[tuple[int, Any, ZeroMeanResult, ObservationProducts]],
        trace_env_id: int | None) -> tuple[int, Any, ZeroMeanResult, ObservationProducts]:
    """Select the one environment used for a case's action-time-series panel."""
    if trace_env_id is None:
        return environments[0]
    for environment in environments:
        if environment[0] == trace_env_id:
            return environment
    available = ", ".join(f"{env_id:03d}" for env_id, *_ in environments)
    raise ValueError(
        f"requested --trace-env {trace_env_id:03d} is not available; "
        f"case has environment(s) {available}"
    )


def plot_cases(cases: list[tuple[str, str, list[tuple[int, Any, ZeroMeanResult, ObservationProducts]], dict[str, int]]],
               output: Path, bins: int, n_show: int, show: bool,
               trace_env_id: int | None = None) -> None:
    """Create one compact three-panel row per case."""
    ncase = len(cases)
    fig, axes = plt.subplots(ncase, 3, squeeze=False, figsize=(15, 3.7 * ncase))
    cmap = plt.get_cmap("tab10")

    for row, (case_name, support_name, environments, nodes) in enumerate(cases):
        ax_mean, ax_pdf, ax_trace = axes[row]
        for colour_index, (env_id, rec, result, _) in enumerate(environments):
            color = cmap(colour_index % 10)
            ax_mean.plot(rec.time, result.removed_mean, color=color,
                         label=f"env {env_id:03d}: removed mean")
            ax_mean.plot(rec.time, result.residual_mean, color=color, ls="--", lw=1.0,
                         label=f"env {env_id:03d}: residual")

        raw, corrected = _pooled_values(environments)
        lower = min(float(raw.min()), float(corrected.min()))
        upper = max(float(raw.max()), float(corrected.max()))
        if lower == upper:
            delta = max(abs(lower), 1.0) * 1.0e-6
            lower, upper = lower - delta, upper + delta
        edges = np.linspace(lower, upper, bins + 1)
        ax_pdf.hist(raw, bins=edges, density=True, histtype="step", lw=1.5,
                    color="#777777", label="raw")
        ax_pdf.hist(corrected, bins=edges, density=True, histtype="step", lw=1.5,
                    color="#D23918", label="GLL zero-mean")

        selected_env_id, selected_rec, selected_result, _ = _trace_environment(
            environments, trace_env_id
        )
        active_indices = np.flatnonzero(selected_result.support)
        trace_indices = np.array([], dtype=np.int64) if n_show == 0 else active_indices[
            np.unique(np.linspace(0, active_indices.size - 1,
                                  min(n_show, active_indices.size)).astype(int))
        ]
        for trace_index, agent_index in enumerate(trace_indices):
            ax_trace.plot(selected_rec.time,
                          selected_result.act_zero_mean[:, agent_index],
                          color=cmap(trace_index % 10), alpha=0.95, lw=1.0,
                          label=f"agent {agent_index}")

        node_text = ", ".join(f"{axis}={nodes[axis]}" for axis in AXES)
        ax_mean.set_title(f"{case_name}: removed GLL mean ({support_name}; {node_text})")
        ax_mean.set_xlabel("time")
        ax_mean.set_ylabel("actuation")
        ax_mean.axhline(0.0, color="0.75", lw=0.8, zorder=0)
        ax_mean.legend(fontsize=8)

        ax_pdf.set_title(f"{case_name}: selected-node distribution")
        ax_pdf.set_xlabel("actuation")
        ax_pdf.set_ylabel("density")
        ax_pdf.legend(fontsize=8)

        ax_trace.set_title(
            f"{case_name}: zero-mean action traces (env {selected_env_id:03d})"
        )
        ax_trace.set_xlabel("time")
        ax_trace.set_ylabel("actuation")
        ax_trace.axhline(0.0, color="0.75", lw=0.8, zorder=0)
        if n_show:
            ax_trace.legend(fontsize=8)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"[ZNMF] figure  : {output}")
    if show:
        plt.show()
    plt.close(fig)


def _symmetric_map_limit(
        cases: list[tuple[str, str, list[tuple[int, Any, ZeroMeanResult, ObservationProducts]], dict[str, int]]]
        ) -> float:
    """One signed colour scale for every raw/corrected action-map panel."""
    values = []
    for _, _, environments, _ in cases:
        for _, _, _, products in environments:
            action_map = products.action_map
            values.extend((action_map["act_raw_mean"],
                           action_map["act_gll_zero_mean_mean"]))
    finite = [np.abs(value[np.isfinite(value)]) for value in values]
    finite = [value for value in finite if value.size]
    return max((float(value.max()) for value in finite), default=1.0)


def plot_observation_products(
        cases: list[tuple[str, str, list[tuple[int, Any, ZeroMeanResult, ObservationProducts]], dict[str, int]]],
        output: Path, show: bool) -> None:
    """Plot raw/corrected action maps and observation component PDFs.

    There is one row per environment because action maps are conditional
    statistics of a trajectory; pooling time series from separate environments
    would create a map that no individual run produced.  The raw and corrected
    maps share one signed colour range to make their difference visible.
    """
    rows = [
        (case_name, env_id, products)
        for case_name, _, environments, _ in cases
        for env_id, _, _, products in environments
    ]
    if not rows:
        return

    limit = _symmetric_map_limit(cases)
    fig, axes = plt.subplots(len(rows), 3, squeeze=False,
                             figsize=(16, 4.2 * len(rows)))
    raw_mesh = corrected_mesh = None
    for row, (case_name, env_id, products) in enumerate(rows):
        raw_ax, corrected_ax, distribution_ax = axes[row]
        action_map = products.action_map
        distribution = products.component_distribution
        common = {
            "cmap": "RdBu_r", "vmin": -limit, "vmax": limit,
            "shading": "flat", "rasterized": True,
        }
        raw_mesh = raw_ax.pcolormesh(
            action_map["xedges"], action_map["yedges"],
            action_map["act_raw_mean"].T, **common
        )
        corrected_mesh = corrected_ax.pcolormesh(
            action_map["xedges"], action_map["yedges"],
            action_map["act_gll_zero_mean_mean"].T, **common
        )
        distribution_ax.stairs(distribution["pdf1"], distribution["edges1"],
                               color="#1764AB", lw=1.8,
                               label=OBSERVATION_LABELS[0])
        distribution_ax.stairs(distribution["pdf2"], distribution["edges2"],
                               color="#D23918", lw=1.8,
                               label=OBSERVATION_LABELS[1])

        descriptor = f"{case_name}, env {env_id:03d}"
        raw_ax.set(title=f"{descriptor}: raw action map",
                   xlabel=OBSERVATION_LABELS[0], ylabel=OBSERVATION_LABELS[1])
        corrected_ax.set(title=f"{descriptor}: GLL zero-mean action map",
                         xlabel=OBSERVATION_LABELS[0], ylabel=OBSERVATION_LABELS[1])
        distribution_ax.set(title=f"{descriptor}: component distributions",
                            xlabel="observation value", ylabel="density")
        distribution_ax.legend(fontsize=8)

    fig.tight_layout()
    if raw_mesh is not None and corrected_mesh is not None:
        raw_colorbar = fig.colorbar(raw_mesh, ax=axes[:, 0], pad=0.02)
        raw_colorbar.set_label("mean raw action")
        corrected_colorbar = fig.colorbar(corrected_mesh, ax=axes[:, 1], pad=0.02)
        corrected_colorbar.set_label("mean GLL zero-mean action")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"[ZNMF] products: {output}")
    if show:
        plt.show()
    plt.close(fig)


def plot_sensitivity_products(
        cases: list[tuple[str, str, list[tuple[int, Any, ZeroMeanResult, ObservationProducts]], dict[str, int]]],
        output: Path, show: bool) -> None:
    """Plot the action response to each input with the other input held near median."""
    rows = [
        (case_name, env_id, products)
        for case_name, _, environments, _ in cases
        for env_id, _, _, products in environments
    ]
    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 2, squeeze=False,
                             figsize=(11, 3.8 * len(rows)))
    for row, (case_name, env_id, products) in enumerate(rows):
        sensitivity = products.sensitivity
        for component, ax in enumerate(axes[row]):
            index = component + 1
            other = 1 - component
            x_values = sensitivity[f"x{index}"]
            raw_values = sensitivity[f"act_raw{index}"]
            corrected_values = sensitivity[f"act_gll_zero_mean{index}"]
            if x_values.size:
                ax.plot(x_values, raw_values, color="#777777", ls="--", lw=1.5,
                        label="raw")
                ax.plot(x_values, corrected_values, color="#D23918", lw=2.0,
                        label="GLL zero-mean")
            else:
                ax.text(0.5, 0.5,
                        f"only {sensitivity[f'n{index}']} held samples\n"
                        f"(need {sensitivity['min_samples']})",
                        transform=ax.transAxes, ha="center", va="center")
            ax.axhline(0.0, color="0.75", lw=0.8, zorder=0)
            ax.set(
                title=(f"{case_name}, env {env_id:03d}: action vs "
                       f"{OBSERVATION_LABELS[component]}"),
                xlabel=OBSERVATION_LABELS[component], ylabel="mean actuation",
            )
            if x_values.size:
                ax.legend(fontsize=8, loc="best")
            ax.text(0.02, 0.03,
                    f"{OBSERVATION_LABELS[other]} near median "
                    f"(band={sensitivity['band']:g} std)",
                    transform=ax.transAxes, fontsize=8, va="bottom")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"[ZNMF] sensitivity: {output}")
    if show:
        plt.show()
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", nargs="+", help="case directories, relative to --runs-dir or absolute")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs",
                        help="base directory for relative case names (default: %(default)s)")
    parser.add_argument("--support", choices=("active", "all"), default="active",
                        help="nodes whose GLL mean is removed (default: %(default)s)")
    parser.add_argument("--gll-nodes", default="auto",
                        help="GLL nodes per coordinate direction, or auto (default: %(default)s)")
    parser.add_argument("--case-options", type=Path,
                        help="YAML mapping of per-case support/gll_nodes overrides")
    parser.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "Figs" / "znmf_actuation",
                        help="directory for the PNG figure (default: %(default)s)")
    parser.add_argument("--export-dir", type=Path,
                        default=SCRIPT_DIR / "processed_data" / "znmf_actuation",
                        help="directory for one MATLAB file per case/environment (default: %(default)s)")
    parser.add_argument("--figure-name", default="gll_zero_mean_actuation.png",
                        help="PNG file name below --outdir (default: %(default)s)")
    parser.add_argument("--products-figure-name", default="gll_zero_mean_observation_products.png",
                        help="action-map/PDF PNG name below --outdir (default: %(default)s)")
    parser.add_argument("--sensitivity-figure-name", default="gll_zero_mean_sensitivity.png",
                        help="one-input sensitivity PNG name below --outdir (default: %(default)s)")
    parser.add_argument("--bins", type=int, default=80,
                        help="number of bins in each action PDF (default: %(default)s)")
    parser.add_argument("--action-map-bins", type=int, default=50,
                        help="bins per observation axis in each action map (default: %(default)s)")
    parser.add_argument("--component-bins", type=int, default=90,
                        help="bins in each observation-component PDF (default: %(default)s)")
    parser.add_argument("--action-map-min-count", type=int, default=1,
                        help="minimum samples required to draw an action-map bin (default: %(default)s)")
    parser.add_argument("--observation-clip", nargs=2, type=float,
                        metavar=("LOW", "HIGH"), default=DEFAULT_OBSERVATION_CLIP,
                        help="percentile limits for action-map axes (default: %(default)s)")
    parser.add_argument("--sensitivity-bins", type=int, default=25,
                        help="percentile bins in each one-input sensitivity sweep (default: %(default)s)")
    parser.add_argument("--sensitivity-band", type=float, default=0.1,
                        help="held-input half-band in standard deviations (default: %(default)s)")
    parser.add_argument("--sensitivity-min-samples", type=int, default=50,
                        help="minimum held-input samples before plotting a sweep (default: %(default)s)")
    parser.add_argument("--n-show", type=int, default=3,
                        help="zero-mean agent traces to show from one environment per case (default: %(default)s)")
    parser.add_argument("--trace-env", type=int,
                        help="environment id used for action traces; default is each case's first environment")
    parser.add_argument("--show", action="store_true", help="display the figure after saving it")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")
    if args.n_show < 0:
        raise ValueError("--n-show must be non-negative")
    if args.action_map_bins < 2:
        raise ValueError("--action-map-bins must be at least 2")
    if args.component_bins < 2:
        raise ValueError("--component-bins must be at least 2")
    if args.action_map_min_count < 1:
        raise ValueError("--action-map-min-count must be at least 1")
    if args.sensitivity_bins < 2:
        raise ValueError("--sensitivity-bins must be at least 2")
    if args.sensitivity_band <= 0.0:
        raise ValueError("--sensitivity-band must be positive")
    if args.sensitivity_min_samples < 1:
        raise ValueError("--sensitivity-min-samples must be at least 1")
    observation_clip = tuple(args.observation_clip)
    if not 0.0 <= observation_clip[0] < observation_clip[1] <= 100.0:
        raise ValueError("--observation-clip must satisfy 0 <= LOW < HIGH <= 100")
    default_nodes = _as_node_count(args.gll_nodes)
    options = _load_case_options(args.case_options)
    plot_data = []

    for case_arg in args.cases:
        case_path = _resolve_case(case_arg, args.runs_dir)
        support, requested_nodes = _case_options(
            options, case_arg, case_path, args.support, default_nodes
        )
        env_dirs = find_env_dirs(case_path)
        if not env_dirs:
            raise FileNotFoundError(f"no drlrec environments found below {case_path}")

        environments = []
        print(f"[ZNMF] case    : {case_path} (support={support}, gll_nodes={requested_nodes})")
        for env_id, env_dir in env_dirs:
            env_path = Path(env_dir)
            rec = load_run(env_path)
            result = zero_mean_action(rec, support=support, gll_nodes=requested_nodes)
            products = process_observation_products(
                rec, result, action_map_bins=args.action_map_bins,
                component_bins=args.component_bins,
                action_map_min_count=args.action_map_min_count,
                observation_clip=observation_clip,
                sensitivity_bins=args.sensitivity_bins,
                sensitivity_band=args.sensitivity_band,
                sensitivity_min_samples=args.sensitivity_min_samples,
            )
            output = export_mat(case_path.name, env_id, env_path, rec, result,
                                products, support, args.export_dir)
            nselected = int(result.support.sum())
            ninactive = rec.nagent - nselected
            print(
                f"[ZNMF] env {env_id:03d}: records={rec.nrec}, agents={rec.nagent}, "
                f"corrected={nselected}, unchanged={ninactive}, "
                f"max |residual|={np.abs(result.residual_mean).max():.3e}\n"
                f"       export  : {output}"
            )
            environments.append((env_id, rec, result, products))

        nodes = environments[0][2].gll_nodes
        if any(item[2].gll_nodes != nodes for item in environments[1:]):
            raise ValueError(f"{case_path}: environments inferred different GLL node counts; use --gll-nodes")
        plot_data.append((case_path.name, support, environments, nodes))

    plot_cases(plot_data, args.outdir / args.figure_name, args.bins, args.n_show,
               args.show, args.trace_env)
    plot_observation_products(plot_data, args.outdir / args.products_figure_name,
                              args.show)
    plot_sensitivity_products(plot_data, args.outdir / args.sensitivity_figure_name,
                              args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
