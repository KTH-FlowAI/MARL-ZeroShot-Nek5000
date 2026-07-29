#!/usr/bin/env python3
"""
CLI wrapper around lib/pol_export: turn an SB3 checkpoint into the .pol
file read by the embedded F77 policy, plus a reference table for the
offline self-test.

    python utils/sb3_to_f77.py runs/mc_nes_nek/logs/best_model.zip \
           --config conf/mini_channel/MC-nes.yml

    python utils/sb3_to_f77.py path/to/any_checkpoint.zip \
           --utau 0.064 --amp 0.064 --out /tmp/actor.pol

The checkpoint is an ordinary positional argument, so any file can be
exported -- there is nothing special about best_model.zip.

The export logic lives in src/lib/pol_export.py so that this CLI and the
prepare step in lib/nek_utils.py share one implementation.
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from lib.pol_export import (build_reference, detect_algo, export_checkpoint,  # noqa: E402
                            load_state_dict, write_chk)


def _scalings_from_yaml(cfg_path):
    """u_tau, action multiplier and bounds, mirroring nek_marl."""
    import yaml
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    r = cfg.get("runner", {})
    utau = float(r["u_tau"])
    if r.get("rescale_actions", False):
        return utau, float(r["ctrl_max_amp"]), -1.0, 1.0
    return utau, 1.0, float(r["ctrl_min_amp"]), float(r["ctrl_max_amp"])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="path to any SB3 .zip checkpoint")
    ap.add_argument("--config", help="run YAML; supplies u_tau and bounds")
    ap.add_argument("--utau", type=float, help="observation scaling")
    ap.add_argument("--amp", type=float, help="action multiplier")
    ap.add_argument("--alow", type=float, help="action-space lower bound")
    ap.add_argument("--ahigh", type=float, help="action-space upper bound")
    ap.add_argument("--algo", help="force TD3/DDPG/SAC instead of auto-detect")
    ap.add_argument("--out", help="output .pol (default: beside the checkpoint)")
    ap.add_argument("--nchk", type=int, default=20000,
                    help="reference table size")
    ap.add_argument("--seed", type=int, default=1998)
    ap.add_argument("--no-chk", action="store_true",
                    help="skip the reference table")
    args = ap.parse_args()

    utau = amp = None
    alow, ahigh = -1.0, 1.0
    if args.config:
        utau, amp, alow, ahigh = _scalings_from_yaml(args.config)
    if args.utau is not None:
        utau = args.utau
    if args.amp is not None:
        amp = args.amp
    if args.alow is not None:
        alow = args.alow
    if args.ahigh is not None:
        ahigh = args.ahigh
    if utau is None or amp is None:
        ap.error("give --config, or both --utau and --amp")

    out = args.out or os.path.join(
        os.path.dirname(args.checkpoint),
        os.path.splitext(os.path.basename(args.checkpoint))[0] + ".pol")

    algo, dims = export_checkpoint(args.checkpoint, out, utau, amp,
                                   alow, ahigh, args.algo)
    print(f"[POL] {algo}  {' -> '.join(str(d) for d in dims)}  "
          f"scl_obs={utau}  scl_act={amp}")
    print(f"[POL] wrote {out}")

    if not args.no_chk:
        chk = os.path.splitext(out)[0] + ".chk"
        phys, act = build_reference(args.checkpoint, args.nchk, utau, amp,
                                    alow, ahigh, (dims[0], 1, 1), args.seed)
        write_chk(chk, phys, act, os.path.abspath(args.checkpoint))
        print(f"[POL] wrote {chk}  ({len(act)} rows, "
              f"action range [{act.min():.6g}, {act.max():.6g}])")


if __name__ == "__main__":
    main()
