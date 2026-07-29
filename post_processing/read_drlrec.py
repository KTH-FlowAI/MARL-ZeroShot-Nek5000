"""
Reader for the embedded-policy trajectory files written by drl/pol_IO.f.

The solver writes one `drlrecNNNNN.bin` per MPI rank, each self-describing:
the agent identity (element, face, GLL indices, coordinates, policy id) sits
in the header, so the ranks can be rejoined without a separate NODE_INFO
file and without knowing how the mesh was partitioned.

Typical use
-----------
    from read_drlrec import load_run

    rec = load_run('runs/mc_nes/eval/env_001')

    rec.time            # (nrec,)            simulation time
    rec.istep           # (nrec,)            solver step
    rec.icycle          # (nrec,)            control cycle index
    rec.obs             # (nrec, nagent, nfld)   u', v' at the sensing plane
    rec.act             # (nrec, nagent)     actuation velocity
    rec.rwd             # (nrec, nagent, 3)  tau_w, |p'v|, 0.5|v^3|
    rec.agents          # DataFrame: nid, ieg, iface, ix, iy, iz, ipol, x, y, z

    rec.reward(alpha=1, beta=1, gamma=1)   # net-gain reward per record
    rec.drag_reduction(dudy_ref=11.75, nu=1/2900)

Agents are ordered by (nid, position within the rank), matching the order
the Fortran wrote them, and that order is stable across records.

@yuningw
"""
import os
import glob
from dataclasses import dataclass, field

import numpy as np

MAGIC_V1 = b"NEKPOLR1"          # original layout, no segment index
MAGIC_V2 = b"NEKPOLR2"          # + int32 segment index after rec_freq
ENDIAN_PROBE = 1234567890

# Identity that survives repartitioning: global element, face and GLL
# indices. Sorting on this makes the agent axis independent of nproc and of
# the order the ranks happened to register their wall points in, which is
# what lets two runs made with different nproc be stitched together.
AGENT_KEY = ["ieg", "iface", "iz", "iy", "ix"]


@dataclass
class DRLRecord:
    """Trajectory of one run, with every rank's agents concatenated."""
    time: np.ndarray
    istep: np.ndarray
    icycle: np.ndarray
    obs: np.ndarray                     # (nrec, nagent, nfld)
    act: np.ndarray                     # (nrec, nagent)
    rwd: np.ndarray                     # (nrec, nagent, nrwd)
    agents: "object"                    # pandas DataFrame
    dt: float = 0.0
    ndrl: int = 0
    rec_freq: int = 1
    files: list = field(default_factory=list)

    @property
    def nrec(self):
        return len(self.time)

    @property
    def nagent(self):
        return self.act.shape[1]

    def reward(self, alpha=1.0, beta=1.0, gamma=1.0, dudy_ref=None, nu=None,
               tau_ref=None):
        """Net-gain reward per record, averaged over agents.

        Mirrors nek_marl._normalize_reward:
            R = alpha*(1 - tau_w/tau_ref) - beta*|p'v|/tau_ref
                                          - gamma*0.5|v^3|/tau_ref
        Give either `tau_ref` directly, or `dudy_ref` and `nu`.
        """
        if tau_ref is None:
            if dudy_ref is None or nu is None:
                raise ValueError("give tau_ref, or both dudy_ref and nu")
            tau_ref = nu * dudy_ref
        tau = self.rwd[:, :, 0].mean(axis=1)
        pw = self.rwd[:, :, 1].mean(axis=1)
        v3 = self.rwd[:, :, 2].mean(axis=1)
        return (alpha * (1.0 - tau / tau_ref)
                - beta * pw / tau_ref
                - gamma * v3 / tau_ref)

    def drag_reduction(self, dudy_ref=None, nu=None, tau_ref=None,
                       t_start=None):
        """Drag reduction in percent, optionally discarding the transient
        before `t_start` (in the same units as `time`)."""
        if tau_ref is None:
            if dudy_ref is None or nu is None:
                raise ValueError("give tau_ref, or both dudy_ref and nu")
            tau_ref = nu * dudy_ref
        sel = slice(None)
        if t_start is not None:
            sel = self.time >= t_start
        tau = self.rwd[sel, :, 0].mean()
        return 100.0 * (1.0 - tau / tau_ref)

    def to_dataframe(self):
        """Long-format DataFrame, one row per (record, agent). Convenient
        for seaborn/groupby; large, so build it only when needed."""
        import pandas as pd
        nrec, nag, nfld = self.obs.shape
        base = {
            "time": np.repeat(self.time, nag),
            "istep": np.repeat(self.istep, nag),
            "icycle": np.repeat(self.icycle, nag),
            "agent": np.tile(np.arange(nag), nrec),
            "act": self.act.reshape(-1),
        }
        for k in range(nfld):
            base[f"obs{k + 1}"] = self.obs[:, :, k].reshape(-1)
        for k, name in enumerate(("rwd_tau", "rwd_pw", "rwd_v3")[:self.rwd.shape[2]]):
            base[name] = self.rwd[:, :, k].reshape(-1)
        df = pd.DataFrame(base)
        meta = self.agents.reset_index(drop=True)
        for col in ("nid", "ieg", "iface", "ix", "iy", "iz", "ipol",
                    "x", "y", "z"):
            if col in meta:
                df[col] = np.tile(meta[col].to_numpy(), nrec)
        return df


def _read_one(path):
    """Read a single per-rank file. Returns (header dict, arrays)."""
    with open(path, "rb") as fh:
        blob = fh.read()

    if blob[:8] == MAGIC_V2:
        version = 2
    elif blob[:8] == MAGIC_V1:
        version = 1
    else:
        raise ValueError(f"{path}: bad magic {blob[:8]!r}, expected "
                         f"{MAGIC_V1!r} or {MAGIC_V2!r}")

    # The probe tells us the byte order the solver wrote with, so a file
    # produced on a big-endian machine still reads correctly here.
    probe_le = np.frombuffer(blob[8:12], "<i4")[0]
    endian = "<" if probe_le == ENDIAN_PROBE else ">"
    if endian == ">":
        probe_be = np.frombuffer(blob[8:12], ">i4")[0]
        if probe_be != ENDIAN_PROBE:
            raise ValueError(f"{path}: endianness probe is {probe_le}, "
                             f"file is corrupt")
    i4, f8 = endian + "i4", endian + "f8"

    off = 12
    nid, nagent, nfld, nrwd, ndrl, recf = np.frombuffer(
        blob[off:off + 24], i4)
    off += 24
    if version >= 2:
        iseg = int(np.frombuffer(blob[off:off + 4], i4)[0])
        off += 4
    else:
        iseg = 0
    dt, t0 = np.frombuffer(blob[off:off + 16], f8)
    off += 16

    # per-agent identity: 6 int32 then 3 float64, interleaved
    ids = np.empty((nagent, 6), dtype=np.int32)
    pos = np.empty((nagent, 3), dtype=np.float64)
    for ia in range(nagent):
        ids[ia] = np.frombuffer(blob[off:off + 24], i4)
        off += 24
        pos[ia] = np.frombuffer(blob[off:off + 24], f8)
        off += 24

    rec_bytes = 8 + 4 + 4 + (nfld + 1 + nrwd) * nagent * 8
    remain = len(blob) - off
    nrec = remain // rec_bytes
    if nrec * rec_bytes != remain:
        # A killed job can leave a partial final record; keep the whole
        # ones rather than refusing to read the file.
        print(f"[DRLREC] {os.path.basename(path)}: trailing "
              f"{remain - nrec * rec_bytes} bytes ignored (partial record)")

    time = np.empty(nrec, dtype=np.float64)
    istep = np.empty(nrec, dtype=np.int32)
    icyc = np.empty(nrec, dtype=np.int32)
    obs = np.empty((nrec, nagent, nfld), dtype=np.float64)
    act = np.empty((nrec, nagent), dtype=np.float64)
    rwd = np.empty((nrec, nagent, nrwd), dtype=np.float64)

    p = off
    for ir in range(nrec):
        time[ir] = np.frombuffer(blob[p:p + 8], f8)[0]
        p += 8
        istep[ir], icyc[ir] = np.frombuffer(blob[p:p + 8], i4)
        p += 8
        n = nfld * nagent * 8
        obs[ir] = np.frombuffer(blob[p:p + n], f8).reshape(nagent, nfld)
        p += n
        n = nagent * 8
        act[ir] = np.frombuffer(blob[p:p + n], f8)
        p += n
        n = nrwd * nagent * 8
        rwd[ir] = np.frombuffer(blob[p:p + n], f8).reshape(nagent, nrwd)
        p += n

    hdr = dict(nid=int(nid), nagent=int(nagent), nfld=int(nfld),
               nrwd=int(nrwd), ndrl=int(ndrl), rec_freq=int(recf),
               dt=float(dt), t0=float(t0), iseg=iseg, version=version,
               ids=ids, pos=pos)
    return hdr, (time, istep, icyc, obs, act, rwd)


def _merge_ranks(parts, files):
    """Merge one segment's per-rank parts into a single DRLRecord, with the
    agent axis put into canonical (partition-independent) order."""
    import pandas as pd

    # Ranks are written in lockstep, so a record-count mismatch means one
    # file was truncated -- typically the rank that was mid-write when the
    # job was killed. Keep the records every rank has.
    nrecs = {len(p[1][0]) for p in parts}
    if len(nrecs) != 1:
        nmin = min(nrecs)
        print(f"[DRLREC] ranks disagree on record count {sorted(nrecs)}; "
              f"truncating all to {nmin}")
    else:
        nmin = nrecs.pop()

    hdr0 = parts[0][0]
    time = parts[0][1][0][:nmin]
    istep = parts[0][1][1][:nmin]
    icyc = parts[0][1][2][:nmin]

    obs = np.concatenate([p[1][3][:nmin] for p in parts], axis=1)
    act = np.concatenate([p[1][4][:nmin] for p in parts], axis=1)
    rwd = np.concatenate([p[1][5][:nmin] for p in parts], axis=1)

    rows = []
    for hdr, _ in parts:
        for ia in range(hdr["nagent"]):
            rows.append({
                "nid": hdr["nid"],
                "ieg": int(hdr["ids"][ia, 0]),
                "iface": int(hdr["ids"][ia, 1]),
                "ix": int(hdr["ids"][ia, 2]),
                "iy": int(hdr["ids"][ia, 3]),
                "iz": int(hdr["ids"][ia, 4]),
                "ipol": int(hdr["ids"][ia, 5]),
                "x": hdr["pos"][ia, 0],
                "y": hdr["pos"][ia, 1],
                "z": hdr["pos"][ia, 2],
            })
    agents = pd.DataFrame(rows)

    # Canonical order. Without this the agent axis depends on nproc and on
    # the order each rank happened to register its wall points, so two runs
    # of the same case would not be comparable column by column.
    order = np.lexsort(tuple(agents[k].to_numpy()
                             for k in reversed(AGENT_KEY)))
    agents = agents.iloc[order].reset_index(drop=True)
    obs, act, rwd = obs[:, order], act[:, order], rwd[:, order]

    dup = agents.duplicated(subset=AGENT_KEY).sum()
    if dup:
        print(f"[DRLREC] WARNING {dup} agents share an identity key "
              f"{AGENT_KEY}; ordering between them is arbitrary")

    return DRLRecord(time=time, istep=istep, icycle=icyc,
                     obs=obs, act=act, rwd=rwd, agents=agents,
                     dt=hdr0["dt"], ndrl=hdr0["ndrl"],
                     rec_freq=hdr0["rec_freq"], files=list(files))


def _discover(folder):
    """Find record files and group them by segment.

    Looks in <folder>/drlrec/ first (where the prepare step puts them) and
    then in <folder> itself, so hand-staged runs and legacy v1 files are
    picked up too. Returns {segment: [paths]}.
    """
    pats = [os.path.join(folder, "drlrec", "drlrec_s*_p*.bin"),
            os.path.join(folder, "drlrec_s*_p*.bin"),
            os.path.join(folder, "drlrec", "drlrec[0-9]*.bin"),
            os.path.join(folder, "drlrec[0-9]*.bin")]           # v1 layout
    files = []
    for p in pats:
        files.extend(glob.glob(p))
    files = sorted(set(files))

    segs = {}
    for f in files:
        base = os.path.basename(f)
        if base.startswith("drlrec_s"):
            seg = int(base[8:13])
        else:
            seg = 0                       # v1 files are all one segment
        segs.setdefault(seg, []).append(f)
    return {k: sorted(v) for k, v in sorted(segs.items())}


def load_segments(folder):
    """Load each segment separately, ordered by its first timestamp."""
    segs = _discover(folder)
    if not segs:
        raise FileNotFoundError(
            f"no drlrec files under {folder} (looked in ./ and ./drlrec/). "
            "Embedded runs only write these when embedded.enabled is set.")

    out = []
    for seg, files in segs.items():
        parts = [_read_one(f) for f in files]
        parts = [p for p in parts if len(p[1][0]) > 0]
        if not parts:
            print(f"[DRLREC] segment {seg} holds no complete record, skipped")
            continue
        out.append(_merge_ranks(parts, files))
    out.sort(key=lambda r: r.time[0])
    return out


def load_run(folder, stitch=True):
    """Load a run, stitching every segment it contains.

    A run killed on wall clock and resumed writes a second segment rather
    than overwriting the first, and the resumed job may use a different
    nproc. Because the agent axis is put into canonical order per segment,
    such segments concatenate directly.

    stitch=False returns only the earliest segment.
    """
    segs = load_segments(folder)
    if len(segs) == 1 or not stitch:
        return segs[0]

    ref = segs[0]
    key0 = ref.agents[AGENT_KEY].to_numpy()
    for s in segs[1:]:
        k = s.agents[AGENT_KEY].to_numpy()
        if k.shape != key0.shape or not np.array_equal(k, key0):
            raise ValueError(
                "segments describe different agent sets and cannot be "
                "stitched -- they are not the same case. Load them "
                "separately with load_segments().")

    # Guard against a resume started from an earlier checkpoint, which
    # would otherwise silently splice two overlapping trajectories.
    for a, b in zip(segs[:-1], segs[1:]):
        if b.time[0] <= a.time[-1]:
            print(f"[DRLREC] WARNING segments overlap in time "
                  f"({a.time[-1]:.4f} -> {b.time[0]:.4f}); the resume "
                  f"probably restarted from an earlier checkpoint. "
                  f"Records are kept as-is, not de-duplicated.")

    return DRLRecord(
        time=np.concatenate([s.time for s in segs]),
        istep=np.concatenate([s.istep for s in segs]),
        icycle=np.concatenate([s.icycle for s in segs]),
        obs=np.concatenate([s.obs for s in segs], axis=0),
        act=np.concatenate([s.act for s in segs], axis=0),
        rwd=np.concatenate([s.rwd for s in segs], axis=0),
        agents=ref.agents, dt=ref.dt, ndrl=ref.ndrl,
        rec_freq=ref.rec_freq,
        files=[f for s in segs for f in s.files])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Summarise a drlrec run")
    ap.add_argument("folder")
    ap.add_argument("--dudy-ref", type=float, default=11.75)
    ap.add_argument("--nu", type=float, default=1.0 / 2900.0)
    ap.add_argument("--t-start", type=float, default=None)
    a = ap.parse_args()

    segs = load_segments(a.folder)
    print(f"segments : {len(segs)}  "
          f"({', '.join(f'{len(s.files)} files/{s.nrec} rec' for s in segs)})")
    r = load_run(a.folder)
    print(f"files    : {len(r.files)}")
    print(f"records  : {r.nrec}   agents: {r.nagent}   "
          f"nfld: {r.obs.shape[2]}")
    print(f"time     : {r.time[0]:.4f} .. {r.time[-1]:.4f}  "
          f"(dt={r.dt}, ndrl={r.ndrl}, rec_freq={r.rec_freq})")
    print(f"obs range: {r.obs.min():+.5e} .. {r.obs.max():+.5e}")
    print(f"act range: {r.act.min():+.5e} .. {r.act.max():+.5e}")
    print(f"DR       : {r.drag_reduction(a.dudy_ref, a.nu, t_start=a.t_start):+.4f} %")
