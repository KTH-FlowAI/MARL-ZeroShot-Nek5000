"""
Diagnostic script: check compatibility between phill.re2 (2D post-processing mesh)
and DATA/c2Dphill0.f* (element centres written by the 3D simulation).

Usage:
    python check_mesh.py [re2_file] [c2d_file]

Defaults:
    re2_file : phill.re2
    c2d_file : DATA/c2Dphill0.f00001
"""

import struct
import sys
import os


# ── helpers ──────────────────────────────────────────────────────────────────

def _unpack(fmt, data, offset=0):
    size = struct.calcsize(fmt)
    return struct.unpack_from(fmt, data, offset), offset + size


def read_re2(path):
    """Return element corner coordinates from a Nek5000 .re2 v002 file.

    Returns
    -------
    xs, ys : list of lists
        Corner x- and y-coordinates for each element (4 corners each for 2D).
    meta   : dict with keys version, nelt, ndim, wdsizi
    """
    with open(path, "rb") as f:
        raw = f.read()

    hdr = raw[:80].decode("ascii", errors="replace")
    version = hdr[0:5]
    nelt    = int(hdr[5:14])
    ndim    = int(hdr[14:17])
    nelgv   = int(hdr[17:26])

    if version not in ("#v001", "#v002"):
        raise ValueError(f"Unsupported re2 version: {version!r}")
    wdsizi = 8 if version == "#v002" else 4
    float_fmt = "d" if wdsizi == 8 else "f"

    # byte-order test at offset 80 (4 bytes, single precision)
    (test,) = struct.unpack_from("<f", raw, 80)
    byte_swap = not (6.5 < test < 6.6)

    data_offset = 84  # 80-byte header + 4-byte endian test
    nc = 2 ** ndim    # corners per element: 4 for 2D
    # record: 1 group (int64 = 2 ints = 1 double) + ndim*nc doubles
    lrs = 1 + ndim * nc  # items per record (9 for 2D wdsizi=8)
    rec_bytes = lrs * wdsizi

    xs, ys = [], []
    offset = data_offset
    endian = ">" if byte_swap else "<"
    for _ in range(nelt):
        chunk = raw[offset: offset + rec_bytes]
        # group: first wdsizi bytes (treat as int64 / int32)
        x_off = 1 * wdsizi
        y_off = (1 + nc) * wdsizi
        xc = list(struct.unpack_from(f"{endian}{nc}{float_fmt}", chunk, x_off))
        yc = list(struct.unpack_from(f"{endian}{nc}{float_fmt}", chunk, y_off))
        xs.append(xc)
        ys.append(yc)
        offset += rec_bytes

    meta = dict(version=version, nelt=nelt, ndim=ndim, wdsizi=wdsizi,
                byte_swap=byte_swap)
    return xs, ys, meta


def read_c2d(path):
    """Return element centres from a Nek5000 c2D field file.

    Returns
    -------
    gnels  : list of global element numbers
    cx, cy : lists of element centre x- and y-coordinates
    meta   : dict with keys wdsizr, nelgr, time, step
    """
    with open(path, "rb") as f:
        raw = f.read()

    hdr = raw[:132].decode("ascii", errors="replace")
    parts = hdr.split()
    wdsizr = int(parts[1])
    nelgr  = int(parts[3])
    time   = float(parts[4])
    step   = int(parts[5])

    float_fmt = "d" if wdsizr == 8 else "f"

    # byte-order test (4 bytes, single precision) at offset 132
    (test,) = struct.unpack_from("<f", raw, 132)
    byte_swap = not (6.5 < test < 6.6)
    endian = ">" if byte_swap else "<"

    offs0 = 136  # 132-byte header + 4-byte endian test

    # global element numbers: nelgr * 4 bytes (integers)
    gnels = list(struct.unpack_from(f"{endian}{nelgr}i", raw, offs0))

    # levels: nelgr * 4 bytes
    # (offs0 + nelgr*4)

    # centres: nelgr * 2 * wdsizr bytes starting at offs0 + 2*nelgr*4
    cnt_offset = offs0 + 2 * nelgr * 4
    vals = struct.unpack_from(f"{endian}{nelgr * 2}{float_fmt}", raw, cnt_offset)
    cx = [vals[i * 2]     for i in range(nelgr)]
    cy = [vals[i * 2 + 1] for i in range(nelgr)]

    meta = dict(wdsizr=wdsizr, nelgr=nelgr, time=time, step=step,
                byte_swap=byte_swap)
    return gnels, cx, cy, meta


def elem_centers_re2(xs, ys):
    """Average corner coordinates to get element centres."""
    cx = [sum(xc) / len(xc) for xc in xs]
    cy = [sum(yc) / len(yc) for yc in ys]
    return cx, cy


def unique_sorted(values, tol=1e-5):
    """Return sorted unique values within tolerance tol."""
    out = []
    for v in sorted(values):
        if not out or abs(v - out[-1]) > tol:
            out.append(v)
    return out


def print_section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


# ── main check ───────────────────────────────────────────────────────────────

def check(re2_path, c2d_path):
    ok = True

    # ── read files ────────────────────────────────────────────────────────────
    print_section(f"re2  : {re2_path}")
    if not os.path.exists(re2_path):
        print(f"  ERROR: file not found")
        return False
    xs_re2, ys_re2, re2_meta = read_re2(re2_path)
    print(f"  version  : {re2_meta['version']}")
    print(f"  nelt     : {re2_meta['nelt']}")
    print(f"  ndim     : {re2_meta['ndim']}")
    print(f"  wdsizi   : {re2_meta['wdsizi']}")
    print(f"  byte_swap: {re2_meta['byte_swap']}")

    all_x_re2 = [v for xc in xs_re2 for v in xc]
    all_y_re2 = [v for yc in ys_re2 for v in yc]
    print(f"  X range  : [{min(all_x_re2):.6f}, {max(all_x_re2):.6f}]")
    print(f"  Y range  : [{min(all_y_re2):.6f}, {max(all_y_re2):.6f}]")

    # unique node positions
    uniq_x = unique_sorted(all_x_re2)
    uniq_y = unique_sorted(all_y_re2)
    print(f"  unique X nodes ({len(uniq_x)}): {[f'{v:.6f}' for v in uniq_x]}")
    print(f"  unique Y nodes ({len(uniq_y)}): {[f'{v:.6f}' for v in uniq_y]}")

    print_section(f"c2D  : {c2d_path}")
    if not os.path.exists(c2d_path):
        print(f"  ERROR: file not found")
        return False
    gnels, cx_c2d, cy_c2d, c2d_meta = read_c2d(c2d_path)
    print(f"  wdsizr   : {c2d_meta['wdsizr']}")
    print(f"  nelgr    : {c2d_meta['nelgr']}")
    print(f"  time     : {c2d_meta['time']:.4f}")
    print(f"  step     : {c2d_meta['step']}")
    print(f"  byte_swap: {c2d_meta['byte_swap']}")
    print(f"  X range  : [{min(cx_c2d):.6f}, {max(cx_c2d):.6f}]")
    print(f"  Y range  : [{min(cy_c2d):.6f}, {max(cy_c2d):.6f}]")

    uniq_cx = unique_sorted(cx_c2d)
    uniq_cy = unique_sorted(cy_c2d)
    print(f"  unique X centres ({len(uniq_cx)}): {[f'{v:.6f}' for v in uniq_cx]}")
    print(f"  unique Y centres ({len(uniq_cy)}): {[f'{v:.6f}' for v in uniq_cy]}")

    # ── compatibility checks ──────────────────────────────────────────────────
    print_section("Compatibility checks")

    # 1. element count
    nelt_re2 = re2_meta["nelt"]
    nelgr    = c2d_meta["nelgr"]
    status = "OK" if nelt_re2 == nelgr else "FAIL"
    if status == "FAIL":
        ok = False
    print(f"  [{status}] element count  re2={nelt_re2}  c2D={nelgr}")

    # 2. x-centre containment: every c2D x-centre must lie within re2 x-range
    x_min_re2, x_max_re2 = min(all_x_re2), max(all_x_re2)
    tol = 1e-4
    x_fails = [v for v in cx_c2d if v < x_min_re2 - tol or v > x_max_re2 + tol]
    status = "OK" if not x_fails else "FAIL"
    if x_fails:
        ok = False
    print(f"  [{status}] c2D X centres inside re2 X range "
          f"[{x_min_re2:.6f}, {x_max_re2:.6f}]"
          + (f"  — {len(x_fails)} point(s) outside" if x_fails else ""))

    # 3. y-centre containment: every c2D y-centre must lie within re2 y-range
    y_min_re2, y_max_re2 = min(all_y_re2), max(all_y_re2)
    y_fails = [v for v in cy_c2d if v < y_min_re2 - tol or v > y_max_re2 + tol]
    status = "OK" if not y_fails else "FAIL"
    if y_fails:
        ok = False
    print(f"  [{status}] c2D Y centres inside re2 Y range "
          f"[{y_min_re2:.6f}, {y_max_re2:.6f}]"
          + (f"  — {len(y_fails)} point(s) outside" if y_fails else ""))
    if y_fails:
        uniq_y_fails = unique_sorted(y_fails)
        print(f"         out-of-range Y values: {[f'{v:.6f}' for v in uniq_y_fails]}")

    # 4. matching x grid lines
    cx_tol = 1e-4
    re2_cx, re2_cy = elem_centers_re2(xs_re2, ys_re2)
    uniq_re2_cx = unique_sorted(re2_cx)
    uniq_c2d_cx = unique_sorted(cx_c2d)
    x_grid_ok = (len(uniq_re2_cx) == len(uniq_c2d_cx) and
                 all(abs(a - b) < cx_tol for a, b in zip(uniq_re2_cx, uniq_c2d_cx)))
    status = "OK" if x_grid_ok else "WARN"
    print(f"  [{status}] X grid lines match  "
          f"re2={[f'{v:.4f}' for v in uniq_re2_cx]}  "
          f"c2D={[f'{v:.4f}' for v in uniq_c2d_cx]}")

    # 5. matching y grid lines
    uniq_re2_cy = unique_sorted(re2_cy)
    uniq_c2d_cy = unique_sorted(cy_c2d)
    y_grid_ok = (len(uniq_re2_cy) == len(uniq_c2d_cy) and
                 all(abs(a - b) < cx_tol for a, b in zip(uniq_re2_cy, uniq_c2d_cy)))
    status = "OK" if y_grid_ok else "FAIL"
    if not y_grid_ok:
        ok = False
    print(f"  [{status}] Y grid lines match")
    if not y_grid_ok:
        print(f"         re2 Y centres : {[f'{v:.6f}' for v in uniq_re2_cy]}")
        print(f"         c2D Y centres : {[f'{v:.6f}' for v in uniq_c2d_cy]}")

    # ── summary ───────────────────────────────────────────────────────────────
    print_section("Summary")
    if ok:
        print("  PASS — re2 mesh and c2D file are compatible.")
    else:
        print("  FAIL — mesh mismatch detected.")
        if y_fails:
            print()
            print("  Most likely cause: phill.re2 covers the wrong Y domain.")
            print(f"    re2 Y max  : {y_max_re2:.6f}")
            print(f"    c2D Y max  : {max(cy_c2d):.6f}")
            print()
            print("  Fix: regenerate phill.re2 from msh2d/channel_2d.box")
            print("       cd msh2d && bash mesh2d.sh")
    print()
    return ok


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    re2_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base, "phill.re2")
    c2d_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(base, "DATA", "c2Dphill0.f00001")

    check(re2_path, c2d_path)
