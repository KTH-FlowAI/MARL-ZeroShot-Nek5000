"""
Reading the KTH `tsrs` point time-series files (``pts<case>0.f#####``).

File layout, as produced by Toolbox/tools/tsrs/tsrs_IO.f::

    [   132 B ] ascii header: '#std' wdsize ldim nelo nptot ntsnap nfld time fid0 nfileo
    [     4 B ] float32 endianness tag (6.54321)
    [ wdsize * ntsnap ] snapshot time list
    [      4 * nptot  ] int32 global point id (1-based)
    [ wdsize * nptot * ldim        ] point coordinates
    [ wdsize * nptot * ntsnap*nfld ] interpolated fields

The point axis is **not** in the order the points were written to ``int_pos``:
pts_redistribute.f hands each point to the rank owning its element, and every
rank dumps its own chunk (tsrs_IO.f:222-250).  The permutation therefore depends
on the mesh partition and the process count.  The global point id array is what
inverts it -- see ``sort_by_glid`` -- so no coordinate matching is ever needed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

#: Nek's iheadersize.
HEADER_SIZE = 132

#: Field order set by tsrs_interpolate (tsrs.f:534-563): velocity, pressure,
#: vorticity.  nfld == 2*ldim + 1.
FIELD_NAMES_3D = ('u', 'v', 'w', 'p', 'omega_x', 'omega_y', 'omega_z')


@dataclass
class PointSeries:
    """One ``pts`` file, still in on-disk point order."""

    glid: np.ndarray   # (npoints,)         int32, 1-based global point id
    pos: np.ndarray    # (npoints, ldim)    float64
    fld: np.ndarray    # (npoints, ntsnap, nfld) float64
    t: np.ndarray      # (ntsnap,)          float64, snapshot times
    time: float        # simulation time at write

    @property
    def npoints(self):
        return self.pos.shape[0]

    @property
    def ntsnap(self):
        return self.fld.shape[1]

    @property
    def nfld(self):
        return self.fld.shape[2]


def read_pts(fname):
    """Read a single ``pts`` file into a :class:`PointSeries`."""
    with open(fname, 'rb') as fh:
        raw = fh.read()

    header = raw[:HEADER_SIZE].split()
    if header[0] != b'#std':
        raise ValueError(f'{fname}: not a Nek std file (tag {header[0]!r})')

    wdsize = int(header[1])
    ldim = int(header[2])
    nelo = int(header[3])      # points in *this* file
    nptot = int(header[4])     # points in the whole set
    ntsnap = int(header[5])
    nfld = int(header[6])
    time = float(header[7])

    if nelo != nptot:
        raise NotImplementedError(
            f'{fname}: multi-file output (nelo={nelo} != nptot={nptot}); '
            'only single-file (ifmpiio) tsrs output is supported')

    tag = raw[HEADER_SIZE:HEADER_SIZE + 4]
    if abs(struct.unpack('<f', tag)[0] - 6.54321) < 1e-4:
        emode = '<'
    elif abs(struct.unpack('>f', tag)[0] - 6.54321) < 1e-4:
        emode = '>'
    else:
        raise ValueError(f'{fname}: bad endianness tag {tag!r}')

    rtype = np.dtype(emode + ('f4' if wdsize == 4 else 'f8'))
    itype = np.dtype(emode + 'i4')

    off = HEADER_SIZE + 4
    t = np.frombuffer(raw, dtype=rtype, count=ntsnap, offset=off)
    off += wdsize * ntsnap

    glid = np.frombuffer(raw, dtype=itype, count=nptot, offset=off)
    off += 4 * nptot

    pos = np.frombuffer(raw, dtype=rtype, count=nptot * ldim, offset=off)
    off += wdsize * nptot * ldim

    fld = np.frombuffer(raw, dtype=rtype, count=nptot * ntsnap * nfld, offset=off)
    off += wdsize * nptot * ntsnap * nfld

    if off != len(raw):
        raise ValueError(
            f'{fname}: trailing/short data -- expected {off} bytes, got {len(raw)}')

    return PointSeries(
        glid=np.array(glid),
        pos=np.array(pos, dtype=np.float64).reshape(nptot, ldim),
        fld=np.array(fld, dtype=np.float64).reshape(nptot, ntsnap, nfld),
        t=np.array(t, dtype=np.float64),
        time=time,
    )


def sort_by_glid(series):
    """
    Undo the MPI point redistribution, returning ``(pos, fld)`` in the order the
    points were written to ``int_pos``.
    """
    glid = series.glid
    expected = np.arange(1, series.npoints + 1, dtype=glid.dtype)
    if not np.array_equal(np.sort(glid), expected):
        raise ValueError(
            'global point ids are not a permutation of 1..npoints; '
            'the file is truncated or the point set is inconsistent')

    perm = np.argsort(glid, kind='stable')
    return series.pos[perm], series.fld[perm]


def infer_shape(series, decimals=9):
    """
    Deduce (ny, nx, nz) from the point coordinates actually stored in the file.

    Used to explain a grid mismatch: a file left over from an earlier
    configuration reports its own plane count and resolution rather than just a
    point total.
    """
    pos = np.round(series.pos, decimals)
    ny = len(np.unique(pos[:, 1]))
    nx = len(np.unique(pos[:, 0]))
    nz = len(np.unique(pos[:, 2]))
    return ny, nx, nz


def to_grid(series, shape):
    """
    Reshape a :class:`PointSeries` onto the structured sensing grid.

    ``shape`` is ``(ny, nx, nz)`` from writer_int_pos.channel_grid().  Returns
    ``(pos, fld)`` with shapes ``(ny, nx, nz, ldim)`` and
    ``(ny, nx, nz, ntsnap, nfld)``.
    """
    ny, nx, nz = shape
    if ny * nx * nz != series.npoints:
        got = infer_shape(series)
        yvals = np.unique(np.round(series.pos[:, 1], 9))
        raise ValueError(
            f'grid mismatch: the config asks for (ny, nx, nz) = {shape} '
            f'= {ny * nx * nz} points, but this file holds {series.npoints} '
            f'points laid out as {got}.\n'
            f'    the file\'s own y planes are {yvals.tolist()}\n'
            f'    this usually means the file predates a change to y_planes / '
            f'Nx / Nz / lx1 -- check whether it is stale')

    pos, fld = sort_by_glid(series)
    ldim = pos.shape[1]
    return (pos.reshape(ny, nx, nz, ldim),
            fld.reshape(ny, nx, nz, series.ntsnap, series.nfld))


def validate_grid(pos, y, x, z, atol=1e-9):
    """
    Assert that the regridded coordinates really are the ones we asked the
    solver to sample.  Cheap, and it turns a silent mis-ordering into an error.
    """
    for axis, (name, want) in enumerate(zip(('y', 'x', 'z'), (y, x, z))):
        # coordinate column: y is pos[...,1], x is pos[...,0], z is pos[...,2]
        col = {'x': 0, 'y': 1, 'z': 2}[name]
        idx = [0, 0, 0]
        idx[axis] = slice(None)
        got = pos[tuple(idx) + (col,)]
        if not np.allclose(got, want, atol=atol):
            raise ValueError(
                f'{name}-coordinates do not match the requested grid '
                f'(max deviation {np.abs(got - want).max():.3e})')

        # and the coordinate must be constant along the other two axes
        spread = np.ptp(pos[..., col], axis=tuple(a for a in range(3) if a != axis))
        if np.max(spread) > atol:
            raise ValueError(
                f'{name} varies along an axis it should not; point ordering is wrong')


def stitch(fnames, shape, y, x, z, fields=None, dtype=np.float32, verbose=True):
    """
    Read a time-ordered list of ``pts`` files and concatenate them along time.

    Returns a dict ready for ``np.savez``: ``fld`` (ny, nx, nz, nt, nfld),
    the coordinate vectors, the time vector ``t`` and the field ``names``.
    """
    names = list(FIELD_NAMES_3D if fields is None else fields)

    chunks, times = [], []
    for fname in fnames:
        series = read_pts(fname)
        pos, fld = to_grid(series, shape)
        validate_grid(pos, y, x, z)

        if series.nfld < len(FIELD_NAMES_3D):
            raise ValueError(f'{fname}: nfld={series.nfld}, expected '
                             f'{len(FIELD_NAMES_3D)}')
        cols = [FIELD_NAMES_3D.index(n) for n in names]
        chunks.append(fld[..., cols].astype(dtype))
        times.append(series.t)
        if verbose:
            print(f'  {fname}: {series.ntsnap} snapshots, '
                  f't = [{series.t[0]:.4f}, {series.t[-1]:.4f}]')

    if not chunks:
        raise ValueError('no files to stitch')

    t = np.concatenate(times)
    if not np.all(np.diff(t) > 0):
        bad = np.flatnonzero(np.diff(t) <= 0)
        raise ValueError(
            f'stitched time vector is not strictly increasing at indices '
            f'{bad[:5].tolist()} -- files overlap, repeat or are out of order')

    return {
        'fld': np.concatenate(chunks, axis=3),
        'y': np.asarray(y), 'x': np.asarray(x), 'z': np.asarray(z),
        't': t,
        'names': np.array(names),
    }


def load(fname):
    """Load a stitched ``.npz`` and return a plain dict."""
    with np.load(fname, allow_pickle=False) as fh:
        data = {k: fh[k] for k in fh.files}
    data['names'] = [str(n) for n in data['names']]
    return data


def field(data, name, iy=None):
    """
    Extract one field from a stitched dict as ``(nx, nz, nt)``.

    ``iy`` selects the wall-normal plane (0 = wall); omit for all planes.
    """
    names = [str(n) for n in data['names']]   # list after load(), ndarray after stitch()
    if name not in names:
        raise KeyError(f'no field {name!r}; available: {names}')
    k = names.index(name)
    return data['fld'][..., k] if iy is None else data['fld'][iy, ..., k]
