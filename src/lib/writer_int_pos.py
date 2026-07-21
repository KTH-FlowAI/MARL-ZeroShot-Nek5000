# Example script for reading int_fld data
# Kept similar to pymech
import struct
import numpy as np
import os
import re


class point:
    """class defining point variables"""

    def __init__(self, ldim):
        self.pos = np.zeros((ldim))


class pset:
    """class containing data of the point collection"""

    def __init__(self, ldim, npoints):
        self.ldim = ldim
        self.npoints = npoints
        self.pset = [point(ldim) for il in range(npoints)]


def set_pnt_pos(data, il, lpos):
    """set position of the single point"""
    lptn = data.pset[il]
    data_pos = getattr(lptn, 'pos')
    for jl in range(data.ldim):
        data_pos[jl] = lpos[jl]


def write_int_pos(fname, wdsize, emode, data):
    """ write point positions to the file"""
    # open file
    outfile = open(fname, 'wb')

    # word size
    if (wdsize == 4):
        realtype = 'f'
    elif (wdsize == 8):
        realtype = 'd'

    # header
    header = '#iv1 %1i %1i %10i ' % (wdsize, data.ldim, data.npoints)
    header = header.ljust(32)
    outfile.write(header.encode('utf-8'))

    # write tag (to specify endianness)
    # etagb = struct.pack(emode+'f', 6.54321)
    # outfile.write(etagb)
    outfile.write(struct.pack(emode+'f', 6.54321))

    # write point positions
    for il in range(data.npoints):
        lptn = data.pset[il]
        data_pos = getattr(lptn, 'pos')
        outfile.write(struct.pack(emode+data.ldim*realtype, *data_pos))


# Canonical ordering of the channel sensing points.  Points are emitted with y
# outermost, then x, then z, so the flat point index of the file is a
# C-contiguous index into an array of shape (ny, nx, nz).  The post-processing
# reader (post_processing/postlib/tsrs.py) imports channel_grid() from here so
# that writer and reader cannot drift apart.
CHANNEL_AXES = ('y', 'x', 'z')

CHANNEL_HALF_HEIGHT = 1.0


def resolve_yplus(yplus):
    """
    Normalise the wall-normal plane specification to a sorted list of y+ values.

    A scalar means "the wall plus that plane", i.e. ``15.0 -> [0.0, 15.0]``,
    which is the historical two-plane behaviour.  A sequence is taken literally,
    so pass ``[0, 15, 30, 50]`` to sample four planes -- include 0.0 explicitly
    if you want the wall.
    """
    if np.isscalar(yplus):
        values = [0.0, float(yplus)]
    else:
        values = [float(v) for v in yplus]
        if not values:
            raise ValueError('yplus is empty: at least one plane is required')

    out = sorted(set(values))
    if len(out) != len(values):
        raise ValueError(f'yplus contains duplicate planes: {values}')
    if out[0] < 0.0:
        raise ValueError(f'negative y+ requested: {out[0]}')
    return out


def channel_grid(Ret, yplus, Lx, Lz, Nx, Nz, lx1):
    """
    Coordinate vectors of the channel sensing planes.

    Ret   : Re_tau
    yplus : scalar (wall + that plane) or sequence of y+ values -- see
            resolve_yplus()

    Returns (pty, ptx, ptz) -- ordered outermost-axis first, matching
    CHANNEL_AXES and the point-emission order of write_channel().
    """
    yp = resolve_yplus(yplus)
    if yp[-1] > Ret:
        # y+ == Ret is the centreline; beyond it findpts has nothing to find
        raise ValueError(
            f'y+ = {yp[-1]} lies outside the channel (centreline is '
            f'y+ = Re_tau = {Ret})')

    ptx = np.linspace(0.0, Lx, Nx*lx1)
    ptz = np.linspace(0.0, Lz, Nz*lx1)
    pty = np.array(yp) / Ret * CHANNEL_HALF_HEIGHT
    return pty, ptx, ptz


def _check_lhis(path, npoints, nproc=None):
    """
    Warn if the point set cannot fit in the solver's static per-rank arrays.

    tsrs stores per-rank point data in arrays dimensioned lhis (SIZE), so the
    busiest rank must hold no more than lhis points after redistribution.
    """
    size_file = os.path.join(path, 'SIZE')
    lhis = None
    try:
        with open(size_file) as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.startswith(('c', 'C', '!', '*')):
                    continue
                m = re.search(r'parameter\s*\(\s*lhis\s*=\s*(\d+)\s*\)',
                              stripped, re.IGNORECASE)
                if m:
                    lhis = int(m.group(1))
    except OSError:
        return

    if lhis is None:
        return
    if nproc is None:
        print(f'[int_pos] {npoints} points; SIZE has lhis = {lhis} '
              f'(max points per rank)')
        return

    per_rank = -(-npoints // nproc)      # ceil, the perfectly balanced case
    if per_rank > lhis:
        print(f'[int_pos] WARNING: {npoints} points over {nproc} ranks needs at '
              f'least {per_rank} points/rank, but SIZE has lhis = {lhis}. '
              f'Raise lhis and recompile, or use fewer planes.', flush=True)
    else:
        print(f'[int_pos] {npoints} points, >= {per_rank}/rank, lhis = {lhis}')


def write_channel(path, Ret, yplus,
                  Lx, Lz, Nx, Nz, lx1, nproc=None):
    """
    Function to write the interploation sensing plane
    Ret : Retau
    yplus: wall unit distance -- a scalar keeps the historical two-plane
           behaviour (wall + that plane); pass a sequence such as
           [0, 15, 30, 50] to sample several planes at once
    Lx, Lz: Domain size
    Nx, Nz: Number of elements
    lx1: Poly order
    nproc: MPI ranks, used only to warn when the point set exceeds lhis

    This point set feeds the tsrs diagnostic only (#ifdef TSRS).  The DRL
    sensing plane is passed to the solver separately as a UPARAM, so adding
    planes here does not change the RL state.
    """

    fname = f'{path}/int_pos'

    #if os.path.exists(fname):
    #    print(f"File Exists!:{fname}", flush=True)
    #    return True

    wdsize = 8
    # little endian
    emode = '<'
    # set of points
    ldim = 3

    # create point coordinates (single source of truth for the point ordering)
    pty, ptx, ptz = channel_grid(Ret, yplus, Lx, Lz, Nx, Nz, lx1)
    npointsy, npointsx, npointsz = len(pty), len(ptx), len(ptz)
    # allocate space
    npoints = npointsx*npointsz*npointsy
    # number of points
    data = pset(ldim, npoints)
    print('Allocated {0} points on {1} plane(s): y+ = {2}'.format(
        npoints, npointsy, resolve_yplus(yplus)))
    _check_lhis(path, npoints, nproc)

    # initialise point position buffer
    lpos = np.zeros(data.ldim)

    # assign point structure
    npoints = 0
    for jy in range(npointsy):
        for ix in range(npointsx):
            for iz in range(npointsz):
                lpos[0] = ptx[ix]
                lpos[1] = pty[jy]
                lpos[2] = ptz[iz]
                set_pnt_pos(data, npoints, lpos)
                npoints = npoints + 1

    # write points to the file
    write_int_pos(fname, wdsize, emode, data)
    print('Written {0} points to the file'.format(npoints))
    return True


if __name__ == "__main__":
    # initialise variables
    fname = '02-run/int_pos'
    wdsize = 8
    # little endian
    emode = '<'
    # big endian
    # emode = '<'
    # set of points
    ldim = 3

    Ret = 180
    channelh = 1.0
    npointsx = 4*6
    npointsz = 4*6
    npointsy = 2

    xmin, xmax = 0.0, 2.67
    zmin, zmax = 0.0, 0.8
    ymin, ymax = 0.0, 15.0 / Ret * channelh

    # create point coordinates
    ptx = np.linspace(xmin, xmax, npointsx)
    ptz = np.linspace(zmin, zmax, npointsz)
    pty = np.linspace(ymin, ymax, npointsy)
    # allocate space
    npoints = npointsx*npointsz*npointsy
    # number of points
    data = pset(ldim, npoints)
    print('Allocated {0} points'.format(npoints))

    # initialise point position buffer
    lpos = np.zeros(data.ldim)

    # assign point structure
    npoints = 0
    for jy in range(npointsy):
        for ix in range(npointsx):
            for iz in range(npointsz):
                lpos[0] = ptx[ix]
                lpos[1] = pty[jy]
                lpos[2] = ptz[iz]
                set_pnt_pos(data, npoints, lpos)
                npoints = npoints + 1

    # write points to the file
    write_int_pos(fname, wdsize, emode, data)
    print('Written {0} points to the file'.format(npoints))
