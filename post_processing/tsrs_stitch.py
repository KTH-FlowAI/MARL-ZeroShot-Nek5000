"""
Stitching Timeseries 
"""
# Example script for reading point time statistics data
# Kept similar to pymech
import struct, os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import yaml 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--case_id", type=int, default=302000)
args = parser.parse_args()

class point:
    """class defining point variables"""

    def __init__(self,ldim,ntsnap,nfld):
        self.glid = np.zeros((1), dtype=np.uint32)
        self.pos = np.zeros((ldim))
        self.fld = np.zeros((ntsnap,nfld))

class pset:
    """class containing data of the point collection"""

    def __init__(self,ldim,ntsnap,nfld,npoints):
        self.ldim = ldim
        self.ntsnap = ntsnap
        self.nfld = nfld
        self.npoints = npoints
        self.time = []
        self.tmlist = np.zeros((ntsnap))
        self.pset = [point(ldim,ntsnap,nfld) for il in range(npoints)]

def read_int(infile,emode,nvar):
    """read integer array"""
    isize = 4
    llist = infile.read(isize*nvar)
    llist = list(struct.unpack(emode+nvar*'i', llist))
    return llist

def read_flt(infile,emode,wdsize,nvar):
    """read real array"""
    if (wdsize == 4):
        realtype = 'f'
    elif (wdsize == 8):
        realtype = 'd'
    llist = infile.read(wdsize*nvar)
    llist = np.frombuffer(llist, dtype=emode+realtype, count=nvar)
    return llist

def read_int_fld(fname):
    """read data from interpolation file"""
    # open file
    infile = open(fname, 'rb')
    # read header
    header = infile.read(132).split()

    # extract word size
    wdsize = int(header[1])

    # identify endian encoding
    etagb = infile.read(4)
    etagL = struct.unpack('<f', etagb)[0]; etagL = int(etagL*1e5)/1e5
    etagB = struct.unpack('>f', etagb)[0]; etagB = int(etagB*1e5)/1e5
    if (etagL == 6.54321):
        emode = '<'
    elif (etagB == 6.54321):
        emode = '>'
    else:
        emode = "<"
        # raise ValueError("The emode is wrong!")

    # get simulation parameters
    ldim = int(header[2])
    npoints = int(header[4])
    ntsnap = int(header[5])
    nfld = int(header[6])
    time = float(header[7])

    # create main data structure
    data = pset(ldim,ntsnap,nfld,npoints)

    # fill simulation parameters
    data.time = time

    # read snapshot time list
    data.tmlist = read_flt(infile,emode,wdsize,data.ntsnap)

    # read global point number
    glidlist = read_int(infile,emode,data.npoints)
    # fill data structure in
    for il in range(data.npoints):
        lptn = data.pset[il]
        data_glid = getattr(lptn,'glid')
        data_glid[0] = glidlist[il]

    # read coordinates
    for il in range(data.npoints):
        lpos = read_flt(infile,emode,wdsize,data.ldim)
        lptn = data.pset[il]
        data_pos = getattr(lptn,'pos')
        for jl in range(data.ldim):
            data_pos[jl] =  lpos[jl]

    # read fields
    for il in range(data.npoints):
        lptn = data.pset[il]
        data_fld = getattr(lptn,'fld')
        for jl in range(data.ntsnap):
            lfld = read_flt(infile,emode,wdsize,data.nfld)
            for kl in range(data.nfld):
                data_fld[jl][kl] = lfld[kl]

    # place for sorting

    return data

def print_sim_data(data):
    """print simulation data"""
    print('Simulation data:')
    print('    ldim = {0}'.format(data.ldim))
    print('    ntsnap = {0}'.format(data.ntsnap))
    print('    nfld = {0}'.format(data.nfld))
    print('    npoints = {0}'.format(data.npoints))
    print('    time = {0}'.format(data.time))
    # print('Time snapshots:')
    # for il in range(data.ntsnap):
    #     print('    time{0} = {1}'.format(il+1,data.tmlist[il]))

def print_point_data(data,il):
    """print data related to a single point"""
    print('Point data, npt = {0}'.format(il+1))
    lptn = data.pset[il]
    data_glid = getattr(lptn,'glid')
    print('Global point number:')
    print(data_glid)
    data_pos = getattr(lptn,'pos')
    print('Point position:')
    print(data_pos)
    data_fld = getattr(lptn,'fld')
    print('Point fields:')
    print(data_fld)

def write_point_data(data,il):
    """write data related to a single point"""
    lptn = data.pset[il]
    data_glid = getattr(lptn,'glid')
    data_pos = getattr(lptn,'pos')
    data_fld = getattr(lptn,'fld')
    return data_glid, data_pos, data_fld
    
    # For proper 2D mesh reconstruction and contour plotting, we need to:
    # 1. Extract the unique coordinate arrays
    # 2. Create proper meshgrid
    # 3. Map the field data to the correct mesh positions
def wall_data(data_pos,data_fld,mask_,ntsnap):
        # Extract unique coordinate arrays from the off-wall points
        x_coords = np.unique(data_pos[mask_, 0])
        z_coords = np.unique(data_pos[mask_, 2])
        y_coords = np.unique(data_pos[mask_, 1])
        
        #print(f"Unique X coordinates: {len(x_coords)} points")
        #print(f"Unique Z coordinates: {len(z_coords)} points") 
        #print(f"Unique Y coordinates: {len(y_coords)} points")
        
        # Create meshgrid for proper contour plotting
        X_mesh, Z_mesh = np.meshgrid(x_coords, z_coords, indexing='ij')
        
        # Initialize field arrays on the meshgrid
        data_dict = {
            'x': X_mesh,
            'z': Z_mesh,
            'y': y_coords,
            'u': np.zeros(shape=(len(x_coords), len(z_coords),ntsnap)),
            'v': np.zeros(shape=(len(x_coords), len(z_coords),ntsnap)),
            'w': np.zeros(shape=(len(x_coords), len(z_coords),ntsnap)),
            'p': np.zeros(shape=(len(x_coords), len(z_coords),ntsnap)),
            #'vortx': np.zeros(shape=(len(x_coords), len(z_coords),ntsnap)),
            #'vorty': np.zeros(shape=(len(x_coords), len(z_coords),ntsnap)),
            #'vortz': np.zeros(shape=(len(x_coords), len(z_coords),ntsnap)),
        }
        
        # Map the field data to the meshgrid
        # We need to find the correct mapping from the original data to the meshgrid
        for i, point_idx in enumerate(mask_):
            # Get the coordinates of this point
            x_val = data_pos[point_idx, 0]
            z_val = data_pos[point_idx, 2]
            
            # Find the indices in the meshgrid
            x_idx = np.where(x_coords == x_val)[0][0]
            z_idx = np.where(z_coords == z_val)[0][0]
            # Map the field data
            data_dict['u'][x_idx, z_idx,:]     = data_fld[point_idx,:,0]  # First snapshot, first field
            data_dict['v'][x_idx, z_idx,:]     = data_fld[point_idx,:,1]  # First snapshot, second field
            data_dict['w'][x_idx, z_idx,:]     = data_fld[point_idx,:,2]  # First snapshot, third field
            data_dict['p'][x_idx, z_idx,:]     = data_fld[point_idx,:,3]  # First snapshot, fourth field
            #data_dict['vortx'][x_idx, z_idx,:] = data_fld[point_idx,:,4]  # First snapshot, fifth field
            #data_dict['vorty'][x_idx, z_idx,:] = data_fld[point_idx,:,5]  # First snapshot, sixth field
            #data_dict['vortz'][x_idx, z_idx,:] = data_fld[point_idx,:,6]  # First snapshot, seventh field

        return data_dict 

def plot_data(data_dict,ID=0,var=['u','v','w']):
    plt.rc("font",family = "serif")
    plt.rc("font",size = 12)
    plt.rc("axes",labelsize =12, linewidth = 1)
    plt.rc("legend",fontsize= 12, handletextpad = 0.1)
    plt.rc("xtick",labelsize = 12)
    plt.rc("ytick",labelsize = 12)
    fig, axs = plt.subplots(3, 1, figsize=(8, 4),sharex=True)
    # Plot 1: Using meshgrid (correct contour plot)
    for i,var_ in enumerate(var):
        contour = axs[i].contourf(data_dict['x'], data_dict['z'], 
          data_dict[var_][:,:,ID], levels=100,cmap='jet')
        fig.colorbar(contour, ax=axs[i], **{'orientation':'vertical','pad':0.02,'shrink':1,
        #'ticks':np.linspace(-5,5,4)
        })
    axs[1].set_ylabel('Z')
    axs[2].set_xlabel('X')
    axs[2].set_ylabel('Z')
    axs[0].set_title(var[0])
    axs[1].set_title(var[1])
    axs[2].set_title(var[2])
    for ax in axs:
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    fig.subplots_adjust(hspace=0.3)
    return fig, axs

def read_data(path_list,case_name,save_path,Nx,Nz,lx1,SMPSTEP,DT):
    Nfile = len(path_list)
    DATA = {'wall':{"x":[],'z':[],'y':[],'u':[],'v':[],'w':[],'p':[],
                # 'vortx':[],'vorty':[],'vortz':[],
                },
            'y15':{"x":[],'z':[],'y':[],'u':[],'v':[],'w':[],'p':[],
            # 'vortx':[],'vorty':[],'vortz':[]
            }}
    for fname in path_list:
        print(f"Reading data from {fname}")
        
        try:
            data = read_int_fld(fname)
        except:
            print(f"Error reading data from {fname}")
            continue
        print_sim_data(data)
        time = data.time
        npts =  int(data.npoints)
        ldim = data.ldim
        ntsnap = data.ntsnap
        nfld = data.nfld
        data_glid = np.zeros((npts), dtype=np.uint32)
        data_pos = np.zeros((npts,ldim))
        data_fld = np.zeros((npts,ntsnap,nfld))
        il = 0
        for jy in range(2):
            for ix in range(Nx):
                for iz in range(Nz):
                    data_glid_, data_pos_, data_fld_ = write_point_data(data,il)
                    data_glid[il] = data_glid_
                    data_pos[il,:] = data_pos_
                    data_fld[il,:,:] = data_fld_
                    il+=1
        mask_wall = np.where(data_pos[:,1]==0.0)[0]
        mask_y15  = np.where(data_pos[:,1]>0.0)[0]
        data_dict_wall = wall_data(data_pos,data_fld,mask_wall,ntsnap)
        data_dict_y15 = wall_data(data_pos,data_fld,mask_y15,ntsnap)
        for key in data_dict_wall.keys():
            DATA['wall'][key].append(data_dict_wall[key])
        for key in data_dict_y15.keys():
            DATA['y15'][key].append(data_dict_y15[key])

    for key in DATA['wall'].keys():
        DATA['wall'][key] = np.concatenate(DATA['wall'][key],axis=-1)
        print(f"Concatenated {key} for wall: {DATA['wall'][key].shape}")
    for key in DATA['y15'].keys():
        DATA['y15'][key] = np.concatenate(DATA['y15'][key],axis=-1)
        print(f"Concatenated {key} for y15: {DATA['y15'][key].shape}")
    wall_p = DATA['wall']['p']
    sio.savemat(os.path.join(save_path,f'tsrs_{case_name}_wall.mat'),DATA['wall'])
    print(f"Saved wall data to {os.path.join(save_path,f'tsrs_{case_name}_wall.mat')}")
    sio.savemat(os.path.join(save_path,f'tsrs_{case_name}_y15.mat'),DATA['y15'])
    print(f"Saved y15 data to {os.path.join(save_path,f'tsrs_{case_name}_y15.mat')}")

if __name__ == "__main__":
    # in genera there should be loop over files and some concatenation mechanism
    case_id = args.case_id
    print(f"Processing case {case_id}")
    print("--------------------------------")
    config_path = os.path.join(f"../runs_old/{case_id}/current_conf.yml") 
    data_path = os.path.join(f"../runs_old/{case_id}")
    path_list = os.listdir(data_path)
    path_list = [os.path.join(data_path,f) for f in path_list if "env" in f]
    path_list.sort()

    if os.path.exists(config_path):
        with open(config_path,'r') as f:
            config=yaml.load(f,Loader=yaml.FullLoader)

    case_name = config['simulation']['CASENAME']
    Nx = config['simulation']['Nx']
    Nz = config['simulation']['Nz']
    lx1 = config['simulation']['lx1']
    Nx, Nz = Nx*lx1, Nz*lx1
    SMPSTEP = config['simulation']['SMPSTEP']
    DT = config['simulation']['dt']
    print(f"Case Name: {case_name}")
    print(f"Nx: {Nx}")
    print(f"Nz: {Nz}")
    print(f"lx1: {lx1}")
    print(f"SMPSTEP: {SMPSTEP}")
    print(f"DT: {DT}")
    print("--------------------------------")

    for path in path_list:
        path_list = os.listdir(path)
        path_list = [os.path.join(path,f) for f in path_list if f"pts{case_name}0.f" in f]
        path_list.sort()
        if len(path_list) > 0:
            read_data(path_list,case_name,path,Nx,Nz,lx1,SMPSTEP,DT)
        else:
            print(f"No data found in {path}")
            continue
