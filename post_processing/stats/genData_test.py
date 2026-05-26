###########################################################
# Post-processing of the 2D statistics of the periodic hill 
###########################################################
# Saleh Rezaeiravesh, salehr@kth.se
# Daniele Massaro, dmassaro@kth.se
#----------------------------------------------------------

import sys
import numpy as np
sys.path.append("./modules/")
from reader_int_fld import read_int_fld
from interface import dbMaker,read_pp_inputs
import turbStats
import scipy.io as sio 
import matplotlib.pyplot as plt 

#1. Read the `int_fld`
# path_int_data='int_fld' (import from matlab)
path_int_data = 'int_fld'
data = read_int_fld(path_int_data)
#print(data.__dict__.keys())   #list of the keys

#3. Read the post-processing input parameters
##params=read_pp_inputs('./inputs_phill_pp.in')

#4. Construct databases (dictionaries) from the data read from 'int_fld'
#   The data contain 'stats' & 'derivative' fields and the coordinates of the interpolation points.
nx=100
ny=1000

dbProfs_in=dbMaker(data,nx,ny)

#5. Compute mean, rms, and budget terms at the interpolation points 
#   (profiles and points on the bottom wall)
# nu=1.0/2800 (import from matlab)
nu = 1.0/2800
print('Viscosity:',nu)
print('File path:',path_int_data)



rho=1
d=turbStats.comp(dbProfs_in,nu,rho)   #profiles at different streamwise locations

#6. Plot post-processed profiles
#   To see a complete set of qoiName: 
# print('Available Quantities:',d.keys())



nx = 100 
ny = 1000 
nu = 1/2800

d['x'] = np.reshape(np.float64(np.nditer(d['x'])),(ny,nx),order='C')
d['yy'] = np.reshape(np.float64(np.nditer(d['y'])),(ny,nx),order='C')
d['U'] = np.reshape(np.float64(np.nditer(d['U'])),(ny,nx),order='C')
d['dUdy'] = np.reshape(np.float64(np.nditer(d['dUdy'])),(ny,nx),order='C')
d['uu'] = np.reshape(np.float64(np.nditer(d['uu'])),(ny,nx),order='C')
d['vv'] = np.reshape(np.float64(np.nditer(d['vv'])),(ny,nx),order='C')
d['ww'] = np.reshape(np.float64(np.nditer(d['ww'])),(ny,nx),order='C')
d['uv'] = np.reshape(np.float64(np.nditer(d['uv'])),(ny,nx),order='C')
d['nu'] = nu 
d['rho'] = 1
d['nx'] = nx
d['ny'] = ny

d['y']  = d['yy'][:,0] + 1

# Mean Vel
d['avg_U']      = np.mean(d['U'],axis=1).astype(np.float64)
d['avg_dUdy']   = np.mean(d['dUdy'],axis=1).astype(np.float64)
d['avg_uu']     = np.mean(d['uu'],axis=1).astype(np.float64)
d['avg_vv']     = np.mean(d['vv'],axis=1).astype(np.float64)
d['avg_ww']     = np.mean(d['ww'],axis=1).astype(np.float64)
d['avg_uv']     = np.mean(d['uv'],axis=1).astype(np.float64)

# friction
d['twall'] = d['rho']*d['nu']*(d['avg_dUdy'][0] - d['avg_dUdy'][-1])/2
d['utau']  = np.sqrt(d['twall']/d['rho'])
d['Ret']   = d['utau']/d['nu']
d['lstar'] = d['nu']/d['utau']

print(f"Ret = {d['Ret']}")
midloc = int(ny/2)

d['yp']     = d['y'][:midloc]/d['lstar']
d['Up']     = (d['avg_U'][:midloc]   + d['avg_U'][-1:-midloc-1:-1])/2/d['utau']
d['uup']     = (d['avg_uu'][:midloc] + d['avg_uu'][-1:-midloc-1:-1])/2/d['utau']
d['vvp']     = (d['avg_vv'][:midloc] + d['avg_vv'][-1:-midloc-1:-1])/2/d['utau']
d['wwp']     = (d['avg_ww'][:midloc] + d['avg_ww'][-1:-midloc-1:-1])/2/d['utau']
d['uvp']     = (d['avg_uv'][:midloc] + d['avg_uv'][-1:-midloc-1:-1])/2/d['utau']

sio.savemat('../results/benchmark.mat',d)