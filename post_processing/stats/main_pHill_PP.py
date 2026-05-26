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
dbProfs=turbStats.comp(dbProfs_in,nu,rho)   #profiles at different streamwise locations

#6. Plot post-processed profiles
#   To see a complete set of qoiName: 
print('Available Quantities:',dbProfs.keys())
