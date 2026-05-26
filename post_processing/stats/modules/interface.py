###########################################################
# Construct database (dict) from the read-in `int_fld` data
###########################################################
# Saleh Rezaeiravesh, salehr@kth.se
#-----------------------------------------------------------

import numpy as np

def dbMaker(data,nx,ny):
    """
    Making database for fields 'F1', 'F2, ...,'D64', where,
      'F': statistics
      'D': derivatives
    For a complete list of these fields, see: 
    https://github.com/KTH-Nek5000/KTH_Examples/blob/master/pipe_PSTAT2D/pp_python/nom_fields.txt  

    Assumption: Two sets of interpolation points are included in the data 
       (also written with the same order as below):
       1. [Profs] A set of `nx` streamwise locations at each of which `ny` points are considered in 
       the vertical direction to extract different profiles.  
      
    Args: 
      `data`: data
      `nx`: number of points in the streamwise direction [Profs]
      `ny`: number of points in the vertical direction [Profs]

    Returns:
      `dbProfs`: a database containing the coordinates and "stats and derivatives" quantities 
      of the `nx`*`ny` profile points
         Ex.: dbProfs['F1'] is a numpy array of shape (`nx`,`ny`) containing the 'F1' field
    """

    nStats=data.nstat
    nDerivs=data.nderiv
    nPts=data.npoints
    f=np.zeros((nPts,nStats))
    g=np.zeros((nPts,nDerivs))
    pos_=np.zeros((nPts,2))
    dbProfs={}

    for j in range(nPts):
        lptn = data.pset[j]
        f[j,:] = getattr(lptn,'stat')
        g[j,:] = getattr(lptn,'deriv')
        pos_[j,:] = getattr(lptn,'pos')

    #coordinate of the interpolating points
    x_=np.reshape(pos_[:nx*ny,0],[nx,ny],'C')
    y_=np.reshape(pos_[:nx*ny,1],[nx,ny],'C')
    dbProfs.update({'x':x_,'y':y_})

    #F stats
    for i in range(nStats):
        fName='F'+str(i+1)
        f2=f[:,i]
        f2Profs=np.reshape(f2[:nx*ny],[nx,ny],'C')            
        dbProfs.update({fName:f2Profs})

    #D Derivatives
    for i in range(nDerivs):
        fName='D'+str(i+1)
        f2=g[:,i]
        f2Profs=np.reshape(f2[:nx*ny],[nx,ny],'C')    
        dbProfs.update({fName:f2Profs})
    return dbProfs

def read_pp_inputs(filename):
    """
    Read the post-processing parameters from `inputs_phill_pp.in` 
    """ 
    params = {}
    f1=open(filename,'r')
    ain=f1.readlines()
    for i in range(len(ain)):
        ain_sep=ain[i].split()

        if len(ain_sep)>1 and ain_sep[1]=='=':
           key_=ain_sep[0] 
           if key_!='npointsx' and key_!='npointsy' and key_!='xmin' and key_!='xmax' and key_!='ymin' and key_!='ymax':
              val_=int(ain_sep[2])
           elif key_!='npointsy' and key_!='xmin' and key_!='xmax' and key_!='ymin' and key_!='ymax':
              val_=int(ain_sep[2]) 
           elif key_!='xmin' and key_!='xmax' and key_!='ymin' and key_!='ymax':
              val_=float(ain_sep[2]) 
           elif key_!='xmax' and key_!='ymin' and key_!='ymax':
              val_=float(ain_sep[2]) 
           elif key_!='ymin' and key_!='ymax':
              val_=float(ain_sep[2]) 
           elif key_!='ymax':
              val_=float(ain_sep[2]) 

           params.update({key_:val_}) 
    return params       
