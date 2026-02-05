"""
Read the variables from the record file
"""
from ast import arg
from logging import root
import os
import re
import numpy as np
import pandas as pd
import scipy.io as sio 
import matplotlib.pyplot as plt
import argparse
from matplotlib import cm 
plt.rc("font",family = "serif")
plt.rc("font",size = 12)
control_blocks = [
(0.25,0.4),(0.4,0.5),(0.5,0.7),(0.7,0.86),
]
cmap = cm.get_cmap('Greys')
N_blocks = len(control_blocks)
region_colors = [cmap(i) for i in np.linspace(0, 1, N_blocks+3)[3:]]
region_configs={}
for il, blocks in enumerate(control_blocks):
  region_configs[blocks] = {
    "xmin":blocks[0],
    "xmax":blocks[1],
    'color':region_colors[il],
    "alpha":0.5}
parser=argparse.ArgumentParser()
parser.add_argument("--case_folder",type=str,default='TD3_1utau_2586_N11')
parser.add_argument("--agent_run_name",type=int,default=405001)
args = parser.parse_args()

base_dir = "./"
root_dir = os.path.join(base_dir,'results')
case_folder = args.case_folder
agent_run_name  = args.agent_run_name
print(f"[IO] case folder = {case_folder}\n agent_run_name = {agent_run_name}",flush=True)


def X2UTAU(x):
  """ interploation of the friction velocity""" 

  pl = np.zeros(10)
  pl[0] = -0.23595416741213884
  pl[1] = 7.531940367787123
  pl[2] = -71.85201540938085
  pl[3] = 368.29571879467534
  pl[4] = -1144.425228828373
  pl[5] = 2260.2692431989067
  pl[6] = -2860.400642033803
  pl[7] = 2250.753528233314
  pl[8] = -1004.3463967380769
  pl[9] = 194.43028958805616
  u_t = 0.0 
  for il, p in enumerate(pl):
    u_t += p * x ** (il)
  return u_t


if __name__ == "__main__":
    folder_path = os.path.join(root_dir, case_folder)
    run_path = os.path.join(base_dir,'runs')
    run_path = os.path.join(run_path, str(agent_run_name))
    run_path = os.path.join(run_path, 'history')
    
    node_csv = pd.read_csv(os.path.join(run_path,'NODE_INFO.csv'))
    node_csv = node_csv.sort_values(by=['x','z'],ascending=True)
    node_csv.to_csv(os.path.join(run_path,'NODE_INFO_sorted.csv'),index=False)
     
    
    x = node_csv['x'].values
    z = node_csv['z'].values
    Nz = 27*12
    Nx = len(x)// Nz
    x = np.reshape(x,(Nx,Nz),order='A')
    z = np.reshape(z,(Nx,Nz),order='A')
    xx, zz = np.meshgrid(np.linspace(x.min(),x.max(),Nx),np.linspace(z.min(),z.max(),Nz))
    
    lst_files = os.listdir(folder_path)
    res_folders = [os.path.join(folder_path, f) for f in lst_files if 'res' in f ]
    res_folders.sort()
    res_folders = [os.path.join(f, 'drl_data') for f in res_folders]

    keyword = 'act'
    data_load = sio.loadmat(os.path.join(folder_path,f"stitched_{keyword}_{len(res_folders)}.mat"),)
    print(f"Load :{data_load[f'{keyword}'].shape}",flush=True)    

    x = data_load['x']
    z = data_load['z']

    u_tau = X2UTAU(x)
    u_tau = u_tau.reshape(Nz,Nx,1,1)
    act = data_load['act'].reshape(Nz,Nx,-1,1)
    act = u_tau * act
    
    fig,axs=plt.subplots(1,1,figsize=(8,4))
    # Subtract the mean 
    # axs.plot(x[0,:], act[:,0,0,0])
    act -= np.mean(act,axis=0,keepdims=True)
    act = np.abs(act)
    act = np.mean(act,axis=0,keepdims=True)
    act = np.mean(act,axis=2,keepdims=True)
    act = act.reshape(-1,1)

    xc = np.linspace(0.25,0.86, len(act))
    ind = np.where((xc<=0.7)&(xc>0.25))[0]
    avg_act = np.mean(act[ind]) 
    print(f"Action Up To 0.7= {avg_act}") 
    axs.plot(xc,(act),label=r"$||v'_{\rm wall}||$")
    axs.plot(xc,u_tau[0,:,0,0],label=r"$u_{\tau}$")
    axs.plot(xc,0.5*act**3,label=r"$\frac{1}{2} ||v'_{\rm wall}||^3$"+f" = {np.mean(0.5*act**3):.2e}" + r"$U_{\infty}$")
    axs.axhline(1e-3,color='k',linestyle='--',label=r"$0.1\% U_{\infty}$")
    axs.set(**{'xlabel':"x/c", "ylabel":"Wall Quantities"})
    axs.legend()
    for b in control_blocks:
      axs.axvspan(**region_configs[b])
    axs.set_xlim(0,0.7)
    fig.savefig("Action.jpg",dpi=300,bbox_inches='tight')


    
    
    # fig,axs = plt.subplots(1,2)
    # clb=axs[0].contourf(xx,zz, data_act[:,:,90,0].T,cmap='RdBu_r',levels=30)
    # fig.colorbar(clb,ax = axs[0])
    # clb=axs[1].contourf(xx,zz,data_obs[:,:,90,1].T,cmap='RdBu_r',levels=30)
    # fig.colorbar(clb,ax = axs[1])
    # axs[0].set_xlabel('x')
    # axs[0].set_ylabel('z')
    # axs[1].set_xlabel('x')
    # axs[1].set_ylabel('z')
    # fig.savefig("Action_Inspect.jpg",dpi=300,bbox_inches='tight')
    
    # fig,axs = plt.subplots(1,2)
    # axs[0].plot(xx,zz, data_act[0,0,:,0],label="1-1")
    # axs[0].plot(xx,zz, data_act[0,10,:,0],label="1-2")
    # axs[0].plot(xx,zz, data_act[0,20,:,0],label="1-3")
    # axs[1].plot(xx,zz,data_act[0,0,:,0],label="3-1")
    # axs[1].plot(xx,zz,data_act[10,0,:,0],label="4-1")
    # axs[1].plot(xx,zz,data_act[20,0,:,0],label="5-1")
    # axs[0].set_xlabel('x')
    # axs[0].set_ylabel('z')
    # axs[1].set_xlabel('x')
    # axs[1].set_ylabel('z')
    # fig.savefig("Action_Inspect.jpg",dpi=300,bbox_inches='tight')
    
    
    # plt.show()
    # fig,axs = plt.subplots(1,1)
    # fig_obs,axs_obs = plt.subplots(1,1)
    # fig_act,axs_act = plt.subplots(1,1)

    # xlocs = [0.33, 0.45, 0.6, 0.78]
    # for xloc in xlocs: 
    #   matched_rows = track_coord(xloc,0.1,node_csv,r=0.01)
    #   if len(matched_rows) > 0: 
    #     first_row = matched_rows.iloc[0]
    #     agent_name = nameAgent(
    #           nid=int(first_row['NID']),
    #           gllid=int(first_row['GLLID']),
    #           iface=int(first_row['FACEID']),
    #           ix=int(first_row['ix']),
    #           iy=int(first_row['iy']),
    #           iz=int(first_row['iz'])
    #       )
    #     dd = data[agent_name]
    #     axs.plot(dd['rew_rec'],label=r"$x_{ss}=$"+f"{xloc:.2f}")    
    #     axs_obs.plot(dd['obs_rec'][:,0],ls='-',label=r"$x_{ss}=$"+f"{xloc:.2f}")    
    #     axs_obs.plot(dd['obs_rec'][:,1],ls='--',label=r"$x_{ss}=$"+f"{xloc:.2f}")    
    #     axs_act.plot(dd['act_rec'],ls='-',label=r"$x_{ss}=$"+f"{xloc:.2f}")    
    # axs.legend()
    # plt.show()

