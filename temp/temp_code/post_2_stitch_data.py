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

parser=argparse.ArgumentParser()
parser.add_argument("--case_folder",type=str,default='TD3_1utau_2586_N11')
parser.add_argument("--agent_run_name",type=int,default=405001)
args = parser.parse_args()

base_dir = "./"
root_dir = os.path.join(base_dir,'results')
case_folder = args.case_folder
agent_run_name  = args.agent_run_name
print(f"[IO] case folder = {case_folder}\n agent_run_name = {agent_run_name}",flush=True)

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

      # Save data       
    data_load = {
        "rew":[],
        "act":[],
        "obs":[],
      }
      
    for ifile, f in enumerate(res_folders):

      for ky in data_load.keys():
        d=sio.loadmat(os.path.join(f,f'stitched_{ky}_{ifile}.mat'),)
        print(f"SHAPE={d[f'{ky}'].shape}")
        data_load[ky].append(d[f"{ky}"])
      
      if ifile ==len(res_folders): 
        data_load['x'] = d['x']
        data_load['z'] = d['z']
        
      print(f"Stitch File Read {ifile}",flush=True) 
    
    for ky in data_load.keys(): 
      data_load[ky] = np.concatenate(data_load[ky],axis=2)
      sio.savemat(os.path.join(folder_path,f"stitched_{ky}_{len(res_folders)}.mat"),
                 { f"{ky}":data_load[ky],
                  'x':d['x'],
                  'z':d['z'],
                 },
      )
      print(f"Save :{data_load[ky].shape}",flush=True)

    
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

