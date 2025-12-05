"""
visualisation of the reward
@yuningw
"""

import numpy as np 
import scipy.io as sio 
import os 
import matplotlib.pyplot as plt 
from lib.plot import plt_setUp, colorplate as cc
import pandas as pd 
import argparse
def smooth_Value(raw_data,window_size):
    """Smooth the value by a convolution"""
    kernel = np.ones(shape=(window_size,))
    data = np.convolve(raw_data,kernel,mode='valid')/window_size
    return data

parser = argparse.ArgumentParser()
parser.add_argument('--id',default=199810,type=int)
parser.add_argument('--mean',action='store_true')
parser.add_argument('--single',action='store_true')
arguments = parser.parse_args()

plt.rc("font",family = "serif")
plt.rc("font",size = 20)
plt.rc("axes",labelsize = 30, linewidth = 2)
plt.rc("legend",fontsize= 20, handletextpad = 0.1)
plt.rc("xtick",labelsize = 25)
plt.rc("ytick",labelsize = 25)

style_dict = {'lw':2.0, 'c':cc.red,}
axs_dict = {'xlabel':r'$t^+$',"ylabel":r"$1 - \left({dUdy}_{\rm ctrl}/{dUdy}_{\rm ref}\right)$"}
fig_kw = {'dpi':300,'bbox_inches':'tight'}

DT   = 1e-3 
utau = 0.061
mu   = 1.0/4400.0
tstar = mu/utau**2
ndrl  = 36
TSTART = 0.3999960000012E+03
# CASE_NUM=1730718665
CASE_NUM=1730724685
CASE_NUM=arguments.id
case_path = f"runs/{CASE_NUM}/train"
history_path = case_path + '/history'
figure_path = history_path + '/figs'
if not os.path.exists(figure_path):
  os.makedirs(figure_path)


# Check all the foler 
run_list = os.listdir(case_path)
run_list = [os.path.join(case_path,f) for f in run_list if 'round' in f]
run_list.sort()
print(run_list)
rwd_list_tot = []
for history_case in run_list: 
  file_list = os.listdir(history_case)
  rwd_list_ = [os.path.join(history_case, f) for f in file_list if 'rewlog' in f]
  rwd_list_.sort()
  rwd_list_tot += rwd_list_

## Check the file in the current history folder
file_list = os.listdir(history_path)
rwd_list = [os.path.join(history_path, f) for f in file_list if 'rewlog' in f]
rwd_list.sort()
numfile = len(rwd_list)
print(f"[IO] NUM FILE ={numfile}")
rwd_list_tot += rwd_list
print(f"[IO] HISTORY IN TOTAL:{len(rwd_list_tot)} FILES")

rew_history = []
if arguments.single:
  for fi_,fname_ in enumerate(rwd_list[:]):
    fi = fi_+1
    data = []
    figname=figure_path + f'/Reward_EP_{fi}.jpg'
    is_exist=os.path.exists(figname)
    if not is_exist :
        fname = f'rewlog_{fi:05d}.npz'
        f_to_read = os.path.join(history_path,fname)
        # print(f"[IO] READING {f_to_read}")
        rwd = np.load(f_to_read,allow_pickle=True)
        data = rwd['rew']
        fig,axs = plt.subplots(1,1,figsize=(8,6))
        Nstep = len(data)
        TEND  = Nstep * DT * ndrl   
        t     = np.linspace(0,TEND,Nstep)
        # Rescale the time 
        tplus = t/tstar
        
        axs.plot(tplus[1:Nstep],data[1:Nstep],c=cc.deepgreen,lw=2.5)
        axs.set(**axs_dict)
        fig.savefig(figname,**fig_kw)
        plt.close()
    else:
        print(f"[IO] FILE EXIST:{figname}")
elif arguments.mean:
  rew_history=[]
  for f_to_read in rwd_list_tot[:]:
    if (f'{1:05d}' not in f_to_read):
      rwd = np.load(f_to_read,allow_pickle=True)
      data = rwd['rew']
      rew_history.append(data[:]*1.)
      window_size = data.shape[0]
  rew_history = np.concatenate(rew_history)
  mean_rew = rew_history.reshape(-1,) 
  NEPISODE=mean_rew.shape[0]
  
  episode_x = np.arange(window_size,NEPISODE+1)/(window_size)
  reward_y  = smooth_Value(mean_rew,window_size)

  i_max_reward = np.argmax(reward_y)
  print(f"ID={arguments.id}")
  print(f'The Max R = {(reward_y[i_max_reward])*100:.2f}% at Eps = {int(episode_x[i_max_reward])}')
  print(f"Current R = {reward_y[-1]*100:.2f}% at Eps = {int(episode_x[-1])}")
  fig,axs = plt.subplots(1,1,figsize=(10,8))
  axs.plot(
          episode_x,
          reward_y,
          **{'c':cc.deepgreen,
          'lw':2.0,'linestyle':"-",'alpha':1.0,
          'marker':"D",'markersize':8.5,'fillstyle':"none","markevery":window_size,})
  axs.grid()
  axs.set(**{'xlabel':"Episodes","ylabel":'Mean reward / episode'})
  fig.savefig(figure_path + f'/moving_avg_reward.jpg',**fig_kw)
  
  ## Save the mean data 
  sio.savemat(figure_path+f'/mean_reward_{CASE_NUM}.mat',
              {
              "epsiode":episode_x,
              "reward":reward_y,
              })  

