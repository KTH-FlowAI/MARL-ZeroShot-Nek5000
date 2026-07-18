"""
Read the record
@yuningw
"""

import numpy as np 
import os 
import matplotlib.pyplot as plt 
from lib.plot import plt_setUp, colorplate as cc
import pandas as pd 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--id',default=199810,type=int)
parser.add_argument('--env',default=2,type=int)
parser.add_argument('--var',default=1731934871,type=int)
arguments = parser.parse_args()

plt.rc("font",family = "serif")
plt.rc("font",size = 20)
plt.rc("axes",labelsize = 30, linewidth = 2)
plt.rc("legend",fontsize= 20, handletextpad = 0.1)
plt.rc("xtick",labelsize = 25)
plt.rc("ytick",labelsize = 25)


oppo_reward_rec = np.load('reward_oppo.npz')['rew']

style_dict = {'lw':2.0, 'c':cc.red,}
axs_dict = {'xlabel':r'$t^+$',"ylabel":r"$1 - \left({dUdy}_{\rm ctrl}/{dUdy}_{\rm ref}\right)[\%]$"}
fig_kw = {'dpi':300,'bbox_inches':'tight'}

CASE_NUM=arguments.id
REC_NUM=arguments.var
env_num=arguments.env
#[MOD] Evaluation results live under runs/<case>/eval/: per-env records in
#[MOD] eval/env_XXX, and the shared NODE_INFO in eval/history (see below).
eval_root = f'runs/{CASE_NUM}/eval'
history_path = f'{eval_root}/env_{env_num:03d}/'
fig_path = f'{eval_root}/env_{env_num:03d}/figs/'
if not os.path.exists(fig_path): os.makedirs(fig_path)
d = np.load(history_path+f"vars_record_{REC_NUM}.npz")

obs_rec = d['obs_rec']
acts_rec = d['acts_rec']
rew_rec = d['rew_rec']
print(f'[IO] FILE LOADED: OBS: {obs_rec.shape}, ACTs: {acts_rec.shape}, REWARD {rew_rec.shape}')

DT   = 2e-3 
utau = 0.063
mu   = 1.0/2800.0
tstar = mu/utau**2
TSTART = 0.1089995999988E+04


#[MOD] NODE_INFO now lives in the shared eval/history (not per-env), because the
#[MOD] evaluation env writes its history under runs/<case>/eval/history.
agent_info = pd.read_csv(f'{eval_root}/history/'+"NODE_INFO.csv").to_dict()
agent_info['NAME']=[]
print(agent_info.keys())
num_agent=len(agent_info['NID'])
print(f'[Agent] NUM AGENT={num_agent}')
agent_dict={}
for il in range(num_agent):
  nid = agent_info['NID'][il]
  gllid=agent_info['GLLID'][il]
  iface=agent_info['FACEID'][il]
  ix=agent_info['ix'][il]
  iy=agent_info['iy'][il]
  iz=agent_info['iz'][il]
  agent_name = f"jet_np{nid:05d}_gid{gllid:05d}_iface{iface}_ix{ix:05d}_iy{iy:05d}_iz{iz:05d}"
  agent_dict[agent_name] = {
                          "obs":obs_rec[:,il,:],
                          "act":acts_rec[:,il,:],
                          "rwd":rew_rec[:,il,],
                            }

# color_bar = plt.get_cmap('plasma')
cmap = plt.get_cmap('plasma')
colors = cmap(np.linspace(0,1,num_agent))
Nstep = obs_rec.shape[0]
TEND  = Nstep * DT * 27  
t     = np.linspace(0,TEND,Nstep)
# Rescale the time 
tplus = t/tstar
ind = np.where(tplus>=500)[0][0]


fig,axs = plt.subplots(1,1,figsize=(12,4))
il=10
axs.plot(tplus[:Nstep],acts_rec[:Nstep,il,0],'-o',c=colors[il],alpha = 0.1)
axs.set(**axs_dict)
axs.set_ylabel('ACTIONS')
fig.savefig(fig_path+f'/ACTION_EP_AVG_{REC_NUM}.jpg',**fig_kw)

oppo_reward_rec =oppo_reward_rec[::20] 
fig,axs = plt.subplots(1,1,figsize=(8,8))
axs.plot(tplus[:Nstep],rew_rec[:Nstep,0],c=cc.deepgreen,lw=3.5)
axs.plot(tplus[:Nstep],oppo_reward_rec[:Nstep],c=cc.red,lw=3.5)
axs.set(**axs_dict)
axs.legend([f'DRL={np.mean(rew_rec[ind:Nstep,0])*100:.2f}[%]',
            f"OC={np.mean(oppo_reward_rec[ind:Nstep])*100:.2f}[%]",
            ])
fig.savefig(fig_path+f'/REWARD_EP_AVG_{REC_NUM}.jpg',**fig_kw)

fig,axs = plt.subplots(1,1,figsize=(6,6))
axs.plot(rew_rec[:Nstep,0],acts_rec[:Nstep,il,0],'o',c=cc.yellow)
axs.set(**{'xlabel':'reward','ylabel':'Actions'})
fig.savefig(fig_path+f'/REW_VS_ACTION_EP_AVG_{REC_NUM}.jpg',**fig_kw)
