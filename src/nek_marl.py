"""
MARL ENV using NEK5000 
@yuningw
"""

# from gym import spaces
# NOTE: This is depends on the verison of STB3 
from gym import spaces
import numpy as np
import functools
import pettingzoo
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn as  parallel_to_aec 
import os,math,time,shutil,sys,datetime
from omegaconf import OmegaConf 
import pandas as pd 
from configs import Config
from pathlib import Path
from mpi4py import MPI
from lib.lglnodes import lglnodes
from lib.nek_utils import (remove_sch)
from lib.reward_logger import RewardLogger, SimpleRewardLogger
def env():
    """
    The env function often wraps the environment in wrappers by default.
   
    """
    env = raw_env()
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(conf:Config, rank_folder):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(conf, rank_folder)
    env = parallel_to_aec(env)
    return env

class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self,conf,rank_folder,sub_comm):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        """
        self.conf = conf
        self.folder=rank_folder
        # Obtain the communication
        self.sub_comm = sub_comm
        self.initialization()

    def initialization(self):
        print(f'------------ INITIALIZATION -------------',flush=True)
        
        #--------------I/O-----------------------
        self.history_path=Path(f"{self.folder}/history")
        self.history_path.mkdir(exist_ok=True)
        self.rstart_folder = Path(f"{os.getcwd()}/{self.conf.simulation.restart_folder}")
        
        # Remove the .sch file if it exists
        remove_sch(self.folder)

        ## We save the config for the run
        ##-----------------------------------
        OmegaConf.save(self.conf,os.path.join(self.history_path,'current_conf.yml'))
        OmegaConf.save(self.conf,os.path.join(self.folder,'current_conf.yml'))
        ##-----------------------------------

        #-----------------------------------------
        # Spawning Nek5000 as MPI process
        #-----------------------------------------
        mpi_info = MPI.Info.Create()
        mpi_info.Set('wdir',f"{os.getcwd()}/{self.folder}")
        mpi_info.Set('bind_to','none')
        if self.conf.simulation.hostfile != '':
            mpi_info.Set('hostfile',self.conf.simulation.hostfile)
            print('[STB3] LOAD HOSTFILE!')
        self.mpi_info = mpi_info
        
        # -----------AGENT--------------
        # NOW, wait for the NODE information to start 
        self.init_agent()
        self.possible_agents = [
                                self.nameAgent(nid,gllid,iface,ix,iy,iz)
                                for nid, gllid,iface,ix,iy,iz 
                                in zip(self.agent_info['NID'],
                                    self.agent_info['GLLID'],
                                    self.agent_info['FACEID'],
                                    self.agent_info['ix'],
                                    self.agent_info['iy'],
                                    self.agent_info['iz'],)]
        
        # print(self.possible_agents)
        # Mapping 
        self.agent_name_mapping = dict(
            zip( self.possible_agents, list(range(len(self.possible_agents))) )
                                    )
        
        
        # -----------STATE--------------
        # STATE BUFFER
        self.full_observation=\
            np.ndarray(shape=(self.conf.runner.npl_state,self.nAgents))
        ## Scaling the data
        self.utau = self.conf.runner.u_tau # u_tau for wing @ x/c =  0.4
        # -----------ACTION--------------
        # Gll node weight 
        _,self.gll_weight,_ = lglnodes(N=self.conf.simulation.lx1-1)
        print(f"[STB3] GLL WEIGHT={self.gll_weight}",flush=True)
        # Action rescaling variables
        self.rescale_actions = self.conf.runner.rescale_actions
        if self.rescale_actions:
            self.rescale_factors= [[self.conf.runner.ctrl_max_amp,self.conf.runner.ctrl_max_amp]]
            self.ctrl_min_amp=-1.0
            self.ctrl_max_amp=1.0
        else:
            # Bounding Control Amplitude
            self.ctrl_min_amp=self.conf.runner.ctrl_min_amp
            self.ctrl_max_amp=self.conf.runner.ctrl_max_amp
            # Output shape
        self.action_shape=[1,]

        # -----------REWARD--------------
        self.reward_fn = self.conf.runner.reward_fn
        # Baseline Reward
        self.baseline_dudy=self.conf.runner.dUdy # For N=5 @ x/c = 0.4
        # tau_wall = nu * dUdy; nu = 1/|viscosity| when viscosity < 0 (NEK convention)
        nu = 1.0 / abs(self.conf.simulation.viscosity)
        self.baseline_tau_wall = nu * self.baseline_dudy
        # Reward history logging variables
        self.restart_index = 0
        self.act_index = 0
        self.reward_log = list()
        
        # Initialize real-time reward logger
        self.reward_logger = RewardLogger(
            log_dir=self.history_path,
            log_name="rewards",
            log_per_agent=False,
            log_aggregated=True,
            flush_frequency=10
        )
 


        print(f"SCALE: dUdy={self.baseline_dudy}\n Utau={self.utau}",flush=True)
        # Reward-related variables
        if self.conf.runner.rew_mode == 'MovingAverage':
            self.reward_history = RingBuffer(length=self.conf.runner.size_history,
                                            dim=(self.nAgents,))
            # The history of the wall shear-stress is initialized with
            # an the reference value
            for i_h in range(self.conf.runner.size_history):
                self.reward_history.data[i_h] = self.baseline_dudy*np.ones((self.nAgents,))
        print(f'------------ FINISH -------------',flush=True)


    def init_agent(self):
        """INIT for NEK5000, get node ID and value""" 
        request=b"INTAL"
        self.sub_comm.Send([request,MPI.CHARACTER],dest=0,tag=tag_dict["COMMAND"]['tag'])
        self.nAgents = 0

        ## Creat Agent info 
        self.agent_info={}
        for k in tag_dict.keys():
            if tag_dict[k]['cate'] =='info':
                self.agent_info[k] = []

        # A hand-shake from NEK, let me know which nid I should recv info
        node_list = np.empty((self.conf.simulation.nproc,),dtype=np.int32)
        print(f'[STB3] REQUEST NODE LIST',flush=True)
        self.sub_comm.Recv([node_list,MPI.INTEGER],0,tag=tag_dict['NID']['tag'])
        # Masking the nid_list
        nid_list = np.arange(self.conf.simulation.nproc,dtype=tag_dict['NID']['py_dtype'])
        nid_list = nid_list[node_list!=0]
        print(f'[STB3] NODE LIST GET:{nid_list}',flush=True)
        
        # RECV on those RANKS only
        for nid in (nid_list):
            rank_data = {}
            rank_data['NID'] = np.array([nid],dtype=tag_dict['NID']['py_dtype']) 
            for k in self.agent_info.keys(): 
                ## Prepare Tensor, NID and NUMCTRL are Scalars
                if 'NID' not in k: 
                    if 'NUMCTRL' not in k:
                        rank_data[k] = np.empty((self.conf.simulation.TOTCTRL,),
                                            dtype=tag_dict[k]['py_dtype'])
                    else: 
                        rank_data[k] = np.empty((1,),
                                            dtype=tag_dict[k]['py_dtype'])
                    
                    ## MPI RECV
                    self.sub_comm.Recv([rank_data[k],tag_dict[k]['mpi_dtype']],nid,tag=nid+tag_dict[k]['tag'])

            ## Resort to match the length 
            numctrl = rank_data['NUMCTRL'][0]
            for k in rank_data.keys():
                if ('NUMCTRL' in k) or ("NID" in k):
                    # print(rank_data[k])
                    rank_data[k] = rank_data[k][0] * np.ones(shape=(numctrl,),dtype=tag_dict[k]['py_dtype'])
                else:
                    rank_data[k] = rank_data[k][:numctrl]
                # print(f'{k}: SHAPE={rank_data[k].shape}',flush=True)
                self.agent_info[k].append(rank_data[k])
            
            self.nAgents +=numctrl
        
        for k in self.agent_info.keys():
            self.agent_info[k] = np.concatenate(self.agent_info[k])
        print(f"[STB3] INIT END, NUM AGENT={self.nAgents}")

        self.uniqID = np.unique(self.agent_info['NID'])
        self.nNID = len(self.uniqID)

        # Dump the information in Pandas
        df = pd.DataFrame(self.agent_info)
        fname=os.path.join(self.history_path,'NODE_INFO.csv')
        df.to_csv(fname)
        print(f"[STB3] DUMP NODE INFO : {fname}")

        return 
    

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(  low = self.ctrl_min_amp,
                            high = self.ctrl_max_amp,
                            shape = self.action_shape,
                            dtype = np.float32)
    
    #### Define the observation space 
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(  low = -np.inf,
                            high = np.inf,
                            shape = (self.conf.runner.npl_state,
                                self.conf.simulation.nzs,
                                self.conf.simulation.nxs),
                                dtype = np.float32)


    def render(self, mode='human', close=False):
        ...

    def close(self):
        print(f"[STB3] CLOSE ENV",flush=True)
        
        # Close reward logger
        if hasattr(self, 'reward_logger'):
            self.reward_logger.close()
            print(f"[STB3] REWARD LOGGER CLOSED",flush=True)
        
        self.end_simulation(farewell=True)
        # Wait until all the operations are completed
        time.sleep(1)

    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        Returns the observations for each agent
        """

        print("[STB3] RESET!",flush=True)
        self.agents = self.possible_agents[:]
        # Close the current SIMSON simulation
        # self.end_simulation()
        # update restart_index
        self.restart_index +=1
        print(f'[STB3] EPISODE={self.restart_index}',flush=True)
        # Re-initialize the action index
        self.act_index = 0

        # Save reward log and re-initialized it
        np.savez(os.path.join(self.history_path,
                            f'rewlog_{self.restart_index:05d}.npz'),
                rew=np.array(self.reward_log))
        
        # Log episode summary
        self.reward_logger.log_episode_summary(episode=self.restart_index)

        self.reward_log = list()
        print('[STB3] SAVE LOG',flush=True)


        remove_sch(self.folder)
        
        ### TODO: Potentially, Add management of logfiles 
        ### but as nek does not generate logfile automatically, it is not worth 
        self.restart_handle()
        # Open a new simulation
        self.start_simulation()
        print('[STB3] Start SIM',flush=True)
        # Return current (initial) state
        time, observation = self.state()
        # Distribute observations to the agents
        observations = self._distribute_field(observation,reward=False)

        return observations

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """

        # Resetting reward value
        rewards = {}

        # Trasform actions to write before re-scaling
        ctrl_value = {}            
        for i_a,agent in enumerate(self.agents):
            ctrl_value[agent] = actions[agent]  
        
        
        # Linear mapping of the actions if the range differs
        # breakpoint()
        if self.rescale_actions:
            for i_a,agent in enumerate(self.agents):
                if actions[agent] < 0:
                    actions[agent] *= self.rescale_factors[0][0]
                if actions[agent] > 0:
                    actions[agent] *= self.rescale_factors[0][1] 

        # Sending the new action values to the environment
        self.action(ctrl_value)

        # Let the solution evolve with the new control values
        rewards = self.evolve()
        
        # Obtain new observation
        flow_time, observation = self.state()

        # Distribute observations to the agents
        observations = self._distribute_field(observation,reward=False)

        # YW: Modified here, If we fixed the time step, it is able to stoped by the maximum amount of steps 

        infos = {agent: {} for agent in self.agents}
        for agent in self.agents:
            infos[agent]['time'] = flow_time        
        
        self.act_index += 1
        
        # Check whether we have approximately reached the maximum simulation time
        if flow_time > self.conf.simulation.tmax:
            dones = {agent : True for agent in self.agents}
        else:
            dones = {agent : False for agent in self.agents}
        
        # Add check regarding the maximum number of interactions
        if self.act_index >= self.conf.runner.nb_interactions:
            print(f'[STEP] ACT_INDEX={self.act_index}; DONES == TRUE',flush=True)
            dones = {agent : True for agent in self.agents}
        
        ## Test if Nek is still working 
        #self.HeartBeat()
        return observations,rewards,dones,infos


    def _distribute_field(self,field:dict,reward=False):
        distributed_fields = {}
        
        ## For READING STATE 
        ### NOTE: fld buffer has shape=[nNID,nfield,TOTCTRL]
        icount=0
        for il, nid in enumerate(self.uniqID): 
            indx = np.where((self.agent_info['NID']==nid))[0]
            for jl, gllid in enumerate(self.agent_info['GLLID'][indx]):
                agent_name = self.nameAgent(nid=nid,gllid=gllid,iface=self.agent_info["FACEID"][indx][jl],
                                            ix=self.agent_info['ix'][indx][jl],
                                            iy=self.agent_info['iy'][indx][jl],
                                            iz=self.agent_info['iz'][indx][jl],
                                            )
                state_     = field['fld'][il,:,jl].reshape(-1,1,1)
                distributed_fields[agent_name] = state_
                icount+=1
        assert icount == self.nAgents,ValueError('[STB3] noAgent NOT MATCH!')
        # print(f"[STB3] REDISTRIBUTED for {icount} Agents")
        
        return distributed_fields


#-------------------
# Requested Methods
#-------------------
    def start_simulation(self):
        # Restart the simulation into loop
        request=b"RSETS"
        self.sub_comm.Send([request, tag_dict['COMMAND']['mpi_dtype']], dest=0, tag=22)
        

    def end_simulation(self,farewell=False):
        """
        Finish the current time loop
        Arg:[bool] farewell :: True==> Turn OFF the current run, False: restart the time loop simulation.
        """
        
        if farewell:
            request=b"TERMN"
            self.sub_comm.Send([request,tag_dict['COMMAND']['mpi_dtype']],dest=0,tag=tag_dict['COMMAND']['tag'])
            # NOTE: Since NEK is disconnected, I did not see the point of disconnect here 
            self.sub_comm.Free()
            sleep_time = 0.01 * self.conf.simulation.nproc
            print(f'[STB3] TERMN ENV, Sleep: {sleep_time}SEC')
            time.sleep(sleep_time)
            MPI.Finalize()

        else:
            print(f"[STB3] Restart the simulation!",flush=True)
            self.start_simulation()

    def state(self):
        #self.HeartBeat()
        # Send request
        request = b'STATE'
        self.sub_comm.Send([request,tag_dict['COMMAND']['mpi_dtype']],dest=0,tag=tag_dict['COMMAND']['tag'])
        
        # Important size constant
        NFLDC=self.conf.runner.npl_state 
        TOTCTRL=self.conf.simulation.TOTCTRL
        # Hand-shake, Current time
        current_time = np.ndarray((1,),dtype=np.float64)
        self.sub_comm.Recv([current_time,MPI.DOUBLE],0,tag=1998)
        current_time = current_time[0]
        # Current State
        current_state = {
                        "NID":np.ndarray(shape=(self.nNID,),dtype=np.int32),
                        "fld":np.ndarray(shape=(self.nNID,NFLDC,TOTCTRL),dtype=np.float64),
                        }
        # Only recv the data from partion with controlled elem
        for ni, nid in enumerate(self.uniqID):
            state_buffer=np.ndarray(shape=(NFLDC,TOTCTRL),dtype=np.float64)
            for t in range(NFLDC):
                buffer= np.ndarray(shape=(TOTCTRL),dtype=np.float64)
                
                self.sub_comm.Recv([buffer,tag_dict['STATE']['mpi_dtype']],
                                    nid,
                                    tag=nid*(t+1)+tag_dict['STATE']['tag'])
                
                state_buffer[t,:] = buffer[:]
                # state_buffer[t,:] = buffer[:]
            current_state['NID'][ni] = nid 
            current_state['fld'][ni,:,:] = self._normalize_state(state_buffer)
            # print(f"[STB3] AT NID= {current_state['NID'][ni]}\n BUFFER:{current_state['fld'][ni,:,:]}",flush=True)
        print('[STB3] STATE RECV',flush=True)

        return current_time, current_state


    def action(self,ctrl_value:dict):
        #self.HeartBeat()
        request=b"CNTRL"
        self.sub_comm.Send([request,tag_dict['COMMAND']['mpi_dtype']],dest=0,tag=tag_dict['COMMAND']['tag'])

        ## Apply the ZNMF condition 
        ctrl_value = self.avg_ZNMF(ctrl_value=ctrl_value)

        ## Sending MPI
        icount = 0 
        for il, nid in enumerate(self.uniqID):
            # Write Buffer 
            indx = np.where((self.agent_info['NID']==nid))[0]
            numctrl = self.agent_info['NUMCTRL'][indx][0]
            act_buffer = np.ndarray(shape=(self.conf.simulation.TOTCTRL),
                                    dtype=tag_dict['ACTION']['py_dtype'])
            for jl, gllid in enumerate(self.agent_info['GLLID'][indx]):
                agent_name = self.nameAgent(nid = nid, gllid=gllid,iface=self.agent_info["FACEID"][indx][jl],
                                            ix=self.agent_info['ix'][indx][jl],iy=self.agent_info['iy'][indx][jl],iz=self.agent_info['iz'][indx][jl],
                                            )
                
                # NOTE: Here we subtract the mean to ensure ZNMF condition            
                act_buffer[jl] = ctrl_value[agent_name]
                icount +=1
            # Send Buffer 
            self.sub_comm.Send([act_buffer,tag_dict['ACTION']['mpi_dtype']],nid,
                                tag=nid+tag_dict['ACTION']['tag'])    
        # Sanity Check 
        assert icount == self.nAgents,ValueError('[STB3] noAgent NOT MATCH!')
        print(f"[STB3] ACTION for {icount} Agents",flush=True)
        return 
    

    def evolve(self):
        #self.HeartBeat()
        request=b"EVOLV"
        self.sub_comm.Send([request,tag_dict['COMMAND']['mpi_dtype']],
                            dest=0,tag=tag_dict['COMMAND']['tag'])
        
        
        # Synchorize with NEK 
        i_evolv = 1 
        while i_evolv <= (self.conf.simulation.ndrl):
            # Recv CFL data, keeping sync between NEK and STB3
            #--------------------------------
            current_cfl = np.ndarray((1,),dtype=np.float64)
            self.sub_comm.Recv([current_cfl,tag_dict['current_cfl']['mpi_dtype']],
                                0,tag=tag_dict['current_cfl']['tag'])
            current_cfl = current_cfl[0]
            
            # print(f"[STB3] i_evlov {i_evolv}/{self.conf.simulation.ndrl} CFL={current_cfl:.3f}",flush=True)
            
            # If we found cfl explode
            if current_cfl >= self.conf.simulation.target_cfl:
                print(f"[WARNING] {i_evolv}/{self.conf.simulation.ndrl} Current {current_cfl} >= {self.conf.simulation.target_cfl}!",flush=True)
                # Exit the entire framework
                self.end_simulation(farewell=True)
                exit()
            #--------------------------------

            # Recv Buffer only at the last step, reduce the work load of MPI.
            #--------------------------------
            if i_evolv == self.conf.simulation.ndrl:
                TOTCTRL = self.conf.simulation.TOTCTRL
                if self.reward_fn == 'net_gain':
                    tau_buf = np.ndarray(shape=(self.nNID, TOTCTRL), dtype=np.float64)
                    pw_buf  = np.ndarray(shape=(self.nNID, TOTCTRL), dtype=np.float64)
                    v3_buf  = np.ndarray(shape=(self.nNID, TOTCTRL), dtype=np.float64)
                    for il, nid in enumerate(self.uniqID):
                        rb = np.ndarray(shape=(TOTCTRL,), dtype=np.float64)
                        self.sub_comm.Recv([rb, MPI.DOUBLE], nid, tag=nid+tag_dict['REWRD']['tag'])
                        tau_buf[il, :] = rb
                        self.sub_comm.Recv([rb, MPI.DOUBLE], nid, tag=nid+tag_dict['REWRD_PW']['tag'])
                        pw_buf[il, :] = rb
                        self.sub_comm.Recv([rb, MPI.DOUBLE], nid, tag=nid+tag_dict['REWRD_V3']['tag'])
                        v3_buf[il, :] = rb
                    tau_dist = self._distribute_field({'fld': np.expand_dims(tau_buf, 1)}, reward=True)
                    pw_dist  = self._distribute_field({'fld': np.expand_dims(pw_buf,  1)}, reward=True)
                    v3_dist  = self._distribute_field({'fld': np.expand_dims(v3_buf,  1)}, reward=True)
                else:
                    ws_stress_buffer = np.ndarray(shape=(self.nNID, TOTCTRL),
                                                  dtype=tag_dict["REWRD"]['py_dtype'])
                    for il, nid in enumerate(self.uniqID):
                        recv_buffer = np.ndarray(shape=(TOTCTRL,), dtype=np.float64)
                        self.sub_comm.Recv([recv_buffer, tag_dict['REWRD']['mpi_dtype']],
                                           nid, tag=nid+tag_dict["REWRD"]['tag'])
                        ws_stress_buffer[il, :] = recv_buffer
                    ws_stress_buffer = self._distribute_field(
                        {'fld': np.expand_dims(ws_stress_buffer, 1)}, reward=True)
            #--------------------------------
            i_evolv +=1
        #---- While Loop End here---------

        # Compute normalized rewards
        rewards = {}
        for il, nid in enumerate(self.uniqID):
            indx = np.where((self.agent_info['NID']==nid))[0]
            for jl, gllid in enumerate(self.agent_info['GLLID'][indx]):
                agent_name = self.nameAgent(nid=nid,gllid=gllid,
                                            iface=self.agent_info["FACEID"][indx][jl],
                                            ix=self.agent_info['ix'][indx][jl],
                                            iy=self.agent_info['iy'][indx][jl],
                                            iz=self.agent_info['iz'][indx][jl],
                                            )
                if self.reward_fn == 'net_gain':
                    i_reward = self._normalize_reward(
                        tau_w=tau_dist[agent_name],
                        pw=pw_dist[agent_name],
                        v3=v3_dist[agent_name],
                    )
                    r_reward = tau_dist[agent_name]  # for logging
                else:
                    r_reward = ws_stress_buffer[agent_name]
                    i_reward = self._normalize_reward(dudy=r_reward)
                rewards[agent_name] = i_reward

        # Logging
        self.reward_log.append(i_reward)
        raw_dict = {agent_name: r_reward.squeeze() for agent_name in rewards.keys()}
        if self.reward_fn == 'net_gain':
            ref = self.baseline_tau_wall
            components = {
                'R_tau': float(1.0 - np.mean([np.mean(tau_dist[a]) for a in rewards]) / ref),
                'R_pw':  float(-np.mean([np.mean(pw_dist[a])  for a in rewards]) / ref),
                'R_v3':  float(-np.mean([np.mean(v3_dist[a])  for a in rewards]) / ref),
            }
        else:
            components = None
        self.reward_logger.log_rewards(
            rewards=rewards,
            dUdy_raw=raw_dict,
            episode=self.restart_index,
            step=self.act_index,
            components=components,
        )
        print(f"[LOGGER] act_index={self.act_index} raw_rwd={r_reward.squeeze():.5f} R={i_reward:.5f} fn={self.reward_fn}",flush=True)
        
        return rewards
        
#######################
# Utility Function 
#####################
    
    @staticmethod
    def nameAgent(nid,gllid,iface,ix,iy,iz):
        """
        Name the agent based on the GRID information 
        nid   [int] RANK 
        gllid [int] Element ID 
        iface [int] number of face from 1~6 
        ix    [int] number of x indices 
        iy    [int] number of y indices 
        iz    [int] number of z indices 
        """
        agent_name = f"jet_np{nid:05d}_"+\
                        f"gid{gllid:05d}_"+\
                        f"iface{iface}_"+\
                        f"ix{ix:05d}_"+\
                        f"iy{iy:05d}_"+\
                        f"iz{iz:05d}" 
        return agent_name 
    

    def avg_ZNMF(self,ctrl_value:dict):
        """
        Python-end Averaging for the zero-net-mass-flux (znmf) condition

        Args:
        ctrl_value [dict] The actions from STB3

        Returns:
        ctrl_value [dict] The actions with ZNMF condition
        """

        # Averaging Scheme

        ## Naive Average
        if self.conf.simulation.znmf_avg == -1:
            mean_action=0;num_=0
            for il, aval in enumerate(ctrl_value.values()):
                mean_action+=aval
                num_ =+il 
            mean_action/=num_
            
            for agent_name in ctrl_value.keys():
                ctrl_value_single     =ctrl_value[agent_name] 
                ctrl_value_znmf       =ctrl_value_single - mean_action 
                ctrl_value[agent_name]=ctrl_value_znmf
                # print(f"[ZNMF] {agent_name}:\n BEF={ctrl_value_single},NOW={ctrl_value_znmf}",flush=True)
        
        ## Weighted Average 
        elif self.conf.simulation.znmf_avg == -2:
            mean_action=0;wxz=0
            for il in range(self.nAgents):
                ix,iz=self.agent_info['ix'][il],self.agent_info['iz'][il]
                agent_name = self.nameAgent(self.agent_info['NID'][il],self.agent_info['GLLID'][il],self.agent_info['FACEID'][il],
                                            self.agent_info['ix'][il],self.agent_info['iy'][il],self.agent_info['iz'][il],)
                i_action= ctrl_value[agent_name]
                wx=self.gll_weight[ix-1];wz=self.gll_weight[iz-1]
                wxz +=wx*wz
                mean_action +=i_action*wx*wz
            mean_action /=wxz

            for agent_name in ctrl_value.keys():
                ctrl_value_single     =ctrl_value[agent_name] 
                ctrl_value_znmf       =ctrl_value_single - mean_action 
                ctrl_value[agent_name]=ctrl_value_znmf
                # print(f"[ZNMF] {agent_name}:\n BEF={ctrl_value_single},NOW={ctrl_value_znmf}",flush=True)
        
        
        elif (self.conf.simulation.znmf_avg==1) or (self.conf.simulation.znmf_avg==0): 
            mean_action = 0.0
            print(f"[ZNMF] DONE BY NEK5000",flush=True)
        
        else:
            raise NotImplementedError('Please Ensure the ZNMF condition!')
        return ctrl_value


    def _normalize_state(self,state):
        """
        Normalize the state by various method 
        """
        if self.conf.runner.normalize_input !="None":
            if self.conf.runner.normalize_input =="utau":
                state /= self.utau

            elif self.conf.runner.normalize_input =="std":
                state /= np.std(state)
            
            elif self.conf.runner.normalize_input =="minmax":
                state =  2* (state - np.min(state))/(np.max(state)-np.min(state)) - 1
            
            else: 
                pass 

        return state 
    
    
    def _assign_reward(self,reward_buffer:np.ndarray):
        """ Assign the reward buffer to Assign the value """
        
        if self.conf.runner.rew_mode=='Homo':
            # We let all the agent share the same reward by taking average
            reward_buffer=np.ones_like(reward_buffer)*\
                        np.mean(reward_buffer.flatten())
            return reward_buffer
        
        elif self.conf.runner.rew_mode=="InHomo":
            # We leave the rewards as they were.
            return reward_buffer
        
        else:
            raise NotImplementedError("[REWARD] Please Choose Available Assign Method!")

    def _normalize_reward(self, dudy=None, tau_w=None, pw=None, v3=None):
        """Compute scalar reward from raw Fortran buffers.

        'dudy' mode  : R = 1 - dUdy / dUdy_ref
        'net_gain'   : R = alpha*(1 - tau_w/ref) + beta*(-pw/ref) + gamma*(-v3/ref)
                         = alpha*R_wallshear + beta*R_pw + gamma*R_v3
        """
        ref = self.baseline_tau_wall
        if self.reward_fn == 'net_gain':
            alpha = self.conf.runner.reward_alpha
            beta  = self.conf.runner.reward_beta
            gamma = self.conf.runner.reward_gamma
            R_wallshear = 1.0 - np.mean(tau_w) / ref
            R_pw        = -np.mean(pw)  / ref
            R_v3        = -np.mean(v3)  / ref
            return alpha * R_wallshear + beta * R_pw + gamma * R_v3
        else:
            return 1.0 - (np.mean(dudy) / self.baseline_dudy)

    
    def restart_handle(self):
        """
            Mangement of rs8/rs6 data for each episode
            if random_init > 0,   we shuffle the RESTARTS to use.
            if random_init == -1, we use the specified No.INIT 
            if random_init == -2 and restart_index==1, Not OverWrite the RSTART for the first run 
        """
        if self.conf.runner.random_init>0: 
            n_init = np.random.randint(low=1,high=self.conf.runner.random_init+1)
            target_folder = os.path.join(self.rstart_folder,f"init_{n_init}")
            rs_list = os.listdir(target_folder)
            rs_list = [f for f in rs_list if 'rs' in f ]
            for rsfile in rs_list: 
                rsfile = os.path.join(target_folder,rsfile)
                shutil.copy(rsfile,dst=self.mpi_info['wdir']+'/')
                print(f"[RSTART] RESET: {rsfile}",flush=True)

        elif self.conf.runner.random_init==-1:
            print(f'[RSTART] NOT SHUFFLE; RANK={self.conf.runner.rank}',flush=True)
            target_folder = os.path.join(self.rstart_folder,f"init_{self.conf.runner.rank}")
            n_init = self.conf.runner.rank
            rs_list = os.listdir(target_folder)
            rs_list = [f for f in rs_list if 'rs' in f ]
            for rsfile in rs_list: 
                rsfile = os.path.join(target_folder,rsfile)
                shutil.copy(rsfile,dst=self.mpi_info['wdir']+'/')
                print(f"[RSTART] RESET: {rsfile}",flush=True)
        
        elif (self.conf.runner.random_init<-1) and (self.restart_index == 1):
            print(f'[RSTART] NOT OVERWIRTE; RANK {self.conf.runner.rank}',flush=True)
            target_folder = os.path.join(self.rstart_folder,f"init_{self.conf.runner.rank}")
            ## But give a Sainty check, ensure at least rs8 files exist
            source_folder = self.mpi_info['wdir']+'/'
            rs_list = os.listdir(target_folder)
            rs_list = [f for f in rs_list if 'rs' in f ]
            
            if len(rs_list) < 3:
                raise ValueError(f"[RSTART] NOT ENOUGH FILE TO RESTART")
            else:
                for rsfile in rs_list:
                    print(f"[RSTART] EXIST: {rsfile}",flush=True)



## A HeartBeat Mechanism in case of emergency exit of NEK 
    def HeartBeat(self):
        # # Sync with subcommnuication
        request = self.sub_comm.Ibarrier()  # Non-blocking barrier to check if workers finish
        flag = request.Test()  # Check if the request is completed
        if flag:
            print("Worker has finished execution.")
            exit()
        print("Worker is still running...")
        time.sleep(1)


class RingBuffer():
    "A n-dimensional ring buffer using numpy arrays"
    def __init__(self, length, dim=1):
        if type(dim) is int:
            dim = (dim,)
        self.data = np.zeros((length,)+dim, dtype='f')

        self.index = 0

    def extend(self, x):
        "adds array x to ring buffer"
        assert x.shape==self.data.shape[1:],'Input array does not match \
            the ring buffer size'
        # breakpoint()
        # x_index = (self.index + np.arange(x.size)) % self.data.size
        x_index = self.index % self.data.shape[0]
        self.data[x_index] = x
        self.index = x_index + 1#[-1] + 1

    def get(self):
        "Returns the first-in-first-out data in the ring buffer"
        idx = (self.index + np.arange(self.data.size)) % self.data.size
        return self.data[idx]

    def average(self):
        "Returns the average of the entries in the ring buffer"
        return np.mean(self.data,axis=0)




################## DEFINE MPI TAGS ###########################

"""
Definition of the variables and their tags 

    The general rule is: 
        TAG = RANK + tag_dict{VAR_NAME}

    For the state/observation is: 
        TAG = RANK * (1+NTYPE)  +  tag_dict{'STAT'}

    For a single message: 
        TAG = tag_dict{VAR_NAME}

"""

tag_dict = {

            "NID"    : {"tag":1996,
                        "mpi_dtype":MPI.INTEGER,
                        'py_dtype':np.int32,
                        'cate':'info',
                        },
            
            "NUMCTRL"   : {"tag":10000,
                        "mpi_dtype":MPI.INTEGER,
                        'py_dtype':np.int32,
                        'cate':'info',
                        },
            
            "GLLID"  : {"tag":20000,
                        "mpi_dtype":MPI.INTEGER,
                        'py_dtype':np.int32,
                        'cate':'info',
                        },
            "FACEID" : {"tag":30000,
                        "mpi_dtype":MPI.INTEGER,
                        'py_dtype':np.int32,
                        'cate':'info',
                        },

            "ix"     : {"tag":40000,
                        "mpi_dtype":MPI.INTEGER,
                        'py_dtype':np.int32,
                        'cate':'info',
                        },
            "iy"     : {"tag":50000,
                        "mpi_dtype":MPI.INTEGER,
                        'py_dtype':np.int32,
                        'cate':'info',
                        },
            "iz"     : {"tag":60000,
                        "mpi_dtype":MPI.INTEGER,
                        'py_dtype':np.int32,
                        'cate':'info',
                        },

        
            "x"      : {"tag":100000,
                        "mpi_dtype":MPI.DOUBLE,
                        'py_dtype':np.float64,
                        'cate':'info',
                        },
            "y"      : {"tag":200000,
                        "mpi_dtype":MPI.DOUBLE,
                        'py_dtype':np.float64,
                        'cate':'info',
                        },
            "z"      : {"tag":300000,
                        "mpi_dtype":MPI.DOUBLE,
                        'py_dtype':np.float64,
                        'cate':'info',
                        },
            
            'current_cfl': {"tag":1999,
                        "mpi_dtype":MPI.DOUBLE,
                        'py_dtype':np.float64,
                        'cate':'request',
                        },

            'current_time': {"tag":1998,
                        "mpi_dtype":MPI.DOUBLE,
                        'py_dtype':np.float64,
                        'cate':'request',
                        },

            'STATE':{"tag":70000,
                        "mpi_dtype":MPI.DOUBLE,
                        'py_dtype':np.float64,
                        'cate':'request',
                        },
            
            'REWRD':{"tag":80000,
                        "mpi_dtype":MPI.DOUBLE,
                        'py_dtype':np.float64,
                        'cate':'request',
                        },

            'REWRD_PW':{"tag":81000,
                        "mpi_dtype":MPI.DOUBLE,
                        'py_dtype':np.float64,
                        'cate':'request',
                        },

            'REWRD_V3':{"tag":82000,
                        "mpi_dtype":MPI.DOUBLE,
                        'py_dtype':np.float64,
                        'cate':'request',
                        },
            

            'ACTION':{"tag":90000,
                        "mpi_dtype":MPI.DOUBLE,
                        "py_dtype":np.float64,
                        'cate':'send',
                        },

            'COMMAND':{ "tag":22,
                        "mpi_dtype":MPI.CHARACTER,
                        "py_dtype":str,
                        'cate':'send',
                        },
            
            }

