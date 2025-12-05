"""
MARL Environment using NEK5000 for Turbulent Channel Flow Control

This module implements a Multi-Agent Reinforcement Learning (MARL) environment
that interfaces with NEK5000 CFD solver for active flow control applications.

@author: yuningw
"""

import os
import time
import shutil
import functools
from pathlib import Path

import numpy as np
import pandas as pd
from gym import spaces
from omegaconf import OmegaConf
from mpi4py import MPI

from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn as parallel_to_aec

from configs import Config
from lib.lglnodes import lglnodes
from lib.nek_utils import remove_sch
from lib.reward_logger import RewardLogger, SimpleRewardLogger


def env(conf: Config, rank_folder: str, sub_comm=None):
    """
    Create and wrap the environment with standard wrappers.
    
    Args:
        conf: Configuration object
        rank_folder: Path to the rank-specific folder
        sub_comm: MPI communicator (optional)
        
    Returns:
        Wrapped environment ready for use
    """
    env = raw_env(conf, rank_folder, sub_comm)
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # This wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide variety of helpful user errors
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(conf: Config, rank_folder: str, sub_comm=None):
    """
    Create raw environment and convert from ParallelEnv to AEC format.
    
    Args:
        conf: Configuration object
        rank_folder: Path to the rank-specific folder
        sub_comm: MPI communicator
        
    Returns:
        AEC-compatible environment
    """
    env = parallel_env(conf, rank_folder, sub_comm)
    env = parallel_to_aec(env)
    return env

def raw_env_clustered(conf: Config, rank_folder: str, sub_comm=None):
    """
    Create raw clustered environment and convert from ParallelEnv to AEC format.
    
    Args:
        conf: Configuration object
        rank_folder: Path to the rank-specific folder
        sub_comm: MPI communicator
        
    Returns:
        AEC-compatible clustered environment
    """
    env = parallel_env_clustered(conf, rank_folder, sub_comm)
    env = parallel_to_aec(env)
    return env

def env_clustered(conf: Config, rank_folder: str, sub_comm=None):
    """
    Create and wrap the clustered environment with standard wrappers.
    
    Args:
        conf: Configuration object
        rank_folder: Path to the rank-specific folder
        sub_comm: MPI communicator (optional)
        
    Returns:
        Wrapped clustered environment ready for use
    """
    env = raw_env_clustered(conf, rank_folder, sub_comm)
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # This wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide variety of helpful user errors
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class parallel_env(ParallelEnv):
    """
    Multi-Agent Reinforcement Learning environment for NEK5000 CFD simulations.
    
    This class implements a parallel environment where multiple agents can control
    different regions of a turbulent channel flow using NEK5000 as the CFD solver.
    """
    
    metadata = {"render_modes": ["human"], "name": "nek_tcf_marl"}

    def __init__(self, conf, rank_folder:str, sub_comm):
        """
        Initialize the parallel environment.
        
        Args:
            conf: Configuration object containing simulation parameters
            rank_folder: Path to the rank-specific working directory
            sub_comm: MPI communicator for inter-process communication
            
        The init method defines the following attributes:
        - possible_agents: List of agent identifiers
        - action_spaces: Action space for each agent
        - observation_spaces: Observation space for each agent
        
        These attributes should not be changed after initialization.
        
        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If MPI communication fails
        """
        
        if not isinstance(rank_folder, str) or not rank_folder:
            raise ValueError("rank_folder must be a non-empty string")
        if sub_comm is None:
            raise ValueError("sub_comm cannot be None")
            
        self.conf = conf
        self.folder = rank_folder
        self.sub_comm = sub_comm
        
        try:
            self.initialization()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize environment: {e}")

    def initialization(self):
        """
        Initialize the environment with configuration, directories, and agent setup.
        
        This method sets up:
        - I/O directories and file management
        - MPI communication setup
        - Agent initialization and mapping
        - State and action space definitions
        - Reward system initialization
        """
        print(f'------------ INITIALIZATION -------------', flush=True)
        
        #--------------I/O-----------------------
        self.history_path = Path(f"{self.folder}/history")
        self.history_path.mkdir(exist_ok=True)
        self.rstart_folder = Path(f"{os.getcwd()}/{self.conf.simulation.restart_folder}")
        
        # Remove the .sch file if it exists
        remove_sch(self.folder)

        ## We save the config for the run
        ##-----------------------------------
        OmegaConf.save(self.conf, os.path.join(self.history_path, 'current_conf.yml'))
        OmegaConf.save(self.conf, os.path.join(self.folder, 'current_conf.yml'))
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
        self.cluster_nodes_to_agents()
        
        print(f"[STB3] POSSIBLE AGENTS={self.possible_agents}", flush=True)
        # Mapping 
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        
        
        # -----------STATE--------------
        # STATE BUFFER
        self.full_observation=\
            np.ndarray(shape=(self.conf.runner.npl_state,self.nNodes))
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

        # Output shape will  be the total number of gll nodes
        self.action_shape=[1,]

        # -----------REWARD--------------
        # Baseline Reward
        self.baseline_dudy=self.conf.runner.dUdy # For N=5 @ x/c = 0.4 
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
        print(f'------------ FINISH -------------',flush=True)


    def init_agent(self):
        """
        Initialize agents by communicating with NEK5000 to get node information.
        
        This method:
        1. Sends initialization request to NEK5000
        2. Receives node list from NEK5000
        3. Collects node information (NID, GLLID, FACEID, coordinates)
        4. Creates agent mapping and saves node info to CSV
        
        Raises:
            ValueError: If node information is incomplete or invalid
        """ 
        request = b"INTAL"
        self.sub_comm.Send([request, MPI.CHARACTER], dest=0, tag=tag_dict["COMMAND"]['tag'])
        self.nNodes = 0

        ## Create Agent info 
        self.node_info = {}
        for k in tag_dict.keys():
            if tag_dict[k]['cate'] == 'info':
                self.node_info[k] = []

        # A hand-shake from NEK, let me know which nid I should recv info
        node_list = np.empty((self.conf.simulation.nproc,), dtype=np.int32)
        print(f'[STB3] REQUEST NODE LIST', flush=True)
        
        try:
            self.sub_comm.Recv([node_list, MPI.INTEGER], 0, tag=tag_dict['NID']['tag'])
        except Exception as e:
            raise RuntimeError(f"Failed to receive node list from NEK5000: {e}")
            
        # Masking the nid_list
        nid_list = np.arange(self.conf.simulation.nproc, dtype=tag_dict['NID']['py_dtype'])
        nid_list = nid_list[node_list != 0]
        print(f'[STB3] NODE LIST GET:{nid_list}', flush=True)
        
        if len(nid_list) == 0:
            raise RuntimeError("No valid nodes received from NEK5000")
        
        # RECV on those RANKS only
        for nid in (nid_list):
            rank_data = {}
            rank_data['NID'] = np.array([nid],dtype=tag_dict['NID']['py_dtype']) 
            for k in self.node_info.keys(): 
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
                self.node_info[k].append(rank_data[k])
            
            self.nNodes +=numctrl
        
        for k in self.node_info.keys():
            self.node_info[k] = np.concatenate(self.node_info[k])
        print(f"[STB3] INIT END, NUM NODES={self.nNodes}")

        self.uniqID = np.unique(self.node_info['NID'])
        self.nNID = len(self.uniqID)

        # Dump the information in Pandas
        df = pd.DataFrame(self.node_info)
        fname = os.path.join(self.history_path, 'NODE_INFO.csv')
        df.to_csv(fname)
        print(f"[STB3] DUMP NODE INFO : {fname}")

        return 
    
    def cluster_nodes_to_agents(self):
        """
        Clustering the node info based on the element ID.
        """
        # list all the elements
        elems = np.unique(self.node_info['GLLID'])
        color_list = np.zeros_like(self.node_info['GLLID'],dtype=np.int32)
        self.agent_info = {}
        for il, elem in enumerate(elems):
            mask = (self.node_info['GLLID'] == elem)
            assert self.conf.simulation.lx1 % self.conf.simulation.nxs == 0 and self.conf.simulation.lx1 % self.conf.simulation.nzs == 0, "lx1 must be divisible by nxs and nzs"
            subset = {k: v[mask] for k, v in self.node_info.items()}
            subset = {k: v.reshape(self.conf.simulation.lx1,self.conf.simulation.lx1) for k, v in subset.items()} # Reshape to 2D based on the polynomial order
            color = np.zeros_like(subset['x'],dtype=np.int32)
            color_i = 1
            # Cluster the elements
            for ixs in range(0,self.conf.simulation.lx1,self.conf.simulation.nxs):
                for izs in range(0,self.conf.simulation.lx1,self.conf.simulation.nzs):
                    #print(subset['x'][ixs:ixs+nxs,izs:izs+nzs])
                    color[ixs:ixs+self.conf.simulation.nxs,izs:izs+self.conf.simulation.nzs] = color_i
                    color_i += 1
            
            subset = {k: v.reshape(self.conf.simulation.lx1*self.conf.simulation.lx1) for k, v in subset.items()}
            subset['icolor'] = color.reshape(self.conf.simulation.lx1*self.conf.simulation.lx1)

            # Now regroup the elements
            for color_i in np.unique(subset['icolor']):
                mask = (subset['icolor'] == color_i)
                subset_i = {k: v[mask] for k, v in subset.items()}
                self.agent_info[self.nameAgent(elem,color_i)] = subset_i
                #print(f"Agent {self.nameAgent(elem,color_i)} has {subset_i}")
            
            # Fill the color list to update the node_info
            color_list[np.where(self.node_info['GLLID'] == elem)] = subset['icolor']

        self.node_info['icolor'] = color_list
        self.possible_agents = list(self.agent_info.keys())
        self.nAgents = len(self.possible_agents)
        # Dump the information in Pandas
        df = pd.DataFrame(self.node_info)
        fname = os.path.join(self.history_path, 'AGENT_INFO.csv')
        df.to_csv(fname)
        # Print the information
        print(f"[STB3] CLUSTERING END, NUM AGENT={self.nAgents}", flush=True)
        return

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(  low =-1.0,
                            high = 1.0,
                            shape = self.action_shape,
                            dtype = np.float32)
    
    #### Define the observation space 
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(  low = -np.inf,
                            high = np.inf,
                            shape = (self.conf.runner.npl_state,
                                1,
                                1),
                                dtype = np.float32)


    def render(self, mode: str = 'human', close: bool = False) -> None:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode (currently only 'human' supported)
            close: Whether to close the rendering
        
        Note:
            This method is a placeholder for future rendering functionality.
        """
        if close:
            return
        # TODO: Implement rendering functionality
        pass

    def close(self) -> None:
        """
        Close the environment and clean up resources.
        
        This method:
        1. Closes the reward logger
        2. Terminates the NEK5000 simulation
        3. Cleans up MPI resources
        """
        print(f"[STB3] CLOSE ENV", flush=True)
        
        # Close reward logger
        if hasattr(self, 'reward_logger'):
            self.reward_logger.close()
            print(f"[STB3] REWARD LOGGER CLOSED", flush=True)
        
        self.end_simulation(farewell=True)
        # Wait until all the operations are completed
        time.sleep(1)

    def reset(self, seed: int = None, return_info: bool = False, options: dict = None) -> dict:
        """
        Reset the environment to initial state.
        
        This method:
        1. Initializes the agents list
        2. Increments the restart index
        3. Saves the previous episode's reward log
        4. Handles restart data management
        5. Starts a new simulation
        6. Returns initial observations
        
        Args:
            seed: Random seed for reproducibility (currently unused)
            return_info: Whether to return additional info (currently unused)
            options: Additional options (currently unused)
            
        Returns:
            Dictionary of initial observations for each agent
            
        Raises:
            RuntimeError: If simulation fails to start
        """
        print("[STB3] RESET!", flush=True)
        self.agents = self.possible_agents[:]
        
        # Update restart index
        self.restart_index += 1
        print(f'[STB3] EPISODE={self.restart_index}', flush=True)
        
        # Re-initialize the action index
        self.act_index = 0

        # Save reward log and re-initialize it
        np.savez(
            os.path.join(self.history_path, f'rewlog_{self.restart_index:05d}.npz'),
            rew=np.array(self.reward_log)
        )
        
        # Log episode summary
        self.reward_logger.log_episode_summary(episode=self.restart_index)

        self.reward_log = list()
        print('[STB3] SAVE LOG', flush=True)

        remove_sch(self.folder)
        
        # TODO: Potentially, Add management of logfiles 
        # but as nek does not generate logfile automatically, it is not worth 
        self.restart_handle()
        
        # Open a new simulation
        self.start_simulation()
        print('[STB3] Start SIM', flush=True)
        
        # Return current (initial) state
        flow_time, observation = self.state()
        # Distribute observations to the agents
        observations = self._distribute_field(observation, reward=False)

        return observations

    def step(self, actions: dict) -> tuple:
        """
        Execute one step of the environment.
        
        Args:
            actions: Dictionary mapping agent names to their actions
            
        Returns:
            Tuple containing:
            - observations: Dictionary of observations for each agent
            - rewards: Dictionary of rewards for each agent
            - dones: Dictionary indicating if each agent is done
            - infos: Dictionary of additional information for each agent
        
        Raises:
            ValueError: If actions are invalid or missing
            RuntimeError: If simulation fails
        """
        if not isinstance(actions, dict):
            raise ValueError("Actions must be a dictionary")
        
        if not all(agent in self.agents for agent in actions.keys()):
            raise ValueError("Actions contain invalid agent names")

        # Resetting reward value
        rewards = {}

        # Transform actions before re-scaling
        ctrl_value = {}            
        for agent in self.agents:
            ctrl_value[agent] = actions[agent]  
        
        # Linear mapping of the actions if the range differs
        if self.rescale_actions:
            for agent in self.agents:
                # if actions[agent] < 0:
                #     actions[agent] *= self.rescale_factors[0][0]
                # if actions[agent] > 0:
                actions[agent] *= self.rescale_factors[0][1] 

        # Sending the new action values to the environment
        self.action(ctrl_value)

        # Let the solution evolve with the new control values
        rewards = self.evolve()
        
        # Obtain new observation
        flow_time, observation = self.state()

        # Distribute observations to the agents
        observations = self._distribute_field(observation, reward=False)

        # Create info dictionary with flow time
        infos = {agent: {'time': flow_time} for agent in self.agents}
        
        self.act_index += 1
        
        # Check termination conditions
        dones = self._check_termination(flow_time)
        
        return observations, rewards, dones, infos
    
    def _check_termination(self, flow_time: float) -> dict:
        """
        Check if the simulation should terminate.
        
        Args:
            flow_time: Current simulation time
            
        Returns:
            Dictionary indicating if each agent is done
        """
        # Check whether we have approximately reached the maximum simulation time
        if flow_time > self.conf.simulation.tmax:
            return {agent: True for agent in self.agents}
        
        # Add check regarding the maximum number of interactions
        if self.act_index >= self.conf.runner.nb_interactions:
            print(f'[STEP] ACT_INDEX={self.act_index}; DONES == TRUE', flush=True)
            return {agent: True for agent in self.agents}
        
        return {agent: False for agent in self.agents}


    def _distribute_field(self, field: dict, reward: bool = False) -> dict:
        """
        Distribute field data to individual agents.

        Args:
            field: Dictionary containing field data with shape [nNID, nfield, TOTCTRL]
            reward: Whether this is reward data (affects distribution logic)

        Returns:
            Dictionary mapping agent names to their field data

        Raises:
            ValueError: If agent count doesn't match expected value
        """
        distributed_fields = {}

        ## For READING STATE 
        ### NOTE: fld buffer has shape=[nNID,nfield,TOTCTRL]
        icount = 0

        # elems = np.unique(self.node_info['GLLID'])
        _, nField, totctrl = field['fld'].shape
        distributed_fields = {v:np.zeros((nField,1,1)) for v in self.possible_agents}
        for il, nid in enumerate(self.uniqID):
            # print(f"[STB3] NID={nid}", flush=True)
            mask_rank = self.node_info['NID'] == nid
            nid_subset = {k: v[mask_rank] for k, v in self.node_info.items()}
            elems = nid_subset['GLLID']
            color_list = nid_subset['icolor']
            for jl, elem in enumerate(elems):
                state_ = field['fld'][il,:,jl].reshape(-1,1,1)
                agent_name = self.nameAgent(gllid=elem,icolor=color_list[jl])
                distributed_fields[agent_name] += state_
                icount += 1
        icount //= self.conf.simulation.nxs * self.conf.simulation.nzs
        for agent_name in self.possible_agents:
            distributed_fields[agent_name] /= self.conf.simulation.nxs * self.conf.simulation.nzs
        assert icount == self.nAgents, ValueError(f'[STB3] Agent count mismatch in field distribution: expected {self.nAgents}, got {icount}')
        print(f"[STB3] REDISTRIBUTED for {icount} Agents")
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
        for il, nid in enumerate(self.uniqID):
            # Write Buffer 
            act_buffer = np.ndarray(shape=(self.conf.simulation.TOTCTRL),
                                    dtype=tag_dict['ACTION']['py_dtype'])
            mask_rank = self.node_info['NID'] == nid
            nid_subset = {k: v[mask_rank] for k, v in self.node_info.items()}
            elems = nid_subset['GLLID']
            color_list = nid_subset['icolor']
            for jl, gllid in enumerate(elems):
                agent_name = self.nameAgent(gllid=gllid,icolor=color_list[jl])
                act_buffer[jl] = ctrl_value[agent_name]
            self.sub_comm.Send([act_buffer,tag_dict['ACTION']['mpi_dtype']],nid,
                                tag=nid+tag_dict['ACTION']['tag'])    
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
                ws_stress_buffer=np.ndarray(shape=(self.nNID,self.conf.simulation.TOTCTRL,),
                                    dtype=tag_dict["REWRD"]['py_dtype'])
                # Recv Data from the expected NID 
                for il, nid in enumerate(self.uniqID):
                    recv_buffer=np.ndarray(shape=(self.conf.simulation.TOTCTRL,),dtype=np.float64)
                    self.sub_comm.Recv([recv_buffer,tag_dict['REWRD']['mpi_dtype']],
                                        nid,tag=nid+tag_dict["REWRD"]['tag'])
                    ws_stress_buffer[il,:] = recv_buffer
                # Expand the dimension to fit the _distribute_field 
                ws_stress_buffer = self._distribute_field({'fld':np.expand_dims(ws_stress_buffer,1)},
                                                            reward=True)
            #--------------------------------
            i_evolv +=1
        #---- While Loop End here---------

        # Scale the dUdy to be reward in 0.0~1.0
        rewards = {}
        mean_reward = 0; num_rwd=0
        for il, nid in enumerate(self.uniqID): 
            indx = np.where((self.node_info['NID']==nid))[0]
            subset = {k: v[indx] for k, v in self.node_info.items()}
            elems = subset['GLLID']
            color_list = subset['icolor']
            for jl, gllid in enumerate(elems):
                agent_name = self.nameAgent(gllid=gllid,icolor=color_list[jl])
                r_reward   = ws_stress_buffer[agent_name]
                i_reward   = float(self._normalize_reward(r_reward))
                rewards[agent_name] = i_reward   
        # Logging reward for debugging and further analysis
        self.reward_log.append(i_reward)
        
        # Real-time reward logging
        dUdy_raw_dict = {agent_name: r_reward.squeeze() for agent_name in rewards.keys()}
        self.reward_logger.log_rewards(
            rewards=rewards,
            dUdy_raw=dUdy_raw_dict,
            episode=self.restart_index,
            step=self.act_index
        )
                
        print(f"[LOGGER] act_index={self.act_index} R={i_reward:.5f} ",flush=True)
        
        return rewards
        
#######################
# Utility Function 
#####################
    
    @staticmethod
    def nameAgent(gllid,icolor):
        """
        Name the agent based on the GRID information 
        gllid [int] Element ID 
        icolor [int] Color ID 
        """
        return f"jet_{gllid:05d}_{icolor:03d}"
    

    def avg_ZNMF(self, ctrl_value: dict) -> dict:
        """
        Apply zero-net-mass-flux (ZNMF) condition to control values.
        
        The ZNMF condition ensures that the net mass flux through the control
        surface is zero, which is physically meaningful for blowing/suction control.
        
        Args:
            ctrl_value: Dictionary mapping agent names to their control values
            
        Returns:
            Dictionary with ZNMF-adjusted control values
            
        Raises:
            NotImplementedError: If unsupported ZNMF averaging method is specified
        """

        # Averaging Scheme

        ## Naive Average
        if self.conf.simulation.znmf_avg == -1:
            mean_action = 0
            num_ = 0
            for il, aval in enumerate(ctrl_value.values()):
                mean_action += aval
                num_ += 1  # Fixed: was =+il, should be += 1 
            mean_action /= num_
            
            for agent_name in ctrl_value.keys():
                ctrl_value_single = ctrl_value[agent_name] 
                ctrl_value_znmf = ctrl_value_single - mean_action 
                ctrl_value[agent_name] = ctrl_value_znmf
                # print(f"[ZNMF] {agent_name}:\n BEF={ctrl_value_single},NOW={ctrl_value_znmf}",flush=True)
        
        ## Weighted Average 
        elif self.conf.simulation.znmf_avg == -2:
            mean_action = 0
            wxz = 0
            for il in range(self.nNodes):
                ix, iz = self.node_info['ix'][il], self.node_info['iz'][il]
                agent_name = self.nameAgent(
                    self.node_info['NID'][il],
                    self.node_info['GLLID'][il],
                    self.node_info['FACEID'][il],
                    self.node_info['ix'][il],
                    self.node_info['iy'][il],
                    self.node_info['iz'][il]
                )
                i_action = ctrl_value[agent_name]
                wx = self.gll_weight[ix-1]
                wz = self.gll_weight[iz-1]
                wxz += wx * wz
                mean_action += i_action * wx * wz
            mean_action /= wxz

            for agent_name in ctrl_value.keys():
                ctrl_value_single = ctrl_value[agent_name] 
                ctrl_value_znmf = ctrl_value_single - mean_action 
                ctrl_value[agent_name] = ctrl_value_znmf
                # print(f"[ZNMF] {agent_name}:\n BEF={ctrl_value_single},NOW={ctrl_value_znmf}",flush=True)
        
        
        elif (self.conf.simulation.znmf_avg == 1) or (self.conf.simulation.znmf_avg == 0): 
            mean_action = 0.0
            print(f"[ZNMF] DONE BY NEK5000", flush=True)
        
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

    def _normalize_reward(self,reward):
        """Normalizing the reward in the range of 0~1""" 
        return (1 - (np.mean(reward)/self.baseline_dudy))
    def restart_handle(self):
        """
            Mangement of rs8/rs6 data for each episode
            if random_init > 0,   we shuffle the RESTARTS to use.
            if random_init == -1, we use the specified No.INIT 
            if random_init == -2 and restart_index==1, Not OverWrite the RSTART for the first run 
        """
        if self.conf.runner.random_init>0: 
            n_init = np.random.randint(low=1,high=self.conf.runner.random_init)
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







################## DEFINE MPI TAGS ###########################

"""
Definition of the variables and their MPI communication tags.

Communication Protocol:
    The general rule is: 
        TAG = RANK + tag_dict{VAR_NAME}

    For the state/observation: 
        TAG = RANK * (1+NTYPE) + tag_dict{'STATE'}

    For a single message: 
        TAG = tag_dict{VAR_NAME}

Each tag entry contains:
    - tag: Base tag number
    - mpi_dtype: MPI data type
    - py_dtype: Python data type
    - cate: Category (info, request, send)
"""

# MPI Communication Tags
# Each tag defines the communication protocol for different data types
tag_dict = {
    # Node identification and control information
    "NID": {
        "tag": 1996,
        "mpi_dtype": MPI.INTEGER,
        'py_dtype': np.int32,
        'cate': 'info',
    },
    
    "NUMCTRL": {
        "tag": 10000,
        "mpi_dtype": MPI.INTEGER,
        'py_dtype': np.int32,
        'cate': 'info',
    },
    
    "GLLID": {
        "tag": 20000,
        "mpi_dtype": MPI.INTEGER,
        'py_dtype': np.int32,
        'cate': 'info',
    },
    
    "FACEID": {
        "tag": 30000,
        "mpi_dtype": MPI.INTEGER,
        'py_dtype': np.int32,
        'cate': 'info',
    },

    # Spatial indices
    "ix": {
        "tag": 40000,
        "mpi_dtype": MPI.INTEGER,
        'py_dtype': np.int32,
        'cate': 'info',
    },
    
    "iy": {
        "tag": 50000,
        "mpi_dtype": MPI.INTEGER,
        'py_dtype': np.int32,
        'cate': 'info',
    },
    
    "iz": {
        "tag": 60000,
        "mpi_dtype": MPI.INTEGER,
        'py_dtype': np.int32,
        'cate': 'info',
    },

    # Spatial coordinates
    "x": {
        "tag": 100000,
        "mpi_dtype": MPI.DOUBLE,
        'py_dtype': np.float64,
        'cate': 'info',
    },
    
    "y": {
        "tag": 200000,
        "mpi_dtype": MPI.DOUBLE,
        'py_dtype': np.float64,
        'cate': 'info',
    },
    
    "z": {
        "tag": 300000,
        "mpi_dtype": MPI.DOUBLE,
        'py_dtype': np.float64,
        'cate': 'info',
    },
    
    # Simulation state requests
    'current_cfl': {
        "tag": 1999,
        "mpi_dtype": MPI.DOUBLE,
        'py_dtype': np.float64,
        'cate': 'request',
    },

    'current_time': {
        "tag": 1998,
        "mpi_dtype": MPI.DOUBLE,
        'py_dtype': np.float64,
        'cate': 'request',
    },

    'STATE': {
        "tag": 70000,
        "mpi_dtype": MPI.DOUBLE,
        'py_dtype': np.float64,
        'cate': 'request',
    },
    
    'REWRD': {
        "tag": 80000,
        "mpi_dtype": MPI.DOUBLE,
        'py_dtype': np.float64,
        'cate': 'request',
    },
    
    # Control actions
    'ACTION': {
        "tag": 90000,
        "mpi_dtype": MPI.DOUBLE,
        "py_dtype": np.float64,
        'cate': 'send',
    },

    # Commands
    'COMMAND': {
        "tag": 22,
        "mpi_dtype": MPI.CHARACTER,
        "py_dtype": str,
        'cate': 'send',
    },
}

