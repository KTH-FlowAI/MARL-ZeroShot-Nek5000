"""
Clustered Parallel Environment for DRL Flow Control with NEK5000

This module provides a clustered version of the parallel environment that groups
GLL nodes into agents based on nxs and nzs parameters, reducing the total number
of agents while maintaining the same MPI interface.

Author: AI Assistant
Date: 2025
"""

import os
import time
import shutil
import functools
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from pettingzoo import ParallelEnv
from mpi4py import MPI

# Import from the original environment
from gym import spaces
from nek_marl import tag_dict, remove_sch, RingBuffer
from lib.lglnodes import lglnodes
from lib.reward_logger import RewardLogger


class parallel_env_clustered(ParallelEnv):
    """
    Clustered parallel environment for DRL flow control.
    
    This environment clusters GLL nodes into agents based on nxs and nzs parameters,
    reducing the total number of agents while maintaining the same MPI interface
    as the original parallel_env.
    """
    
    metadata = {"render_modes": ["human"], "name": "nek_clustered_v1"}

    def __init__(self, conf, rank_folder, sub_comm, nxs=2, nzs=2):
        """
        Initialize the clustered parallel environment.
        
        Args:
            conf: Configuration object
            rank_folder: Rank folder for the environment
            sub_comm: MPI sub-communicator
            nxs: Number of nodes to cluster in x-direction (default: 2)
            nzs: Number of nodes to cluster in z-direction (default: 2)
        """
        self.conf = conf
        self.folder = rank_folder
        self.sub_comm = sub_comm
        self.nxs = nxs
        self.nzs = nzs
        
        # Validate clustering parameters
        if nxs <= 0 or nzs <= 0:
            raise ValueError("nxs and nzs must be positive integers")
        
        # Update configuration to reflect clustering
        self.conf.simulation.nxs = nxs
        self.conf.simulation.nzs = nzs
        
        self.initialization()

    def initialization(self):
        """Initialize the clustered environment."""
        print(f'------------ CLUSTERED INITIALIZATION -------------', flush=True)
        
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
        mpi_info.Set('wdir', f"{os.getcwd()}/{self.folder}")
        mpi_info.Set('bind_to', 'none')
        if self.conf.simulation.hostfile != '':
            mpi_info.Set('hostfile', self.conf.simulation.hostfile)
            print('[STB3] LOAD HOSTFILE!')
        self.mpi_info = mpi_info
        
        # -----------AGENT--------------
        # Initialize agents and create clusters
        self.init_agent()
        self.create_clusters()
        
        # -----------STATE--------------
        # STATE BUFFER
        self.full_observation = np.ndarray(shape=(self.conf.runner.npl_state, self.nAgents))
        ## Scaling the data
        self.utau = self.conf.runner.u_tau # u_tau for wing @ x/c =  0.4
        
        # -----------ACTION--------------
        # Gll node weight 
        _, self.gll_weight, _ = lglnodes(N=self.conf.simulation.lx1-1)
        print(f"[STB3] GLL WEIGHT={self.gll_weight}", flush=True)
        # Action rescaling variables
        self.rescale_actions = self.conf.runner.rescale_actions
        if self.rescale_actions:
            self.rescale_factors = [[self.conf.runner.ctrl_max_amp, self.conf.runner.ctrl_max_amp]]
            self.ctrl_min_amp = -1.0
            self.ctrl_max_amp = 1.0
        else:
            # Bounding Control Amplitude
            self.ctrl_min_amp = self.conf.runner.ctrl_min_amp
            self.ctrl_max_amp = self.conf.runner.ctrl_max_amp
        # Output shape
        self.action_shape = [1,]

        # -----------REWARD--------------
        # Baseline Reward
        self.baseline_dudy = self.conf.runner.dUdy # For N=5 @ x/c = 0.4 
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

        print(f"SCALE: dUdy={self.baseline_dudy}\n Utau={self.utau}", flush=True)
        # Reward-related variables
        if self.conf.runner.rew_mode == 'MovingAverage':
            self.reward_history = RingBuffer(length=self.conf.runner.size_history,
                                            dim=(self.nAgents,))
            # The history of the wall shear-stress is initialized with
            # an the reference value
            for i_h in range(self.conf.runner.size_history):
                self.reward_history.data[i_h] = self.baseline_dudy * np.ones((self.nAgents,))
        print(f'------------ FINISH -------------', flush=True)

    def init_agent(self):
        """Initialize agents and get node information from NEK5000."""
        request = b"INTAL"
        self.sub_comm.Send([request, MPI.CHARACTER], dest=0, tag=tag_dict["COMMAND"]['tag'])
        self.original_nAgents = 0

        ## Create Agent info 
        self.original_agent_info = {}
        for k in tag_dict.keys():
            if tag_dict[k]['cate'] == 'info':
                self.original_agent_info[k] = []

        # A hand-shake from NEK, let me know which nid I should recv info
        node_list = np.empty((self.conf.simulation.nproc,), dtype=np.int32)
        print(f'[STB3] REQUEST NODE LIST', flush=True)
        self.sub_comm.Recv([node_list, MPI.INTEGER], 0, tag=tag_dict['NID']['tag'])
        # Masking the nid_list
        nid_list = np.arange(self.conf.simulation.nproc, dtype=tag_dict['NID']['py_dtype'])
        nid_list = nid_list[node_list != 0]
        print(f'[STB3] NODE LIST GET:{nid_list}', flush=True)
        
        # RECV on those RANKS only
        for nid in (nid_list):
            rank_data = {}
            rank_data['NID'] = np.array([nid], dtype=tag_dict['NID']['py_dtype']) 
            for k in self.original_agent_info.keys(): 
                ## Prepare Tensor, NID and NUMCTRL are Scalars
                if 'NID' not in k: 
                    if 'NUMCTRL' not in k:
                        rank_data[k] = np.empty((self.conf.simulation.TOTCTRL,),
                                            dtype=tag_dict[k]['py_dtype'])
                    else: 
                        rank_data[k] = np.empty((1,),
                                            dtype=tag_dict[k]['py_dtype'])
                    
                    ## MPI RECV
                    self.sub_comm.Recv([rank_data[k], tag_dict[k]['mpi_dtype']], nid, tag=nid+tag_dict[k]['tag'])

            ## Resort to match the length 
            numctrl = rank_data['NUMCTRL'][0]
            for k in rank_data.keys():
                if ('NUMCTRL' in k) or ("NID" in k):
                    rank_data[k] = rank_data[k][0] * np.ones(shape=(numctrl,), dtype=tag_dict[k]['py_dtype'])
                else:
                    rank_data[k] = rank_data[k][:numctrl]
                self.original_agent_info[k].append(rank_data[k])
            
            self.original_nAgents += numctrl
        
        for k in self.original_agent_info.keys():
            self.original_agent_info[k] = np.concatenate(self.original_agent_info[k])
        print(f"[STB3] INIT END, NUM ORIGINAL AGENT={self.original_nAgents}")

        self.uniqID = np.unique(self.original_agent_info['NID'])
        self.nNID = len(self.uniqID)

        # Dump the information in Pandas
        df = pd.DataFrame(self.original_agent_info)
        fname = os.path.join(self.history_path, 'NODE_INFO_ORIGINAL.csv')
        df.to_csv(fname)
        print(f"[STB3] DUMP ORIGINAL NODE INFO : {fname}")

        return 

    def create_clusters(self):
        """Create clusters from the original agent information based on spatial coordinates."""
        print(f"[CLUSTER] Creating clusters with nxs={self.nxs}, nzs={self.nzs}", flush=True)
        
        # Group nodes by spectral element (GLLID) and face (FACEID)
        element_groups = {}
        for i, (nid, gllid, iface, ix, iy, iz, x, y, z) in enumerate(zip(
            self.original_agent_info['NID'],
            self.original_agent_info['GLLID'],
            self.original_agent_info['FACEID'],
            self.original_agent_info['ix'],
            self.original_agent_info['iy'],
            self.original_agent_info['iz'],
            self.original_agent_info['x'],
            self.original_agent_info['y'],
            self.original_agent_info['z']
        )):
            key = (nid, gllid, iface)
            if key not in element_groups:
                element_groups[key] = []
            element_groups[key].append({
                'original_index': i,
                'nid': nid,
                'gllid': gllid,
                'iface': iface,
                'ix': ix,
                'iy': iy,
                'iz': iz,
                'x': x,
                'y': y,
                'z': z
            })
        
        # Create clusters within each element
        self.cluster_mapping = {}  # cluster_agent_name -> list of original node indices
        self.node_to_cluster = {}  # original_node_index -> cluster_agent_name
        self.clustered_agent_info = {
            'NID': [],
            'GLLID': [],
            'FACEID': [],
            'ix': [],
            'iy': [],
            'iz': [],
            'NUMCTRL': []
        }
        
        cluster_index = 0
        for (nid, gllid, iface), nodes in element_groups.items():
            # Sort nodes by x coordinate first, then z coordinate for spatial adjacency
            nodes.sort(key=lambda x: (x['x'], x['z']))
            
            print(f"[CLUSTER] Element {gllid}: {len(nodes)} nodes")
            print(f"[CLUSTER] X range: [{min(n['x'] for n in nodes):.3f}, {max(n['x'] for n in nodes):.3f}]")
            print(f"[CLUSTER] Z range: [{min(n['z'] for n in nodes):.3f}, {max(n['z'] for n in nodes):.3f}]")
            
            # Group nodes into clusters based on spatial adjacency
            n_nodes = len(nodes)
            nodes_per_cluster = self.nxs * self.nzs
            
            if n_nodes % nodes_per_cluster != 0:
                print(f"[WARNING] Element {gllid} has {n_nodes} nodes, not divisible by {nodes_per_cluster}")
                # Adjust clustering to fit available nodes
                actual_clusters = n_nodes // nodes_per_cluster
                if actual_clusters == 0:
                    actual_clusters = 1
                    nodes_per_cluster = n_nodes
                else:
                    nodes_per_cluster = n_nodes // actual_clusters
            
            # Create clusters using spatial adjacency
            clusters = self._create_spatial_clusters(nodes, nxs=self.nxs, nzs=self.nzs)
            
            # Create cluster agents
            for cluster_idx, cluster_nodes in enumerate(clusters):
                # Create cluster agent name
                cluster_agent_name = self._name_cluster_agent(
                    nid, gllid, iface, cluster_idx
                )
                
                # Store mapping
                original_indices = [node['original_index'] for node in cluster_nodes]
                self.cluster_mapping[cluster_agent_name] = original_indices
                
                for node_idx in original_indices:
                    self.node_to_cluster[node_idx] = cluster_agent_name
                
                # Store cluster agent info (use first node as representative)
                first_node = cluster_nodes[0]
                self.clustered_agent_info['NID'].append(first_node['nid'])
                self.clustered_agent_info['GLLID'].append(first_node['gllid'])
                self.clustered_agent_info['FACEID'].append(first_node['iface'])
                self.clustered_agent_info['ix'].append(first_node['ix'])
                self.clustered_agent_info['iy'].append(first_node['iy'])
                self.clustered_agent_info['iz'].append(first_node['iz'])
                self.clustered_agent_info['NUMCTRL'].append(len(cluster_nodes))
                
                # Print cluster information for debugging
                x_coords = [n['x'] for n in cluster_nodes]
                z_coords = [n['z'] for n in cluster_nodes]
                print(f"[CLUSTER] Cluster {cluster_idx}: {len(cluster_nodes)} nodes")
                print(f"[CLUSTER]   X: [{min(x_coords):.3f}, {max(x_coords):.3f}]")
                print(f"[CLUSTER]   Z: [{min(z_coords):.3f}, {max(z_coords):.3f}]")
                
                cluster_index += 1
        
        # Convert to numpy arrays
        for key in self.clustered_agent_info:
            self.clustered_agent_info[key] = np.array(self.clustered_agent_info[key])
        
        self.nAgents = len(self.cluster_mapping)
        self.possible_agents = list(self.cluster_mapping.keys())
        
        # Create agent name mapping
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        
        # Update agent_info for compatibility
        self.agent_info = self.clustered_agent_info
        
        print(f"[CLUSTER] Created {self.nAgents} clustered agents from {self.original_nAgents} nodes")
        
        # Dump clustered information
        df_clustered = pd.DataFrame(self.clustered_agent_info)
        fname_clustered = os.path.join(self.history_path, 'NODE_INFO_CLUSTERED.csv')
        df_clustered.to_csv(fname_clustered)
        print(f"[CLUSTER] DUMP CLUSTERED NODE INFO : {fname_clustered}")

    def _name_cluster_agent(self, nid: int, gllid: int, iface: int, cluster_idx: int) -> str:
        """Generate a name for a clustered agent."""
        return f"cluster_np{nid:05d}_gid{gllid:05d}_iface{iface}_c{cluster_idx:02d}"

    def _create_spatial_clusters(self, nodes, nxs, nzs):
        """
        Create spatial clusters by grouping nodes based on their ix and iz indices.
        Groups nodes in a specific pattern: (1,2,1,2), (3,4,3,4), (5,6,5,6), etc.
        
        Args:
            nodes: List of node dictionaries with ix, iz indices
            nxs: Number of nodes to cluster in x-direction
            nzs: Number of nodes to cluster in z-direction
            
        Returns:
            List of clusters, where each cluster is a list of nodes
        """
        if len(nodes) == 0:
            return []
        
        # Sort nodes by ix first, then iz for consistent clustering
        sorted_nodes = sorted(nodes, key=lambda n: (n['ix'], n['iz']))
        
        print(f"[CLUSTER] Sorting {len(sorted_nodes)} nodes by ix, iz indices")
        
        # Find unique ix and iz values
        ix_values = sorted(list(set(n['ix'] for n in sorted_nodes)))
        iz_values = sorted(list(set(n['iz'] for n in sorted_nodes)))
        
        print(f"[CLUSTER] Unique ix values: {ix_values}")
        print(f"[CLUSTER] Unique iz values: {iz_values}")
        
        # Create a 2D grid mapping based on ix, iz indices
        ix_to_idx = {ix: i for i, ix in enumerate(ix_values)}
        iz_to_idx = {iz: i for i, iz in enumerate(iz_values)}
        
        # Create grid of nodes
        grid = {}
        for node in sorted_nodes:
            ix_idx = ix_to_idx[node['ix']]
            iz_idx = iz_to_idx[node['iz']]
            grid[(ix_idx, iz_idx)] = node
        
        # Group nodes into clusters using the specific pattern
        clusters = []
        visited = set()
        
        # Create clusters by grouping specific ix and iz combinations
        # Pattern: (1,2,1,2), (3,4,3,4), (5,6,5,6), etc.
        # This means we group ix ranges and iz ranges together in a diagonal pattern
        for i in range(0, len(ix_values), nxs):
            if i + nxs <= len(ix_values):
                cluster = []
                
                # Collect nodes in the nxs x nzs region with aligned ranges
                for dix in range(nxs):
                    for diz in range(nzs):
                        ix_idx = i + dix
                        iz_idx = i + diz  # Use same starting index for both ix and iz
                        
                        if (ix_idx < len(ix_values) and iz_idx < len(iz_values) and 
                            (ix_idx, iz_idx) in grid and (ix_idx, iz_idx) not in visited):
                            node = grid[(ix_idx, iz_idx)]
                            cluster.append(node)
                            visited.add((ix_idx, iz_idx))
                
                # Only add clusters that have the target size
                if len(cluster) == nxs * nzs:
                    clusters.append(cluster)
        
        # Handle remaining nodes that couldn't form complete clusters
        remaining_nodes = []
        for node in sorted_nodes:
            ix_idx = ix_to_idx[node['ix']]
            iz_idx = iz_to_idx[node['iz']]
            if (ix_idx, iz_idx) not in visited:
                remaining_nodes.append(node)
        
        # Group remaining nodes into clusters of approximately the right size
        if remaining_nodes:
            # Sort remaining nodes by ix, then iz
            remaining_nodes.sort(key=lambda n: (n['ix'], n['iz']))
            
            # Create clusters from remaining nodes
            for i in range(0, len(remaining_nodes), nxs * nzs):
                cluster = remaining_nodes[i:i + nxs * nzs]
                if cluster:
                    clusters.append(cluster)
        
        print(f"[CLUSTER] Created {len(clusters)} spatial clusters")
        for i, cluster in enumerate(clusters):
            x_coords = [n['x'] for n in cluster]
            z_coords = [n['z'] for n in cluster]
            ix_values = [n['ix'] for n in cluster]
            iz_values = [n['iz'] for n in cluster]
            print(f"[CLUSTER]   Cluster {i}: {len(cluster)} nodes")
            print(f"[CLUSTER]     X: [{min(x_coords):.3f}, {max(x_coords):.3f}]")
            print(f"[CLUSTER]     Z: [{min(z_coords):.3f}, {max(z_coords):.3f}]")
            print(f"[CLUSTER]     ix: {sorted(ix_values)}")
            print(f"[CLUSTER]     iz: {sorted(iz_values)}")
            print(f"[CLUSTER]     Node indices: {[n['original_index'] for n in cluster]}")
        
        return clusters

    # this cache ensures that same space object is returned for the same agent
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=self.ctrl_min_amp,
                         high=self.ctrl_max_amp,
                         shape=self.action_shape,
                         dtype=np.float32)
    
    #### Define the observation space 
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low=-np.inf,
                         high=np.inf,
                         shape=(self.conf.runner.npl_state,
                                self.conf.simulation.nzs,
                                self.conf.simulation.nxs),
                         dtype=np.float32)

    def render(self, mode='human', close=False):
        pass

    def close(self):
        print(f"[STB3] CLOSE CLUSTERED ENV", flush=True)
        
        # Close reward logger
        if hasattr(self, 'reward_logger'):
            self.reward_logger.close()
            print(f"[STB3] REWARD LOGGER CLOSED", flush=True)
        
        self.end_simulation(farewell=True)
        # Wait until all the operations are completed
        time.sleep(1)

    def reset(self, seed=None, return_info=False, options=None):
        """Reset the environment and return clustered observations."""
        print("[STB3] CLUSTERED RESET!", flush=True)
        self.agents = self.possible_agents[:]
        # update restart_index
        self.restart_index += 1
        print(f'[STB3] EPISODE={self.restart_index}', flush=True)
        # Re-initialize the action index
        self.act_index = 0

        # Save reward log and re-initialized it
        np.savez(os.path.join(self.history_path,
                            f'rewlog_{self.restart_index:05d}.npz'),
                rew=np.array(self.reward_log))
        
        # Log episode summary
        self.reward_logger.log_episode_summary(episode=self.restart_index)

        self.reward_log = list()
        print('[STB3] SAVE LOG', flush=True)

        remove_sch(self.folder)
        
        self.restart_handle()
        # Open a new simulation
        self.start_simulation()
        print('[STB3] Start SIM', flush=True)
        # Return current (initial) state
        time, observation = self.state()
        # Distribute observations to the agents
        observations = self._distribute_field_clustered(observation, reward=False)

        return observations

    def step(self, actions):
        """Step the environment with clustered actions and return clustered results."""
        # Resetting reward value
        rewards = {}

        # Transform actions to write before re-scaling
        ctrl_value = {}            
        for i_a, agent in enumerate(self.agents):
            ctrl_value[agent] = actions[agent]  
        
        # Linear mapping of the actions if the range differs
        if self.rescale_actions:
            for i_a, agent in enumerate(self.agents):
                if actions[agent] < 0:
                    actions[agent] *= self.rescale_factors[0][0]
                if actions[agent] > 0:
                    actions[agent] *= self.rescale_factors[0][1] 

        # Distribute actions to individual nodes
        distributed_actions = self._distribute_actions_to_nodes(ctrl_value)

        # Sending the new action values to the environment
        self.action(distributed_actions)

        # Let the solution evolve with the new control values
        rewards = self.evolve()
        
        # Obtain new observation
        flow_time, observation = self.state()

        # Distribute observations to the agents
        observations = self._distribute_field_clustered(observation, reward=False)

        infos = {agent: {} for agent in self.agents}
        for agent in self.agents:
            infos[agent]['time'] = flow_time        
        
        self.act_index += 1
        
        # Check whether we have approximately reached the maximum simulation time
        if flow_time > self.conf.simulation.tmax:
            dones = {agent: True for agent in self.agents}
        else:
            dones = {agent: False for agent in self.agents}
        
        # Add check regarding the maximum number of interactions
        if self.act_index >= self.conf.runner.nb_interactions:
            print(f'[STEP] ACT_INDEX={self.act_index}; DONES == TRUE', flush=True)
            dones = {agent: True for agent in self.agents}
        
        return observations, rewards, dones, infos

    def _distribute_field_clustered(self, field: dict, reward=False):
        """Distribute field data to clustered agents."""
        distributed_fields = {}
        
        ## For READING STATE 
        ### NOTE: fld buffer has shape=[nNID,nfield,TOTCTRL]
        icount = 0
        for il, nid in enumerate(self.uniqID): 
            indx = np.where((self.original_agent_info['NID'] == nid))[0]
            for jl, gllid in enumerate(self.original_agent_info['GLLID'][indx]):
                # Find which cluster this node belongs to
                original_agent_name = self._name_original_agent(
                    nid=nid, gllid=gllid, iface=self.original_agent_info["FACEID"][indx][jl],
                    ix=self.original_agent_info['ix'][indx][jl],
                    iy=self.original_agent_info['iy'][indx][jl],
                    iz=self.original_agent_info['iz'][indx][jl]
                )
                
                # Find the cluster this node belongs to
                cluster_agent_name = None
                for cluster_name, node_indices in self.cluster_mapping.items():
                    for node_idx in node_indices:
                        if (self.original_agent_info['NID'][node_idx] == nid and
                            self.original_agent_info['GLLID'][node_idx] == gllid and
                            self.original_agent_info['FACEID'][node_idx] == self.original_agent_info["FACEID"][indx][jl] and
                            self.original_agent_info['ix'][node_idx] == self.original_agent_info['ix'][indx][jl] and
                            self.original_agent_info['iy'][node_idx] == self.original_agent_info['iy'][indx][jl] and
                            self.original_agent_info['iz'][node_idx] == self.original_agent_info['iz'][indx][jl]):
                            cluster_agent_name = cluster_name
                            break
                    if cluster_agent_name:
                        break
                
                if cluster_agent_name and cluster_agent_name not in distributed_fields:
                    # Initialize cluster observation
                    state_ = field['fld'][il, :, jl].reshape(-1, 1, 1)
                    distributed_fields[cluster_agent_name] = [state_]
                elif cluster_agent_name:
                    # Add to existing cluster observation
                    state_ = field['fld'][il, :, jl].reshape(-1, 1, 1)
                    distributed_fields[cluster_agent_name].append(state_)
                
                icount += 1
        
        # Convert lists to proper clustered arrays
        for cluster_name, state_list in distributed_fields.items():
            if state_list:
                # Stack observations along the spatial dimensions
                stacked_obs = np.stack(state_list, axis=-1)  # (npl_state, 1, n_nodes)
                
                # Reshape to (npl_state, nzs, nxs)
                npl_state = stacked_obs.shape[0]
                if stacked_obs.shape[-1] == self.nxs * self.nzs:
                    clustered_obs = stacked_obs.reshape(npl_state, self.nzs, self.nxs)
                else:
                    # Handle case where we don't have exact nxs * nzs nodes
                    target_size = self.nxs * self.nzs
                    if stacked_obs.shape[-1] < target_size:
                        # Pad with zeros
                        pad_size = target_size - stacked_obs.shape[-1]
                        padding = np.zeros((npl_state, 1, pad_size))
                        stacked_obs = np.concatenate([stacked_obs, padding], axis=-1)
                    elif stacked_obs.shape[-1] > target_size:
                        # Truncate
                        stacked_obs = stacked_obs[:, :, :target_size]
                    
                    clustered_obs = stacked_obs.reshape(npl_state, self.nzs, self.nxs)
                
                distributed_fields[cluster_name] = clustered_obs
            else:
                # Create zero observation if no data available
                npl_state = self.conf.runner.npl_state
                distributed_fields[cluster_name] = np.zeros((npl_state, self.nzs, self.nxs))
        
        return distributed_fields

    def _name_original_agent(self, nid, gllid, iface, ix, iy, iz):
        """Generate name for original agent (for compatibility)."""
        return f"jet_np{nid:05d}_gid{gllid:05d}_iface{iface}_ix{ix:05d}_iy{iy:05d}_iz{iz:05d}"

    def _distribute_actions_to_nodes(self, clustered_actions: dict) -> dict:
        """Distribute actions from clustered agents to individual nodes."""
        distributed_actions = {}
        
        for cluster_agent_name, action in clustered_actions.items():
            if cluster_agent_name in self.cluster_mapping:
                node_indices = self.cluster_mapping[cluster_agent_name]
                
                # Distribute the same action to all nodes in the cluster
                for node_idx in node_indices:
                    original_agent_name = self._name_original_agent(
                        nid=self.original_agent_info['NID'][node_idx],
                        gllid=self.original_agent_info['GLLID'][node_idx],
                        iface=self.original_agent_info['FACEID'][node_idx],
                        ix=self.original_agent_info['ix'][node_idx],
                        iy=self.original_agent_info['iy'][node_idx],
                        iz=self.original_agent_info['iz'][node_idx]
                    )
                    distributed_actions[original_agent_name] = action.copy()
        
        return distributed_actions

    # Delegate MPI communication methods to the original implementation
    def start_simulation(self):
        request = b"RSETS"
        self.sub_comm.Send([request, tag_dict['COMMAND']['mpi_dtype']], dest=0, tag=22)

    def end_simulation(self, farewell=False):
        if farewell:
            request = b"TERMN"
            self.sub_comm.Send([request, tag_dict['COMMAND']['mpi_dtype']], dest=0, tag=tag_dict['COMMAND']['tag'])
            self.sub_comm.Free()
            sleep_time = 0.01 * self.conf.simulation.nproc
            print(f'[STB3] TERMN ENV, Sleep: {sleep_time}SEC')
            time.sleep(sleep_time)
            MPI.Finalize()
        else:
            print(f"[STB3] Restart the simulation!", flush=True)
            self.start_simulation()

    def state(self):
        # Send request
        request = b'STATE'
        print(f"[STB3] Sending STATE request", flush=True)
        self.sub_comm.Send([request, tag_dict['COMMAND']['mpi_dtype']], dest=0, tag=tag_dict['COMMAND']['tag'])
        print(f"[STB3] STATE request sent", flush=True)
        
        # Important size constant
        NFLDC = self.conf.runner.npl_state 
        TOTCTRL = self.conf.simulation.TOTCTRL
        # Hand-shake, Current time
        current_time = np.ndarray((1,), dtype=np.float64)
        self.sub_comm.Recv([current_time, MPI.DOUBLE], 0, tag=1998)
        current_time = current_time[0]
        # Current State
        current_state = {
            "NID": np.ndarray(shape=(self.nNID,), dtype=np.int32),
            "fld": np.ndarray(shape=(self.nNID, NFLDC, TOTCTRL), dtype=np.float64),
        }
        # Only recv the data from partion with controlled elem
        for ni, nid in enumerate(self.uniqID):
            state_buffer = np.ndarray(shape=(NFLDC, TOTCTRL), dtype=np.float64)
            for t in range(NFLDC):
                buffer = np.ndarray(shape=(TOTCTRL), dtype=np.float64)
                
                self.sub_comm.Recv([buffer, tag_dict['STATE']['mpi_dtype']],
                                    nid,
                                    tag=nid*(t+1)+tag_dict['STATE']['tag'])
                
                state_buffer[t, :] = buffer[:]
            current_state['NID'][ni] = nid 
            current_state['fld'][ni, :, :] = self._normalize_state(state_buffer)
        print('[STB3] STATE RECV', flush=True)

        return current_time, current_state

    def action(self, ctrl_value: dict):
        request = b"CNTRL"
        self.sub_comm.Send([request, tag_dict['COMMAND']['mpi_dtype']], dest=0, tag=tag_dict['COMMAND']['tag'])

        ## Apply the ZNMF condition 
        ctrl_value = self.avg_ZNMF(ctrl_value=ctrl_value)

        ## Sending MPI
        icount = 0 
        for il, nid in enumerate(self.uniqID):
            # Write Buffer 
            indx = np.where((self.original_agent_info['NID'] == nid))[0]
            numctrl = self.original_agent_info['NUMCTRL'][indx][0]
            act_buffer = np.ndarray(shape=(self.conf.simulation.TOTCTRL),
                                    dtype=tag_dict['ACTION']['py_dtype'])
            for jl, gllid in enumerate(self.original_agent_info['GLLID'][indx]):
                original_agent_name = self._name_original_agent(
                    nid=nid, gllid=gllid, iface=self.original_agent_info["FACEID"][indx][jl],
                    ix=self.original_agent_info['ix'][indx][jl],
                    iy=self.original_agent_info['iy'][indx][jl],
                    iz=self.original_agent_info['iz'][indx][jl]
                )
                
                act_buffer[jl] = ctrl_value[original_agent_name]
                icount += 1
            # Send Buffer 
            self.sub_comm.Send([act_buffer, tag_dict['ACTION']['mpi_dtype']], nid,
                                tag=nid+tag_dict['ACTION']['tag'])    
        # Sanity Check 
        assert icount == self.original_nAgents, ValueError('[STB3] noAgent NOT MATCH!')
        print(f"[STB3] ACTION for {icount} Agents", flush=True)
        
        # Add a small delay to ensure Nek has processed CNTRL before sending EVOLV
        import time
        time.sleep(0.05)  # 50ms delay to ensure Nek processes CNTRL
        print(f"[STB3] CNTRL synchronization complete", flush=True)
        return 

    def evolve(self):
        request = b"EVOLV"
        print(f"[STB3] Sending EVOLV request", flush=True)
        self.sub_comm.Send([request, tag_dict['COMMAND']['mpi_dtype']],
                            dest=0, tag=tag_dict['COMMAND']['tag'])
        print(f"[STB3] EVOLV request sent", flush=True)
        
        # Synchronize with NEK 
        i_evolv = 1 
        while i_evolv <= (self.conf.simulation.ndrl):
            # Recv CFL data, keeping sync between NEK and STB3
            print(f"[STB3] Waiting for CFL data (step {i_evolv}/{self.conf.simulation.ndrl})", flush=True)
            current_cfl = np.ndarray((1,), dtype=np.float64)
            self.sub_comm.Recv([current_cfl, tag_dict['current_cfl']['mpi_dtype']],
                                0, tag=tag_dict['current_cfl']['tag'])
            current_cfl = current_cfl[0]
            print(f"[STB3] Received CFL data: {current_cfl:.6f}", flush=True)
            
            # If we found cfl explode
            if current_cfl >= self.conf.simulation.target_cfl:
                print(f"[WARNING] {i_evolv}/{self.conf.simulation.ndrl} Current {current_cfl} >= {self.conf.simulation.target_cfl}!", flush=True)
                # Exit the entire framework
                self.end_simulation(farewell=True)
                exit()
            
            # Recv Buffer only at the last step, reduce the work load of MPI.
            if i_evolv == self.conf.simulation.ndrl:
                print(f"[STB3] Receiving reward data from NIDs with NUMCTRL > 0", flush=True)
                ws_stress_buffer = np.ndarray(shape=(self.nNID, self.conf.simulation.TOTCTRL,),
                                    dtype=tag_dict["REWRD"]['py_dtype'])
                # Recv Data only from NIDs that have control points (NUMCTRL > 0)
                for il, nid in enumerate(self.uniqID):
                    indx = np.where((self.original_agent_info['NID'] == nid))[0]
                    numctrl = self.original_agent_info['NUMCTRL'][indx][0]
                    if numctrl > 0:
                        print(f"[STB3] Waiting for reward data from NID {nid} (NUMCTRL={numctrl}), tag {nid+tag_dict['REWRD']['tag']}", flush=True)
                        recv_buffer = np.ndarray(shape=(self.conf.simulation.TOTCTRL,), dtype=np.float64)
                        self.sub_comm.Recv([recv_buffer, tag_dict['REWRD']['mpi_dtype']],
                                            nid, tag=nid+tag_dict["REWRD"]['tag'])
                        print(f"[STB3] Received reward data from NID {nid}", flush=True)
                        ws_stress_buffer[il, :] = recv_buffer
                    else:
                        print(f"[STB3] Skipping NID {nid} (NUMCTRL={numctrl}) - no reward data expected", flush=True)
                        # Fill with zeros for NIDs with no control points
                        ws_stress_buffer[il, :] = 0.0
                # Expand the dimension to fit the _distribute_field 
                ws_stress_buffer = self._distribute_field_clustered({'fld': np.expand_dims(ws_stress_buffer, 1)},
                                                            reward=True)
            i_evolv += 1
        #---- While Loop End here---------

        # Scale the dUdy to be reward in 0.0~1.0
        rewards = {}
        mean_reward = 0; num_rwd = 0
        for il, nid in enumerate(self.uniqID): 
            indx = np.where((self.original_agent_info['NID'] == nid))[0]
            for jl, gllid in enumerate(self.original_agent_info['GLLID'][indx]):
                # Find which cluster this node belongs to
                cluster_agent_name = None
                for cluster_name, node_indices in self.cluster_mapping.items():
                    for node_idx in node_indices:
                        if (self.original_agent_info['NID'][node_idx] == nid and
                            self.original_agent_info['GLLID'][node_idx] == gllid and
                            self.original_agent_info['FACEID'][node_idx] == self.original_agent_info["FACEID"][indx][jl] and
                            self.original_agent_info['ix'][node_idx] == self.original_agent_info['ix'][indx][jl] and
                            self.original_agent_info['iy'][node_idx] == self.original_agent_info['iy'][indx][jl] and
                            self.original_agent_info['iz'][node_idx] == self.original_agent_info['iz'][indx][jl]):
                            cluster_agent_name = cluster_name
                            break
                    if cluster_agent_name:
                        break
                
                if cluster_agent_name:
                    r_reward = ws_stress_buffer[cluster_agent_name]
                    i_reward = self._normalize_reward(r_reward)
                    if cluster_agent_name not in rewards:
                        rewards[cluster_agent_name] = i_reward

        # Logging reward for debugging and further analysis
        if rewards:
            self.reward_log.append(list(rewards.values())[0])
        
        # Real-time reward logging
        dUdy_raw_dict = {agent_name: r_reward.squeeze() for agent_name in rewards.keys()}
        self.reward_logger.log_rewards(
            rewards=rewards,
            dUdy_raw=dUdy_raw_dict,
            episode=self.restart_index,
            step=self.act_index
        )
        
        print(f"[LOGGER] act_index={self.act_index} R={list(rewards.values())[0]:.5f} ", flush=True)
        
        return rewards

    # Utility methods from original implementation
    def avg_ZNMF(self, ctrl_value: dict):
        """Apply zero-net-mass-flux condition."""
        if self.conf.simulation.znmf_avg == -1:
            mean_action = 0; num_ = 0
            for il, aval in enumerate(ctrl_value.values()):
                mean_action += aval
                num_ = +il 
            mean_action /= num_
            
            for agent_name in ctrl_value.keys():
                ctrl_value_single = ctrl_value[agent_name] 
                ctrl_value_znmf = ctrl_value_single - mean_action 
                ctrl_value[agent_name] = ctrl_value_znmf
        
        elif self.conf.simulation.znmf_avg == -2:
            mean_action = 0; wxz = 0
            for il in range(self.original_nAgents):
                ix, iz = self.original_agent_info['ix'][il], self.original_agent_info['iz'][il]
                original_agent_name = self._name_original_agent(
                    self.original_agent_info['NID'][il], self.original_agent_info['GLLID'][il], 
                    self.original_agent_info['FACEID'][il], self.original_agent_info['ix'][il], 
                    self.original_agent_info['iy'][il], self.original_agent_info['iz'][il]
                )
                i_action = ctrl_value[original_agent_name]
                wx = self.gll_weight[ix-1]; wz = self.gll_weight[iz-1]
                wxz += wx*wz
                mean_action += i_action*wx*wz
            mean_action /= wxz

            for agent_name in ctrl_value.keys():
                ctrl_value_single = ctrl_value[agent_name] 
                ctrl_value_znmf = ctrl_value_single - mean_action 
                ctrl_value[agent_name] = ctrl_value_znmf
        
        elif (self.conf.simulation.znmf_avg == 1) or (self.conf.simulation.znmf_avg == 0): 
            mean_action = 0.0
            print(f"[ZNMF] DONE BY NEK5000", flush=True)
        
        else:
            raise NotImplementedError('Please Ensure the ZNMF condition!')
        return ctrl_value

    def _normalize_state(self, state):
        """Normalize the state by various method."""
        if self.conf.runner.normalize_input != "None":
            if self.conf.runner.normalize_input == "utau":
                state /= self.utau
            elif self.conf.runner.normalize_input == "std":
                state /= np.std(state)
            elif self.conf.runner.normalize_input == "minmax":
                state = 2 * (state - np.min(state))/(np.max(state)-np.min(state)) - 1
        return state 

    def _normalize_reward(self, reward):
        """Normalizing the reward in the range of 0~1."""
        return (1 - (np.mean(reward)/self.baseline_dudy))

    def restart_handle(self):
        """Management of rs8/rs6 data for each episode."""
        if self.conf.runner.random_init > 0: 
            n_init = np.random.randint(low=1, high=self.conf.runner.random_init)
            target_folder = os.path.join(self.rstart_folder, f"init_{n_init}")
            rs_list = os.listdir(target_folder)
            rs_list = [f for f in rs_list if 'rs' in f ]
            for rsfile in rs_list: 
                rsfile = os.path.join(target_folder, rsfile)
                shutil.copy(rsfile, dst=self.mpi_info['wdir']+'/')
                print(f"[RSTART] RESET: {rsfile}", flush=True)

        elif self.conf.runner.random_init == -1:
            print(f'[RSTART] NOT SHUFFLE; RANK={self.conf.runner.rank}', flush=True)
            target_folder = os.path.join(self.rstart_folder, f"init_{self.conf.runner.rank}")
            n_init = self.conf.runner.rank
            rs_list = os.listdir(target_folder)
            rs_list = [f for f in rs_list if 'rs' in f ]
            for rsfile in rs_list: 
                rsfile = os.path.join(target_folder, rsfile)
                shutil.copy(rsfile, dst=self.mpi_info['wdir']+'/')
                print(f"[RSTART] RESET: {rsfile}", flush=True)
        
        elif (self.conf.runner.random_init < -1) and (self.restart_index == 1):
            print(f'[RSTART] NOT OVERWIRTE; RANK {self.conf.runner.rank}', flush=True)
            target_folder = os.path.join(self.rstart_folder, f"init_{self.conf.runner.rank}")
            ## But give a Sainty check, ensure at least rs8 files exist
            source_folder = self.mpi_info['wdir']+'/'
            rs_list = os.listdir(target_folder)
            rs_list = [f for f in rs_list if 'rs' in f ]
            
            if len(rs_list) < 3:
                raise ValueError(f"[RSTART] NOT ENOUGH FILE TO RESTART")
            else:
                for rsfile in rs_list:
                    print(f"[RSTART] EXIST: {rsfile}", flush=True)

    def get_cluster_info(self):
        """Get information about the clustering."""
        return {
            'nxs': self.nxs,
            'nzs': self.nzs,
            'original_nAgents': self.original_nAgents,
            'clustered_nAgents': self.nAgents,
            'cluster_mapping': self.cluster_mapping,
            'node_to_cluster': self.node_to_cluster
        }
