"""
Node Cluster Wrapper for DRL Environment

This module provides a wrapper class that clusters GLL nodes into agents
while maintaining the same MPI interface. The clustering is done based on
nxs and nzs parameters, grouping nodes within each spectral element.

Author: AI Assistant
Date: 2025
"""

import numpy as np
import functools
from typing import Dict, List, Tuple, Any
try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces
from pettingzoo import ParallelEnv


class NodeClusterWrapper:
    """
    A wrapper class that clusters GLL nodes into agents for DRL training.
    
    This wrapper maintains the same MPI interface as the original environment
    but groups nodes within each spectral element into clusters based on
    nxs and nzs parameters. Each cluster becomes a single agent.
    
    Key features:
    - Clusters nodes within each spectral element (GLLID)
    - Aggregates state data from clustered nodes
    - Distributes actions from clustered agents to individual nodes
    - Aggregates rewards from clustered nodes
    - Maintains same MPI communication interface
    """
    
    def __init__(self, env: ParallelEnv, nxs: int = 1, nzs: int = 1):
        """
        Initialize the node cluster wrapper.
        
        Args:
            env: The original parallel environment
            nxs: Number of nodes to cluster in x-direction (default: 1)
            nzs: Number of nodes to cluster in z-direction (default: 1)
        """
        self.env = env
        self.nxs = nxs
        self.nzs = nzs
        
        # Validate clustering parameters
        if nxs <= 0 or nzs <= 0:
            raise ValueError("nxs and nzs must be positive integers")
        
        # Get original agent information
        self.original_agent_info = env.agent_info.copy()
        self.original_possible_agents = env.possible_agents.copy()
        self.original_nAgents = env.nAgents
        
        # Create clustered agent mapping
        self._create_cluster_mapping()
        
        # Update environment attributes
        self._update_env_attributes()
        
        print(f"[CLUSTER] Created {self.nAgents} clustered agents from {self.original_nAgents} nodes")
        print(f"[CLUSTER] Clustering: nxs={nxs}, nzs={nzs}")
    
    def _create_cluster_mapping(self):
        """
        Create mapping between original nodes and clustered agents.
        
        This method groups nodes within each spectral element (GLLID) into
        clusters based on nxs and nzs parameters.
        """
        # Group nodes by spectral element (GLLID) and face (FACEID)
        element_groups = {}
        for i, (nid, gllid, iface, ix, iy, iz) in enumerate(zip(
            self.original_agent_info['NID'],
            self.original_agent_info['GLLID'],
            self.original_agent_info['FACEID'],
            self.original_agent_info['ix'],
            self.original_agent_info['iy'],
            self.original_agent_info['iz']
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
                'iz': iz
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
            # Sort nodes by ix, then iz for consistent clustering
            nodes.sort(key=lambda x: (x['ix'], x['iz']))
            
            # Group nodes into clusters
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
            
            # Create clusters
            for cluster_idx in range(0, n_nodes, nodes_per_cluster):
                cluster_nodes = nodes[cluster_idx:cluster_idx + nodes_per_cluster]
                
                # Create cluster agent name
                cluster_agent_name = self._name_cluster_agent(
                    nid, gllid, iface, cluster_idx // nodes_per_cluster
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
    
    def _name_cluster_agent(self, nid: int, gllid: int, iface: int, cluster_idx: int) -> str:
        """
        Generate a name for a clustered agent.
        
        Args:
            nid: Node ID (rank)
            gllid: GLL element ID
            iface: Face ID
            cluster_idx: Cluster index within the element
            
        Returns:
            Agent name string
        """
        return f"cluster_np{nid:05d}_gid{gllid:05d}_iface{iface}_c{cluster_idx:02d}"
    
    def _update_env_attributes(self):
        """
        Update the wrapped environment's attributes to reflect clustering.
        """
        # Update agent-related attributes
        self.env.agent_info = self.clustered_agent_info
        self.env.possible_agents = self.possible_agents
        self.env.nAgents = self.nAgents
        self.env.agent_name_mapping = self.agent_name_mapping
        
        # Update observation space shape to include clustering dimensions
        self.env.conf.simulation.nxs = self.nxs
        self.env.conf.simulation.nzs = self.nzs
    
    def _cluster_state_data(self, original_observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Cluster state data from individual nodes to clustered agents.
        
        Args:
            original_observations: Dictionary mapping original agent names to observations
            
        Returns:
            Dictionary mapping clustered agent names to clustered observations
        """
        clustered_observations = {}
        
        for cluster_agent_name, node_indices in self.cluster_mapping.items():
            # Get observations for all nodes in this cluster
            cluster_observations = []
            for node_idx in node_indices:
                original_agent_name = self.original_possible_agents[node_idx]
                if original_agent_name in original_observations:
                    cluster_observations.append(original_observations[original_agent_name])
            
            if cluster_observations:
                # Stack observations along the spatial dimensions
                # Original shape: (npl_state, 1, 1) -> Clustered shape: (npl_state, nzs, nxs)
                stacked_obs = np.stack(cluster_observations, axis=-1)  # (npl_state, 1, n_nodes)
                
                # Reshape to (npl_state, nzs, nxs)
                npl_state = stacked_obs.shape[0]
                if stacked_obs.shape[-1] == self.nxs * self.nzs:
                    clustered_obs = stacked_obs.reshape(npl_state, self.nzs, self.nxs)
                else:
                    # Handle case where we don't have exact nxs * nzs nodes
                    # Pad or truncate as needed
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
                
                clustered_observations[cluster_agent_name] = clustered_obs
            else:
                # Create zero observation if no data available
                npl_state = self.env.conf.runner.npl_state
                clustered_observations[cluster_agent_name] = np.zeros((npl_state, self.nzs, self.nxs))
        
        return clustered_observations
    
    def _distribute_actions(self, clustered_actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Distribute actions from clustered agents to individual nodes.
        
        Args:
            clustered_actions: Dictionary mapping clustered agent names to actions
            
        Returns:
            Dictionary mapping original agent names to distributed actions
        """
        distributed_actions = {}
        
        for cluster_agent_name, action in clustered_actions.items():
            if cluster_agent_name in self.cluster_mapping:
                node_indices = self.cluster_mapping[cluster_agent_name]
                
                # Distribute the same action to all nodes in the cluster
                for node_idx in node_indices:
                    original_agent_name = self.original_possible_agents[node_idx]
                    distributed_actions[original_agent_name] = action.copy()
        
        return distributed_actions
    
    def _cluster_rewards(self, original_rewards: Dict[str, float]) -> Dict[str, float]:
        """
        Cluster rewards from individual nodes to clustered agents.
        
        Args:
            original_rewards: Dictionary mapping original agent names to rewards
            
        Returns:
            Dictionary mapping clustered agent names to clustered rewards
        """
        clustered_rewards = {}
        
        for cluster_agent_name, node_indices in self.cluster_mapping.items():
            # Collect rewards for all nodes in this cluster
            cluster_rewards = []
            for node_idx in node_indices:
                original_agent_name = self.original_possible_agents[node_idx]
                if original_agent_name in original_rewards:
                    cluster_rewards.append(original_rewards[original_agent_name])
            
            if cluster_rewards:
                # Average the rewards from all nodes in the cluster
                clustered_rewards[cluster_agent_name] = np.mean(cluster_rewards)
            else:
                clustered_rewards[cluster_agent_name] = 0.0
        
        return clustered_rewards
    
    def _cluster_dones(self, original_dones: Dict[str, bool]) -> Dict[str, bool]:
        """
        Cluster done flags from individual nodes to clustered agents.
        
        Args:
            original_dones: Dictionary mapping original agent names to done flags
            
        Returns:
            Dictionary mapping clustered agent names to done flags
        """
        clustered_dones = {}
        
        for cluster_agent_name, node_indices in self.cluster_mapping.items():
            # If any node in the cluster is done, the cluster is done
            cluster_done = any(
                original_dones.get(self.original_possible_agents[node_idx], False)
                for node_idx in node_indices
            )
            clustered_dones[cluster_agent_name] = cluster_done
        
        return clustered_dones
    
    def _cluster_infos(self, original_infos: Dict[str, dict]) -> Dict[str, dict]:
        """
        Cluster info dictionaries from individual nodes to clustered agents.
        
        Args:
            original_infos: Dictionary mapping original agent names to info dicts
            
        Returns:
            Dictionary mapping clustered agent names to info dicts
        """
        clustered_infos = {}
        
        for cluster_agent_name, node_indices in self.cluster_mapping.items():
            # Collect info from all nodes in this cluster
            cluster_infos = []
            for node_idx in node_indices:
                original_agent_name = self.original_possible_agents[node_idx]
                if original_agent_name in original_infos:
                    cluster_infos.append(original_infos[original_agent_name])
            
            if cluster_infos:
                # Use the first node's info as representative
                clustered_infos[cluster_agent_name] = cluster_infos[0].copy()
            else:
                clustered_infos[cluster_agent_name] = {}
        
        return clustered_infos
    
    # Delegate all other methods to the wrapped environment
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped environment."""
        return getattr(self.env, name)
    
    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset the environment and return clustered observations.
        """
        # Call original reset
        original_observations = self.env.reset(seed=seed, return_info=return_info, options=options)
        
        # Cluster the observations
        clustered_observations = self._cluster_state_data(original_observations)
        
        return clustered_observations
    
    def step(self, actions):
        """
        Step the environment with clustered actions and return clustered results.
        """
        # Distribute actions to individual nodes
        distributed_actions = self._distribute_actions(actions)
        
        # Call original step
        observations, rewards, dones, infos = self.env.step(distributed_actions)
        
        # Cluster the results
        clustered_observations = self._cluster_state_data(observations)
        clustered_rewards = self._cluster_rewards(rewards)
        clustered_dones = self._cluster_dones(dones)
        clustered_infos = self._cluster_infos(infos)
        
        return clustered_observations, clustered_rewards, clustered_dones, clustered_infos
    
    def action_space(self, agent):
        """
        Return the action space for a clustered agent.
        """
        return self.env.action_space(agent)
    
    def observation_space(self, agent):
        """
        Return the observation space for a clustered agent.
        """
        return self.env.observation_space(agent)
    
    def render(self, mode='human', close=False):
        """
        Render the environment.
        """
        return self.env.render(mode=mode, close=close)
    
    def close(self):
        """
        Close the environment.
        """
        return self.env.close()
    
    def get_cluster_info(self):
        """
        Get information about the clustering.
        
        Returns:
            Dictionary containing clustering information
        """
        return {
            'nxs': self.nxs,
            'nzs': self.nzs,
            'original_nAgents': self.original_nAgents,
            'clustered_nAgents': self.nAgents,
            'cluster_mapping': self.cluster_mapping,
            'node_to_cluster': self.node_to_cluster
        }
