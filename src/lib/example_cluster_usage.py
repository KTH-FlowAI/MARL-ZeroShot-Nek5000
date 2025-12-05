"""
Example usage of the NodeClusterWrapper for DRL flow control.

This script demonstrates how to use the NodeClusterWrapper to cluster
GLL nodes into agents while maintaining the same MPI interface.

Author: AI Assistant
Date: 2025
"""

import numpy as np
from node_cluster_wrapper import NodeClusterWrapper
from nek_marl import parallel_env
from configs import Config
import os


def create_clustered_environment(conf, rank_folder, sub_comm, nxs=2, nzs=2):
    """
    Create a clustered environment from the original parallel environment.
    
    Args:
        conf: Configuration object
        rank_folder: Rank folder for the environment
        sub_comm: MPI sub-communicator
        nxs: Number of nodes to cluster in x-direction
        nzs: Number of nodes to cluster in z-direction
    
    Returns:
        Clustered environment wrapper
    """
    # Create original environment
    original_env = parallel_env(conf, rank_folder, sub_comm)
    
    # Wrap with clustering
    clustered_env = NodeClusterWrapper(original_env, nxs=nxs, nzs=nzs)
    
    return clustered_env


def demonstrate_clustering():
    """
    Demonstrate the clustering functionality with a simple example.
    """
    print("=== Node Clustering Demonstration ===")
    
    # Example configuration (you would load your actual config)
    conf = Config()
    conf.simulation.nxs = 1  # Original setting
    conf.simulation.nzs = 1  # Original setting
    conf.runner.npl_state = 2  # Number of state variables
    
    # Simulate agent information (normally comes from NEK5000)
    n_original_agents = 12  # Example: 12 individual nodes
    n_elements = 3  # Example: 3 spectral elements
    nodes_per_element = n_original_agents // n_elements
    
    # Create mock agent info
    agent_info = {
        'NID': np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
        'GLLID': np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]),
        'FACEID': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        'ix': np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]),
        'iy': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        'iz': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        'NUMCTRL': np.ones(n_original_agents, dtype=int)
    }
    
    # Create mock environment class for demonstration
    class MockEnv:
        def __init__(self, agent_info, conf):
            self.agent_info = agent_info
            self.conf = conf
            self.possible_agents = [f"jet_np{i//4:05d}_gid{i//4+1:05d}_iface1_ix{i%4+1:05d}_iy1_iz1" 
                                  for i in range(n_original_agents)]
            self.nAgents = n_original_agents
            self.agent_name_mapping = dict(zip(self.possible_agents, range(n_original_agents)))
        
        def action_space(self, agent):
            return None
        
        def observation_space(self, agent):
            return None
        
        def reset(self, **kwargs):
            # Return mock observations
            observations = {}
            for i, agent in enumerate(self.possible_agents):
                observations[agent] = np.random.randn(conf.runner.npl_state, 1, 1)
            return observations
        
        def step(self, actions):
            # Return mock step results
            observations = self.reset()
            rewards = {agent: np.random.randn() for agent in self.possible_agents}
            dones = {agent: False for agent in self.possible_agents}
            infos = {agent: {} for agent in self.possible_agents}
            return observations, rewards, dones, infos
        
        def close(self):
            pass
    
    # Create mock environment
    mock_env = MockEnv(agent_info, conf)
    
    # Test different clustering configurations
    clustering_configs = [
        (1, 1, "No clustering (original)"),
        (2, 1, "Cluster 2 nodes in x-direction"),
        (1, 2, "Cluster 2 nodes in z-direction"),
        (2, 2, "Cluster 2x2 nodes"),
        (4, 1, "Cluster all 4 nodes in x-direction")
    ]
    
    for nxs, nzs, description in clustering_configs:
        print(f"\n--- {description} (nxs={nxs}, nzs={nzs}) ---")
        
        try:
            # Create clustered environment
            clustered_env = NodeClusterWrapper(mock_env, nxs=nxs, nzs=nzs)
            
            print(f"Original agents: {mock_env.nAgents}")
            print(f"Clustered agents: {clustered_env.nAgents}")
            print(f"Clustered agent names: {clustered_env.possible_agents[:3]}...")  # Show first 3
            
            # Test reset
            observations = clustered_env.reset()
            print(f"Observation shape for first agent: {observations[clustered_env.possible_agents[0]].shape}")
            
            # Test step
            actions = {agent: np.random.randn() for agent in clustered_env.possible_agents}
            obs, rewards, dones, infos = clustered_env.step(actions)
            print(f"Reward for first agent: {rewards[clustered_env.possible_agents[0]]:.3f}")
            
            # Show cluster mapping
            cluster_info = clustered_env.get_cluster_info()
            print(f"Cluster mapping example: {list(cluster_info['cluster_mapping'].items())[:2]}")
            
        except Exception as e:
            print(f"Error with nxs={nxs}, nzs={nzs}: {e}")


def integration_example():
    """
    Example of how to integrate the clustering wrapper with your existing code.
    """
    print("\n=== Integration Example ===")
    
    # This is how you would modify your existing training script
    example_code = '''
# In your training script (e.g., run.py or train.py):

from src.lib.node_cluster_wrapper import NodeClusterWrapper
from src.lib.sb3_utils import init_env

def create_clustered_env(conf, run_folder, sub_comm, nxs=2, nzs=2):
    """Create a clustered environment for training."""
    
    # Create original environment
    env, nAgents, action_space_sample, observation_space_sample = init_env(conf, run_folder, sub_comm)
    
    # Wrap with clustering
    clustered_env = NodeClusterWrapper(env, nxs=nxs, nzs=nzs)
    
    # Update configuration to reflect clustering
    conf.simulation.nxs = nxs
    conf.simulation.nzs = nzs
    
    return clustered_env, clustered_env.nAgents, action_space_sample, observation_space_sample

# Usage in your main training loop:
if __name__ == "__main__":
    # Load your configuration
    conf = load_config("your_config.yml")
    
    # Set clustering parameters
    nxs = 2  # Cluster 2 nodes in x-direction
    nzs = 2  # Cluster 2 nodes in z-direction
    
    # Create clustered environment
    env, nAgents, action_space, obs_space = create_clustered_env(conf, run_folder, sub_comm, nxs, nzs)
    
    # Train with clustered environment
    # ... your training code here ...
    '''
    
    print(example_code)


if __name__ == "__main__":
    # Run demonstration
    demonstrate_clustering()
    
    # Show integration example
    integration_example()
    
    print("\n=== Summary ===")
    print("The NodeClusterWrapper allows you to:")
    print("1. Cluster GLL nodes into agents based on nxs and nzs parameters")
    print("2. Maintain the same MPI interface as the original environment")
    print("3. Aggregate state data from clustered nodes")
    print("4. Distribute actions from clustered agents to individual nodes")
    print("5. Aggregate rewards from clustered nodes")
    print("\nTo use in your code:")
    print("- Import: from src.lib.node_cluster_wrapper import NodeClusterWrapper")
    print("- Wrap your environment: clustered_env = NodeClusterWrapper(env, nxs=2, nzs=2)")
    print("- Use clustered_env exactly like the original environment")
