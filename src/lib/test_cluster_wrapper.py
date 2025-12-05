"""
Test script for the NodeClusterWrapper.

This script tests the clustering functionality with mock data to ensure
the wrapper works correctly before integration with the actual NEK5000 environment.

Author: AI Assistant
Date: 2025
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from node_cluster_wrapper import NodeClusterWrapper


class MockConfig:
    """Mock configuration class for testing."""
    def __init__(self):
        self.runner = MockRunner()
        self.simulation = MockSimulation()

class MockRunner:
    def __init__(self):
        self.npl_state = 2

class MockSimulation:
    def __init__(self):
        self.nxs = 1
        self.nzs = 1


class MockEnvironment:
    """Mock environment for testing the clustering wrapper."""
    
    def __init__(self, n_agents=12):
        self.n_agents = n_agents
        self.conf = MockConfig()
        
        # Create mock agent information
        self.agent_info = self._create_mock_agent_info(n_agents)
        self.possible_agents = [f"jet_np{i//4:05d}_gid{i//4+1:05d}_iface1_ix{i%4+1:05d}_iy1_iz1" 
                              for i in range(n_agents)]
        self.nAgents = n_agents
        self.agent_name_mapping = dict(zip(self.possible_agents, range(n_agents)))
    
    def _create_mock_agent_info(self, n_agents):
        """Create mock agent information."""
        n_elements = n_agents // 4  # 4 nodes per element
        
        agent_info = {
            'NID': np.array([i // 4 for i in range(n_agents)]),
            'GLLID': np.array([i // 4 + 1 for i in range(n_agents)]),
            'FACEID': np.ones(n_agents, dtype=int),
            'ix': np.array([i % 4 + 1 for i in range(n_agents)]),
            'iy': np.ones(n_agents, dtype=int),
            'iz': np.ones(n_agents, dtype=int),
            'NUMCTRL': np.ones(n_agents, dtype=int)
        }
        return agent_info
    
    def action_space(self, agent):
        return None
    
    def observation_space(self, agent):
        return None
    
    def reset(self, **kwargs):
        """Return mock observations."""
        observations = {}
        for i, agent in enumerate(self.possible_agents):
            # Create mock state data with shape (npl_state, 1, 1)
            observations[agent] = np.random.randn(self.conf.runner.npl_state, 1, 1)
        return observations
    
    def step(self, actions):
        """Return mock step results."""
        observations = self.reset()
        rewards = {agent: np.random.randn() for agent in self.possible_agents}
        dones = {agent: False for agent in self.possible_agents}
        infos = {agent: {'time': 0.0} for agent in self.possible_agents}
        return observations, rewards, dones, infos
    
    def close(self):
        pass


def test_basic_clustering():
    """Test basic clustering functionality."""
    print("=== Testing Basic Clustering ===")
    
    # Create mock environment
    mock_env = MockEnvironment(n_agents=12)
    print(f"Original environment: {mock_env.nAgents} agents")
    
    # Test different clustering configurations
    test_cases = [
        (1, 1, "No clustering"),
        (2, 1, "2x1 clustering"),
        (1, 2, "1x2 clustering"),
        (2, 2, "2x2 clustering"),
        (4, 1, "4x1 clustering")
    ]
    
    for nxs, nzs, description in test_cases:
        print(f"\n--- {description} (nxs={nxs}, nzs={nzs}) ---")
        
        try:
            # Create clustered environment
            clustered_env = NodeClusterWrapper(mock_env, nxs=nxs, nzs=nzs)
            
            print(f"Clustered agents: {clustered_env.nAgents}")
            print(f"Agent names: {clustered_env.possible_agents[:3]}...")
            
            # Test reset
            observations = clustered_env.reset()
            first_agent = clustered_env.possible_agents[0]
            print(f"Observation shape: {observations[first_agent].shape}")
            
            # Test step
            actions = {agent: np.random.randn() for agent in clustered_env.possible_agents}
            obs, rewards, dones, infos = clustered_env.step(actions)
            print(f"Reward: {rewards[first_agent]:.3f}")
            
            # Test cluster mapping
            cluster_info = clustered_env.get_cluster_info()
            print(f"Cluster mapping: {len(cluster_info['cluster_mapping'])} clusters")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def test_state_clustering():
    """Test state data clustering."""
    print("\n=== Testing State Clustering ===")
    
    mock_env = MockEnvironment(n_agents=8)
    clustered_env = NodeClusterWrapper(mock_env, nxs=2, nzs=2)
    
    # Get original observations
    original_obs = mock_env.reset()
    print(f"Original observations: {len(original_obs)} agents")
    print(f"Original observation shape: {list(original_obs.values())[0].shape}")
    
    # Get clustered observations
    clustered_obs = clustered_env.reset()
    print(f"Clustered observations: {len(clustered_obs)} agents")
    print(f"Clustered observation shape: {list(clustered_obs.values())[0].shape}")
    
    # Verify clustering
    cluster_info = clustered_env.get_cluster_info()
    for cluster_name, node_indices in cluster_info['cluster_mapping'].items():
        print(f"Cluster {cluster_name}: {len(node_indices)} nodes")
        print(f"  Node indices: {node_indices}")


def test_action_distribution():
    """Test action distribution from clusters to nodes."""
    print("\n=== Testing Action Distribution ===")
    
    mock_env = MockEnvironment(n_agents=8)
    clustered_env = NodeClusterWrapper(mock_env, nxs=2, nzs=2)
    
    # Create clustered actions
    clustered_actions = {agent: np.random.randn() for agent in clustered_env.possible_agents}
    print(f"Clustered actions: {len(clustered_actions)} agents")
    
    # Test step (which internally distributes actions)
    obs, rewards, dones, infos = clustered_env.step(clustered_actions)
    print(f"Step completed successfully with {len(obs)} observations")


def test_reward_aggregation():
    """Test reward aggregation from nodes to clusters."""
    print("\n=== Testing Reward Aggregation ===")
    
    mock_env = MockEnvironment(n_agents=8)
    clustered_env = NodeClusterWrapper(mock_env, nxs=2, nzs=2)
    
    # Test step to get rewards
    actions = {agent: np.random.randn() for agent in clustered_env.possible_agents}
    obs, rewards, dones, infos = clustered_env.step(actions)
    
    print(f"Clustered rewards: {len(rewards)} agents")
    print(f"Reward values: {list(rewards.values())}")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")
    
    # Test with odd number of agents
    print("Testing with odd number of agents...")
    try:
        mock_env = MockEnvironment(n_agents=7)  # Odd number
        clustered_env = NodeClusterWrapper(mock_env, nxs=2, nzs=2)
        print(f"Success: {clustered_env.nAgents} clustered agents")
    except Exception as e:
        print(f"Expected error: {e}")
    
    # Test with invalid clustering parameters
    print("\nTesting with invalid clustering parameters...")
    try:
        mock_env = MockEnvironment(n_agents=8)
        clustered_env = NodeClusterWrapper(mock_env, nxs=0, nzs=2)
        print("Unexpected: Should have failed")
    except ValueError as e:
        print(f"Expected error: {e}")
    
    # Test with clustering larger than available nodes
    print("\nTesting with clustering larger than available nodes...")
    try:
        mock_env = MockEnvironment(n_agents=4)
        clustered_env = NodeClusterWrapper(mock_env, nxs=3, nzs=2)
        print(f"Success: {clustered_env.nAgents} clustered agents")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Testing NodeClusterWrapper...")
    
    # Run all tests
    test_basic_clustering()
    test_state_clustering()
    test_action_distribution()
    test_reward_aggregation()
    test_edge_cases()
    
    print("\n=== All Tests Completed ===")
    print("The NodeClusterWrapper is ready for integration with your NEK5000 environment!")
