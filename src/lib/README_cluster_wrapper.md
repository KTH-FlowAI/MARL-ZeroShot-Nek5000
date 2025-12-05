# Node Cluster Wrapper for DRL Flow Control

This module provides a wrapper class that clusters GLL nodes into agents for DRL training while maintaining the same MPI interface as the original NEK5000 environment.

## Overview

The `NodeClusterWrapper` allows you to group multiple GLL nodes within each spectral element into single agents, reducing the total number of agents while maintaining the same MPI communication interface. This is particularly useful when you want to:

- Reduce the computational complexity of multi-agent reinforcement learning
- Group spatially related nodes for more coherent control strategies
- Scale up to larger problems with many control points

## Key Features

- **Maintains MPI Interface**: No changes needed to the NEK5000 MPI communication
- **Flexible Clustering**: Configurable clustering in x and z directions (`nxs`, `nzs`)
- **State Aggregation**: Combines state data from clustered nodes
- **Action Distribution**: Distributes actions from clustered agents to individual nodes
- **Reward Aggregation**: Averages rewards from clustered nodes
- **Backward Compatibility**: Works as a drop-in replacement for the original environment

## Files

- `node_cluster_wrapper.py`: Main wrapper class implementation
- `example_cluster_usage.py`: Example usage and integration guide
- `test_cluster_wrapper.py`: Test script to verify functionality
- `README_cluster_wrapper.md`: This documentation file

## Usage

### Basic Usage

```python
from src.lib.node_cluster_wrapper import NodeClusterWrapper
from src.lib.sb3_utils import init_env

# Create original environment
env, nAgents, action_space, obs_space = init_env(conf, run_folder, sub_comm)

# Wrap with clustering (cluster 2x2 nodes)
clustered_env = NodeClusterWrapper(env, nxs=2, nzs=2)

# Use clustered environment exactly like the original
observations = clustered_env.reset()
actions = {agent: np.random.randn() for agent in clustered_env.possible_agents}
obs, rewards, dones, infos = clustered_env.step(actions)
```

### Integration with Existing Code

Modify your existing training script:

```python
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

# In your main training loop:
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
```

## Clustering Parameters

- `nxs`: Number of nodes to cluster in the x-direction (default: 1)
- `nzs`: Number of nodes to cluster in the z-direction (default: 1)

### Examples

- `nxs=1, nzs=1`: No clustering (original behavior)
- `nxs=2, nzs=1`: Cluster 2 nodes in x-direction
- `nxs=1, nzs=2`: Cluster 2 nodes in z-direction  
- `nxs=2, nzs=2`: Cluster 2x2 nodes (4 nodes per agent)
- `nxs=4, nzs=1`: Cluster all 4 nodes in x-direction

## Data Flow

### State Clustering
1. Original state data: `(npl_state, 1, 1)` per node
2. Clustered state data: `(npl_state, nzs, nxs)` per cluster
3. Nodes within each spectral element are grouped and their states are stacked

### Action Distribution
1. Clustered actions: Single action per cluster
2. Distributed actions: Same action applied to all nodes in the cluster
3. Maintains zero-net-mass-flux (ZNMF) condition through the original environment

### Reward Aggregation
1. Individual node rewards: One reward per node
2. Clustered rewards: Average reward across all nodes in the cluster
3. Preserves reward structure while reducing agent count

## Configuration Updates

The wrapper automatically updates the environment configuration:

```python
# Original configuration
conf.simulation.nxs = 1
conf.simulation.nzs = 1

# After clustering with nxs=2, nzs=2
conf.simulation.nxs = 2
conf.simulation.nzs = 2
```

## Error Handling

The wrapper handles various edge cases:

- **Odd number of nodes**: Automatically adjusts clustering to fit available nodes
- **Invalid clustering parameters**: Raises `ValueError` for invalid `nxs` or `nzs`
- **Insufficient nodes**: Handles cases where clustering parameters exceed available nodes

## Testing

Run the test script to verify functionality:

```bash
cd /home/yuninw/codes/drl/nek_drl
python src/lib/test_cluster_wrapper.py
```

## Performance Considerations

- **Memory**: Clustered observations use more memory per agent due to increased dimensions
- **Computation**: State clustering and action distribution add minimal overhead
- **Scalability**: Reduces the number of agents, potentially improving training efficiency

## Limitations

- Clustering is done within each spectral element (GLLID) - nodes from different elements are not clustered together
- The clustering must be compatible with the number of nodes per element
- State dimensions increase from `(npl_state, 1, 1)` to `(npl_state, nzs, nxs)`

## Example Results

For a typical case with 12 nodes in 3 spectral elements:

| Clustering | Original Agents | Clustered Agents | Observation Shape |
|------------|----------------|------------------|-------------------|
| 1x1 (none) | 12 | 12 | (npl_state, 1, 1) |
| 2x1 | 12 | 6 | (npl_state, 1, 2) |
| 1x2 | 12 | 6 | (npl_state, 2, 1) |
| 2x2 | 12 | 3 | (npl_state, 2, 2) |
| 4x1 | 12 | 3 | (npl_state, 1, 4) |

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're in the correct directory and have the required dependencies
2. **Clustering Error**: Check that `nxs * nzs` divides evenly into the number of nodes per element
3. **Memory Error**: Large clustering parameters may increase memory usage significantly

### Debug Information

Use the `get_cluster_info()` method to inspect clustering:

```python
cluster_info = clustered_env.get_cluster_info()
print(f"Original agents: {cluster_info['original_nAgents']}")
print(f"Clustered agents: {cluster_info['clustered_nAgents']}")
print(f"Cluster mapping: {cluster_info['cluster_mapping']}")
```

## Future Enhancements

- Support for cross-element clustering
- Weighted reward aggregation based on node importance
- Adaptive clustering based on flow features
- Integration with different clustering strategies (e.g., k-means, spatial proximity)
