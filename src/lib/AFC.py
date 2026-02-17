"""
The classical approaches for wing control 
"""

import numpy as np 

class AFC:
  """
  A general object for classical AFC approaches 
  """
  def __init__(self,agent_list) -> None:
    self.agent_list = agent_list

  def policy(self,observation):
    """
    The control policy depends on the observation
    """
    action = None 
    if action == None: 
      raise NotImplementedError('[ERROR] Please Add the policy function!')
    return action
  
  def predict(self,observations,state,episode_start,deterministic):
    """
    Return the action to ENV based on the policy 
    """
    if isinstance(observations, dict):
      actions = {agent: self.policy(observations[agent]) for agent in observations.keys()}
      return actions, None

    if isinstance(observations, (list, tuple, np.ndarray)):
      if self.agent_list is None:
        agents = list(range(len(observations)))
      else:
        agents = self.agent_list
      if len(agents) != len(observations):
        raise ValueError("[ERROR] Observations length does not match agent_list length.")
      actions = {agent: self.policy(obs) for agent, obs in zip(agents, observations)}
      return actions, None

    raise TypeError("[ERROR] observations must be dict, list, tuple, or numpy.ndarray.")



class OppoCtrl(AFC):
  """
  Opposition control implementation
  """

  def __init__(self, agent_list, ctrl_max_amp) -> None:
    super().__init__(agent_list)
    self.alpha = ctrl_max_amp

  def policy(self, observation):
    """
    v  = -alpha *(v - <v>)
    """
    ## We define the V-Vel <==> 1 
    # print(f"[OC] OBS SHAPE={observation.shape}")
    action = -1.0 * self.alpha * observation[1,0,0] 
    return action

class BLCtrl(AFC):
  """
  Steady uniform blowing/suction
  """
  def __init__(self, agent_list, ctrl_max_amp) -> None:
    super().__init__(agent_list)
    self.alpha = ctrl_max_amp

  def policy(self, observation):
    """
    v  = Psi
    """
    ## On the suction side the  
    # action = np.array([1 * self.alpha*observation,])  
    action = np.array([1 * self.alpha,])  
    return action

class SinWave():
  """
  Imposing Sinsoidal wave to check the mean
  """

  def __init__(self, agent_list, ctrl_max_amp,
                Kx, Kz, Lx, Lz) -> None:
    self.alpha = ctrl_max_amp
    self.agent_list = agent_list
    self.Kx, self.Kz = Kx, Kz 
    self.Lx, self.Lz = Lx, Lz 

  def load_node_info(self, Node_Info,): 
    """Get the node info to impose the Sinsoidal Wave"""
    self.Node_Info = Node_Info
    print(f"Node Info {self.Node_Info}",flush=True)
    return 

  def policy(self, observation):
    ## We define the V-Vel <==> 1
    x,z = observation
    kx = self.Kx*2*np.pi * x / self.Lx 
    kz = self.Kz*2*np.pi * z / self.Lz 
    action = np.array([self.alpha * np.sin(kx), ])
    return action
  def predict(self,observations,state,episode_start,deterministic):
    obs={}
    for il,agent_name_ in enumerate(self.agent_list):
            agent_name = self.nameAgent(nid=self.Node_Info['NID'][il],
                                    gllid=self.Node_Info['GLLID'][il],
                                    iface=self.Node_Info['FACEID'][il],
                                    ix=self.Node_Info['ix'][il],
                                    iy=self.Node_Info['iy'][il],
                                    iz=self.Node_Info['iz'][il],
                                            )
            x,z = self.Node_Info['x'][il], self.Node_Info['z'][il]
            obs[agent_name] = (x,z)
    actions = { agent:self.policy(obs[agent]) for agent in obs.keys()}
    return actions, None

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
    
