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
  
  def predict(self,observations:dict,state,episode_start,deterministic):
    """
    Return the action to ENV based on the policy 
    """
    actions = { agent:self.policy(observations[agent]) for agent in observations.keys()}
    return actions, None



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
    action = np.array([-1 * self.alpha,])  
    return action

