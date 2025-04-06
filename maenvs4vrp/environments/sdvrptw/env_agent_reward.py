import torch
from tensordict import TensorDict
from maenvs4vrp.core.env_agent_reward import RewardFn

from typing import Optional, List


class DenseReward(RewardFn):
    """
    SDVRPTW dense reward class.
    """

    def __init__(self):
        """
        Constructor.

        Args: 
            n/a.

        Returns:
            None.
        """
        self.env = None
        self.pending_penalty = -10

    def set_env(self, env):
        """
        Set environment.

        Args:
            env(AECEnv): Environment.

        Returns:
            None.
        """
        self.env = env

    def get_reward(self, action):
        """
        Get reward and penalty.

        Args:
            action(torch.Tensor): Tensor with agent moves.

        Returns:
            reward(torch.Tensor): Reward.
            penalty(torch.Tensor): Penalty.
        """

        reward = -self.env.td_state['cur_agent']['cur_ttime'].clone()
        penalty = torch.zeros_like(action, dtype = torch.float, device=self.env.device)

        # compute penalty if env has unvisited nodes 
        is_last_step = self.env.td_state['is_last_step']
        
        depot2nodes = torch.pairwise_distance(self.env.td_state['depot_loc'], self.env.td_state['coords'], eps=0, keepdim = False)
        if self.env.n_digits is not None:
            depot2nodes = torch.floor(self.env.n_digits * depot2nodes) / self.env.n_digits
        penalty[is_last_step] = self.pending_penalty * ((depot2nodes * self.env.td_state['nodes']['active_nodes_mask']).sum(-1, keepdim = True).float()[is_last_step])

        return reward, penalty



class SparseReward(RewardFn):
    """
    SDVRPTW sparse reward class.
    """

    def __init__(self):
        """
        Constructor.

        Args: 
            n/a.

        Returns:
            None.
        """
        self.env = None
        self.pending_penalty = -10

    def set_env(self, env):
        """
        Set environment.

        Args:
            env(Environment): Environment.

        Returns:
            None.
        """
        self.env = env

    def get_reward(self, action):
        """
        Get reward and penalty.

        Args:
            action(torch.Tensor): Tensor with agent moves.

        Returns:
            reward(torch.Tensor): Reward.
            penalty(torch.Tensor): Penalty.
        """

        reward = torch.zeros_like(action, dtype = torch.float, device=self.env.device)
        penalty = torch.zeros_like(action, dtype = torch.float, device=self.env.device)

        # compute penalty if env has unvisited nodes 
        is_last_step = self.env.td_state['is_last_step']
        
        depot2nodes = torch.pairwise_distance(self.env.td_state['depot_loc'], self.env.td_state['coords'], eps=0, keepdim = False)
        if self.env.n_digits is not None:
            depot2nodes = torch.floor(self.env.n_digits * depot2nodes) / self.env.n_digits

        final_reward = -self.env.td_state['agents']['cum_ttime'].sum(1, keepdim = True)
        penalty[is_last_step] = self.pending_penalty * ((depot2nodes * self.env.td_state['nodes']['active_nodes_mask']).sum(-1, keepdim = True).float()[is_last_step])
        
        reward[is_last_step] = final_reward[is_last_step]
        return reward, penalty
