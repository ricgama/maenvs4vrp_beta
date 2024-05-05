import torch
from tensordict import TensorDict
from maenvs4vrp.core.env_agent_reward import RewardFn

from typing import Optional, List


class DenseReward(RewardFn):
    """Reward class.
    """

    def __init__(self):
        """Constructor

        """
        self.env = None
        self.pending_penalty = -1

    def set_env(self, env):
        self.env = env

    def get_reward(self, action):
        """
        
        """     
        raise NotImplementedError()




