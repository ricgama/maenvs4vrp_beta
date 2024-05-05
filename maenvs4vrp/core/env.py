from maenvs4vrp.core.env_generator_builder import InstanceBuilder
from maenvs4vrp.core.env_observation_builder import ObservationBuilder
from maenvs4vrp.core.env_agent_selector import BaseSelector
from maenvs4vrp.core.env_agent_reward import RewardFn


from typing import Any, Dict, Iterable, Iterator, TypeVar, Tuple, Optional

import torch
from tensordict.tensordict import TensorDict

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
AgentID = str

ObsDict = Dict[AgentID, ObsType]
ActionDict = Dict[AgentID, ActionType]

class AECEnv():
    """The env module defines the base Environment class.
    
    Based on https://pettingzoo.farama.org/api/aec/

    https://arxiv.org/abs/2009.13051

    References: 
        https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/utils/env.py
        https://gitlab.aicrowd.com/flatland/flatland/-/blob/master/flatland/envs/rail_env.py
        The module is responsible for defining and manage the simulations environment.        

        
    """

    DEFAULT_SEED = 2925
    def __init__(self,
            instance_generator_object: InstanceBuilder,  
            obs_builder_object: ObservationBuilder,
            agent_selector_object: BaseSelector,
            reward_evaluator: RewardFn,
            seed:int = None,               
            device: Optional[str] = None,
            batch_size: torch.Size = None,
            ):        
        """
        
        Args:
            instance_generator_object: constructor instances. 
            obs_builder_object: constructor of observations. 
            agent_iterator_object: next agent generator.
            seed (int): initialise the seeds of the random number generators

        """

        if seed is None:
            self._set_seed(self.DEFAULT_SEED)
        else:
            self._set_seed(seed)

        self.agent_selector = agent_selector_object

        self.inst_generator = instance_generator_object
        self.inst_generator._set_seed(self.seed)
        self.obs_builder = obs_builder_object
        self.obs_builder.set_env(self)      
        self.reward_evaluator = reward_evaluator
        self.reward_evaluator.set_env(self)   
              
        if device == None:
            self.device = instance_generator_object.device
        else:
            self.device = device

        if batch_size is None:
            self.batch_size =  self.inst_generator.batch_size
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)
            instance_generator_object.batch_size = torch.Size(batch_size)


        self.num_nodes = self.inst_generator.max_num_nodes
        self.num_agents = self.inst_generator.max_num_agents

        self.nodes_static_feat_dim = self.obs_builder.get_nodes_static_feat_dim()
        self.nodes_dynamic_feat_dim = self.obs_builder.get_nodes_dynamic_feat_dim()
        self.agent_feat_dim = self.obs_builder.get_agent_feat_dim()
        self.agents_feat_dim = self.obs_builder.get_other_agents_feat_dim()
        self.global_feat_dim = self.obs_builder.get_global_feat_dim()


    def _set_seed(self, seed: Optional[int]):
        """Sets the random seed used by the environment."""
        self.seed = seed
        rng = torch.manual_seed(self.seed)
        self.rng = rng


    def observe(self, agent_name:AgentID)-> TensorDict:
        """
        Args:
            agent_name (str): current AgentID str identification

        Returns
            observations (TensorDict): current agent observaions and masks dict.
        """
        raise NotImplementedError()


    def sample_action(self, agent_name:AgentID)-> TensorDict:
        """
        Samples next action for the active agent.

        Args:
            agent_name (str): current agent str identification

        Returns
            action (TensorDict): New action for the current agent
        """
        raise NotImplementedError()

    def reset(self) -> TensorDict:
        """
        Resets the environment to a starting state and returns infos dict.

        Returns
            TensorDict: environment information TensorDict.
        """
        raise NotImplementedError()


    def step(self, action:ActionType)-> TensorDict:
        """Environment step.
        Performs an environment step for active agent. 
        Returns observation, reward, done, info TensorDict for the agent.

        Parameters
        ----------
        Args:
            action (int): action to execute.

         Returns
            TensorDict: observations , step reward, done, environment information.

        """
        raise NotImplementedError()
