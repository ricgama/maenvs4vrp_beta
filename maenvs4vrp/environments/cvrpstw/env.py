"""

# Capacitated Vehicle Routing Problem with Soft Time Windows (CVRPSTW) environment 

### Version History

* v0: Initial versions release (0.1.0)

"""

import torch
from tensordict import TensorDict

from typing import Tuple, Optional

import warnings

from maenvs4vrp.core.env_generator_builder import InstanceBuilder
from maenvs4vrp.core.env_observation_builder import ObservationBuilder
from maenvs4vrp.core.env_agent_selector import BaseSelector
from maenvs4vrp.core.env_agent_reward import RewardFn
from maenvs4vrp.core.env import AECEnv


class Environment(AECEnv):
    """
    # Capacitated Vehicle Routing Problem with Soft Time Windows (CVRPSTW) environment 

    ### Version History

    * v0: Initial versions release (0.1.0)

    Args:
        instance_generator_object: constructor instances. 
        obs_builder_object: constructor of observations. 
        agent_iterator_object: next agent generator.
        seed (int): initialise the seeds of the random number generators

    """
    def __init__(self,
                instance_generator_object: InstanceBuilder,  
                obs_builder_object: ObservationBuilder,
                agent_selector_object: BaseSelector,
                reward_evaluator: RewardFn,
                seed=None,         
                device: Optional[str] = None,
                batch_size: torch.Size = None):

        self.version = 'v0'
        self.env_name = 'cvrpstw'
        # seed the environment
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
        self.env_nsteps = 0

        if device is None:
            self.device = self.inst_generator.device
        else:
            self.device = device
            self.inst_generator.device = device

        if batch_size is None:
            self.batch_size =  self.inst_generator.batch_size
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)
            self.inst_generator.batch_size = torch.Size(batch_size)
            
        self.td_state = TensorDict({}, batch_size=self.batch_size, device=self.device)

        
    def observe(self, is_reset=False)-> TensorDict:
        """
        Args:

        Returns
            observations (TensorDict): current agent observaions and masks dict
        """

        self._update_feasibility()
        td_observations = self.obs_builder.get_observations(is_reset=is_reset)
        td_observations['action_mask'] = self.td_state['cur_agent']['action_mask'].clone()
        td_observations['agents_mask'] = self.td_state['agents']['active_agents_mask'].clone()

        return td_observations

    
    def sample_action(self, td: TensorDict)-> TensorDict:
        """Helper function to select a random action from available actions"""
        action = torch.multinomial(self.td_state['cur_agent']["action_mask"].float(), 1).to(self.device)
        td['action'] = action
        return td


    def reset(self, 
              num_agents:int|None=None, 
              num_nodes:int|None=None, 
              capacity:int|None=None, 
              service_times:float|None=None, 
              instance_name:str|None=None, 
              sample_type:str='random',
              batch_size: Optional[torch.Size] = None,
              n_augment: Optional[int] = None,
              seed:int|None=None)-> TensorDict:
        """

        Resets the environment and returns infos dict.

        Returns
            infos (dict): environment information dict.

        """

        if seed is not None:
            self._set_seed(seed)

        if batch_size is None:
            batch_size = self.batch_size 
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)
            self.inst_generator.batch_size = torch.Size(batch_size)

        instance_info = self.inst_generator.sample_instance(num_agents=num_agents, 
                                                            num_nodes=num_nodes, 
                                                            capacity=capacity, 
                                                            service_times=service_times, 
                                                            instance_name=instance_name, 
                                                            sample_type=sample_type, 
                                                            batch_size=batch_size,
                                                            n_augment=n_augment,
                                                            seed=seed)

        self.num_nodes = instance_info['num_nodes']
        self.num_agents = instance_info['num_agents']

        if 'n_digits' in instance_info:
            self.n_digits = instance_info['n_digits'] 
        else:
            self.n_digits = None

        self.td_state = instance_info['data']

        self.td_state['done'] = torch.zeros(*batch_size, dtype=torch.bool)
        self.td_state['is_last_step'] = torch.zeros(*batch_size, dtype=torch.bool)
        self.td_state['depot_loc'] = self.td_state['coords'].gather(1, self.td_state['depot_idx'][:,:,None].expand(-1, -1, 2))

        self.td_state['max_tour_duration'] =  self.td_state['end_time'] - self.td_state['start_time']

        self.td_state['Pmax'] = instance_info['Pmax'] * self.td_state['max_tour_duration'] # transform to time 
        self.td_state['Wmax'] = instance_info['Wmax'] * self.td_state['max_tour_duration'] # transform to time

        self.early_penalty = instance_info['early_penalty']
        self.late_penalty = instance_info['late_penalty' ]

        self.td_state['tw_low_limit'] = self.td_state['tw_low'] - self.td_state['Pmax']
        self.td_state['tw_high_limit'] = self.td_state['tw_high'] + self.td_state['Pmax']
        self.td_state['arrive_limit'] = self.td_state['tw_low'] - self.td_state['Pmax'] - self.td_state['Wmax']

        time2depot = torch.pairwise_distance(self.td_state['depot_loc'], 
                                                              self.td_state['coords'], eps=0, keepdim = False)
        if self.n_digits is not None:
            time2depot = torch.floor(self.n_digits * time2depot) / self.n_digits

        self.td_state['time2depot'] = time2depot

        self.td_state['nodes'] = TensorDict(
                                    source={'cur_demands': self.td_state['demands'].clone(),
                                            'active_nodes_mask': torch.ones((*batch_size, self.num_nodes),dtype=torch.bool, device=self.device)},
                                    batch_size=batch_size, device=self.device)
        self.td_state['agents'] =  TensorDict(
                                    source={'capacity': self.td_state['capacity'],
                                            'cur_load': self.td_state['capacity'].clone() * torch.ones((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cur_time': self.td_state['start_time'].unsqueeze(1).clone() * torch.ones((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cur_node': self.td_state['depot_idx'] * torch.ones((*batch_size, self.num_agents), dtype = torch.int64, device=self.device),
                                            'cur_ttime': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cum_ttime': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'visited_nodes': torch.zeros((*batch_size, self.num_agents, self.num_nodes), dtype=torch.bool, device=self.device),
                                            'feasible_nodes': torch.ones((*batch_size, self.num_agents, self.num_nodes), dtype=torch.bool, device=self.device),
                                            'active_agents_mask': torch.ones((*batch_size, self.num_agents), dtype=torch.bool, device=self.device),
                                            'agents_step': torch.zeros((*batch_size, self.num_agents), dtype=torch.int32, device=self.device)},
                                    batch_size=batch_size, device=self.device)

        self.td_state['cur_agent_idx'] = torch.zeros((*batch_size, 1), dtype = torch.int64, device=self.device)

        self.td_state['cur_agent'] = TensorDict({
                                'action_mask': self.td_state['agents']['feasible_nodes'].gather(1, self.td_state['cur_agent_idx'][:,:,None].expand(-1, -1, self.num_nodes)).squeeze(1),
                                'cur_load': self.td_state['agents']['cur_load'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_time': self.td_state['agents']['cur_time'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_node': self.td_state['agents']['cur_node'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_ttime': self.td_state['agents']['cur_ttime'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cum_ttime': self.td_state['agents']['cum_ttime'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                }, batch_size=batch_size)

        self.agent_selector.set_env(self)
        self.obs_builder.set_env(self)
        self.reward_evaluator.set_env(self)

        agent_step = self.td_state['agents']['agents_step'].gather(1, self.td_state['cur_agent_idx']).clone()
        done = self.td_state['done'].clone()
        reward = torch.zeros_like(done, dtype = torch.float, device=self.device)
        penalty = torch.zeros_like(done, dtype = torch.float, device=self.device)

        td_observations = self.observe(is_reset=True)
        self.env_nsteps = 0
        return TensorDict(
            {
                "agent_step": agent_step,
                "observations": td_observations,
                "cur_agent_idx":self.td_state['cur_agent_idx'].clone(),
                "reward": reward,
                "penalty":penalty,
                "done": done,
            },
            batch_size=batch_size, device=self.device)


    def _update_feasibility(self):
        raise NotImplementedError()
    

    def _update_done(self, action):
        former_done = self.td_state['done'].clone()

        # update done agents
        self.td_state['agents']['active_agents_mask'].scatter_(1, self.td_state['cur_agent_idx'], 
                                                                    ~action.eq(self.td_state['depot_idx']))
        
        self.td_state['done'] = (~self.td_state['agents']['active_agents_mask']).all(dim=-1)

        # update served nodes
        self.td_state['nodes']['active_nodes_mask'].scatter_(1, action, action.eq(self.td_state['depot_idx']))
        self.td_state['is_last_step'] = self.td_state['done'].eq(~former_done)


    def _update_state(self, action):
        loc = self.td_state['coords'].gather(1, self.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2))
        next_loc = self.td_state['coords'].gather(1, action[:,:,None].expand(-1, -1, 2))

        ptime = self.td_state['cur_agent']['cur_time'].clone()
        time2j = torch.pairwise_distance(loc, next_loc, eps=0, keepdim = False)
        if self.n_digits is not None:
            time2j = torch.floor(self.n_digits * time2j) / self.n_digits
        tw = self.td_state['tw_low_limit'].gather(1, action)
        service_time = self.td_state['service_time'].gather(1, action)

        arrivej = ptime + time2j
        waitj = torch.clip(tw-arrivej, min=0)

        time_update = arrivej + waitj + service_time
        # update agent cur node
        self.td_state['cur_agent']['cur_node'] = action
        self.td_state['agents']['cur_node'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_node'])
        # update agent cur time
        self.td_state['cur_agent']['cur_time'] = time_update

        # is agent is done set agent time to end_time
        agents_done = self.td_state['agents']['active_agents_mask'].gather(1, self.td_state['cur_agent_idx']).clone()
        self.td_state['cur_agent']['cur_time'] = torch.where(agents_done, 
                                                             self.td_state['cur_agent']['cur_time'], self.td_state['end_time'])
        self.td_state['agents']['cur_time'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_time'])

        # update agent cum traveled time
        self.td_state['cur_agent']['cur_ttime'] = time2j
        self.td_state['cur_agent']['cum_ttime'] += time2j
        self.td_state['agents']['cur_ttime'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_ttime'])
        self.td_state['agents']['cum_ttime'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cum_ttime'])

        # update agent load and node demands
        self.td_state['cur_agent']['cur_load'] -= self.td_state['demands'].gather(1, action)
        # is agent is done set agent cur_load to 0
        self.td_state['cur_agent']['cur_load'] = torch.where(agents_done, 
                                                             self.td_state['cur_agent']['cur_load'], 0.)
        
        self.td_state['nodes']['cur_demands'].scatter_(1, action, torch.zeros_like(action, dtype = torch.float))
        self.td_state['agents']['cur_load'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_load'])
        # update visited nodes
        r = torch.arange(*self.td_state.batch_size, device=self.device)
        self.td_state['agents']['visited_nodes'][r, self.td_state['cur_agent_idx'].squeeze(-1), action.squeeze(-1)] = True
        # update agent step
        r = torch.arange(*self.td_state.batch_size, device=self.device)
        self.td_state['agents']['agents_step'][r, self.td_state['cur_agent_idx'].squeeze(-1)] += 1

        # if all done activate first agent to guarantee batch consistency during agent sampling
        self.td_state['agents']['active_agents_mask'][self.td_state['agents']['active_agents_mask'].sum(1).eq(0), 0] = True
        
    def _update_cur_agent(self, cur_agent_idx):

        self.td_state['cur_agent_idx'] =  cur_agent_idx
        self.td_state['cur_agent'] = TensorDict({
                                'action_mask': self.td_state['agents']['feasible_nodes'].gather(1, self.td_state['cur_agent_idx'][:,:,None].expand(-1, -1, self.num_nodes)).squeeze(1).clone(),
                                'cur_load': self.td_state['agents']['cur_load'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_time': self.td_state['agents']['cur_time'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_node': self.td_state['agents']['cur_node'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_ttime': self.td_state['agents']['cur_ttime'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cum_ttime': self.td_state['agents']['cum_ttime'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                }, batch_size=self.td_state.batch_size, device=self.device)
        
        
    def step(self, td: TensorDict) -> TensorDict:
        """Environment step.
        Performs an environment step for active agent. 

        Parameters
        ----------
        Args:
            TensorDict
         Returns
            TensorDict

        """
        action = td["action"]
        assert self.td_state['cur_agent']['action_mask'].gather(1, action).all(), f"not feasible action"

        self._update_done(action)
        done = self.td_state['done'].clone()
        is_last_step = self.td_state['is_last_step'].clone()

        # get reward and penalty
        reward, penalty = self.reward_evaluator.get_reward(action)

        # update env state    
        self._update_state(action)

        # select and update cur agent
        cur_agent_idx =  self.agent_selector._next_agent()
        self._update_cur_agent(cur_agent_idx)
        agent_step = self.td_state['agents']['agents_step'].gather(1, self.td_state['cur_agent_idx']).clone()

        # new observations
        td_observations = self.observe()
        self.env_nsteps += 1
        td.update(
            {
                "agent_step": agent_step,
                "observations": td_observations,
                "reward": reward,
                "penalty":penalty,  
                "cur_agent_idx":cur_agent_idx,              
                "done": done,
                "is_last_step": is_last_step
            },
        )
        return td

