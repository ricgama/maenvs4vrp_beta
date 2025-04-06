import torch
from tensordict import TensorDict

from typing import Optional

from maenvs4vrp.core.env_generator_builder import InstanceBuilder
from maenvs4vrp.core.env_observation_builder import ObservationBuilder
from maenvs4vrp.core.env_agent_selector import BaseSelector
from maenvs4vrp.core.env_agent_reward import RewardFn
from maenvs4vrp.core.env import AECEnv

from maenvs4vrp.utils.utils import gather_by_index

class Environment(AECEnv):
    """
    MTVRP environment generator class.
    """
    def __init__(
        self,
        instance_generator_object: InstanceBuilder,
        obs_builder_object: ObservationBuilder,
        agent_selector_object: BaseSelector,
        reward_evaluator: RewardFn,
        seed=None,
        device: Optional[str] = None,
        batch_size: torch.Size = None
    ):

        """
        Constructor.

        Args:
            instance_generator_object(InstanceBuilder): Generator instance.
            obs_builder_object(ObservationBuilder): Observations instance.
            agent_selector_object(BaseSelector): Agent selector instance
            reward_evaluator(RewardFn): Reward evaluator instance.
            seed(int): Random number generator seed. Defaults to None.
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to None.
            batch_size(torch.Size): Batch size. Defaults to None.
        """
        self.version = 'v0'
        self.env_name = 'mtvrp'

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

        self.td_state = TensorDict({}, batch_size=self.batch_size, device=self.device) #Environment TensorDict

    def observe(self, is_reset=False)-> TensorDict:
        """
        Retrieve agent environment observations.

        Args:
            is_reset(bool): If the environment is on reset. Defauts to False.

        Returns
            td_observations(TensorDict): Current agent observaions and masks dictionary.
        """

        self._update_feasibility()
        td_observations = self.obs_builder.get_observations(is_reset=is_reset)
        td_observations['action_mask'] = self.td_state['cur_agent']['action_mask'].clone()
        td_observations['agents_mask'] = self.td_state['agents']['active_agents_mask'].clone()
        return td_observations
    
    def sample_action(
        self,
        td: TensorDict
    ) -> TensorDict:
        
        """
        Compute a random action from avaliable actions to current agent.
        
        Args:
            td(TensorDict): Tensor environment instance.

        Returns:
            td(TensorDict): Tensor environment instance with updated action.
        """
        
        action = torch.multinomial(self.td_state['cur_agent']["action_mask"].float(), 1).to(self.device)
        td['action'] = action
        return td

    def reset(
        self,
        num_agents: int = 2,
        num_nodes: int = 15,
        min_coords: float = 0.0,
        max_coords: float = 3.0,
        capacity: int = 50,
        service_times: float = 0.2,
        min_demands: int = 0,
        max_demands: int = 10,
        min_backhaul: int = 0,
        max_backhaul: int = 10,
        max_tw_depot: float = 12,
        backhaul_ratio: float = 0.2,
        backhaul_class: int = 1,
        sample_backhaul_class: bool = False,
        distance_limit: float = 2.8,
        speed: float = 1.0,
        subsample: bool = True,
        variant_preset: str = None,
        use_combinations: bool = False,
        force_visit: bool = True,
        batch_size: Optional[torch.Size] = None,
        n_augment: Optional[int] = 2,
        sample_type: str = 'random',
        seed: int = None,
        device: Optional[str] = "cpu"
    ):
        
        """
        Reset the environment and load agent information into dictionary.

        Args:
            num_agents(int): Total number of agents. Defaults to None.
            num_nodes(int): Total number of nodes. Defaults to None.
            min_coords(float): Minimum coordinates of nodes. Defaults to 0.0.
            max_coords(float): Maximum coordinates of nodes. Defaults to 3.0.
            capacity(int): Total capacity for each agent. Defaults to None.
            service_times(float): Service time in the nodes. Defaults to None.
            min_demands(int): Minimum linehaul demands of nodes. Defaults to 0.
            max_demands(int): Maximum linehaul demands of nodes. Defualts to 10.
            min_backhaul(int): Minimum backhaul demands of nodes. Defaults to 0.
            max_backhaul(int): Maximum backhaul demands of nodes. Defaults to 10.
            max_tw_depot(float): High time window of depot. Defaults to 12.0.
            backhaul_ratio(float): Ratio of backhaul demands. Defaults to 0.2.
            backhaul_class(int): If problem is mixed. 1 for unmixed and 2 for mixed. Defaults to 1.
            sample_backhaul_class(bool): If true, problem is randomly mixed in each batch. Defaults to False.
            distance_limit(float): Route distance limit. Defaults to 2.8.
            speed(float): Agents' speed. Defaults to 1.0.
            subsample(bool): If True, subsamples variant. Defaults to True.
            variant_preset(str): Selects variant preset. It must be used with subsample=True. Defaults to None.
            use_combinations(bool): If True, inverts parameters used on table. Otherwise, it uses them as presented. Defaults to False.
            force_visit(bool): If True, it won't allow the agent to go back to depot if there are other feasible nodes. Defaults to True. 
            batch_size(torch.Size, optional): Batch size. Defaults to None.
            n_augment(int, optional): Data augmentation. Defaults to None.
            sample_type(str): Sample type. It can be "random", "augment" or "saved". Defaults to "random".
            seed(int): Random number generator seed. Defaults to None. 
            device(str, optional): Type of processing. It can be "cpu" or "gpu". Defaults to "cpu".

        Returns:
            TensorDict: Environment information dictionary.
        """
        
        if seed is not None:
            self._set_seed(seed)

        if batch_size is None:
            batch_size = self.batch_size 
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)
            self.inst_generator.batch_size = torch.Size(batch_size)

        if force_visit is not None:
            self.force_visit = force_visit

        instance_info = self.inst_generator.sample_instance(
            num_agents = num_agents,
            num_nodes = num_nodes,
            min_coords = min_coords,
            max_coords = max_coords,
            capacity = capacity,
            service_times = service_times,
            min_demands = min_demands,
            max_demands = max_demands,
            min_backhaul = min_backhaul,
            max_backhaul = max_backhaul,
            max_tw_depot = max_tw_depot,
            backhaul_ratio = backhaul_ratio,
            backhaul_class = backhaul_class,
            sample_backhaul_class = sample_backhaul_class,
            distance_limit = distance_limit,
            speed = speed,
            subsample = subsample,
            variant_preset = variant_preset,
            use_combinations = use_combinations,
            force_visit=force_visit,
            sample_type=sample_type,
            n_augment=n_augment,
            batch_size = batch_size,
            seed = seed,
            device = device
        )

        self.num_nodes = instance_info['num_nodes']
        self.num_agents = instance_info['num_agents']

        if 'n_digits' in instance_info:
            self.n_digits = instance_info['n_digits'] 
        else:
            self.n_digits = None

        self.td_state = instance_info['data'] #Data from instance goes into env td_state

        self.td_state['done'] = torch.zeros(*batch_size, dtype=torch.bool)
        self.td_state['is_last_step'] = torch.zeros(*batch_size, dtype=torch.bool)
        self.td_state['depot_loc'] = self.td_state['coords'].gather(1, self.td_state['depot_idx'][:,:,None].expand(-1, -1, 2))

        self.td_state['max_tour_duration'] =  self.td_state['end_time'] - self.td_state['start_time']

        distance2depot = torch.pairwise_distance(self.td_state['depot_loc'], 
                                                              self.td_state['coords'], eps=0, keepdim = False)

        self.td_state['speed'] = torch.full((*self.batch_size, 1), speed)

        time2depot = distance2depot / self.td_state['speed']

        if self.n_digits is not None:
            distance2depot = torch.floor(self.n_digits * time2depot) / self.n_digits
            time2depot = torch.floor(self.n_digits * time2depot) / self.n_digits

        self.td_state['nodes'] = TensorDict(
                                    source={'linehaul_demands': self.td_state['linehaul_demands'].clone(),
                                            'backhaul_demands': self.td_state['backhaul_demands'].clone(),
                                            'distance2depot': distance2depot,
                                            'time2depot': time2depot,
                                            'active_nodes_mask': torch.ones((*batch_size, self.num_nodes),dtype=torch.bool, device=self.device)},
                                    batch_size=batch_size, device=self.device)
        
        self.td_state['agents'] =  TensorDict(
                                    source={'capacity': self.td_state['capacity'],
                                            'cur_time': self.td_state['start_time'].unsqueeze(1).clone() * torch.ones((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cur_node': self.td_state['depot_idx'] * torch.ones((*batch_size, self.num_agents), dtype = torch.int64, device=self.device),
                                            'cur_ttime': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'cum_ttime': torch.zeros((*batch_size, self.num_agents), dtype = torch.float, device=self.device),
                                            'visited_nodes': torch.zeros((*batch_size, self.num_agents, self.num_nodes), dtype=torch.bool, device=self.device),
                                            'feasible_nodes': torch.ones((*batch_size, self.num_agents, self.num_nodes), dtype=torch.bool, device=self.device),
                                            'active_agents_mask': torch.ones((*batch_size, self.num_agents), dtype=torch.bool, device=self.device),
                                            'cur_step': torch.zeros((*batch_size, self.num_agents), dtype=torch.int32, device=self.device),
                                            'route_length': torch.zeros((*batch_size, self.num_agents), dtype=torch.float, device=self.device)},
                                    batch_size=batch_size, device=self.device)
        
        cur_agent_idx = torch.zeros((*batch_size, 1), dtype = torch.int64, device=self.device)
        self.td_state['cur_agent_idx'] = cur_agent_idx

        self.td_state['cur_agent'] = TensorDict({
                                'action_mask': self.td_state['agents']['feasible_nodes'].gather(1, cur_agent_idx[:,:,None].expand(-1, -1, self.num_nodes)).squeeze(1),
                                'cur_time': self.td_state['agents']['cur_time'].gather(1, cur_agent_idx).clone(),
                                'cur_node': self.td_state['agents']['cur_node'].gather(1, cur_agent_idx).clone(),
                                'cur_ttime': self.td_state['agents']['cur_ttime'].gather(1, cur_agent_idx).clone(),
                                'cum_ttime': self.td_state['agents']['cum_ttime'].gather(1, cur_agent_idx).clone(),
                                'cur_route_length': self.td_state['agents']['route_length'].gather(1, cur_agent_idx).clone(),
                                'cur_step': self.td_state['agents']['cur_step'].gather(1, cur_agent_idx).clone(),
                                'used_capacity_linehaul': torch.zeros((*batch_size, 1), device=self.device),
                                'used_capacity_backhaul': torch.zeros((*batch_size, 1), device=self.device)
                                }, batch_size=batch_size)

        self.td_state['backhaul_class'] = torch.full((*batch_size, 1), backhaul_class, device=self.device)

        self.td_state['solution'] = TensorDict({}, batch_size=batch_size)

        self.agent_selector.set_env(self)
        self.obs_builder.set_env(self)
        self.reward_evaluator.set_env(self)

        #Set environment do agent selector, reward e observations
        agent_step = self.td_state['cur_agent']['cur_step']
        done = self.td_state['done'].clone()
        reward = torch.zeros_like(done, dtype = torch.float, device=self.device)
        penalty = torch.zeros_like(done, dtype = torch.float, device=self.device)

        #td observations
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

        """
        Update actions feasibility.
        
        Args:
            n/a.

        Returns:
            None.
        """

        active_nodes = self.td_state['nodes']['active_nodes_mask'].clone() #Active nodes. Agent can only visit node if it's active
        loc = self.td_state['coords'].gather(1, self.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2)) #Current agent location
        ptime = self.td_state['cur_agent']['cur_time'].clone() #Agent current time

        distance2j = torch.pairwise_distance(loc, self.td_state["coords"], eps=0, keepdim = False) #Distance between current agent and nodes
        if self.n_digits is not None:
            distance2j = torch.floor(self.n_digits * distance2j) / self.n_digits

        time2depot = self.td_state['nodes']['time2depot'].clone() #Time from nodes to depot
        distance2depot = self.td_state['nodes']['distance2depot'].clone()
        time2arrive = distance2j / self.td_state['speed']
        arrival_time = ptime + time2arrive #Arrival time. Current time + time 2 arrive (distance / speed)

        #Constraint 1. Can arrive to node in time.
        c1 = arrival_time <= self.td_state['tw_high']
        #Constraint 2. If problem is closed, if agent can arrive to depot in time.
        c2 = (torch.max(arrival_time, self.td_state['tw_low']) + self.td_state['service_times'].squeeze(-1) + time2depot) * ~self.td_state['open_routes'] < self.td_state['end_time']
        #Constraint 3. Does agent exceed distance limit.
        c3 = self.td_state['cur_agent']['cur_route_length'] + distance2j + (distance2depot * ~self.td_state['open_routes']) <= self.td_state['distance_limits']

        #Demands constraints

        #Capacity

        exceeds_cap_linehaul = self.td_state['linehaul_demands'] + self.td_state['cur_agent']['used_capacity_linehaul'] > self.td_state['agents']['capacity']
        exceeds_cap_backhaul = self.td_state['backhaul_demands'] + self.td_state['cur_agent']['used_capacity_backhaul'] > self.td_state['agents']['capacity']

        '''
        Backhaul class 1. Node either linehaul or backhaul. Linehauls before backhauls.
        '''

        linehaul_missing = ((self.td_state['linehaul_demands'] * active_nodes).sum(-1) > 0).unsqueeze(-1)
        is_carrying_backhaul = gather_by_index(src=self.td_state['backhaul_demands'], idx=self.td_state['cur_agent']['cur_node'], dim=1, squeeze=False) > 0
        meets_demand_constraint_backhaul_1 = (linehaul_missing & ~exceeds_cap_linehaul & ~is_carrying_backhaul & (self.td_state['linehaul_demands'] > 0)) | (~exceeds_cap_backhaul & (self.td_state['backhaul_demands'] > 0))

        '''
        Backhaul class 2. Mixed linehauls and backhauls
        '''

        cannot_serve_linehaul = self.td_state['linehaul_demands'] > self.td_state['capacity'] - self.td_state['cur_agent']['used_capacity_linehaul']
        meets_demand_constraint_backhaul_2 = ~exceeds_cap_linehaul & ~exceeds_cap_backhaul & ~cannot_serve_linehaul

        #Demand constraints according to backhaul class

        meet_demand_constraints = ((self.td_state['backhaul_class'] == 1) & meets_demand_constraint_backhaul_1) | ((self.td_state['backhaul_class'] == 2) & meets_demand_constraint_backhaul_2)
        can_visit = active_nodes & c1 & c2 & c3 & meet_demand_constraints
        self.can_visit = can_visit

        if self.force_visit:
            can_visit[:, 0] = ~((self.td_state['cur_agent']['cur_node'] == 0).squeeze(-1) & (can_visit[:, 1:].sum(-1) > 0)) #Don't visit depot if not in there and there are still nodes i can visit
        # else:
        #  can_visit[:, 0] = True

        self.td_state['cur_agent'].update({'action_mask': can_visit})
        self.td_state['agents']['feasible_nodes'].scatter_(1, 
                                            self.td_state['cur_agent_idx'][:,:,None].expand(-1,-1,self.num_nodes), can_visit.unsqueeze(1))

        return can_visit

    def _update_done(
        self,
        action
    ):
        
        """
        Update done state.

        Args:
            action(torch.Tensor): Tensor with agent moves.

        Returns:
            None.
        """
        
        former_done = self.td_state['done'].clone()

        # update done agents
        self.td_state['agents']['active_agents_mask'].scatter_(1, self.td_state['cur_agent_idx'], 
                                                                    ~action.eq(self.td_state['depot_idx']))
        
        self.td_state['done'] = (~self.td_state['agents']['active_agents_mask']).all(dim=-1)
        self.td_state['done'][former_done] = True
        # update served nodes
        self.td_state['nodes']['active_nodes_mask'].scatter_(1, action, action.eq(self.td_state['depot_idx']))
        self.td_state['is_last_step'] = self.td_state['done'].eq(~former_done)

    def _update_state(self, action):

        """
        Update agent state.

        Args:
            action(torch.Tensor): Tensor with agent moves.

        Returns:
            None.
        """

        loc = self.td_state['coords'].gather(1, self.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2))
        next_loc = self.td_state['coords'].gather(1, action[:,:,None].expand(-1, -1, 2))

        ptime = self.td_state['cur_agent']['cur_time'].clone()
        distance2j = torch.pairwise_distance(loc, next_loc, eps=0, keepdim = False)
        time2j = distance2j / self.td_state['speed']
        if self.n_digits is not None:
            time2j = torch.floor(self.n_digits * time2j) / self.n_digits
        tw = self.td_state['tw_low'].gather(1, action)
        service_time = self.td_state['service_times'].gather(1, action)

        arrivej = ptime + time2j
        waitj = torch.clip(tw-arrivej, min=0)

        time_update = arrivej + waitj + service_time

        # update agent cur node
        is_open_and_getting_to_depot = (self.td_state['open_routes']) & (action.eq(self.td_state['depot_idx']))
        mask = ~is_open_and_getting_to_depot

        self.td_state['cur_agent']['cur_node'].masked_scatter_(mask, action[mask])

        self.td_state['agents']['cur_node'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_node'])
        # update agent cur time
        self.td_state['cur_agent']['cur_time'] = time_update

        #Current route length

        self.td_state['cur_agent']['cur_route_length'] = (self.td_state['cur_agent']['cur_node'] != 0) * (self.td_state['cur_agent']['cur_route_length'] + distance2j)
        self.td_state['agents']['route_length'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_route_length'])

        # if agent is done set agent time to end_time
        agents_done = ~self.td_state['agents']['active_agents_mask'].gather(1, self.td_state['cur_agent_idx']).clone()

        self.td_state['cur_agent']['cur_time'] = torch.where(agents_done, self.td_state['end_time'].unsqueeze(-1), 
                                                             self.td_state['cur_agent']['cur_time'])
        self.td_state['agents']['cur_time'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_time'])

        # update agent cum traveled time
        self.td_state['cur_agent']['cur_ttime'] = time2j
        self.td_state['cur_agent']['cum_ttime'] += time2j
        self.td_state['agents']['cur_ttime'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_ttime'])
        self.td_state['agents']['cum_ttime'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cum_ttime'])
        
        self.td_state['nodes']['linehaul_demands'].scatter_(1, action, torch.zeros_like(action, dtype = torch.float))
        self.td_state['nodes']['backhaul_demands'].scatter_(1, action, torch.zeros_like(action, dtype = torch.float))
        # update visited nodes
        r = torch.arange(*self.td_state.batch_size, device=self.device)
        self.td_state['agents']['visited_nodes'][r, self.td_state['cur_agent_idx'].squeeze(-1), action.squeeze(-1)] = True

        # update agent step
        self.td_state['cur_agent']['cur_step'] = torch.where(~agents_done, self.td_state['cur_agent']['cur_step']+1, 
                                                             self.td_state['cur_agent']['cur_step'])
        self.td_state['agents']['cur_step'].scatter_(1, self.td_state['cur_agent_idx'], self.td_state['cur_agent']['cur_step'])

        # if all done activate first agent to guarantee batch consistency during agent sampling
        self.td_state['agents']['active_agents_mask'][self.td_state['agents']['active_agents_mask'].sum(1).eq(0), 0] = True
        self._update_feasibility()

    def _update_cur_agent(self, cur_agent_idx):

        """
        Update current agent.

        Args:
            cur_agent_idx(torch.Tensor): Current agent id.

        Returns:
            None.
        """

        self.td_state['cur_agent_idx'] =  cur_agent_idx

        selected_demand_linehaul = gather_by_index(src=self.td_state['linehaul_demands'], idx=self.td_state['cur_agent']['cur_node'], dim=1, squeeze=False)

        selected_demand_backhaul = gather_by_index(src=self.td_state['backhaul_demands'], idx=self.td_state['cur_agent']['cur_node'], dim=1, squeeze=False)

        cur_node = self.td_state['agents']['cur_node'].gather(1, self.td_state['cur_agent_idx']).clone()
        used_capacity_linehaul = (cur_node != 0) * (self.td_state['cur_agent']['used_capacity_linehaul'] + selected_demand_linehaul)
        used_capacity_backhaul = (cur_node != 0) * (self.td_state['cur_agent']['used_capacity_backhaul'] + selected_demand_backhaul)

        self.td_state['cur_agent'] = TensorDict({
                                'action_mask': self.td_state['agents']['feasible_nodes'].gather(1, self.td_state['cur_agent_idx'][:,:,None].expand(-1, -1, self.num_nodes)).squeeze(1).clone(),
                                'cur_agent_idx': cur_agent_idx,
                                'cur_route_length': self.td_state['agents']['route_length'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_time': self.td_state['agents']['cur_time'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_node': self.td_state['agents']['cur_node'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_ttime': self.td_state['agents']['cur_ttime'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cum_ttime': self.td_state['agents']['cum_ttime'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'cur_step': self.td_state['agents']['cur_step'].gather(1, self.td_state['cur_agent_idx']).clone(),
                                'used_capacity_linehaul': used_capacity_linehaul,
                                'used_capacity_backhaul': used_capacity_backhaul
                                }, batch_size=self.td_state.batch_size, device=self.device)
        
    def _update_solution(self, action):

        """
        Update agents and actions in solution.

        Args:
            action(torch.Tensor): Tensor with agent moves.

        Returns: 
            None.
        """

        # update solution dic
        if 'actions' in self.td_state['solution'].keys():
            self.td_state['solution','actions'] = torch.concat( [self.td_state['solution','actions'], action], dim=-1)
        else:
            self.td_state['solution','actions'] = action

        if 'agents' in self.td_state['solution'].keys():
            self.td_state['solution','agents'] = torch.concat( [self.td_state['solution','agents'], self.td_state['cur_agent_idx']], dim=-1)
        else:
            self.td_state['solution','agents'] = self.td_state['cur_agent_idx']

    def step(
        self,
        td: TensorDict
    ) -> TensorDict:
        
        """
        Perform an environment step for active agent.

        Args:
            td(TensorDict): Tensor environment instance.

        Returns:
            td(TensorDict): Updated tensor environment instance.
        """
        
        action = td['action']
        assert self.td_state['cur_agent']['action_mask'].gather(1, action).all(), f"not feasible action"

        self._update_done(action)
        done = self.td_state['done'].clone()
        is_last_step = self.td_state['is_last_step'].clone()

        # update env state    
        self._update_state(action)

        # update solution dic
        self. _update_solution(action)
        
        # get reward and penalty
        reward, penalty = self.reward_evaluator.get_reward(action)

        # select and update cur agent
        cur_agent_idx =  self.agent_selector._next_agent()
        self._update_cur_agent(cur_agent_idx)
        agent_step = self.td_state['cur_agent']['cur_step']

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

    def check_solution_validity(self):

        """
        Check if solution is valid.

        Args:
            N7a.

        Returns:
            None.
        """

        distance2depot = torch.pairwise_distance(self.td_state['coords'], self.td_state['coords'][..., 0:1, :], eps=0, keepdim = False)
        a = self.td_state['tw_low'] + distance2depot + self.td_state['service_times'] #Time 2 serve node and get back to depot
        b = self.td_state['time_windows'][..., 0, 1, None] #Depot late tw

        #Can agent serve node and get back to depot?
        assert torch.all(a <= b), "Agent cannot serve node and get back to depot."

        #Actions cycle assert. Curr_node starts at 0 (depot) and iteratively keeps going onto the next. 
        curr_node = torch.zeros(*self.batch_size, dtype=torch.int64, device=self.device)
        curr_time = torch.zeros(*self.batch_size, dtype=torch.float32, device=self.device)
        curr_length = torch.zeros(*self.batch_size, dtype=torch.float32, device=self.device)
        visited_nodes = torch.zeros(*self.batch_size, self.num_nodes, dtype=torch.int64, device=self.device)

        for ii in range(self.td_state['solution']['actions'].size(1)):
            next_node = self.td_state['solution']['actions'][:, ii]
            curr_loc = gather_by_index(self.td_state['coords'], curr_node)
            next_loc = gather_by_index(self.td_state['coords'], next_node)
            dist = torch.pairwise_distance(curr_loc, next_loc)
            
            fill = visited_nodes.gather(1, next_node.unsqueeze(-1))
            visited_nodes.scatter_(1, next_node.unsqueeze(-1), fill + 1)

            curr_length = curr_length + dist * ~(self.td_state['open_routes'].squeeze(-1) & (next_node == 0)) #Update curr_length
            assert torch.all(curr_length <= self.td_state['distance_limits'].squeeze(-1)), "Route length exceeds distance limit."
            curr_length[next_node == 0] = 0.0 #Reset length for depot

            curr_time = torch.max(curr_time + dist, gather_by_index(self.td_state['time_windows'], next_node)[..., 0]) #Curr time either time to get to node or early tw
            assert torch.all(curr_time <= gather_by_index(self.td_state['time_windows'], next_node)[..., 1]), "Agent must perform service before node's time window closes."

            curr_time = curr_time + gather_by_index(self.td_state['service_times'], next_node)
            curr_node = next_node
            curr_time[next_node == 0] = 0.0

        visited_nodes_exc_depot = visited_nodes[:, 1:]
        assert(torch.all((visited_nodes_exc_depot == 0) | (visited_nodes_exc_depot == 1))), "Nodes were visited more than once!"

        demand_l = self.td_state['linehaul_demands'].gather(1, self.td_state['solution']['actions'])
        demand_b = self.td_state['backhaul_demands'].gather(1, self.td_state['solution']['actions'])

        used_cap_l = torch.zeros_like(self.td_state['linehaul_demands'][:, 0]) #Starts at 0
        used_cap_b = torch.zeros_like(self.td_state['backhaul_demands'][:, 0]) #Starts at 0

        for ii in range(self.td_state['solution']['actions'].size(1)):
            #reset at depot
            used_cap_l = used_cap_l * (self.td_state['solution']['actions'][:, ii] != 0)
            used_cap_b = used_cap_b * (self.td_state['solution']['actions'][:, ii] != 0)

            used_cap_l += demand_l[:, ii]
            used_cap_b += demand_b[:, ii]

            #Backhaul class 1 (unmixed), agents cannot supply linehaul if carrying backhaul
            assert(
                (self.td_state['backhaul_class'] == 2) |
                (used_cap_b == 0) |
                ((self.td_state['backhaul_class'] == 1) & ~(demand_l[:, ii] > 0))
            ).all(), "Cannot pickup linehaul while carrying backhaul in unmixed problems."

            #Backhaul class 2 (mixed), agents cannot supply linehaul, if backhaul load + linehaul demand in node exceeds agent's capacity

            assert(
                (self.td_state['backhaul_class'] == 1) |
                (used_cap_b == 0) |
                ((self.td_state['backhaul_class'] == 2) & (used_cap_b + demand_l[:, ii] <= self.td_state['capacity']))
            ).all(), "Cannot supply linehaul, not enough load."

            #Loads must not exceed capacity
            assert(
                used_cap_l <= self.td_state['capacity']
            ).all(), "Used more linehaul than capacity: {}/{}".format(used_cap_l, self.td_state['capacity'])

            assert(
                used_cap_b <= self.td_state['capacity']
            ).all(), "Used more backhaul than capacity: {}/{}".format(used_cap_b, self.td_state['capacity'])