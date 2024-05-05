import torch
from tensordict import TensorDict

from maenvs4vrp.core.env_observation_builder import ObservationBuilder
from maenvs4vrp.core.env import AECEnv

from typing import Optional, Dict


class Observations(ObservationBuilder):
    """Observations class

    Every featute on POSSIBLE_NODES_STATIC_FEATURES, POSSIBLE_NODES_DYNAMIC_FEATURES, POSSIBLE_SELF_FEATURES
    POSSIBLE_OTHER_AGENTS_FEATURES and POSSIBLE_GLOBAL_FEATURES list have to have their corresponding method.

    Ex: 
    'x_coordinate' -> get_feat_x_coordinate
    'time2open_div_end_time' -> get_feat_time2open_div_end_time
    
    """

    POSSIBLE_NODES_STATIC_FEATURES = ['x_coordinate', 'y_coordinate', 'tw_low',
                                    'tw_high', 'profits', 'service_time', 'tw_high_minus_tw_low_div_max_dur',
                                    'x_coordinate_min_max', 'y_coordinate_min_max', 'is_depot']

    POSSIBLE_NODES_DYNAMIC_FEATURES = ['time2open_div_end_time', 'time2close_div_end_time', 'arrive2node_div_end_time',
                                'time2open_after_step_div_end_time', 'time2close_after_step_div_end_time',
                                'time2end_after_step_div_end_time', 'fract_time_after_step_div_end_time']

    POSSIBLE_AGENT_FEATURES = ['x_coordinate', 'y_coordinate','x_coordinate_min_max', 'y_coordinate_min_max', 'frac_current_time', 
                                'frac_current_profit', 'arrivedepot_div_end_time', 'frac_feasible_nodes']

    POSSIBLE_OTHER_AGENTS_FEATURES = ['x_coordinate_min_max', 'y_coordinate_min_max', 'frac_current_time', 
                                'frac_current_profit', 'dist2agent_div_end_time', 'frac_feasible_nodes']

    POSSIBLE_GLOBAL_FEATURES = ['frac_profits', 'frac_colect_profits',
                                'frac_done_agents']


    def __init__(self, feature_list:Dict = None):
        super().__init__()
        """
        Args:
            feature_list (Dict): dictionary containing observation features list to be available to the agent;

        """

        self.default_feature_list = {'nodes_static': {'x_coordinate_min_max': {'feat': 'x_coordinate_min_max','norm': None},
                                                      'y_coordinate_min_max': {'feat': 'y_coordinate_min_max', 'norm': None}},
                                'nodes_dynamic': ['time2open_div_end_time',
                                                'time2close_div_end_time',
                                                'arrive2node_div_end_time'],
                                'agent': ['x_coordinate_min_max',
                                        'y_coordinate_min_max',
                                        'frac_current_time'],
                                'other_agents': ['frac_current_time',
                                                'frac_feasible_nodes'],
                                'global': ['frac_profits',
                                           'frac_colect_profits',
                                           'frac_done_agents']}

        if feature_list is None:
            feature_list = self.default_feature_list

        self.feature_list = feature_list
        self.possible_nodes_static_features = self.POSSIBLE_NODES_STATIC_FEATURES
        self.possible_nodes_dynamic_features = self.POSSIBLE_NODES_DYNAMIC_FEATURES
        self.possible_agent_features = self.POSSIBLE_AGENT_FEATURES
        self.possible_agents_features = self.POSSIBLE_OTHER_AGENTS_FEATURES
        self.possible_global_features = self.POSSIBLE_GLOBAL_FEATURES

    def set_env(self, env:AECEnv):        
        super().set_env(env)

    
    def get_nodes_static_feat_dim(self):
        """
        Returns:
            int: nodes static features dimentions.
        """
        return sum([self.feature_list.get('nodes_static', []).get(f).get('dim', 1) \
                    for f in self.feature_list.get('nodes_static')])

    def get_nodes_dynamic_feat_dim(self):
        """
        Returns:
            int: nodes dynamic features dimentions.
        """
        return len(self.feature_list.get('nodes_dynamic', []))

    def get_nodes_feat_dim(self):
        """
        Returns:
            int: nodes static + dynamic features dimentions.
        """
        return self.get_nodes_static_feat_dim()+self.get_nodes_dynamic_feat_dim()

    def get_agent_feat_dim(self):
        """
        Returns:
            int: current active agent features dimentions.
        """
        return len(self.feature_list.get('agent', []))

    def get_other_agents_feat_dim(self):
        """
        Returns:
            int: all agents features dimentions.
        """
        return len(self.feature_list.get('other_agents', []))

    def get_global_feat_dim(self):
        """
        Returns:
            int: global state features dimentions.
        """
        return len(self.feature_list.get('global', []))


    ## static features
    def get_feat_x_coordinate(self):
        """ static feature
        Args:

        Returns: 
            npt.NDArray: x coordinates of instance nodes.
        """
        return self.env.td_state["coords"][:, :, 0]

    def get_feat_y_coordinate(self):
        """ static feature
        Args:

        Returns: 
            npt.NDArray: y coordinates of instance nodes.
        """
        return self.env.td_state["coords"][:, :, 1]

    def get_feat_x_coordinate_min_max(self):
        """ static feature
        Args:

        Returns: 
            npt.NDArray: min-max normalized x coordinates of instance nodes.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        feat = ncoord[:,:, 0]
        return feat

    def get_feat_y_coordinate_min_max(self):
        """ static feature
        Args:

        Returns:
            npt.NDArray: min-max normalized y coordinates of instance nodes.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        feat = ncoord[:, :, 1]
        return feat

    def get_feat_tw_low(self):
        """ static feature
        Args:

        Returns: 
            npt.NDArray: nodes' time window starting times.
        """
        return self.env.td_state['tw_low']

    def get_feat_tw_high(self):
        """ static feature
        Args:

        Returns: 
            npt.NDArray: nodes' time window end times.
        """
        return self.env.td_state['tw_high']

    def get_feat_profits(self):
        """ static feature
        Args:

        Returns: 
            npt.NDArray: nodes' profits.
        """
        return self.env.td_state['profits']

    def get_feat_service_time(self):
        """ static feature
        Args:

        Returns: 
            npt.NDArray: nodes' service time.
        """
        return self.env.td_state['service_time']

    def get_feat_tw_high_minus_tw_low_div_max_dur(self):
        """ static feature
        Args:

        Returns: 
            npt.NDArray: nodes' time window amplitude divided by max tour duration.
        """
        tw_high = self.get_feat_tw_high()
        tw_low = self.get_feat_tw_low()
        return (tw_high-tw_low) / self.env.td_state['max_tour_duration']

    def get_feat_is_depot(self):
        """ static feature
        Args:

        Returns: 
            npt.NDArray: is depot bool
        """
        return self.env.td_state['is_depot']
        

    ## dynamic features
    def get_feat_time2open_div_end_time(self):
        """ dynamic feature
        Args:
            agent (Agent): Aent object.

        Returns: 
            npt.NDArray: nodes' time to open divided by end time.
        """
        feat = (self.env.td_state['tw_low'] - self.env.td_state['cur_agent']['cur_time']) / self.env.td_state['end_time'].unsqueeze(dim=-1)
        return feat

    def get_feat_time2close_div_end_time(self):
        """ dynamic feature
        Args:

        Returns: 
            npt.NDArray: nodes' time to close divided by end time.
        """
        feat = (self.env.td_state['tw_high'] - self.env.td_state['cur_agent']['cur_time']) / self.env.td_state['end_time'].unsqueeze(dim=-1)
        return feat

    def get_feat_arrive2node_div_end_time(self):
        """ dynamic feature
        Args:

        Returns: 
            npt.NDArray: agent arrive time to nodes divided by end time.
        """
        loc = self.env.td_state['coords'].gather(1, self.env.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2))
        ptime = self.env.td_state['cur_agent']['cur_time'].clone()
        time2j = torch.pairwise_distance(loc, self.env.td_state["coords"], eps=0, keepdim = False)
        arrivej = ptime + time2j
        return arrivej / self.env.td_state['end_time'].unsqueeze(dim=-1)
    
    def get_feat_time2open_after_step_div_end_time(self):
        """ dynamic feature
        Args:

        Returns: 
            npt.NDArray: nodes' time to open, after agent step, divided by end time.
        """

        arrivej = self.get_feat_arrive2node_div_end_time() * self.env.td_state['end_time'].unsqueeze(dim=-1)
        feat = (self.env.td_state['tw_low'] - arrivej) 
        return feat / self.env.td_state['end_time'].unsqueeze(dim=-1)
    
    def get_feat_time2close_after_step_div_end_time(self):
        """ dynamic feature
        Args:

        Returns: 
            npt.NDArray: nodes' time to close, after agent step, divided by end time.
        """
        arrivej = self.get_feat_arrive2node_div_end_time() * self.env.td_state['end_time'].unsqueeze(dim=-1)
        feat = (self.env.td_state['tw_high'] - arrivej) 
        return feat / self.env.td_state['end_time'].unsqueeze(dim=-1)

    def get_feat_time2end_after_step_div_end_time(self):
        """ dynamic feature
        Args:

        Returns: 
            npt.NDArray: time end, after agent step to node, divided by end time.
        """
        arrivej = self.get_feat_arrive2node_div_end_time() * self.env.td_state['end_time'].unsqueeze(dim=-1)
        feat = (self.env.td_state['end_time'].unsqueeze(dim=-1) - arrivej)
        return feat / self.env.td_state['end_time'].unsqueeze(dim=-1)
       
    def get_feat_fract_time_after_step_div_end_time(self):
        """ dynamic feature
        Args:

        Returns: 
            npt.NDArray: fraction of time left, after agent step to node.
        """
        arrivej = self.get_feat_arrive2node_div_end_time() * self.env.td_state['end_time'].unsqueeze(dim=-1)
        feat = (arrivej - self.env.td_state['start_time'].unsqueeze(dim=-1))
        return feat / self.env.td_state['end_time'].unsqueeze(dim=-1)
    

    ## Agent features
    def get_feat_agent_x_coordinate(self):
        """ active agent feature
        Args:

        Returns: 
            int: agent current x location.
        """
        loc = self.env.td_state["coords"].gather(1, self.env.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 0]
        return feat

    def get_feat_agent_y_coordinate(self):
        """ active agent feature
        Args:

        Returns 
            int: agent current y location.
        """
        loc = self.env.td_state["coords"].gather(1, self.env.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 1]
        return feat
    
    def get_feat_agent_x_coordinate_min_max(self):
        """ active agent feature
        Args:

        Returns: 
            int: agent current min-max normalized x location.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        loc = ncoord.gather(1, self.env.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 0]
        return feat

    def get_feat_agent_y_coordinate_min_max(self):
        """ active agent feature
        Args:

        Returns 
            int: agent current min-max normalized y location.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        loc = ncoord.gather(1, self.env.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 1]
        return feat

    def get_feat_agent_frac_current_time(self):
        """ active agent feature
        Args:

        Returns: 
            int: agent fraction of time elapsed.
        """
        feat =  (self.env.td_state['cur_agent']['cur_time'] - self.env.td_state['start_time'].unsqueeze(1)) 
        return feat / self.env.td_state['max_tour_duration'].unsqueeze(1)

    def get_feat_agent_frac_current_profit(self):
        """ active agent feature
        Args:

        Returns: 
            int: agent fraction of cum profit.
        """
        feat =  self.env.td_state['cur_agent']['cum_profit']  / self.env.td_state['profits'].sum(dim=-1).unsqueeze(1)
        return feat

    def get_feat_agent_arrivedepot_div_end_time(self):
        """ active agent feature
        Args:

        Returns: 
            int: agent time to depot divided by end time.
        """
        loc = self.env.td_state['coords'].gather(1, self.env.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2))
        ptime = self.env.td_state['cur_agent']['cur_time'].clone()
        time2depot = torch.pairwise_distance(loc, self.env.td_state['depot_loc'], eps=0, keepdim = False)
        arrivej = ptime + time2depot

        feat = (arrivej - self.env.td_state['start_time'].unsqueeze(1))

        return feat / self.env.td_state['end_time'].unsqueeze(1)

    def get_feat_agent_frac_feasible_nodes(self):
        """ active agent feature
        Args:

        Returns: 
            int: fraction of feasible nodes, in order to the total number of instance nodes.
        """
        feat = self.env.td_state['cur_agent']['action_mask'].sum(dim=1).unsqueeze(1)
        return feat / self.env.num_nodes

    

    ## Other agents features

    def get_feat_agents_x_coordinate_min_max(self):
        """ active agent feature
        Args:

        Returns: 
            int: agent current min-max normalized x location.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        loc = ncoord.gather(1, self.env.td_state['agents']['cur_node'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 0]
        return feat
    
    def get_feat_agents_y_coordinate_min_max(self):
        """ active agent feature
        Args:

        Returns: 
            int: agent current min-max normalized y location.
        """
        ncoord = self._min_max_normalization2d(self.env.td_state["coords"])
        loc = ncoord.gather(1, self.env.td_state['agents']['cur_node'][:,:,None].expand(-1, -1, 2))
        feat = loc[:, :, 1]
        return feat
    
    def get_feat_agents_frac_current_time(self):
        """ agents features
        Args:

        Returns: 
            npt.NDArray: agents fraction of elapsed time.
        """
        feats = self.env.td_state['agents']['cur_time'] / self.env.td_state['end_time'].unsqueeze(dim=-1)
        return feats
    
    def get_feat_agents_frac_current_profit(self):
        """ agents features
        Args:

        Returns: 
            npt.NDArray: agents fraction of cum profit
        """
        feats = self.env.td_state['agents']['cum_profit'] / self.env.td_state['profits'].sum(dim=-1).unsqueeze(1)
        return feats


    def get_feat_agents_frac_feasible_nodes(self):
        """ agents features
        Args:

        Returns: 
            npt.NDArray: fraction of feasible nodes, in order to the total number of instance nodes.
        """
        feat = self.env.td_state['agents']['feasible_nodes'].sum(dim=-1)
        return feat / self.env.num_nodes
    
    def get_feat_agents_dist2agent_div_end_time(self):
        """ agents features
        Args:

        Returns: 
            npt.NDArray: agents distance to active agent divided by end time.
        """
        locs = self.env.td_state["coords"].gather(1, self.env.td_state['agents']['cur_node'][:,:,None].expand(-1, -1, 2))
        loc = self.env.td_state['coords'].gather(1, self.env.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2))
        
        feat = torch.pairwise_distance(loc, locs, eps=0, keepdim = False)
        return feat  / self.env.td_state['end_time'].unsqueeze(dim=-1)
    
    ## Global features

    def get_feat_global_frac_done_agents(self):
        """global features

        Args:

        Returns: 
            int: fraction of done agents.
        """
        feat = self.env.td_state['agents']['active_agents_mask'].sum(dim=1).unsqueeze(1)
        return 1 - (feat / self.env.num_agents)

    def get_feat_global_frac_profits(self):
        """global features

        Args:

        Returns: 
            int: fraction of remaining profits
        """
        feat = self.env.td_state['nodes']['cur_profits'].sum(dim=-1).unsqueeze(1)
        return feat / self.env.td_state['profits'].sum(dim=-1).unsqueeze(1)

    def get_feat_global_frac_colect_profits(self):
        """global features

        Args:

        Returns: 
            int: fraction of fleet colect profits
        """
        feat = self.env.td_state['agents']['cum_profit'].sum(dim=-1).unsqueeze(1)
        return feat /  self.env.td_state['profits'].sum(dim=-1).unsqueeze(1)
    
    # --------------------------------------------------------------------------------------
    def compute_static_features(self):
        """
        Args:
            agent (Agent): Agent object.

        Returns:
            npt.NDArray: observed nodes static features array
        """
        features_static = self.feature_list.get('nodes_static')
        features_static_set = set([features_static.get(f).get('feat') for f in features_static])
        undefined_feat = features_static_set-set(self.possible_nodes_static_features)
        assert_msg = f'{undefined_feat} are not defined, choose from {str(self.possible_nodes_static_features)}'
        assert len(undefined_feat)==0, assert_msg

        features = list()
        for f in features_static:
            f_feat = features_static.get(f).get('feat')
            dim = features_static.get(f).get('dim')
            if dim:
                feature = eval(f'self.get_feat_{f_feat}')(dim)
            else:
                feature = eval(f'self.get_feat_{f_feat}')()
            f_norm = features_static.get(f).get('norm')
            norm_feature = self._normalize_feature(feature, f_norm)
            features.append(norm_feature)
        return self._concat_features(features)

    def compute_dynamic_features(self):
        """
        Args:
            agent (Agent): Agent object.

        Returns:
            npt.NDArray: observed nodes dynamic features array
        """
        features_dynamic = self.feature_list.get('nodes_dynamic')
        undefined_feat = set(features_dynamic)-set(self.possible_nodes_dynamic_features)
        assert_msg = f'{undefined_feat} are not defined, choose from {str(self.possible_nodes_dynamic_features)}'
        assert len(undefined_feat)==0, assert_msg
        features = list()
        for f in features_dynamic:
            features.append(eval(f'self.get_feat_{f}')())
        return self._concat_features(features)

    def compute_agent_features(self):
        """
        Args:
            agent (Agent): Agent object.

        Returns:
            npt.NDArray: observed current active agent features array
        """
        features_self = self.feature_list.get('agent')
        undefined_feat = set(features_self)-set(self.possible_agent_features)
        assert_msg = f'{undefined_feat} are not defined, choose from {str(self.possible_agent_features)}'
        assert len(undefined_feat)==0, assert_msg
        features = list()
        for f in features_self:
            features.append(eval(f'self.get_feat_agent_{f}')())
        return self._concat_features(features).squeeze(1)
    
    def compute_agents_features(self):
        """
        Args:
            agent (Agent): Agent object.

        Returns:
            npt.NDArray: observed agents features array 
        """
        features_agents = self.feature_list.get('other_agents')
        undefined_feat = set(features_agents)-set(self.possible_agents_features)
        assert_msg = f'{undefined_feat} are not defined, choose from {str(self.possible_agents_features)}'
        assert len(undefined_feat)==0, assert_msg
        features = list()
        for f in features_agents:
            features.append(eval(f'self.get_feat_agents_{f}')())
        return self._concat_features(features)

    def compute_global_features(self):
        """
        Args:
            agent (Agent): Agent object.

        Returns:
            npt.NDArray: global state array 
        """
        features_global = self.feature_list.get('global')
        undefined_feat = set(features_global)-set(self.possible_global_features)
        assert_msg = f'{undefined_feat} are not defined, choose from {str(self.possible_global_features)}'
        assert len(undefined_feat)==0, assert_msg
        features = list()
        for f in features_global:
            features.append(eval(f'self.get_feat_global_{f}')())
        return self._concat_features(features).squeeze(dim=1)
    
    
    def get_observations(self, is_reset=False)-> TensorDict:
        """Called whenever an environment observation has to be computed for the active agent
        Args:

        Returns:
            Dict: dictionary containing agent observations 
        """
        observations = TensorDict({}, batch_size=self.env.batch_size, device=self.env.device)
        if is_reset:
            static_feat = self.compute_static_features()
            mask_static_feat = self.env.td_state['cur_agent']['action_mask'].unsqueeze(dim=-1) * static_feat
            observations['node_static_obs'] =  mask_static_feat   

        if self.feature_list.get('nodes_dynamic'):
            dynamic_feat = self.compute_dynamic_features()
            mask_dynamic_feat = self.env.td_state['cur_agent']['action_mask'].unsqueeze(dim=-1) * dynamic_feat
            observations['node_dynamic_obs'] =  mask_dynamic_feat   
          
        if self.feature_list.get('agent'):
            agent_feat = self.compute_agent_features()
            observations['agent_obs'] = agent_feat

        if self.feature_list.get('other_agents'):
            agents_feat = self.compute_agents_features()
            mask_agents_feat = self.env.td_state['agents']['active_agents_mask'].unsqueeze(dim=-1) * agents_feat
            observations['other_agents_obs'] = mask_agents_feat

        if self.feature_list.get('global'):
            global_feat = self.compute_global_features()
            observations['global_obs'] = global_feat
            
        return observations

    @staticmethod
    def _concat_features(features):
        return torch.cat(\
                [f.unsqueeze(dim=-1) if f.dim()==2 else f for f in features],
                              dim=-1)

    def _normalize_feature(self, x, norm):
        if norm == 'min_max':
            return self._min_max_normalization(x)
        elif norm == 'standardize':
            return self._standardize(x)
        elif norm == None:
            return x

    @staticmethod
    def _min_max_normalization(x):
        max_x = torch.max(x, dim=1, keepdim=True)[0]
        min_x = torch.min(x, dim=1, keepdim=True)[0]
        return (x - min_x) / (max_x - min_x)

    @staticmethod
    def _min_max_normalization2d(x):
        max_x = torch.max(x)
        min_x = torch.min(x)
        return (x - min_x) / (max_x - min_x)

    @staticmethod
    def _standardize(x):
        means = x.mean(dim=1, keepdim=True)
        stds = x.std(dim=1, keepdim=True)
        return (x - means) / stds
