from typing import Optional, Dict, List


class ObservationBuilder:
    """ObservationBuilder base class.
    """

    POSSIBLE_NODES_STATIC_FEATURES:List[str] = []
    POSSIBLE_NODES_DYNAMIC_FEATURES:List[str] = []
    POSSIBLE_AGENT_FEATURES:List[str] = []
    POSSIBLE_OTHER_AGENTS_FEATURES:List[str] = []
    POSSIBLE_GLOBAL_FEATURES:List[str] = []

    def __init__(self, feature_list:Dict = None):
        """Constructor

        Args:
            feature_list (Dict): dictionary containing observation features list to be available to the agent;

        """
        self.env = None
        self.default_feature_list:Dict = {}

        self.possible_nodes_static_features = self.POSSIBLE_NODES_STATIC_FEATURES
        self.possible_nodes_dynamic_features = self.POSSIBLE_NODES_DYNAMIC_FEATURES
        self.possible_agent_features = self.POSSIBLE_AGENT_FEATURES
        self.possible_agents_features = self.POSSIBLE_OTHER_AGENTS_FEATURES
        self.possible_global_features = self.POSSIBLE_GLOBAL_FEATURES

    def set_env(self, env):
        self.env = env

    def get_static_feat_dim(self):
        """
        Returns:
            int: nodes static features dimentions.
        """
        raise NotImplementedError()

    def get_dynamic_feat_dim(self):
        """
        Returns:
            int: nodes dynamic features dimentions.
        """
        raise NotImplementedError()

    def get_nodes_feat_dim(self):
        """
        Returns:
            int: nodes static + dynamic features dimentions.
        """
        raise NotImplementedError()

    def get_agent_feat_dim(self):
        """
        Returns:
            int: current active agent features dimentions.
        """
        raise NotImplementedError()

    def get_other_agents_feat_dim(self):
        """
        Returns:
            int: all agents features dimentions.
        """
        raise NotImplementedError()

    def get_global_feat_dim(self):
        """
        Returns:
            int: global state features dimentions.
        """
        raise NotImplementedError()

    def compute_static_features(self):
        """
        Args:
            agent (Agent): agent object.

        Returns:
            npt.NDArray: observed nodes static features array
        """
        raise NotImplementedError()

    def compute_dynamic_features(self):
        """
        Args:
            agent (Agent): agent object.

        Returns:
            npt.NDArray: observed nodes dynamic features array
        """
        raise NotImplementedError()

    def compute_agent_features(self):
        """
        Args:
            agent (Agent): agent object.

        Returns:
            npt.NDArray: observed current active agent features array
        """
        raise NotImplementedError()

    def compute_agents_features(self):
        """
        Args:
            agent (Agent): agent object.

        Returns:
            npt.NDArray: observed agents features array 
        """
        raise NotImplementedError()

    def compute_global_features(self):
        """
        Args:
            agent (Agent): agent object.

        Returns:
            npt.NDArray: global state array 
        """
        raise NotImplementedError()

    def get_observations(self):
        """Called whenever an environment observation has to be computed for the active agent
        Args:
            agent (Agent): agent object.

        Returns:
            Dict: dictionary containing all agent observations 
        """
        raise NotImplementedError()


