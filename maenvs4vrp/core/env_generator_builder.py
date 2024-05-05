
from typing import Dict, Optional

import torch
from tensordict.tensordict import TensorDict


class InstanceBuilder(object):

    """
    Basic instance generator class

    """

    
    DEFAULT_SEED = 2925
    def __init__(self, instance_type:str=None, 
                 set_of_instances:set=None, 
                 num_services:int=None, 
                 num_agents:int=None, 
                 seed:int=None,
                 device: str = "cpu",
                 batch_size: torch.Size = None) -> None:
        """Constructor

        Args:
            instance_type (str): instance type . Defaults to None;
            set_of_instances (set):  set of instances file names. Defaults to None;
            num_services (int):  Total number of services. Defaults to None;
            num_agents (int):  Total number of agents. Defaults to None;
            seed (int): random number generator seed. Defaults to None;

        """

        self.max_n_services = None
        self.max_n_vehicles = None
        # seed the generation process
        if seed is None:
            self._set_seed(self.DEFAULT_SEED)
        else:
            self._set_seed(seed)

        self.device = device
        if batch_size is None:
            batch_size = [1]
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        self.batch_size = torch.Size(batch_size)

    def _set_seed(self, seed: Optional[int]):
        """Sets the random seed used by the environment."""
        self.seed = seed
        rng = torch.manual_seed(self.seed)
        self.rng = rng

    def read_instance_data(self, instance_name:str)-> Dict:
        """
        Reads instance data
        Args:
            instance_name (str): instance file name.

        Returns: 
            Dict: Instance data 
        """
        raise NotImplementedError()

    def get_instance(self, instance_name:str, preloaded:bool=False)-> Dict:
        """
        Combine read instance file and parse to Dict

        Args:
            instance_name (str): instance file name.
            preloaded (bool): if instance data has been pre-loaded. Defaults to False.

        Returns: 
            Dict: Instance data
        """
        raise NotImplementedError()

    def load_set_of_instances(self, set_of_instances:set=None, already_loaded:bool=None):
        """
        Loads every instance on set_of_instances set

        Args:
            set_of_instances (set): set of instances file names. Defaults to None.
            already_loaded (bool): if instance data has been pre-loaded. Defaults to None.

        """
        raise NotImplementedError()

    def get_instance_preloaded(self) -> Dict:
        """
        Args:
            action (np.ndarray): the action to execute.

        Returns:
            Dict: Instance data
        """
        raise NotImplementedError()



    def random_sample_instance(self, num_agents:int=None, num_services:int=None, seed:int=None) -> Dict:
        """
        Samples one instance from insance space

        Args:
            num_services (int):  Total number of services. Defaults to None;
            num_agents (int):  Total number of agents. Defaults to None;
            seed (int): random number generator seed. Defaults to None;

        Returns:
            Dict: Instance data.
        """
        raise NotImplementedError()

    def sample_name_from_set(self, seed:int=None)-> str:
        """
        Samples one instance from insance set

        Args:
            seed (int): random number generator seed. Defaults to None;

        Returns:
            str: instance name.
        """
        raise NotImplementedError()

    def sample_instance(self, num_agents:int=None, num_services:int=None, instance_name:str=None, random_sample:bool=True, seed:int=None)-> Dict:
        """
        Samples one instance from insance space

        Args:
            num_services (int):  Total number of services. Defaults to None;
            num_agents (int):  Total number of agents. Defaults to None;
            instance_name (str):  instance name. Defaults to None;
            random_sample (bool):  True to sample instance and False to use original instance data. Defaults to None;
            seed (int): random number generator seed. Defaults to None;

        Returns:
            Dict: Instance data.
        """
        raise NotImplementedError()
