import torch
from tensordict import TensorDict

import os
from os import path

from typing import Optional, Dict

import numpy as np

from maenvs4vrp.core.env_generator_builder import InstanceBuilder

BENCHMARK_INSTANCES_PATH = 'mtvrp/data/benchmark'

VARIANT_PRESETS = [
    'cvrp', 'ovrp', 'ovrpb', 'ovrpbl', 'ovrpbltw', 'ovrpbtw',
    'ovrpl', 'ovrpltw', 'ovrpmb', 'ovrpmbl', 'ovrpmbltw', 'ovrpmbtw',
    'ovrptw', 'vrpb', 'vrpbl', 'vrpbltw', 'vrpbtw', 'vrpl',
    'vrpltw', 'vrpmb', 'vrpmbl', 'vrpmbltw', 'vrpmbtw', 'vrptw'
    ]

class BenchmarkInstanceGenerator(InstanceBuilder):

    @classmethod
    def get_list_of_benchmark_instances(cls):
        """
        Get list of possible instances from benchmark files.

        Args:
            n/a.

        Returns:
            None.
        """
        dataset = ['50_test', '100_test','50_validation', '100_validation']
        base_dir = path.dirname(os.path.dirname(os.path.abspath(__file__)))
        inst_dic = {}
        for pset in dataset:
            data_files = []
            numb, settype = pset.split('_')
            for problem in VARIANT_PRESETS:
                full_dir = path.join(base_dir, BENCHMARK_INSTANCES_PATH, problem)

                data_dir = os.listdir(os.path.join(full_dir, settype))
                data_files += [BENCHMARK_INSTANCES_PATH+'/'+problem+'/'+settype+'/'+item.split('.')[0] for item in data_dir if numb in item]

            inst_dic[pset] = data_files
        return inst_dic

    def __init__(
        self,
        problem_type:set = 'all',
        instance_type:str = None,
        set_of_instances:set = None,
        device: Optional[str] = 'cpu',
        batch_size: Optional[torch.Size] = 1000,
        seed: int = None
    ) -> None:

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

        if problem_type is None or 'all':
            problem_type = VARIANT_PRESETS
        assert problem_type is not None and len(problem_type)>0, f"Set of problem variants is not > 0."
        assert all(item in VARIANT_PRESETS for item in problem_type), f"Invalid variant preset."
        assert instance_type in ['50_test', '100_test','50_validation', '100_validation'] or instance_type is None or instance_type == '', f"Instance type must be '50_test', '100_test','50_validation', '100_validation'." 
        assert len(set_of_instances)>0, f"Set of instances not > 0."

        if set_of_instances:
            self.problem_type = problem_type
            self.instance_type = instance_type
            self.set_of_instances = set_of_instances
            self.load_set_of_instances()

    def load_set_of_instances(self, set_of_instances:set=None):
        """
        Load every instance on set_of_instances set.
        
        Args:
            set_of_instances(set): Set of instances file names. Defaults to None.

        Returns:
            None.
        """
        if set_of_instances:
            self.set_of_instances = set_of_instances
        self.instances_data = dict()
        for instance_name in self.set_of_instances:
            instance = self.read_parse_instance_data(instance_name)
            self.instances_data[instance_name] = instance  


    def read_parse_instance_data(self, instance_name:str)-> Dict:
        """
        Read instance data from file.

        Args:
            instance_name(str): Instance file name.

        Returns: 
            Dict: Instance data.
        """

        base_dir = path.dirname(path.dirname(path.abspath(__file__)))
        file_path = '{path_to_generated_instances}/{instance}.npz' \
                        .format(path_to_generated_instances=base_dir,
                                instance=instance_name)

        instance = dict()
        instance['name'] = instance_name

        loaded_data = np.load(file_path)
        np_instance = {key: loaded_data[key] for key in loaded_data.files}

        data = TensorDict({}, batch_size=self.batch_size, device=self.device)
        for key in np_instance:
            data[key] = torch.from_numpy(np_instance[key])

        instance['num_agents'] = 1
        instance['num_nodes'] = data['locs'].shape[1]

        num_agents = instance['num_agents']
        num_nodes = instance['num_nodes'] 
        
        batch_size = data['locs'].shape[0]
        instance['batch_size'] = batch_size

        instance['name'] = instance['name'] + '_samp'

        new_data = TensorDict({}, batch_size=self.batch_size, device=self.device)

        batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(-1)

        new_data['coords'] = data['locs'] #There're always coords
        zeros =  torch.zeros((*self.batch_size, 1), dtype = torch.int64, device=self.device)
        new_data['linehaul_demands'] = torch.concat([zeros, data['demand_linehaul']], dim=1) #There're always linehauls
        new_data['capacity'] = data['vehicle_capacity'] #There're always capacities
        self.depot_idx = 0
        new_data['depot_idx'] = self.depot_idx * torch.ones((*self.batch_size, 1), dtype = torch.int64, device=self.device)
        new_data['speed'] = data['speed'] #There's always speeed etc.
        if 'demand_backhaul' in data.keys():
            zeros =  torch.zeros((*self.batch_size, 1), dtype = torch.int64, device=self.device)
            new_data['backhaul_demands'] = torch.concat([zeros, data['demand_backhaul']], dim=1)
        else:
            new_data['backhaul_demands'] = torch.zeros((*self.batch_size, num_nodes), dtype=torch.float32, device=self.device)
        if 'backhaul_class' in data.keys():
            new_data['backhaul_class'] = data['backhaul_class'].squeeze(-1)
        if 'time_windows' in data.keys():
            new_data['time_windows'] = data['time_windows']
        else:
            new_data['time_windows'] = torch.zeros((*self.batch_size, num_nodes, 2), dtype=torch.float32, device=self.device)
            new_data['time_windows'][:,:,1] = float('inf')
        if 'service_time' in data.keys():
            new_data['service_times'] = data['service_time']
        if 'distance_limit' in data.keys():
            new_data['distance_limits'] = data['distance_limit']
        else:
            new_data['distance_limits'] = torch.full((*self.batch_size, 1), float('inf'))
        if 'open_route' in data.keys():
            new_data['open_routes'] = data['open_route']
        else:
            new_data['open_routes'] = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device=self.device)
        if 'time_windows' in data.keys():
            new_data['end_time'] = data['time_windows'][:,:,1].gather(1, torch.zeros((*self.batch_size, 1), 
                                                                        dtype=torch.int64, device=self.device)).squeeze(-1)
            new_data['start_time'] = data['time_windows'][:,:,0].gather(1, torch.zeros((*self.batch_size, 1), 
                                                                        dtype=torch.int64, device=self.device)).squeeze(-1)
            new_data['tw_low'] = data['time_windows'][:,:,0]
            new_data['tw_high'] = data['time_windows'][:,:,1]
        else:
            new_data['end_time'] = torch.full((*self.batch_size, 1), float('inf'))
            new_data['start_time'] = torch.zeros((*self.batch_size, 1), dtype=torch.int64, device=self.device)
            new_data['tw_low'] = torch.zeros((*self.batch_size, num_nodes), dtype=torch.float32, device=self.device)
            new_data['tw_high'] = torch.full((*self.batch_size, num_nodes), float('inf'))

        new_data['is_depot'] = torch.zeros((*self.batch_size, num_nodes), dtype=torch.bool, device=self.device)
        new_data['is_depot'][:, self.depot_idx] = True

        instance['data'] = new_data

        return instance
    
    
    def get_instance(self, instance_name:str, num_agents:int=None) -> Dict:
        """
        Get an instance with custom number of agents.

        Args:
            instance_name(str): Instance file name.
            num_agents(int): Number of agents. Defaults to None.

        Returns:
            Dict: Instance data.

        """
        instance = self.instances_data.get(instance_name)

        if num_agents is not None:
            assert num_agents>0, f"number of agents must be grater them 0!"
            instance['num_agents'] = num_agents

        return instance
    
    
    def random_sample_instance(self, 
                               instance_name:str=None,
                               num_agents: int = None,
                               num_nodes: int = None,
                               min_coords: float = None,
                               max_coords: float = None,
                               capacity: int = None,
                               service_times: float = None,
                               min_demands: int = None,
                               max_demands: int = None,
                               min_backhaul: int = None,
                               max_backhaul: int = None,
                               max_tw_depot: float = None,
                               backhaul_ratio: float = None,
                               backhaul_class: int = None,
                               sample_backhaul_class: bool = False,
                               distance_limit: float = None,
                               speed: float = None,
                               subsample: bool = True,
                               variant_preset=None,
                               use_combinations: bool = False,
                               force_visit: bool = True,
                               batch_size: Optional[torch.Size] = None,
                               seed: int = None,
                               device: Optional[str] = None)-> Dict:
        """
        Sample one instance from instance space, randomly adjusting the nodes.

        Args:
            instance_name(str): Instance file name. Defaults to None.
            num_agents(int):  Total number of agents. Defaults to None.
            num_nodes(int):  Total number of nodes. Defaults to None.
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            Dict: Instance data.
        """
        if seed is not None:
            self._set_seed(seed)

        new_instance = dict()
        instance = self.get_instance(instance_name, num_agents)

        new_instance['num_agents'] = instance['num_agents']

        if num_nodes is not None:
            num_nodes = min(num_nodes, instance['num_nodes'])
            new_instance['num_nodes'] = num_nodes
        else:
            num_nodes = instance['num_nodes']
            new_instance['num_nodes'] = instance['num_nodes']

        batch_size = instance['batch_size']

        idxs = torch.arange(0, num_nodes, device=self.device).expand(batch_size, num_nodes)
        depots = idxs[:, 0:1]
        non_depots = idxs[:, 1:]
        indices = torch.argsort(torch.rand(*non_depots.shape), dim=-1)
        index = torch.cat([depots, indices], dim=1)
        index = index[:, :num_nodes]

        new_data = TensorDict({}, batch_size=self.batch_size, device=self.device)

        batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(-1)

        data = instance['data']

        new_data['coords'] = data['coords'][batch_idx, index] #There're always coords
        new_data['linehaul_demands'] = data['linehaul_demands'][batch_idx, index] #There're always linehauls
        new_data['capacity'] = data['capacity'] #There're always capacities
        self.depot_idx = 0
        new_data['depot_idx'] = self.depot_idx * torch.ones((*self.batch_size, 1), dtype = torch.int64, device=self.device)
        new_data['speed'] = data['speed'] #There's always speeed etc.
        if 'backhaul_demands' in data.keys():
            new_data['backhaul_demands'] = data['backhaul_demands'][batch_idx, index]
        if 'backhaul_class' in data.keys():
            new_data['backhaul_class'] = data['backhaul_class'].squeeze(-1)
        if 'time_windows' in data.keys():
            new_data['time_windows'] = data['time_windows'][batch_idx, index]
        if 'service_times' in data.keys():
            new_data['service_times'] = data['service_times'][batch_idx, index]
        if 'distance_limits' in data.keys():
            new_data['distance_limits'] = data['distance_limits']
        else:
            new_data['distance_limits'] = torch.full((*self.batch_size, 1), float('inf'))
        if 'open_routes' in data.keys():
            new_data['open_routes'] = data['open_routes']
        if 'time_windows' in data.keys():
            new_data['end_time'] = data['time_windows'][:,:,1].gather(1, torch.zeros((*self.batch_size, 1), 
                                                                        dtype=torch.int64, device=self.device)).squeeze(-1)
            new_data['start_time'] = data['time_windows'][:,:,0].gather(1, torch.zeros((*self.batch_size, 1), 
                                                                        dtype=torch.int64, device=self.device)).squeeze(-1)
            new_data['tw_low'] = new_data['time_windows'][:,:,0]
            new_data['tw_high'] = new_data['time_windows'][:,:,1]
        else:
            new_data['end_time'] = torch.full((*self.batch_size, 1), float('inf'))
            new_data['start_time'] = torch.zeros((*self.batch_size, 1), dtype=torch.int64, device=self.device)
            new_data['tw_low'] = torch.zeros((*self.batch_size, num_nodes), dtype=torch.float32, device=self.device)
            new_data['tw_high'] = torch.full((*self.batch_size, num_nodes), float('inf'))

        new_data['is_depot'] = torch.zeros((*self.batch_size, num_nodes), dtype=torch.bool, device=self.device)
        new_data['is_depot'][:, self.depot_idx] = True

        new_instance['data'] = new_data

        return new_instance
    
    def sample_name_from_set(self, seed:int=None)-> str:
        """
        Sample one instance from instance set.

        Args:
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            str: Instance sample name.
        """
        if seed is not None:
            self._set_seed(seed)
        assert len(self.set_of_instances)>0, f"set_of_instances has to have at least one instance!"

        return list(self.set_of_instances)[torch.randint(0, len(self.set_of_instances), (1,)).item()]
    
    def sample_instance(self,
                        sample_type: str = 'random',
                        instance_name:str=None,
                        num_agents: int = None,
                        num_nodes: int = None,
                        min_coords: float = None,
                        max_coords: float = None,
                        capacity: int = None,
                        service_times: float = None,
                        min_demands: int = None,
                        max_demands: int = None,
                        min_backhaul: int = None,
                        max_backhaul: int = None,
                        max_tw_depot: float = None,
                        backhaul_ratio: float = None,
                        backhaul_class: int = None,
                        sample_backhaul_class: bool = False,
                        distance_limit: float = None,
                        speed: float = None,
                        subsample: bool = True,
                        variant_preset=None,
                        use_combinations: bool = False,
                        force_visit: bool = True,
                        batch_size: Optional[torch.Size] = None,
                        seed: int = None,
                        n_augment: Optional[int] = None,
                        device: Optional[str] = None)-> Dict:
        """
        Sample one instance from instance space.

        Args:
            num_agents(int): Total number of agents. Defaults to None.
            num_nodes(int): Total number of nodes. Defaults to None.
            capacity(int): Capacity of the agents. Defaults to None.
            service_times(float): Service time in the nodes. Defaults to None.
            instance_name(str): Instance name. Defaults to None.
            sample_type(str): Sample type. It can be "random" or something else for "first n". Defaults to "random".
            batch_size(torch.Size or None): Batch size. Defaults to None.
            n_augment(int, optional): Data augmentation. Defaults to None.
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            Dict: Instance data.
        """
        if seed is not None:
            self._set_seed(seed)

        if instance_name==None:
            instance_name = self.sample_name_from_set(seed=seed)
        else:
            instance_name = instance_name

        if sample_type=='random':
            instance = self.random_sample_instance( instance_name=instance_name,
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
                                                    force_visit = force_visit,
                                                    batch_size = batch_size,
                                                    seed = seed,
                                                    device = device)
        else:
            instance = self.get_instance(instance_name, num_agents=num_agents)

        return instance