import torch
from tensordict import TensorDict

from typing import Optional, Union, Callable, Dict

from maenvs4vrp.core.env_generator_builder import InstanceBuilder

from torch.distributions import Uniform

import os

import pickle

GENERATED_INSTANCES_PATH = 'mtvrp/data/generated'

VARIANT_PRESETS = [
    'cvrp', 'ovrp', 'ovrpb', 'ovrpbl', 'ovrpbltw', 'ovrpbtw',
    'ovrpl', 'ovrpltw', 'ovrpmb', 'ovrpmbl', 'ovrpmbltw', 'ovrpmbtw',
    'ovrptw', 'vrpb', 'vrpbl', 'vrpbltw', 'vrpbtw', 'vrpl',
    'vrpltw', 'vrpmb', 'vrpmbl', 'vrpmbltw', 'vrpmbtw', 'vrptw'
    ]

VARIANT_PROBS_PRESETS = { #Variant Probabilities
        "all": {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5},
        "single_feat": {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5},
        "single_feat_otw": {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5, "OTW": 0.5},
        "cvrp": {"O": 0.0, "TW": 0.0, "L": 0.0, "B": 0.0},
        "ovrp": {"O": 1.0, "TW": 0.0, "L": 0.0, "B": 0.0},
        "vrpb": {"O": 0.0, "TW": 0.0, "L": 0.0, "B": 1.0},
        "vrpl": {"O": 0.0, "TW": 0.0, "L": 1.0, "B": 0.0},
        "vrptw": {"O": 0.0, "TW": 1.0, "L": 0.0, "B": 0.0},
        "ovrptw": {"O": 1.0, "TW": 1.0, "L": 0.0, "B": 0.0},
        "ovrpb": {"O": 1.0, "TW": 0.0, "L": 0.0, "B": 1.0},
        "ovrpl": {"O": 1.0, "TW": 0.0, "L": 1.0, "B": 0.0},
        "vrpbl": {"O": 0.0, "TW": 0.0, "L": 1.0, "B": 1.0},
        "vrpbtw": {"O": 0.0, "TW": 1.0, "L": 0.0, "B": 1.0},
        "vrpltw": {"O": 0.0, "TW": 1.0, "L": 1.0, "B": 0.0},
        "ovrpbl": {"O": 1.0, "TW": 0.0, "L": 1.0, "B": 1.0},
        "ovrpbtw": {"O": 1.0, "TW": 1.0, "L": 0.0, "B": 1.0},
        "ovrpltw": {"O": 1.0, "TW": 1.0, "L": 1.0, "B": 0.0},
        "vrpbltw": {"O": 0.0, "TW": 1.0, "L": 1.0, "B": 1.0},
        "ovrpbltw": {"O": 1.0, "TW": 1.0, "L": 1.0, "B": 1.0},
    }

class InstanceGenerator(InstanceBuilder):

    """
    Instance generation class.
    """

    @classmethod
    def get_list_of_benchmark_instances(cls):

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        benchmark_instances = {}

        generated = os.listdir(os.path.join(base_dir, GENERATED_INSTANCES_PATH))
        
        for folder in generated:
            benchmark_instances[folder] = {}
            for problem_type in VARIANT_PRESETS:
                val_path = os.path.join(GENERATED_INSTANCES_PATH, folder, problem_type, 'validation')
                test_path = os.path.join(GENERATED_INSTANCES_PATH, folder, problem_type, 'test')

                benchmark_instances[folder][problem_type] = {
                    'validation': [val_path + '/' + s.split('.')[0] for s in os.listdir(os.path.join(base_dir, val_path))],
                    'test':[test_path + '/' + s.split('.')[0] for s in os.listdir(os.path.join(base_dir, test_path))]
                }
    
        return benchmark_instances


    def __init__(
        self,
        instance_type:str = 'validation',
        set_of_instances:set = None,
        device: Optional[str] = "cpu",
        batch_size: Optional[torch.Size] = None,
        seed: int = None
    ) -> None:

        """
        Constructor.
        """

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

        assert instance_type in ['test', 'validation'] or instance_type is None or instance_type == '', f"Instance type must be 'test', 'validation', '' or None." #If None or empty, it loads both test and validation
        self.set_of_instances = set_of_instances

        if set_of_instances:
            self.instance_type = instance_type
            self.load_set_of_instances()

    def load_set_of_instances(
        self,
        set_of_instances:set = None
    ):
        
        if set_of_instances:
            self.set_of_instances = set_of_instances
        self.instances_data = dict()
        for instance_name in self.set_of_instances:
            instance = self.read_instance_data(instance_name)
            self.instances_data[instance_name] = instance
    
    def read_instance_data(self, instance_name:str):
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        generated_file = '{path_to_generated_instances}/{instance}.pkl' \
                        .format(path_to_generated_instances=base_dir,
                                instance=instance_name)
        with open(generated_file, 'rb') as fp:
            instance = pickle.load(fp)
        self.batch_size = instance['data'].batch_size
        instance['data'] = instance['data'].to(self.device)
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

    def subsample_variant(
        self,
        prob_open_routes: float = 0.5,
        prob_time_windows: float = 0.5,
        prob_limit: float = 0.5,
        prob_backhaul: float = 0.5,
        td: TensorDict = None,
        variant_preset = None,
    ) -> torch.Tensor:
        
        if variant_preset is not None:
            variant_probs = VARIANT_PROBS_PRESETS.get(variant_preset)
            assert variant_probs is not None, f"Variant preset {variant_preset} not found! \
                                                Avaliable presets are {self.VARIANT_PROBS_PRESETS.keys()} with probabilities {self.VARIANT_PROBS_PRESETS.values()}"
            print("Using preset", variant_preset)
        else:
            variant_probs = {
                "O": prob_open_routes,
                "TW": prob_time_windows,
                "L": prob_limit,
                "B": prob_backhaul
            }

        for key, prob in variant_probs.items():
            assert 0 <= prob <= 1, f"Probability {key} must be between 0 and 1"

        self.variant_probs = variant_probs
        self.variant_preset = variant_preset

        variant_probs = torch.Tensor(list(self.variant_probs.values())) #Convert dict into tensor

        if self.use_combinations:
            keep_mask = torch.rand(*self.batch_size, 4) >= variant_probs #O, TW, L, B
        else:
            if self.variant_preset in list(VARIANT_PROBS_PRESETS.keys()) and self.variant_preset not in ("all", "cvrp", "single_feat", "single_feat_otw"):
                cvrp_prob = 0
            else:
                cvrp_prob = 0.5

            if self.variant_preset in ("all", "cvrp", "single_feat", "single_feat_otw"):
                indexes = torch.distributions.Categorical(
                    torch.Tensor(list(self.variant_probs.values()) + [cvrp_prob])[
                        None
                    ].repeat(*self.batch_size, 1)
                ).sample()

                if self.variant_preset == "single_feat_otw":
                    keep_mask = torch.zeros((*self.batch_size, 6), dtype=torch.bool)
                    keep_mask[torch.arange(*self.batch_size), indexes] = True

                    keep_mask[:, :2] |= keep_mask[:, 4:5]
                else:
                    keep_mask = torch.zeros((*self.batch_size, 5), dtype=torch.bool)
                    keep_mask[torch.arange(*self.batch_size), indexes] = True

            else:

                keep_mask = torch.zeros((*self.batch_size, 4), dtype=torch.bool)
                indexes = torch.nonzero(variant_probs).squeeze()
                keep_mask[:, indexes] = True
        
        td = self._default_open(td, ~keep_mask[:, 0])
        td = self._default_time_windows(td, ~keep_mask[:, 1])
        td = self._default_distance_limit(td, ~keep_mask[:, 2])
        td = self._default_backhaul(td, ~keep_mask[:, 3])

        self.keep_mask = keep_mask

        return td    
        
    def get_time_windows(
        self,
        coords: torch.Tensor = None,
        speed: torch.Tensor = None,
        seed: int = None
    ) -> torch.Tensor:
        
        """
        Get time windows.
        """
        
        if seed is not None:
            self._set_seed(seed)

        batch_size = coords.shape[0]
        num_nodes = coords.shape[1] - 1 #No depot

        a, b, c = 0.15, 0.18, 0.2

        service_time = a + (b-a) * torch.rand(batch_size, num_nodes)
        tw_length = b + (c-b) * torch.rand(batch_size, num_nodes)
        
        x = coords[:, 0:1] #Depots
        y = coords[:, 1:] #Everything else

        d_0i = (x-y).norm(p=2, dim=-1)

        h_max = (self.max_tw_depot - service_time - tw_length) / d_0i * speed - 1
        
        tw_start = (1 + (h_max - 1) * torch.rand(batch_size, num_nodes)) * d_0i / speed
        tw_end = tw_start + tw_length

        time_windows = torch.stack(
            (
                torch.cat((torch.zeros(batch_size, 1), tw_start), -1),
                torch.cat((torch.full((batch_size, 1), self.max_tw_depot), tw_end), -1),
            ),
            dim=-1,
        )

        service_time = torch.cat((torch.zeros(batch_size, 1), service_time), dim=-1)

        return time_windows, service_time
    
    def get_distance_limits(
        self,
        coords: torch.Tensor
    ):
        
        max_dist = torch.max(torch.cdist(coords[:, 0:1], coords[:, 1:]).squeeze(-2), dim=1)[0]
        dist_lower_bound = 2 * max_dist + 1e-6
        max_distance_limit = torch.maximum(
            torch.full_like(dist_lower_bound, self.distance_limit),
            dist_lower_bound + 1e-6,
        )

        return torch.distributions.Uniform(dist_lower_bound, max_distance_limit).sample()[
            ..., None
        ] # uniform distribution

    def random_generate_instance(
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
        variant_preset=None,
        use_combinations: bool = False,
        force_visit: bool = True,
        batch_size: Optional[torch.Size] = None,
        seed: int = None,
        device: Optional[str] = "cpu"
    ) -> TensorDict:

        """
        Generate random instance.
        """

        if seed is not None:
            self._set_seed(seed)

        if num_agents is not None:
            assert num_agents>0, f"Number of agents must be greater than 0!"
        if num_nodes is not None:
            assert num_nodes>0, f"Number of nodes must be greater than 0!"
        if capacity is not None:
            assert capacity>0, f"Capacity must be greater than 0!"
        if service_times is not None:
            assert service_times>0, f"Service times must be greater than 0!"
        if max_tw_depot is not None:
            assert max_tw_depot>0, f"Service times must be greater than 0!"
        if distance_limit is not None:
            assert distance_limit>0, f"Distance limit must be greater than 0!"
        if speed is not None:
            assert speed>0, f"Speed must be greater than 0!"
        if backhaul_class is not None:
            assert backhaul_class in (1, 2), f"Backhaul class must be in [1, 2]!"

        if batch_size is None:
            batch_size = self.batch_size
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        self.batch_size = torch.Size(batch_size)

        if device is None:
            device = "cpu"
        self.device = device

        if service_times is not None:
            self.service_times = service_times

        if num_nodes is not None:
            self.num_nodes = num_nodes

        if capacity is not None:
            self.capacity = capacity

        if max_tw_depot is not None:
            self.max_tw_depot = max_tw_depot

        if speed is not None:
            self.speed = speed

        if backhaul_class is not None:
            self.backhaul_class = backhaul_class

        if sample_backhaul_class is not None:
            self.sample_backhaul_class = sample_backhaul_class

        if subsample is not None:
            self.subsample = subsample

        if use_combinations is not None:
            self.use_combinations = use_combinations

        if distance_limit is not None:
            self.distance_limit = distance_limit

        if variant_preset is not None:
            self.variant_preset = variant_preset

        if force_visit is not None:
            self.force_visit = force_visit

        instance = TensorDict({}, batch_size=self.batch_size, device=self.device)

        self.depot_idx = 0

        #Depot generation. All 0.

        instance['depot_idx'] = self.depot_idx * torch.ones((*self.batch_size, 1), dtype = torch.int64, device=self.device)

        #Coords unfiform generation

        coords = torch.FloatTensor(*batch_size, num_nodes, 2).uniform_(min_coords, max_coords) #Nodes. (x,y)

        instance['coords'] = coords
        self.coords = coords

        #Capacity

        vehicle_capacity = torch.full((*batch_size, 1), self.capacity, dtype=torch.float32)

        #Demands: linehaul and backhaul

        linehaul_demand = torch.FloatTensor(*batch_size, num_nodes).uniform_(min_demands, max_demands)
        backhaul_demand = torch.FloatTensor(*batch_size, num_nodes).uniform_(min_backhaul, max_backhaul)

        linehaul_demand[:, self.depot_idx] = 0
        backhaul_demand[:, self.depot_idx] = 0

        is_linehaul = torch.rand(*batch_size, num_nodes) > backhaul_ratio

        linehaul_demand *= is_linehaul #1 linehaul 0 backhaul
        backhaul_demand *= ~is_linehaul #1 backhaul 0 linehaul

        #Linehaul, backhaul and capacity assignment

        instance['linehaul_demands'] = linehaul_demand
        instance['backhaul_demands'] = backhaul_demand
        instance['capacity'] = vehicle_capacity

        #Unscaled capacity

        instance['original_capacity'] = torch.full((*batch_size, 1), self.capacity, dtype=torch.float32)

        #Open routes

        instance['open_routes'] = torch.ones(*batch_size, 1, dtype=torch.bool)

        #Speed

        instance['speed'] = torch.full((*batch_size, 1), self.speed, dtype=torch.float32)

        #Backhaul Class. If sample true it's random. Otherwise it's defined in constructor.

        if (self.sample_backhaul_class):
            instance['backhaul_class'] = torch.randint(1, 3, (*self.batch_size,))
        else:
            instance['backhaul_class'] = torch.full((*batch_size, 1), self.backhaul_class)

        #Time windows and service times

        time_windows, service_times = self.get_time_windows(self.coords, self.speed, seed)

        instance['time_windows'] = time_windows
        instance['service_times'] = service_times

        instance['tw_low'] = time_windows[:, :, 0]
        instance['tw_high'] = time_windows[:, :, 1]

        instance['is_depot'] = torch.zeros((*self.batch_size, num_nodes), dtype=torch.bool, device=self.device)
        instance['is_depot'][:, self.depot_idx] = True

        #Start time and end time
    
        instance['start_time'] = time_windows[:, :, 0].gather(1, torch.zeros((*self.batch_size, 1), 
                                                                          dtype=torch.int64, device=self.device)).squeeze(-1)
        instance['end_time'] = time_windows[:, :, 1].gather(1, torch.zeros((*self.batch_size, 1), 
                                                                        dtype=torch.int64, device=self.device)).squeeze(-1)

        #Distance limits

        distance_limits = self.get_distance_limits(coords=self.coords)
        instance['distance_limits'] = distance_limits
        
        instance_info = {'name': 'random_instance',
                         'num_nodes': num_nodes,
                         'num_agents': num_agents,
                         'data': instance}
        
        if self.subsample:
            instance_info = self.subsample_variant(td=instance_info, variant_preset=self.variant_preset)
            return instance_info
        else:
            return instance_info

    def augment_generate_instance(
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
        max_tw_depot: float = 4.6,
        backhaul_ratio: float = 0.2,
        backhaul_class: int = 1,
        sample_backhaul_class: bool = False,
        distance_limit: float = 2.8,
        speed: float = 1.0,
        subsample: bool = True,
        variant_preset=None,
        use_combinations: bool = False,
        force_visit: bool = True,
        batch_size: Optional[torch.Size] = None,
        n_augment:int = 2,
        seed: int = None,
        device: Optional[str] = "cpu"
    ) -> TensorDict:
        
        if seed is not None:
            self._set_seed(seed)

        if num_agents is not None:
            assert num_agents>0, f"Number of agents must be greater than 0!"
        if num_nodes is not None:
            assert num_nodes>0, f"Number of nodes must be greater than 0!"
        if capacity is not None:
            assert capacity>0, f"Capacity must be greater than 0!"
        if service_times is not None:
            assert service_times>0, f"Service times must be greater than 0!"
        if max_tw_depot is not None:
            assert max_tw_depot>0, f"Service times must be greater than 0!"
        if distance_limit is not None:
            assert distance_limit>0, f"Distance limit must be greater than 0!"
        if speed is not None:
            assert max_tw_depot>0, f"Speed must be greater than 0!"
        if backhaul_class is not None:
            assert backhaul_class in (1, 2), f"Backhaul class must be in [1, 2]!"

        if batch_size is None:
            batch_size = self.batch_size
        else:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        self.batch_size = torch.Size(batch_size)

        if device is None:
            device = "cpu"

        if service_times is not None:
            self.service_times = service_times

        if num_nodes is not None:
            self.num_nodes = num_nodes

        if num_agents is not None:
            self.num_agents = num_agents

        if capacity is not None:
            self.capacity = capacity

        if max_tw_depot is not None:
            self.max_tw_depot = max_tw_depot

        if speed is not None:
            self.speed = speed

        if backhaul_class is not None:
            self.backhaul_class = backhaul_class

        if sample_backhaul_class is not None:
            self.sample_backhaul_class = sample_backhaul_class

        if subsample is not None:
            self.subsample = subsample

        if use_combinations is not None:
            self.use_combinations = use_combinations

        if distance_limit is not None:
            self.distance_limit = distance_limit

        if force_visit is not None:
            self.force_visit = force_visit

        self.variant_preset = variant_preset

        assert self.batch_size.numel()%n_augment == 0, f"Batch size must be divisible by n_augment!"
        s_batch_size = self.batch_size.numel() // n_augment #Same batch size
        self.s_batch_size = torch.Size([s_batch_size])

        instance_info_s = self.random_generate_instance( #Generate random instance
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
            batch_size = self.s_batch_size,
            seed = seed,
            device = device
        )

        self.batch_size = torch.Size(batch_size)

        instance = TensorDict({}, batch_size=self.batch_size, device=self.device)

        for key in instance_info_s['data'].keys():
            if len(instance_info_s['data'][key].shape) == 3: #3 dimension tensors
                instance[key] = instance_info_s['data'][key].repeat(n_augment, 1, 1)
            elif len(instance_info_s['data'][key].shape) == 2: #2 dimension tensors
                instance[key] = instance_info_s['data'][key].repeat(n_augment, 1)

        instance_info = {'name':'augmented_instance',
                         'num_nodes': num_nodes,
                         'num_agents': num_agents,
                         'data':instance}
        
        return instance_info
    
    def sample_name_from_set(self, seed:int=None)-> str:
        """
        Sample one instance from instance set.

        Args:
            seed(int): Random number generator seed. Defaults to None.

        Returns:
            str: Instance name.
        """
        if seed is not None:
            self._set_seed(seed)
        assert len(self.set_of_instances)>0, f"set_of_instances has to have at least one instance!"

        return list(self.set_of_instances)[torch.randint(0, len(self.set_of_instances), (1,)).item()]
    
    def sample_instance(
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
        max_tw_depot: float = 4.6,
        backhaul_ratio: float = 0.2,
        backhaul_class: int = 1,
        sample_backhaul_class: bool = False,
        distance_limit: float = 2.8,
        speed: float = 1.0,
        subsample: bool = True,
        variant_preset=None,
        use_combinations: bool = True,
        force_visit: bool = True,
        batch_size: Optional[torch.Size] = None,
        n_augment: Optional[int] = 2,
        sample_type: str = 'random',
        instance_name: str = None,
        seed: int = None,
        device: Optional[str] = "cpu"
    ):
        
        if seed is not None:
            self._set_seed(seed)

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        self.variant_preset = variant_preset

        if self.set_of_instances is None:
            random_sample = True
        else:
            random_sample = False

        if instance_name==None and random_sample==False:
            instance_name = self.sample_name_from_set(seed=seed)
        elif instance_name==None and random_sample==True:
            instance_name = 'random_instance'
        else:
            instance_name = instance_name

        if sample_type == 'random':
            instance_info = self.random_generate_instance(
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
                batch_size = batch_size,
                seed = seed,
                device = device
            )
        
        elif sample_type == 'augment':
            instance_info = self.augment_generate_instance(
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
                n_augment=n_augment,
                batch_size = batch_size,
                seed = seed,
                device = device
            )

        elif sample_type=='saved':
            instance_info = self.get_instance(instance_name, num_agents=num_agents)

        return instance_info
        

    @staticmethod
    def _default_open(td, remove):
        td['data']['open_routes'][remove] = False
        return td

    @staticmethod
    def _default_time_windows(td, remove):
        default_tw = torch.zeros_like(td['data']['time_windows'])
        default_tw[..., 1] = float('inf')
        td['data']['time_windows'][remove] = default_tw[remove]
        td['data']['service_times'][remove] = torch.zeros_like(td['data']['service_times'][remove])
        return td
    
    @staticmethod
    def _default_distance_limit(td, remove):
        td['data']['distance_limits'][remove] = float('inf')
        return td
    
    @staticmethod
    def _default_backhaul(td, remove):
        td['data']['linehaul_demands'][remove] = (
            td['data']['linehaul_demands'][remove] + td['data']['backhaul_demands'][remove]
        )
        td['data']['backhaul_demands'][remove] = 0
        return td
    
if __name__ == "__main__":

    MIXED_PROBLEMS = ["ovrpmb", "ovrpmbl", "ovrpmbltw", "ovrpmbtw",
                      "vrpmb", "vrpmbl", "vrpmbltw", "vrpmbtw"]

    number_instances = 64
    print("Starting validation/test sets generation...")
    print()

    for num_nodes, n_agent in [(101, 25), (51, 25)]:
        generator = InstanceGenerator(batch_size=32, seed=0)

        for problem in VARIANT_PRESETS:

            for k in range(number_instances):

                #If problem is mixed, sample instance with another preset and backhaul_class=2
                if problem not in MIXED_PROBLEMS:
                    instance =  generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, variant_preset=problem)
                else:
                    if problem == "ovrpmb":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, variant_preset="ovrpb", backhaul_class=2)
                    elif problem == "ovrpmbl":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, variant_preset="ovrpbl", backhaul_class=2)
                    elif problem == "ovrpmbltw":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, variant_preset="ovrpbltw", backhaul_class=2)
                    elif problem == "ovrpmbtw":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, variant_preset="ovrpbtw", backhaul_class=2)
                    elif problem == "vrpmb":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, variant_preset="vrpb", backhaul_class=2)
                    elif problem == "vrpmbl":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, variant_preset="vrpbl", backhaul_class=2)
                    elif problem == "vrpmbltw":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, variant_preset="vrpbltw", backhaul_class=2)
                    elif problem == "vrpmbtw":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, variant_preset="vrpbtw", backhaul_class=2)
                    else:
                        raise Exception("Error generating validation set.")

                name = f'generated_val_servs_{num_nodes-1}_agents_{n_agent}_{problem}_{k}'
                instance['name'] = name
                print("Generating validation data.")
                if not os.path.exists(f'data/generated/servs_{num_nodes-1}_agents_{n_agent}/{problem}/validation'):
                    os.makedirs(f'data/generated/servs_{num_nodes-1}_agents_{n_agent}/{problem}/validation')
                    print(f"Creating directory: data/generated/servs_{num_nodes-1}_agents_{n_agent}/validation")
                with open(f'data/generated/servs_{num_nodes-1}_agents_{n_agent}/{problem}/validation/'+name+'.pkl', 'wb') as fp:
                    pickle.dump(instance, fp, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"Dumped data into: data/generated/servs_{num_nodes-1}_agents_{n_agent}/{problem}/validation/{name}.pkl")

                #If problem is mixed, sample instance with another preset and backhaul_class=2
                if problem not in MIXED_PROBLEMS:
                    instance =  generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, variant_preset=problem)
                else:
                    if problem == "ovrpmb":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, variant_preset="ovrpb", backhaul_class=2)
                    elif problem == "ovrpmbl":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, variant_preset="ovrpbl", backhaul_class=2)
                    elif problem == "ovrpmbltw":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, variant_preset="ovrpbltw", backhaul_class=2)
                    elif problem == "ovrpmbtw":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, variant_preset="ovrpbtw", backhaul_class=2)
                    elif problem == "vrpmb":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, variant_preset="vrpb", backhaul_class=2)
                    elif problem == "vrpmbl":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, variant_preset="vrpbl", backhaul_class=2)
                    elif problem == "vrpmbltw":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, variant_preset="vrpbltw", backhaul_class=2)
                    elif problem == "vrpmbtw":
                        instance = generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes, variant_preset="vrpbtw", backhaul_class=2)
                    else:
                        raise Exception("Error generating validation set.")
                
                name = f'generated_test_servs_{num_nodes-1}_agents_{n_agent}_{problem}_{k}'
                instance['name'] = name
                print("Generating test data.")
                if not os.path.exists(f'data/generated/servs_{num_nodes-1}_agents_{n_agent}/{problem}/test'):
                    os.makedirs(f'data/generated/servs_{num_nodes-1}_agents_{n_agent}/{problem}/test')
                    print(f"Creating directory: data/generated/servs_{num_nodes-1}_agents_{n_agent}/{problem}/test")
                with open(f'data/generated/servs_{num_nodes-1}_agents_{n_agent}/{problem}/test/'+name+'.pkl', 'wb') as fp:
                    pickle.dump(instance, fp, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"Dumped data into: data/generated/servs_{num_nodes-1}_agents_{n_agent}/{problem}/test/{name}.pkl")

    print('Generation completed.')