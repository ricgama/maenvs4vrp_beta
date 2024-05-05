import torch
from tensordict import TensorDict

import os
from os import path
import pickle

from typing import Dict, Optional
from maenvs4vrp.core.env_generator_builder import InstanceBuilder

GENERATED_INSTANCES_PATH = 'toptw/data/generated'

class InstanceGenerator(InstanceBuilder):
    """
    class for TOPTW benchmark instances generation
    
    """
    @classmethod
    def get_list_of_benchmark_instances(cls):
        base_dir = path.dirname(path.dirname(path.abspath(__file__)))

        return {'validation': [s.split('.')[0] for s in os.listdir(path.join(base_dir, GENERATED_INSTANCES_PATH, 'validation'))],
                'test':[s.split('.')[0] for s in os.listdir(path.join(base_dir, GENERATED_INSTANCES_PATH, 'test'))]}
        
    def __init__(self, 
                 instance_type:str='validation', 
                 set_of_instances:set=None, 
                 device: Optional[str] = "cpu",
                 batch_size: Optional[torch.Size] = None,
                 seed:int=None) -> None:
        """    

        Args:       
            instance_type (str):  instance type. Can be "validation" or "test";
            set_of_instances (bool):  set of instances file names;
            seed (int): random number generator seed. Defaults to None;
        """

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

        self.max_num_agents = 20
        self.max_num_nodes = 100

        assert instance_type in ["validation", "test"], f"instance unknown type"
        self.set_of_instances = set_of_instances
        if set_of_instances:
            self.instance_type = instance_type
            self.load_set_of_instances()
            

    def read_instance_data(self, instance_name:str)-> Dict:
        """
        Reads instance data
        Args:
            instance_name (str): instance file name.

        Returns: 
            Dict: Instance data 
        """

        base_dir = path.dirname(path.dirname(path.abspath(__file__)))
        path_to_file = path.join(base_dir, GENERATED_INSTANCES_PATH, self.instance_type)
        generated_file = '{path_to_generated_instances}/{instance}.pkl' \
                        .format(path_to_generated_instances=path_to_file,
                                instance=instance_name)
        with open(generated_file, 'rb') as fp:
            instance = pickle.load(fp)
        return instance


    def get_instance(self, instance_name:str, num_agents:int=None) -> Dict:
        """
        Returns:
            Dict: Instance data

        """
        instance = self.instances_data.get(instance_name)

        if num_agents is not None:
            assert num_agents>0, f"number of agents must be grater them 0!"
            instance['num_agents'] = num_agents

        return instance
            
    def load_set_of_instances(self, set_of_instances:set=None):
        """
        Loads every instance on set_of_instances set
        
        Args:
            set_of_instances (set): set of instances file names. Defaults to None.

        """
        if set_of_instances:
            self.set_of_instances = set_of_instances
        self.instances_data = dict()
        for instance_name in self.set_of_instances:
            instance = self.read_instance_data(instance_name)
            self.instances_data[instance_name] = instance


    def get_time_windows(self, instance:TensorDict=None, coords:torch.Tensor=None,  batch_size:torch.Size=None, seed:int=None)-> torch.tensor:
        """
        Args:
            dist_mat (np.array): distance matrix;
            seed (int): random number generator seed. Defaults to None;

        Returns: 
            np.array: nodes time windows;
        """

        if seed is not None:
            self._set_seed(seed)

        time_windows = torch.zeros((*batch_size, self.max_num_nodes, 2), device=self.device)
        dist_depot = torch.pairwise_distance(coords[:, 0, None], coords[:], keepdim = True)

        depot_start, depot_end = 0, 3  

        inf = depot_start + dist_depot
        sup = depot_end - dist_depot - self.service_times

        time_centers = inf.squeeze(-1) + torch.rand(*batch_size, self.max_num_nodes, device=self.device) * (sup-inf).squeeze(-1)
        time_half_width = torch.empty((*batch_size, self.max_num_nodes), device=self.device).uniform_(self.service_times / 2 , depot_end / 3)
        time_windows[:, :, 0] = torch.clip(time_centers - time_half_width, depot_start, depot_end)
        time_windows[:, :, 1] = torch.clip(time_centers + time_half_width, depot_start, depot_end)
        time_windows[:, self.depot_idx, 0] = depot_start
        time_windows[:, self.depot_idx, 1] = depot_end

        return time_windows


    def random_generate_instance(self, num_agents:int=20, 
                                 num_nodes:int=100, 
                                 service_times:int=0.2, 
                                 profits:str='distance',
                                 batch_size: Optional[torch.Size] = None,
                                 seed:int=None)-> TensorDict:
        """
        TOPTW random instances, for which the time windows generation follow:

        @inproceedings{li2021learning,
                        title={Learning to delegate for large-scale vehicle routing},
                        author={Sirui Li and Zhongxia Yan and Cathy Wu},
                        booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
                        year={2021}
                        }
        see: https://github.com/mit-wu-lab/learning-to-delegate/blob/main/generate_initial.py

        and protif generation follow:
        @inproceedings{kool2018attention,
                        title={Attention, Learn to Solve Routing Problems!},
                        author={Kool, Wouter and van Hoof, Herke and Welling, Max},
                        booktitle={International Conference on Learning Representations},
                        year={2018}
                        }


        Args:
            num_services (int, optional):  Total number of services. Defaults to 100.
            num_agents (int, optional): Total number of agents. Defaults to 20.
            profits (str): can be 'constant', 'uniform' or 'distance'
            service_times (int, optional): Total time of service. Defaults to 0.2.
            seed (int, optional): random number generator seed. Defaults to None.

        Returns:
            Dict: Instance data
        """
        if seed is not None:
            self._set_seed(seed)

        if num_agents is not None:
            assert num_agents>0, f"number of agents must be grater them 0!"
            self.max_num_agents = num_agents
        if num_nodes is not None:
            assert num_nodes>0, f"number of services must be grater them 0!"
            self.max_num_nodes = num_nodes
        if service_times is not None:
            self.service_times = service_times

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        instance = TensorDict({}, batch_size=self.batch_size, device=self.device)
        
        self.depot_idx = 0
        instance['depot_idx'] = self.depot_idx * torch.ones((*self.batch_size, 1), dtype = torch.int64, device=self.device)

        coords = torch.rand(*self.batch_size, self.max_num_nodes, 2, dtype = torch.float, device=self.device) 
        instance['coords'] = coords

        service_times = self.service_times * torch.ones((*self.batch_size, num_nodes), dtype = torch.float, device=self.device)
        service_times[:, self.depot_idx] = 0
        instance['service_time'] = service_times

        time_windows = self.get_time_windows(instance, coords, self.batch_size, seed)

        instance['tw_low'] =  time_windows[:, :, 0].clone()
        instance['tw_high'] = time_windows[:, :, 1].clone()

        instance['is_depot'] = torch.zeros((*self.batch_size, num_nodes), dtype=torch.bool, device=self.device)
        instance['is_depot'][:, self.depot_idx] = True

        instance['start_time'] = time_windows[:, :, 0].gather(1, torch.zeros((*self.batch_size, 1), 
                                                                          dtype=torch.int64, device=self.device)).squeeze(-1)
        instance['end_time'] = time_windows[:, :, 1].gather(1, torch.zeros((*self.batch_size, 1), 
                                                                        dtype=torch.int64, device=self.device)).squeeze(-1)

        if profits == 'constant':
            profits = torch.ones((*self.batch_size, num_nodes), dtype = torch.float, device=self.device) # constant 
        elif profits == 'uniform':
            profits = torch.randint(low = 1, high=100, size = (*self.batch_size, num_nodes), dtype = torch.float, device=self.device) / 100 # uniform 
        elif profits == 'distance':
            depot_loc = coords.gather(1, instance['depot_idx'][:,:,None].expand(-1, -1, 2))
            depot2nodes = torch.pairwise_distance(depot_loc, coords, eps=0, keepdim = False)
            profits = (1+ torch.floor(99 * depot2nodes / torch.max(depot2nodes, dim=1, keepdim = True).values)) / 100 # distance 
        profits[:, self.depot_idx] = 0

        instance['profits'] = profits

        instance_info = {'name':'random_instance',
                         'num_nodes': self.max_num_nodes,
                         'num_agents':self.max_num_agents,
                         'data':instance}
        return instance_info

    def augment_generate_instance(self, num_agents:int=20, 
                                 num_nodes:int=100, 
                                 service_times:int=0.2, 
                                 profits:str='distance',
                                 batch_size: Optional[torch.Size] = None,
                                 n_augment:int = 2,
                                 seed:int=None)-> TensorDict:
        """
        CVRPTW random instances generated following:

        @inproceedings{li2021learning,
                        title={Learning to delegate for large-scale vehicle routing},
                        author={Sirui Li and Zhongxia Yan and Cathy Wu},
                        booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
                        year={2021}
                        }
        see: https://github.com/mit-wu-lab/learning-to-delegate/blob/main/generate_initial.py

        Args:
            num_services (int, optional):  Total number of services. Defaults to 100.
            num_agents (int, optional): Total number of agents. Defaults to 20.
            capacity (int, optional): Total capacity for each agent. Defaults to 50.
            service_times (int, optional): Total time of service. Defaults to 0.2.
            seed (int, optional): random number generator seed. Defaults to None.

        Returns:
            Dict: Instance data
        """
        if seed is not None:
            self._set_seed(seed)

        if num_agents is not None:
            assert num_agents>0, f"number of agents must be grater them 0!"
            self.max_num_agents = num_agents
        if num_nodes is not None:
            assert num_nodes>0, f"number of services must be grater them 0!"
            self.max_num_nodes = num_nodes
        if service_times is not None:
            self.service_times = service_times

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)

        assert self.batch_size.numel()%n_augment == 0, f"batch_size must be divisible by n_augment"
        
        s_batch_size = self.batch_size.numel() // n_augment
        self.s_batch_size = torch.Size([s_batch_size])
        
        instance = TensorDict({}, batch_size=self.batch_size, device=self.device)
        
        self.depot_idx = 0

        depot_idx = self.depot_idx * torch.ones((*self.s_batch_size, 1), dtype = torch.int64, device=self.device)
        instance['depot_idx'] = depot_idx.repeat(n_augment, 1)

        coords = torch.rand(*self.s_batch_size, self.max_num_nodes, 2, dtype = torch.float, device=self.device) 
        instance['coords'] = coords.clone().repeat(n_augment, 1, 1) 

        service_times = self.service_times * torch.ones((*self.s_batch_size, num_nodes), dtype = torch.float, device=self.device).repeat(n_augment, 1)
        service_times[:, self.depot_idx] = 0
        instance['service_time'] = service_times

        time_windows = self.get_time_windows(instance, coords, self.s_batch_size, seed)
        time_windows = time_windows.repeat(n_augment, 1, 1)

        instance['tw_low'] =  time_windows[:, :, 0].clone()
        instance['tw_high'] = time_windows[:, :, 1].clone()

        instance['is_depot'] = torch.zeros((*self.s_batch_size, num_nodes), dtype=torch.bool, device=self.device).repeat(n_augment, 1)
        instance['is_depot'][:, self.depot_idx] = True

        instance['start_time'] = time_windows[:, :, 0].gather(1, torch.zeros((*self.batch_size, 1), 
                                                                          dtype=torch.int64, device=self.device)).squeeze(-1)
        instance['end_time'] = time_windows[:, :, 1].gather(1, torch.zeros((*self.batch_size, 1), 
                                                                        dtype=torch.int64, device=self.device)).squeeze(-1)

        if profits == 'constant':
            profits = torch.ones((*self.s_batch_size, num_nodes), dtype = torch.float, device=self.device) # constant 
        elif profits == 'uniform':
            profits = torch.randint(low = 1, high=100, size = (*self.s_batch_size, num_nodes), dtype = torch.float, device=self.device) / 100 # uniform 
        elif profits == 'distance':
            depot_loc = coords.gather(1, depot_idx[:,:,None].expand(-1, -1, 2))
            depot2nodes = torch.pairwise_distance(depot_loc, coords, eps=0, keepdim = False)
            profits = (1+ torch.floor(99 * depot2nodes / torch.max(depot2nodes, dim=1, keepdim = True).values)) / 100 # distance 
        profits[:, self.depot_idx] = 0

        instance['profits'] = profits.repeat(n_augment, 1)

        instance_info = {'name':'random_instance',
                         'num_nodes': self.max_num_nodes,
                         'num_agents':self.max_num_agents,
                         'data':instance}
        return instance_info
    
    def sample_name_from_set(self, seed:int=None)-> str:
        """
        Samples one instance from instance set

        Args:
            seed (int): random number generator seed. Defaults to None;

        Returns:
            str: instance name.
        """
        if seed is not None:
            self._set_seed(seed)
        assert len(self.set_of_instances)>0, f"set_of_instances has to have at least one instance!"

        return list(self.set_of_instances)[torch.randint(0, len(self.set_of_instances), (1,)).item()]

    def sample_instance(self, 
                        num_agents=None, 
                        num_nodes=None, 
                        service_times=0.2,
                        profits:str='constant',
                        instance_name:str=None, 
                        sample_type:str='random',
                        batch_size: Optional[torch.Size] = None,
                        n_augment: Optional[int] = None,
                        seed:int=None)-> Dict:
        """
        Samples one instance from instance space

        Args:
            num_agents (int): Total number of agents. Defaults to 20.
            num_nodes (int):  Total number of nodes. Defaults to 100.
            capacity (int): Total capacity for each agent. Defaults to 50.
            service_times (int): Total time of service. Defaults to 0.2.            
            instance_name (str):  instance name. Defaults to None;
            random_sample (bool):  True to sample instance and False to use original instance data. Defaults to None;
            seed (int): random number generator seed. Defaults to None;

        Returns:
            Dict: Instance data.
        """
        if seed is not None:
            self._set_seed(seed)

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


        if num_agents is None:
            num_agents = 20
        if num_nodes is None:
            num_nodes = 100
        if service_times is None:
            service_times = 0.2

        if batch_size is not None:
            batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
            self.batch_size = torch.Size(batch_size)
           
        if sample_type=='random':
            instance_info = self.random_generate_instance(num_agents=num_agents, 
                                                     num_nodes=num_nodes, 
                                                     profits=profits, 
                                                     service_times=service_times,
                                                     batch_size = batch_size,
                                                     seed=seed)
        elif sample_type=='augment':
            instance_info = self.augment_generate_instance(num_agents=num_agents, 
                                                     num_nodes=num_nodes, 
                                                     profits=profits, 
                                                     service_times=service_times,
                                                     batch_size = batch_size,
                                                     n_augment = n_augment,
                                                     seed=seed)           
        elif sample_type=='saved':
            instance_info = self.get_instance(instance_name, num_agents=num_agents)

        return instance_info

if __name__ == '__main__':

    number_instances = 128
    print('starting valid/test sets generation')

    if not os.path.exists('data/generated/test'):
        os.makedirs('data/generated/test')
    if not os.path.exists('data/generated/validation'):
        os.makedirs('data/generated/validation')

    # valid/test sets generation
    for num_nodes, n_agent in [(26, 10)]:
        generator = InstanceGenerator(batch_size=1, seed=0)
        for k in range(number_instances):
            instance =  generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes)
            name = f'generated_val_servs_{num_nodes-1}_agents_{n_agent}_{k}'
            instance['name'] = name
            with open('data/generated/validation/'+name+'.pkl', 'wb') as fp:
                pickle.dump(instance, fp, protocol=pickle.HIGHEST_PROTOCOL)

            instance =  generator.sample_instance(num_agents=n_agent, num_nodes=num_nodes)
            name = f'generated_test_servs_{num_nodes-1}_agents_{n_agent}_{k}'
            instance['name'] = name
            with open('data/generated/test/'+name+'.pkl', 'wb') as fp:
                pickle.dump(instance, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print('done')
