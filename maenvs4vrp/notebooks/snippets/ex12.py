class DenseReward(DenseReward):
    """Reward class.
    """

    def get_reward(self, action):
        """
        
        """

        loc = self.env.td_state['coords'].gather(1, self.env.td_state['cur_agent']['cur_node'][:,:,None].expand(-1, -1, 2))
        next_loc = self.env.td_state['coords'].gather(1, action[:,:,None].expand(-1, -1, 2))

        ptime = self.env.td_state['cur_agent']['cur_time'].clone()
        time2j = torch.pairwise_distance(loc, next_loc, eps=0, keepdim = False)
        if self.env.n_digits is not None:
            time2j = torch.floor(self.env.n_digits * time2j) / self.env.n_digits

        arrivej = ptime + time2j

        tw_low = self.env.td_state['tw_low'].gather(1, action)
        tw_high = self.env.td_state['tw_high'].gather(1, action)

        penalty = -(self.env.early_penalty * torch.clip(tw_low-arrivej, min=0, max=None) + \
                    self.env.late_penalty * torch.clip(arrivej - tw_high, min=0, max=None))

        reward = -time2j
        
        # compute extra penalty if env has unvisited nodes 
        is_last_step = self.env.td_state['is_last_step']
        
        depot2nodes = 2*torch.pairwise_distance(self.env.td_state['depot_loc'], self.env.td_state['coords'], eps=0, keepdim = False)
        if self.env.n_digits is not None:
            depot2nodes = torch.floor(self.env.n_digits * depot2nodes) / self.env.n_digits
        penalty[is_last_step] += self.pending_penalty * ((depot2nodes * self.env.td_state['nodes']['active_nodes_mask']).sum(-1, keepdim = True).float()[is_last_step])

        return reward, penalty