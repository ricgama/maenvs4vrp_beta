
class RewardFn:
    """Reward function base class.
    """

    def __init__(self):
        """Constructor

        """
        self.env = None

    def set_env(self, env):
        self.env = env

    def get_reward(self):
        """
        computes reward.
        """

        raise NotImplementedError()
    
