
class BaseSelector():
    """ Agent Iterator base class.
    """

    def __init__(self):
        """Constructor
                
        Args:
        """


    def set_env(self, env):
        """
        Args:
            env (AECEnv): environment object
        """
        self.env = env

    def _next_agent(self):
        """Returns the next agent

        Returns:
            Tensor: next agent 
        """
        raise NotImplementedError()
