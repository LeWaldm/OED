import torch

class Design_Network(torch.nn.Module):
    def __init__(self, encoder, emitter) -> None:
        super().__init__()
        assert isinstance(encoder, torch.nn.Module)
        assert isinstance(emitter, torch.nn.Module)
        self.encoder = encoder  # (xi=Nxd, y=Nxk) -> Nxr
        self.emitter = emitter  # (repr=Nxr) -> Nxd
        self.design_dim = encoder.design_dim
        self.y_dim = encoder.y_dim
        self.repr_dim = encoder.repr_dim

    def forward(self, xi, y):
        """ 
            xi: N x design_dim, 
            y: N x y_dim
        """
        self.buffer_mean_representation = self.buffer_mean_representation + self.encoder(xi,y)
        return self.emitter(self.buffer_mean_representation)
    
    def reset_buffer(self, batch_size):
        self.buffer_mean_representation = torch.zeros((batch_size,self.repr_dim))


class Experimenter:
    def __init__(self) -> None:
        self.obs_data = {'design_params':[], 'design':[], 'y':[]}
            # contains lists where each element corresponds to an executed design

    def params2design(self,design_params):
        """ get design from design_params in form that can be inputted into densities"""
        raise NotImplementedError()

    def execute_design(self,design_params):
        """ Executes experiment identified by design_params """
        raise NotImplementedError()

    def get_initial_design_params(self):
        """ starting point for optimization, returns: design_params """
        raise NotImplementedError()

    def get_obs_data(self):
        """ Get all designs and their outcomes that have been observed so far"""
        return self.obs_data

    def get_eval_data(self,exclude_obs=False):
        """ Get evaluation data (design, outcome), returns: y,design"""
        raise NotImplementedError()
    
    def verbose_designs(self):
        """ givee some information about executed designs, e.g. plotting"""
        raise NotImplementedError()

    def get_candidate_designs(self):
        """
        returns: list of tuples, each tuple specifies candidate design, first coordinate 
        is design_params, second is design
        """
        raise NotImplementedError()


def get_data_dir():
    return '/Users/user1/Documents/data.nosync/'
