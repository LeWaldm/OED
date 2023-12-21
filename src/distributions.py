import torch

class Distr_interface:
    def __init__(self):
        self.params:dict = {}
        self.reparam_trick = False  # whether can calculate gradients through sampling
        self.params_constraints = {}  # constraints on parameters, e.g. positive (c.f. src.data_utils.optim2design)

class Distr(Distr_interface):
    """
    Intended to model a prior.
    Multivariate distribution over p-dim variables thetas.
    """

    def log_prob(self,thetas) -> torch.tensor:
        """
        calculate probabilities of thetas p(theta)

        params: 
            thetas: nxp

        returns: nx1
        """
        raise NotImplementedError()

    def sample(self,n) -> torch.tensor:
        """
        sample n samples from distribution

        params:
            n: integer

        returns: nxp
        """
        raise NotImplementedError()
    
    def predict_mle(self) -> torch.tensor:
        """
        returns the MLE of the distribution

        returns: 1xp
        """
        raise NotImplementedError()
    
class Conditional_distr(Distr_interface):
    """
    Intended to model the likelihood.
    Generic conditional distribution class over k-dim outcome y 
    given d-dim single experimental design design and p-dim parameters of interest theta, i.e. 
        p(y | design, theta)
    """

    def sample(self,design,thetas) -> torch.tensor:
        """
        sample from distribution for given single design and each given theta

        params:
            design: any shape
            thetas: txp

        returns: txk
        """
        raise NotImplementedError()
    
    def log_prob(self,ys,design,thetas) -> torch.tensor:
        """
        calculate density (pmf) of distribution

        params:
            ys: nx...
            design: any shape
            thetas: nxp or 1xp (latter  will be broadcasted)

        returns: nx1
        """
        raise NotImplementedError()

    def predict_mle(self,design,thetas) -> torch.tensor:
        """
        Calculate MLE of distribution given designs,thetas (i.e. predictions of designs under theta) 
        
        params:
            design: any shape
            thetas: nxp or 1xp (latter  will be broadcasted to nxp)
        
        returns: nx1
        """
        raise NotImplementedError()

class Conditional_Distr_multixi(Distr_interface):
    """ same as Conditional_distr but allowing multiple designs xi """

    def sample(self,designs,thetas) -> torch.tensor:
        """
        sample from distribution for given single design and each given theta

        params:
            designs: nx...
            theta: nx...

        returns: nxk
        """
        raise NotImplementedError()

    def log_prob(self,ys,designs,thetas) -> torch.tensor:
        """
        calculate density (pmf) of distribution

        params:
            ys: nx...
            designs: nx...
            thetas: nx...

        returns: nx1
        """
        raise NotImplementedError()
