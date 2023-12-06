from copy import deepcopy
from multiprocessing import log_to_stderr
from uu import Error
from pandas import isna
import torch
import numpy as np
from scipy.stats import bernoulli,multivariate_normal

class Distr_interface:
    def __init__(self):
        # just to define instance variables, not sure how to do it properly
        self.params:dict = {}
        self.is_torch = False
        self.reparam_trick = False

class Distr(Distr_interface):
    """
    Multivariate distribution over p-dim variables thetas.
    Intended to model a prior distribution.
    """

    def log_probs(self,thetas):
        """
        calculate probabilities of thetas p(theta)

        params: 
            thetas: nxp

        returns: nx1
        """
        raise NotImplementedError()

    def sample(self,n):
        """
        sample n samples from distribution

        params:
            n: integer

        returns: nxp
        """
        raise NotImplementedError()
    
    def predict_mle(self):
        """
        returns the MLE of the distribution

        returns: 1xp
        """
        raise NotImplementedError()

class Conditional_distr(Distr_interface):
    """
    Generic conditional distribution class over k-dim outcome y 
    given d-dim experimental design design and p-dim parameters of interest theta, i.e. 
        p(y | design, theta)

    Thetas denote optional parameters of the distribution to use,
    if empty use paremters of self.
    """

    def sample(self,design,thetas=None):
        """
        sample from distribution for given design and each given theta

        params:
            design: d
            thetas: txp

        returns: txk
        """
        raise NotImplementedError()
    
    def log_probs(self,ys,design,thetas=None):
        """
        calculate density (pmf) of distribution for inputs

        params:
            ys: nxk
            design: d
            thetas: nxp or 1xp (latter  will be broadcasted)

        returns: nx1
        """
        raise NotImplementedError()
    
    def predict_mle(self,designs,thetas=None):
        """
        Calculate MLE of distribution given designs,thetas (i.e. predictions of designs under theta) 
        
        params:
            designs: nxd
            thetas: nxp or 1xp (latter  will be broadcasted to nxp)
        
        returns: nx1
        """
        raise NotImplementedError()

class Circle_predictive(Conditional_distr):
    """
    designs: nx(npoints*2)
    ys: nx(npoints)
    """

    def __init__(self, with_weights=False):
        super().__init__()
        self.params = {}
        self.reparam_trick = False
        self.eps = 1e-8
        self.with_weights = with_weights
    
    def _parse_params(self,thetas):

        if len(thetas.shape) == 1:
            r = thetas[0].reshape((1,1))
            c = thetas[1:3].reshape((1,-1))
        elif len(thetas.shape) == 2:
            r = thetas[:,0]
            c = thetas[:,1:3]
        else:
            raise ValueError()
        return c,r

    def log_probs(self, ys, design, thetas):

        c,r = self._parse_params(thetas)
        n = ys.shape[0]
        if c.shape[0] == 1:
            r = r.repeat(ys.shape[0],1)
            c = c.repeat(ys.shape[0],1) 
        else:
            assert c.shape[0] == ys.shape[0]
        if self.with_weights:
            weights = design[:,0]
            design = design[:,1:]
        points = design.reshape((-1,2))
        npoints = points.shape[0]
        r = torch.repeat_interleave(r, repeats=npoints, dim=0)
        c = torch.repeat_interleave(c, repeats=npoints, dim=0)
        points = points.repeat(n,1)
        
        logits = torch.norm(points - c, dim=1) - r
        bern_ps = torch.sigmoid(-logits).clamp(self.eps, 1-self.eps)
        if torch.any(bern_ps.isnan()):
            print('nan in bern_ps')
        log_probs_points = torch.distributions.Binomial(probs=bern_ps)\
            .log_prob(ys.reshape((-1)))
        log_probs = log_probs_points.reshape((n,npoints))
        if self.with_weights:
            log_probs = log_probs * weights.reshape((1,-1))
        log_probs = log_probs.sum(axis=1)
        return log_probs
    
    def sample(self, design, thetas):

        c,r = self._parse_params(thetas)
        t = c.shape[0]
        if self.with_weights:
            weights = design[:,0]
            design = design[:,1:]

        points = design.reshape((-1,2))
        npoints = points.shape[0]
        points = points.repeat(t,1)
        c = torch.repeat_interleave(c,npoints,0)
        r = torch.repeat_interleave(r,npoints,0)

        logits = torch.norm(points - c, dim=1) - r
        bern_ps = torch.sigmoid(-logits).clamp(self.eps, 1-self.eps)
        samples = torch.distributions.Binomial(total_count=1,probs=bern_ps)\
            .sample(torch.tensor([1]))\
            .reshape((t,-1))
        
        return samples
    
    def predict_mle(self, designs, thetas):
        c,r = self._parse_params(thetas)
        if self.with_weights:
            weights = designs[:,0]
            designs = designs[:,1:]
        logits = torch.norm(designs - c, dim=1) - r
        bern_ps = torch.sigmoid(-logits).clamp(self.eps, 1-self.eps)
        return bern_ps


class Circle_prior_log(Distr):
    """
    models 3 pw independent rvs, first is exp(snormal) other two snormal.
    Standard deviation is given as log.
    """

    def __init__(self, 
                 mu = np.array([0.,0.,0.]), 
                 log_std = np.array([1.,1.,1.])):
        super().__init__()
        self.params = {}
        self.params['mu'] = mu
        self.params['log_std'] = log_std
        self.reparam_trick = True
        self.min_log = torch.log(torch.tensor(1e-8))

    def log_probs(self,thetas):
        r = thetas[:,0]
        c = thetas[:,1:]
        mu = self.params['mu']
        std = torch.exp(self.params['log_std'])

        prob_r = torch.distributions.Normal(
            loc=mu[0], scale=std[0]
        ).log_prob(torch.log(r))

        prob_c = torch.distributions.MultivariateNormal(
            loc=mu[1:], covariance_matrix=torch.diag(std[1:]**2)
        ).log_prob(c)

        res = (prob_c + prob_r)
        res_clamp = res.clamp(self.min_log)
        # if torch.any(res != res_clamp):
        #     print('Clamped log_probs')
        return res_clamp

    def sample(self, n):
        mu = self.params['mu']
        std = torch.exp(self.params['log_std'])

        rvs = torch.randn((n,3)) * std.reshape((1,-1)) + mu
        return torch.hstack( (torch.exp(rvs[:,0]).reshape((-1,1)), rvs[:,1:]) )

    def predict_mle(self):
        return self.params['mu'].reshape((1,-1))