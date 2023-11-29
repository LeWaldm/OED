from copy import deepcopy
from multiprocessing import log_to_stderr
from uu import Error
import torch
import numpy as np
from scipy.stats import bernoulli,multivariate_normal

class Distr_interface:
    def __init__(self):
        # just to define instance variables, not sure how to do it properly
        self.params:dict = {}
        self.is_torch = False
        self.reparam_trick = False

    def set_torch(self,to_torch,device='cpu'):
        if to_torch == self.is_torch:
            return
        if to_torch:
            for k,v in self.params.items():
                self.params[k] = torch.from_numpy(v).to(device)
        else:
            for k,v in self.params.items():
                self.params[k] = v.detach().cpu().numpy()
        self.is_torch = to_torch

class Distr(Distr_interface):
    """
    Multivariate distribution over p-dim variables thetas.
    Intended to model a prior distribution.
    """

    def probs(self,theta):
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

class Conditional_distr(Distr_interface):
    """
    Generic conditional distribution class over 1-dim outcome y 
    given d-dim experimental design e and p-dim parameters of interest theta, i.e. 
        p(y | e, theta)

    Thetas denote optional parameters of the distribution to use,
    if empty use paremters of self.
    """

    def sample(self,xs,thetas=None):
        """
        sample from distribution for each given xs and each given theta

        params:
            xs: nxd
            thetas: txp or 1xp (latter  will be broadcasted)

        returns: nxt
        """
        raise NotImplementedError()
    
    def probs(self,ys,xs,thetas=None):
        """
        calculate density (pmf) of distribution for inputs

        params:
            ys: nx1
            xs: nxd
            thetas: nxp or 1xp (latter  will be broadcasted)

        returns: nx1
        """
        raise NotImplementedError()
    
    def predict(self,xs,thetas=None):
        """
        Calculate MLE of distribution given xs,thetas (i.e. predictions of xs under theta) 
        
        params:
            xs: nxd
            thetas: nxp or 1xp (latter  will be broadcasted to nxp)
        
        returns: nx1
        """
        raise NotImplementedError()

class Circle_predictive(Conditional_distr):
    """
    d = 2, p = 3
    """

    def __init__(self, 
                 r = np.array([1]), 
                 c = np.array([0,0])):
        super().__init__()
        self.params = {}
        self.params['r'] = r
        self.params['c'] = c
        self.reparam_trick = False
        self.is_torch = False
    
    def _parse_params(self,thetas):
        if thetas is None:
            c, r = self.params['c'], self.params['r']
        else:
            if len(thetas.shape) == 1:
                r = thetas[0].reshape((1,1))
                c = thetas[1:3].reshape((1,-1))
            elif len(thetas.shape) == 2:
                r = thetas[:,0]
                c = thetas[:,1:3]
            else:
                raise ValueError()
        return c,r

    def probs(self, ys, xs, thetas=None):
        assert ys.shape[0] == xs.shape[0]
        switch_to_torch = not self.is_torch
        if switch_to_torch:
            self.set_torch(True,device='cpu')
            xs = torch.from_numpy(xs)
            ys = torch.from_numpy(ys)
            if thetas is not None:
                thetas = torch.from_numpy(thetas)
        c,r = self._parse_params(thetas)

        bern_ps = 1 / (1 + torch.exp( torch.norm(xs - c, dim=1) - r))
        log_probs = torch.distributions.Binomial(probs=bern_ps).log_prob(ys.reshape((-1)))
        probs = torch.exp(log_probs)

        if switch_to_torch:
            self.set_torch(False)
            probs = probs.cpu().detach().numpy()
        return probs
    
    def sample(self, xs, thetas=None):
        switch_to_torch = not self.is_torch
        if switch_to_torch:
            self.set_torch(True, device='cpu')
            xs = torch.from_numpy(xs)
            if thetas is not None:
                thetas = torch.from_numpy(thetas)
        c,r = self._parse_params(thetas)

        n = xs.shape[0]
        t = thetas.shape[0]
        c = c.repeat((n,1))
        r = r.repeat((n,))
        xs = xs.repeat_interleave(repeats=t,dim=0)

        bern_ps = 1 / (1 + torch.exp( torch.norm(xs - c, dim=1) - r))
        samples = torch.distributions.Binomial(probs=bern_ps)\
            .sample(torch.tensor([1]))\
            .reshape((n,t))
        
        if switch_to_torch:
            self.set_torch(False)
            samples = samples.detach().cpu().numpy()
        return samples
    
    def predict(self, xs, thetas=None):
        c,r = self._parse_params(thetas)
        if self.is_torch:
            return 1 / (1 + torch.exp( torch.norm(xs - c, dim=1) - r))
        else:
            return 1 / (1 + np.exp( np.linalg.norm(xs - c, axis=1) - r))
    
# class Circle_prior(Distr):
#     """
#     models 3 pw independent rvs, first is exp(snormal) other two snormal
#     """

#     def __init__(self, 
#                  mu = np.array([0.,0.,0.]), 
#                  std = np.array([1.,1.,1.])):
#         super().__init__()
#         self.params = {}
#         self.params['mu'] = mu
#         self.params['std'] = std
#         self.reparam_trick = True
#         self.is_torch =  False

#     def probs(self,thetas):
#         r = thetas[:,0].reshape((-1,1))
#         c = thetas[:,1:]
#         mu = self.params['mu']
#         std = self.params['std']
#         if self.is_torch:
#             prob_c = torch.exp(torch.distributions.MultivariateNormal(
#                 loc=mu[1:], covariance_matrix=torch.diag(std[1:]**2)
#             ).log_prob(c))

#             prob_r = torch.exp(torch.distributions.Normal(
#                 loc=mu[0], scale=std[0]
#             ).log_prob(r.log()))
#         else:
#             prob_c = multivariate_normal.pdf(
#                 c, mean=mu[1:], cov= np.diag(std[1:]**2))
#             prob_r = multivariate_normal.pdf(
#                 np.log(r), mean=mu[0].reshape((1,)), cov=(std[0]**2).reshape((1,1)) )
#         return prob_c * prob_r

#     def sample(self, n):
#         mu = self.params['mu']
#         std = self.params['std']
#         if self.is_torch:
#             rvs = torch.randn((n,3)) @ std.reshape((-1,1)) + mu
#             return torch.hstack( (torch.exp(rvs[:,0]).reshape((-1,1)), rvs[:,1:]) )
#         else:
#             rvs = np.random.normal(size=(n,3)) @ std.reshape((-1,1)) + mu
#             return np.hstack([np.exp(rvs[:,0]).reshape((-1,1)), rvs[:,1:]])


class Circle_prior_log(Distr):
    """
    models 3 pw independent rvs, first is exp(snormal) other two snormal
    """

    def __init__(self, 
                 mu = np.array([0.,0.,0.]), 
                 log_std = np.array([1.,1.,1.])):
        super().__init__()
        self.params = {}
        self.params['mu'] = mu
        self.params['log_std'] = log_std
        self.reparam_trick = True
        self.is_torch = False

    def probs(self,thetas):
        r = thetas[:,0].reshape((-1,1))
        c = thetas[:,1:]
        mu = self.params['mu']
        log_std = self.params['log_std']
        eps = 1e-8
        if self.is_torch:
            std = torch.exp(log_std)
            prob_c = torch.exp(torch.distributions.MultivariateNormal(
                loc=mu[1:], covariance_matrix=torch.diag(std[1:]**2)
            ).log_prob(c))

            prob_r = torch.exp(torch.distributions.Normal(
                loc=mu[0], scale=std[0]**2
            ).log_prob(torch.log(r))).reshape((-1))
            res = (prob_c * prob_r).clamp(eps, 1-eps)
        else:
            std = np.exp(log_std)
            prob_c = multivariate_normal.pdf(
                c, mean=mu[1:], cov= np.diag(std[1:]**2))
            prob_r = multivariate_normal.pdf(
                np.log(r), mean=mu[0].reshape((1,)), cov=(std[0]**2).reshape((1,1)) )
            res = np.clip(prob_c * prob_r, eps, 1-eps)
        return res

    def sample(self, n):
        mu = self.params['mu']
        log_std = self.params['log_std']
        if self.is_torch:
            std = torch.exp(log_std)
            rvs = torch.randn((n,3)) @ std.reshape((-1,1)) + mu
            return torch.hstack( (torch.exp(rvs[:,0]).reshape((-1,1)), rvs[:,1:]) )
        else:
            std = np.exp(log_std)
            rvs = np.random.normal(size=(n,3)) @ std.reshape((-1,1)) + mu
            return np.hstack([np.exp(rvs[:,0]).reshape((-1,1)), rvs[:,1:]])
        

# class DiagonalGauss_guide(Distr):
#     """
#     Guide Gaussian distribution with diagonal covariance matrix
#     """
#     def __init__(self,d):
#         self.d = d
#         self.mu = torch.zeros(d)
#         self.std = torch.ones(d).reshape((1,-1))
#         self.params = [self.mu, self.std]
#         self.reparam_trick = True

#     def sample(self,n):
#         rvs = torch.normal(size=(n,self.d))
#         return rvs * self.std + self.mu
    
class Constraint_wrapper(Distr):
    """
    Possible constraints: 'positive'
    """

    def __init__(self, distr:Distr, constraints={}) -> None:
        """
        constraints: dict with key matching key in distr.params and value
            the constraint to ensure ('positive')
        """
        self.distr = distr
        self.params = deepcopy(distr.params)
        switch_to_numpy = distr.is_torch
        if switch_to_numpy:
            distr.set_torch(False)

        for k,v in constraints.items():
            if not k in self.params:
                raise ValueError('Parameter not found in distr.')
            if v == 'positive':
                self.params[k] = np.log(self.params[k])

        if switch_to_numpy:
            distr.set_torch(True)

        raise NotImplementedError()
    
    def probs(self, thetas):
        raise NotImplementedError() 

    def sample(self,n):
        return self.distr.sample(n)