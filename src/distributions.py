import torch

EPSILON = 1e-8

class Distr_interface:
    def __init__(self):
        # just to define instance variables, not sure how to do it properly
        self.params:dict = {}
        self.reparam_trick = False
        self.params_constraints = {}

class Distr(Distr_interface):
    """
    Multivariate distribution over p-dim variables thetas.
    Intended to model a prior distribution.
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
            c = thetas[1:].reshape((1,-1))
        elif len(thetas.shape) == 2:
            r = thetas[:,0]
            c = thetas[:,1:]
        else:
            raise ValueError()
        return c,r

    def log_prob(self, ys, design, thetas):

        if self.with_weights:
            weights = design[:,0]
            idx = (weights > torch.tensor(0.1)).detach()
            weights = weights[idx]
            design = design[idx,1:]
            ys = ys[:,idx]

        c,r = self._parse_params(thetas)
        n = ys.shape[0]
        if c.shape[0] == 1:
            r = r.repeat(ys.shape[0],1)
            c = c.repeat(ys.shape[0],1) 
        else:
            assert c.shape[0] == ys.shape[0]
        points = design.reshape((-1,2))
        npoints = points.shape[0]
        r = torch.repeat_interleave(r, repeats=npoints, dim=0)
        c = torch.repeat_interleave(c, repeats=npoints, dim=0)
        points = points.repeat(n,1)
        
        if c.shape[1] == 2:
            logits = torch.norm(points - c, dim=1) - r
        else: # with temperature
            logits = c[:,2] * (torch.norm(points - c[:,:2], dim=1) - r)
        bern_ps = torch.sigmoid(-logits).clamp(self.eps, 1-self.eps)
        log_prob_points = torch.distributions.Binomial(probs=bern_ps)\
            .log_prob(ys.reshape((-1)))
        log_prob = log_prob_points.reshape((n,npoints))
        if self.with_weights:
            log_prob = log_prob * weights.reshape((1,-1))
        log_prob = log_prob.sum(axis=1)
        return log_prob
    
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

        if c.shape[1] == 2:
            logits = torch.norm(points - c, dim=1) - r
        else: # with temperature
            logits = c[:,2] * (torch.norm(points - c[:,:2], dim=1) - r)
        bern_ps = torch.sigmoid(-logits).clamp(self.eps, 1-self.eps)
        samples = torch.distributions.Binomial(total_count=1,probs=bern_ps)\
            .sample(torch.tensor([1]))\
            .reshape((t,-1))
        return samples
    
    def predict_mle(self, design, thetas):
        c,r = self._parse_params(thetas)
        if self.with_weights:
            weights = design[:,0]
            design = design[:,1:]
        if c.shape[1] == 2:
            logits = torch.norm(design - c, dim=1) - r
        else: # with temperature
            logits = c[:,2] * (torch.norm(design - c[:,:2], dim=1) - r)
        bern_ps = torch.sigmoid(-logits).clamp(self.eps, 1-self.eps)
        return bern_ps


class Circle_prior(Distr):
    """
    models radius, center, and eventually temperature
    Temperature needs to be given extra (since positive constraint). 
    Std of temperature is in std.
    """

    def __init__(self, 
                 mu = torch.tensor([0.,0.,0.,0.]),
                 std = torch.tensor([1.,1.,1.,1.]),
    ):
        super().__init__()
        self.params = {}
        self.params['mu'] = mu
        self.params['std'] = std
        self.params_constraints['std'] = 'positive'

        self.reparam_trick = True
        self.min_log = torch.log(torch.tensor(EPSILON))

    def log_prob(self,thetas):
        r = thetas[:,0]
        c = thetas[:,1:3]
        temp = thetas[:,3]
        mu = self.params['mu']
        std = self.params['std']

        prob_r = torch.distributions.Normal(
            loc=mu[0], scale=std[0]
        ).log_prob(torch.log(r))

        prob_c = torch.distributions.MultivariateNormal(
            loc=mu[1:3], covariance_matrix=torch.diag(std[1:3]**2)
        ).log_prob(c)

        prob_temp = torch.distributions.Normal(
            loc=mu[3], scale=std[3]
        ).log_prob(torch.log(temp))

        res = (prob_c + prob_r + prob_temp).clamp(self.min_log)
        return res

    def sample(self, n):
        mu = self.params['mu']
        std = self.params['std']

        rvs = torch.randn((n,mu.shape[0])) * std.reshape((1,-1)) + mu
        return torch.hstack((
            torch.exp(rvs[:,0]).reshape((-1,1)), 
            rvs[:,1:3],
            torch.exp(rvs[:,3]).reshape((-1,1)) ))

    def predict_mle(self):
        mu = self.params['mu']
        out = torch.hstack((torch.exp(mu[0]), mu[1:3], torch.exp(mu[3]) ))\
            .reshape((1,-1))
        return out