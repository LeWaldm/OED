from matplotlib.pylab import f
from matplotlib.rcsetup import validate_stringlist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, pairwise_distances, roc_auc_score
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.metrics import r2_score
import seaborn as sns
from scipy.stats import multivariate_normal as mvn
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull
import torch
# from DiffCBED.strategies.multi_perturbation_ed.represent import posterior
from data_utils import Experimenter

import matplotlib
from tqdm import tqdm
from itertools import product
from scipy.stats import bernoulli,multivariate_normal
from copy import deepcopy

from distributions import Distr, Conditional_distr


def eig_NMC(
        design,
        prior:Distr,
        predictive:Conditional_distr,
        n_outer=100,
        n_inner=50,
        **kwargs
):
    """
    Computes Expected Information Gain (EIG) with Nested Monte Carlo (NMC)
    which is unbiased.
    """

    # compute eig
    thetas = prior.sample(n_outer * (1+n_inner))
    thetas = thetas.reshape((n_outer,1+n_inner, thetas.shape[1]))
    ys = predictive.sample(design, thetas=thetas[:,0,:])
    log_probs = predictive.log_probs(
        torch.repeat_interleave(ys, repeats=(1+n_inner),dim=0),
        design,
        thetas=thetas.reshape((-1,thetas.shape[2])))
    log_probs = log_probs.reshape((n_outer,1+n_inner))
    lp_nom = log_probs[:,0]
    lp_denom = torch.logsumexp(log_probs[:,1:],axis=1) - torch.log(torch.tensor(n_inner))
    vals = lp_nom - lp_denom
    eig = vals.mean()

    # compute eig for optim with REINFORCE (minus for maximization)
    loss_optim = -torch.mean(vals + vals.detach() * lp_nom)
    return eig, loss_optim

def eig_PCE(
        design,
        prior:Distr,
        predictive:Conditional_distr,
        n_outer=100,
        n_inner=10,
        **kwargs
):
    """
    the Prior Contrastive Estimator is a lower bound on the EIG 
    which is a lower bound
    """

    # compute pce
    thetas = prior.sample(n_outer * (1+n_inner))
    thetas = thetas.reshape((n_outer,1+n_inner, thetas.shape[1]))
    ys = predictive.sample(design, thetas=thetas[:,0,:])
    log_probs = predictive.log_probs(
        torch.repeat_interleave(ys, repeats=(1+n_inner),dim=0),
        design,
        thetas=thetas.reshape((-1,thetas.shape[2])))
    log_probs = log_probs.reshape((n_outer,1+n_inner))
    lp_nom = log_probs[:,0]
    lp_denom = torch.logsumexp(log_probs,axis=1) - torch.log(torch.tensor(1+n_inner))
    vals = lp_nom - lp_denom
    pce = vals.mean()

    # compute pce for optim with REINFORCE (minus for maximization)
    loss_optim = -torch.mean(vals + vals.detach() * lp_nom)
    return pce, loss_optim

def eig_varNMC(
        e,
        prior:Distr,
        predictive:Conditional_distr,

        n_steps_varinf = 100,

):
    """
    variational NMC is an upper bound on the EIG which is 
    asymptotically (for nsamples->infty) unbiased
    """

    # learn proposal distribution
    propsal = None

    # final nmc estimator
    eig = None

    raise NotImplementedError()
    return eig

def eig_ACE():
    """
    the Adaptive Contrastive Estimator is a lower bound on the EIG 
    which is asymptotically (for nsamples->infty) unbiased

    TODO: should be very simple adaptation of eig_varNMC
    """
    raise NotImplementedError()



def variational_inference(
        obs_data:dict,
        prior:Distr,
        predictive:Conditional_distr,
        guide:Distr,
        n_steps_varinf = 500,
        nsamples_elbo = 30,
        print_every = 50,
        with_previous_prior=True,
        **kwargs):
    """
    Standard variational inference to find posterior using elbo.

    Variational distribution is named 'guide' as in pyro.
    """

    assert guide.reparam_trick == True

    # actual optimization
    params = guide.params.values()
    for p in params:
        p.requires_grad = True
    optim = torch.optim.Adam(params, lr=1e-2)
    for s in range(n_steps_varinf):

        # compute (negative) elbo
        thetas = guide.sample(nsamples_elbo)
        log_denom = guide.log_probs(thetas)
        log_prior = prior.log_probs(thetas)
        if with_previous_prior:
            y = obs_data['y'][-1].reshape((1,-1))
            design = obs_data['design'][-1]
            log_predictive = predictive.log_probs(
                    y.repeat(nsamples_elbo,1), design, thetas=thetas)
        else:
            log_predictive = 0.0
            for y,design in zip(obs_data['y'],obs_data['design']):
                log_predictive += predictive.log_probs(
                    y.repeat(nsamples_elbo), design, thetas=thetas)
        log_numerator = log_predictive + log_prior
        neg_elbo = -torch.mean(log_numerator - log_denom) # negative s.t. minimization

        # torch step
        optim.zero_grad()
        neg_elbo.backward()
        optim.step()

        if s%print_every == 0:
            print(f'{s+1}/{n_steps_varinf}: elbo = {-neg_elbo:.6f}')

    return guide


def eig_cont_optim(
        experiment:Experimenter,
        prior,
        predictive,
        eig_method,
        optim_args = {},    
        n_steps_optim=500,  
        print_every=50,
        initial_design_params = None,
        **kwargs
):
    """
    Find optimal design by maximizing EIG w.r.t. continuous design_params
    """
    assert eig_method in [eig_PCE] # check that lower bound on EIG

    # prepare optim_params for potentially constrained optimization
    def optim2design(optim_params):
        design_params = {}
        for k,v in optim_params.items():
            if not k in optim_args:
                design_params[k] = v
            elif optim_args[k][0] == 'minmax': # assumes ('minmax', min, max)
                design_params[k] = torch.sigmoid(optim_params[k]) \
                    * (optim_args[k][2] - optim_args[k][1]) + optim_args[k][1]
            else:
                raise NotImplementedError()
        return design_params
    def design2optim(design_params):
        optim_params = {}
        for k,v in design_params.items():
            if not k in optim_args:
                optim_params[k] = v
            elif optim_args[k][0] == 'minmax':
                optim_params[k] = torch.logit((design_params[k] - optim_args[k][1]) \
                    / (optim_args[k][2] - optim_args[k][1]))
            else:
                raise NotImplementedError()
            optim_params[k] = optim_params[k].detach().clone()
        return optim_params
    if initial_design_params is None:
        initial_design_params = experiment.get_initial_design_params()
    optim_params = design2optim(initial_design_params)

    # prepare optimizer
    for v in optim_params.values():
        v.requires_grad = True
    optim = torch.optim.Adam(optim_params.values(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=1e-2, total_steps=n_steps_optim)

    # main loop
    for s in range(n_steps_optim):
        design_params = optim2design(optim_params)
        _,design = experiment.params2design(design_params)
        eig,loss = eig_method(design,prior,predictive, **kwargs)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if s % print_every == 0:
            print(f'{s+1}/{n_steps_optim}, eig: {eig}')
        lr_scheduler.step()

    for v in design_params.values():
        v = v.detach().clone()
    return eig, design_params


def eig_discrete_optim(
        experiment:Experimenter,
        prior:Distr,
        predictive:Conditional_distr,
        eig_calc_method,
        **kwargs
):
    """
    Find "optimal" design by computing EIG for all designs in discrete design space
    and choosing the best one (i.e. highest EIG)
    """
    
    candidate_designs = experiment.get_candidate_designs()
    best_eig = -torch.inf
    best_design_params = None

    for (design_params,design) in tqdm(candidate_designs):
        curr_eig, _ = eig_calc_method(design, prior, predictive, **kwargs)
        if curr_eig > best_eig:
            best_eig = curr_eig
            best_design_params = design_params

    return best_eig, best_design_params


def OED_fit(
        experiment:Experimenter,
        prior:Distr,
        predictive:Conditional_distr,
        n_designs,
        eig_calc_method,
        eig_optim_method,
        with_previous_prior=True,
        verbose=True,
        **kwargs
):
    # prepare metrics
    best_eigs = []
    metrics = [roc_auc_score, f1_score]
    metric_is_binary = [False, True]
    metric_names = ['roc_auc','f1']
    metric_values = [[] for _ in range(len(metrics))]
    design_params = None # initial design_params, will be specified by experiment

    # main loop
    for iter in range(n_designs):
        print(f'------- Iteration {iter+1}/{n_designs} -------')

        # find best design
        eig, design_params = eig_optim_method(
            experiment,prior,predictive,eig_calc_method, 
            initial_design_params = design_params,
            **kwargs
        )
        experiment.execute_design(design_params)
        best_eigs.append(eig)
        print(f'Executed design; best_eig: {eig}, design_params: {design_params}')

        # fit "posterior"
        obs_data = experiment.get_obs_data()
        for i in obs_data['design_params']:
            print(i)
        guide = deepcopy(prior)
        posterior = variational_inference(
            obs_data,prior,predictive,guide,with_previous_prior=with_previous_prior,**kwargs)
        
        # evaluate with MAP params
        y,design = experiment.get_eval_data()
        thetas_MAP = posterior.predict_mle()
        pred_probs = predictive.predict_mle(design, thetas_MAP).detach().cpu().numpy()
        preds_int = (pred_probs > 0.5).astype(np.int8).reshape((-1))
        for i in range(len(metrics)):
            if metric_is_binary[i]:
                val = metrics[i](y,preds_int)
            else:
                val = metrics[i](y,pred_probs)
            metric_values[i].append(val)

        # plotting
        if verbose:
            print(f'thetas_MAP: {thetas_MAP}')
            experiment.verbose_designs(
                design_eval=design,
                pred_probs=pred_probs,
                metric_values=metric_values,
                metric_names=metric_names
            )

        # prepare next loop
        if with_previous_prior:
            prior = posterior

    return experiment, thetas_MAP, metric_values, metric_names