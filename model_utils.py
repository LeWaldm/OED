import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.metrics import r2_score
import seaborn as sns
from scipy.stats import multivariate_normal as mvn
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull
import torch
from data_utils import Experimenter

from tqdm import tqdm
from itertools import product
from scipy.stats import bernoulli,multivariate_normal
from copy import deepcopy
from tqdm import tqdm
import warnings

from distributions import Distr, Conditional_distr

EPSILON = 1e-8


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
        reinforce=False,
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

    # compute surrogate loss for optim with REINFORCE (minus for maximization)
    log_probs = predictive.log_probs(
        torch.repeat_interleave(ys.detach(), repeats=(1+n_inner),dim=0),
        design,
        thetas=thetas.reshape((-1,thetas.shape[2]))) # detached ys
    log_probs = log_probs.reshape((n_outer,1+n_inner))
    lp_nom = log_probs[:,0]
    lp_denom = torch.logsumexp(log_probs,axis=1) - torch.log(torch.tensor(1+n_inner))
    vals = lp_nom - lp_denom
    
    if reinforce:
        surrogate_loss = -torch.mean(vals + vals.detach().clone() * lp_nom)
    else:
        surrogate_loss = -pce
    return pce, surrogate_loss

def variational_inference(
        obs_data:dict,
        prior:Distr,
        predictive:Conditional_distr,
        guide:Distr,
        n_steps_varinf = 500,
        nsamples_elbo = 30,
        print_every = 50,
        with_previous_prior=True,
        verbose='print',
        **kwargs):
    """
    Standard variational inference to find posterior using elbo.

    Variational distribution is named 'guide' as in pyro.
    """

    assert guide.reparam_trick == True
    assert verbose in [None, 'print', 'plot']

    # prepare optimization
    optim_params = design2optim(guide.params, guide.params_constraints)
    for p in optim_params.values():
        p.requires_grad = True
    optim = torch.optim.Adam(optim_params.values(), lr=1e-2)
    vals_elbo = []

    # optimization
    for s in tqdm(range(n_steps_varinf)):

        # manually update guide params
        guide.params = optim2design(optim_params, guide.params_constraints)

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
                    y.repeat(nsamples_elbo,1), design, thetas=thetas)
        log_numerator = log_predictive + log_prior
        neg_elbo = -torch.mean(log_numerator - log_denom) # negative s.t. minimization

        # torch step
        optim.zero_grad()
        neg_elbo.backward()
        optim.step()

        if s%print_every == 0 and verbose == 'print':
            print(f'{s+1}/{n_steps_varinf}: elbo = {-neg_elbo:.6f}')
        elif verbose == 'plot':
            vals_elbo.append(-neg_elbo.detach().cpu().numpy())

    # clean up optimization
    for k,v in optim_params.items():
        optim_params[k] = v.detach().clone()
    guide.params = optim2design(optim_params, guide.params_constraints)

    # plotting
    if verbose == 'plot':
        plt.figure(figsize=(2, 4))
        plt.plot(vals_elbo)
        plt.xlabel('optim step')
        plt.ylabel('ELBO')
        plt.show()

    return guide


def optim2design(optim_params, optim_args):
    design_params = {}
    for k,v in optim_params.items():
        if not k in optim_args:
            design_params[k] = optim_params[k]
        elif optim_args[k] == 'positive':
            design_params[k] = torch.exp(optim_params[k])
        elif optim_args[k][0] == 'minmax': # assumes ('minmax', min, max)
            design_params[k] = torch.sigmoid(optim_params[k]) \
                * (optim_args[k][2] - optim_args[k][1]) + optim_args[k][1]
        else:
            raise NotImplementedError(f'Unknown value in optim_args: {v}')
    return design_params

def design2optim(design_params, optim_args):
    optim_params = {}
    for k,v in design_params.items():
        if not k in optim_args:
            optim_params[k] = design_params[k]
        elif optim_args[k] == 'positive':    
            optim_params[k] = torch.log(design_params[k].clamp(EPSILON))
        elif optim_args[k][0] == 'minmax':
            optim_params[k] = torch.logit((design_params[k] - optim_args[k][1]) \
                / (optim_args[k][2] - optim_args[k][1]))
        else:
            raise NotImplementedError(f'Unknown value in optim_args: {v}')
        optim_params[k] = optim_params[k].detach().clone()
    return optim_params

def eig_cont_optim(
        experiment:Experimenter,
        prior,
        predictive,
        eig_method,
        optim_args = {},    
        n_steps_optim=500,  
        print_every=50,
        initial_design_params = None,
        verbose=None,
        **kwargs
):
    """
    Find optimal design by maximizing EIG w.r.t. continuous design_params
    """
    assert eig_method in [eig_PCE] # check that lower bound on EIG
    assert verbose in [None, 'print', 'plot']
    check_grads = True
    grad_update = True
    if not grad_update:
        warnings.warn('grad_update=False, so no optimization is performed')

    # prepare optim
    if initial_design_params is None:
        initial_design_params = experiment.get_initial_design_params()
    optim_params = design2optim(initial_design_params,optim_args)
    for k,v in optim_params.items():
        v.requires_grad = True

    optim = torch.optim.SGD(optim_params.values(), lr=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=1e-2, total_steps=n_steps_optim)
    eig_vals = []

    # main loop
    grads_optim = {k:[] for k in optim_params.keys()}
    grads_design = {k:[] for k in optim_params.keys()}
    for s in tqdm(range(n_steps_optim)):
        design_params = optim2design(optim_params,optim_args)
        _,design = experiment.params2design(design_params)
        eig,loss = eig_method(design,prior,predictive, **kwargs)
        optim.zero_grad()
        loss.backward()

        # check grads
        if check_grads:
            optim_params_grads = {}
            for k,v in optim_params.items():
                grad = v.grad.detach().clone()
                if not grad_update:
                    v.grad = None
                optim_params_grads[k] = grad
            design_params_grads = optim2design(optim_params_grads, optim_args)
            for k in design_params_grads.keys():
                grads_design[k].append(design_params_grads[k].detach().cpu().numpy())
                grads_optim[k].append(optim_params_grads[k].detach().cpu().numpy())
        if grad_update:
            optim.step()
            lr_scheduler.step()
            # print(f'{s+1}/{n_steps_optim}, design_params: {design_params}')
        if s % print_every == 0 and verbose=='print':
            print(f'{s+1}/{n_steps_optim}, eig: {eig}')
        elif verbose == 'plot':
            eig_vals.append(eig.detach().cpu().numpy())

    # plot grads
    if check_grads:
        plt.subplots(nrows=len(grads_design), ncols=2, figsize=(4, 6))
        plt.suptitle('grads')
        for i,k in enumerate(grads_design.keys()):
            plt.subplot(len(grads_design),2,2*i+1)
            plt.plot(grads_design[k])
            plt.ylabel(f'Design: {k}')
            plt.subplot(len(grads_optim),2,2*i+2)
            plt.plot(grads_optim[k])
            plt.ylabel(f'Optim: {k}')

    # clean up optim
    for v in design_params.values():
        v = v.detach().clone()

    # plot eig_vals
    if verbose == 'plot':
        plt.figure(figsize = (2, 4))
        plt.plot(eig_vals)
        plt.xlabel('optim step')
        plt.ylabel('EIG')
        plt.show()

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

    intercept = []
    slope = []
    eigs = []

    for (design_params,design) in tqdm(candidate_designs):
        intercept.append(design_params['intercept'])   
        slope.append(design_params['slope'])   
        curr_eig, _ = eig_calc_method(design, prior, predictive, **kwargs)
        # nmc, design_params = eig_NMC(
        #     design, prior, predictive, **kwargs
        # )
        # print(f'PCE: {curr_eig}, NMC: {nmc}')
        if curr_eig > best_eig:
            best_eig = curr_eig
            best_design_params = design_params
        eigs.append(curr_eig.detach().cpu().numpy())

    # plotting
    if True:
        plt.figure(figsize=(2, 4))
        plt.scatter(intercept, slope, c=eigs)
        plt.xlabel('intercept')
        plt.ylabel('slope')
        plt.colorbar()
        plt.show()

    return best_eig, best_design_params


def OED_fit(
        experiment:Experimenter,
        prior:Distr,
        predictive:Conditional_distr,
        n_designs,
        eig_calc_method,
        eig_optim_method,
        with_previous_prior=True,
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