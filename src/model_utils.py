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
from src.data_utils import Design_Network, Experimenter

from tqdm import tqdm
from itertools import product
from scipy.stats import bernoulli,multivariate_normal
from copy import deepcopy
from tqdm import tqdm
import warnings

from src.distributions import Conditional_Distr_multixi, Distr, Conditional_distr

EPSILON = 1e-8

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
    warnings.warn('Need to adjust to PCE, gradietn estimation might be incorrect!')

    # compute eig
    thetas = prior.sample(n_outer * (1+n_inner))
    thetas = thetas.reshape((n_outer,1+n_inner, thetas.shape[1]))
    ys = predictive.sample(design, thetas=thetas[:,0,:])
    log_prob = predictive.log_prob(
        torch.repeat_interleave(ys, repeats=(1+n_inner),dim=0),
        design,
        thetas=thetas.reshape((-1,thetas.shape[2])))
    log_prob = log_prob.reshape((n_outer,1+n_inner))
    lp_nom = log_prob[:,0]
    lp_denom = torch.logsumexp(log_prob[:,1:],axis=1) - torch.log(torch.tensor(n_inner))
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
    log_prob = predictive.log_prob(
        torch.repeat_interleave(ys.detach(), repeats=(1+n_inner),dim=0),
        design,
        thetas=thetas.reshape((-1,thetas.shape[2]))) # detached ys
    log_prob = log_prob.reshape((n_outer,1+n_inner))
    lp_nom = log_prob[:,0]
    lp_denom = torch.logsumexp(log_prob,axis=1) - torch.log(torch.tensor(1+n_inner))
    vals = lp_nom - lp_denom
    pce = vals.mean()

    # compute surrogate loss for optim
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
        log_denom = guide.log_prob(thetas)
        log_prior = prior.log_prob(thetas)
        if with_previous_prior:
            y = obs_data['y'][-1].reshape((1,-1))
            design = obs_data['design'][-1]
            log_predictive = predictive.log_prob(
                    y.repeat(nsamples_elbo,1), design, thetas=thetas)
        else:
            log_predictive = 0.0
            for y,design in zip(obs_data['y'],obs_data['design']):
                log_predictive += predictive.log_prob(
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


def train_DAD_design_policy(
        T,
        prior:Distr,
        likelihood:Conditional_Distr_multixi,
        design_network:Design_Network,
        n_simulations=100,
        batch_size=1,
        L = 50,
        print_every=1,
        verbose = 'plot'
):
    assert likelihood.reparam_trick

    # rollout likelihood over history
    def likelihood_history(history_xi, history_y, thetas):
        """
        history_xi: batch_size x T x design_dim
        history_y:  batch_size x T x y_dim
        thetas:     batch_size x 1+L x theta_dim

        returns: batch_size x 1+L
        """
        lp_prior = prior.log_prob(thetas.reshape((-1,thetas.shape[2])))\
            .reshape((batch_size,1+L))
        
        xi = history_xi.unsqueeze(1).repeat((1,1+L,1,1))
        y = history_y.unsqueeze(1).repeat((1,1+L,1,1))
        thetas = thetas.unsqueeze(2).repeat((1,1,T,1))
        lp_lik = likelihood.log_prob(
            y.reshape((-1,y.shape[3])),
            xi.reshape((-1,xi.shape[3])),
            thetas.reshape((-1,thetas.shape[3])))
        lp_lik = lp_lik.reshape((batch_size,1+L,T)).sum(axis=2)
        log_prob = lp_lik + lp_prior
        # xi = history_xi.reshape((-1,history_xi.shape[2]))
        # y = history_y.reshape((-1,history_y.shape[2]))
        # log_prob = likelihood.log_prob_multixi(
        #     torch.repeat_interleave(y, repeats=k, dim=0),
        #     torch.repeat_interleave(xi, repeats=k, dim=0),
        #     thetas.repeat((batch_size*T,1)),
        # )
        # log_prob = log_prob.reshape((batch_size,T,k)).sum(axis=1)

        return log_prob

    # prepare optimization
    for p in design_network.parameters():
        p.requires_grad = True
    optim = torch.optim.Adam(design_network.parameters(), lr=1e-4)
    vals = []

    # main loop
    for n in range(n_simulations):
        
        # generate batched history
        theta_0 = prior.sample(batch_size)
        history_xi = []
        history_y = []
        design_network.reset_buffer(batch_size)
        next_xi = torch.zeros((batch_size,design_network.design_dim))
        next_y = torch.zeros((batch_size,design_network.y_dim))
        for t in range(T):
            next_xi = design_network(next_xi, next_y)
            next_y = likelihood.sample(next_xi, theta_0)
            history_xi.append(next_xi.unsqueeze(1))
            history_y.append(next_y.unsqueeze(1))
        history_xi = torch.cat(history_xi, dim=1)  # batch_size x T x design_dim
        history_y = torch.cat(history_y, dim=1)    # batch_size x T x y_dim

        # compute loss
        thetas = prior.sample(batch_size * L).reshape((batch_size,L,-1))
        thetas = torch.cat((theta_0.unsqueeze(1), thetas), dim=1)  # batch_size x (1+L) x theta_dim
        log_prob = likelihood_history(
            history_xi, 
            history_y, 
            thetas
        ) # batch_size x (1+L) 
        lp_nom = log_prob[:,0]
        lp_denom = torch.logsumexp(log_prob,axis=1) - torch.log(torch.tensor(1+L))
        gL = lp_nom - lp_denom
        loss = -gL.mean()

        # param update
        optim.zero_grad()
        loss.backward()
        optim.step()
        if n%print_every == 0 and verbose == 'print':
            print(f'{n+1}/{n_simulations}: elbo = {-loss:.6f}')
        if verbose == 'plot':
            vals.append(-loss.detach().cpu().numpy())

    # verbose
    if verbose == 'plot':
        plt.figure(figsize=(2, 4))
        plt.plot(vals)
        plt.xlabel('optim step')
        plt.ylabel('EIG')
        plt.show()