import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, r2_score
import torch
from src.data_utils import Design_Network, Experimenter

from tqdm import tqdm
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
    return eig_PCE(design,prior,predictive,n_outer,n_inner,is_NMC=True,**kwargs)

def eig_PCE(
        design,
        prior:Distr,
        predictive:Conditional_distr,
        n_outer = 100,
        n_inner = 10,
        reinforce = False,
        is_NMC = False,
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
    if is_NMC:
        lp_denom = torch.logsumexp(log_prob[:,1:],axis=1) - torch.log(torch.tensor(n_inner))
    else:
        lp_denom = torch.logsumexp(log_prob,axis=1) - torch.log(torch.tensor(1+n_inner))
    vals = lp_nom - lp_denom
    pce = vals.mean()

    # compute surrogate loss for optim
    if reinforce:
        surrogate_loss = -torch.mean(vals + vals.detach().clone() * lp_nom)
    else:
        surrogate_loss = -pce
    return pce, surrogate_loss


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

def variational_inference(
        obs_data:dict,
        prior:Distr,
        predictive:Conditional_distr,
        guide:Distr,
        n_steps = 500,
        nsamples_elbo = 30,
        print_every = 50,
        with_previous_prior=True,
        verbose=False,
        **kwargs):
    """
    Standard variational inference to find posterior using elbo.

    Variational distribution is named 'guide' as in pyro.
    """

    assert guide.reparam_trick == True

    # prepare optimization
    optim_params = design2optim(guide.params, guide.params_constraints)
    for p in optim_params.values():
        p.requires_grad = True
    optim = torch.optim.Adam(optim_params.values(), lr=1e-2)
    vals_elbo = []

    # optimization
    for s in (bar := tqdm(range(n_steps))):

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

        if s%print_every == 0:
            bar.set_description(f'elbo = {-neg_elbo:.6f}')
        if verbose:
            vals_elbo.append(-neg_elbo.detach().cpu().numpy())

    # clean up optimization
    for k,v in optim_params.items():
        optim_params[k] = v.detach().clone()
    guide.params = optim2design(optim_params, guide.params_constraints)

    # plotting
    if verbose:
        plt.figure(figsize=(2, 4))
        plt.plot(vals_elbo)
        plt.xlabel('optim step')
        plt.ylabel('ELBO')
        plt.show()

    return guide

def eig_cont_optim(
        experiment:Experimenter,
        prior,
        predictive,
        eig_method,
        optim_args = {},    
        n_steps=500,  
        print_every=50,
        initial_design_params = None,
        verbose = True,
        check_grads = False,  # no gradient update and plot gradients for each parameter
        **kwargs
):
    """
    Find optimal design by maximizing EIG w.r.t. continuous design_params
    """
    grad_update = not check_grads
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
        optim, max_lr=1e-2, total_steps=n_steps)
    eig_vals = []

    # main loop
    grads_optim = {k:[] for k in optim_params.keys()}
    grads_design = {k:[] for k in optim_params.keys()}
    for s in (bar := tqdm(range(n_steps))):
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
        if s % print_every == 0:
            bar.set_description(f'EIG = {eig:.6f}')
        if verbose:
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
    if verbose:
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
        verbose = False,
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
        curr_eig, _ = eig_calc_method(design, prior, predictive)
        if curr_eig > best_eig:
            best_eig = curr_eig
            best_design_params = design_params
        eigs.append(curr_eig.detach().cpu().numpy())

    # plotting
    if verbose:
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
        eig_calc_method,    # e.g. eig_NMC, eig_PCE
        eig_optim_method,   # e.g. eig_discrete_optim, eig_cont_optim
        varinf_method,      # e.g. partial(variational_inference, ...)
        with_previous_prior = True,
        verbose = True,
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
        print(f'\n------- Iteration {iter+1}/{n_designs} -------')

        # find best design
        print('Finding best design...')
        eig, design_params = eig_optim_method(
            experiment,prior,predictive,eig_calc_method, 
            initial_design_params = design_params)
        experiment.execute_design(design_params)
        best_eigs.append(eig)
        print(f'Executed design; best_eig: {eig:.6f}, design_params: {design_params}')

        # fit "posterior"
        print('Fitting posterior...')
        obs_data = experiment.get_obs_data()
        guide = deepcopy(prior)
        posterior = varinf_method(
            obs_data,prior,predictive,guide,with_previous_prior=with_previous_prior,**kwargs)
        thetas_MAP = posterior.predict_mle()
        print(f'Fitted posterior; thetas_MAP: {thetas_MAP}')

        # evaluate with MAP params
        y,design = experiment.get_eval_data()
        pred_probs = predictive.predict_mle(design, thetas_MAP).detach().cpu().numpy()
        preds_int = (pred_probs > 0.5).astype(np.int8).reshape((-1))
        for i in range(len(metrics)):
            if metric_is_binary[i]:
                val = metrics[i](y,preds_int)
            else:
                val = metrics[i](y,pred_probs)
            metric_values[i].append(val)

        # plotting
        if verbose or iter == n_designs-1:
            print('Plotting results...')
            experiment.verbose_designs(
                design_eval=design,
                pred_probs=pred_probs,
                metric_values=metric_values,
                metric_names=metric_names
            )
            print('Plotted results.')

        # prepare next loop
        if with_previous_prior:
            prior = posterior

    return experiment, thetas_MAP, metric_values, metric_names


def train_DAD_design_policy(
        T,
        prior:Distr,
        likelihood:Conditional_Distr_multixi,
        design_network:Design_Network,
        n_steps = 100,
        n_outer = 50,
        L = 20,
        print_every = 1,
        verbose = True
):
    assert likelihood.reparam_trick

    # rollout likelihood over history
    def likelihood_history(history_xi, history_y, thetas):
        """
        history_xi: n_outer x T x design_dim
        history_y:  n_outer x T x y_dim
        thetas:     n_outer x 1+L x theta_dim

        returns: n_outer x 1+L
        """
        lp_prior = prior.log_prob(thetas[:,0,:]).reshape((n_outer,1))
        
        xi = history_xi.unsqueeze(1).repeat((1,1+L,1,1))
        y = history_y.unsqueeze(1).repeat((1,1+L,1,1))
        thetas = thetas.unsqueeze(2).repeat((1,1,T,1))
        lp_lik = likelihood.log_prob(
            y.reshape((-1,y.shape[3])),
            xi.reshape((-1,xi.shape[3])),
            thetas.reshape((-1,thetas.shape[3])))
        lp_lik = lp_lik.reshape((n_outer,1+L,T)).sum(axis=2)
        log_prob = lp_lik + lp_prior
        return log_prob

    # prepare optimization
    for p in design_network.parameters():
        p.requires_grad = True
    optim = torch.optim.Adam(design_network.parameters(), lr=5e-5, betas=(0.8,0.998))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=1)
    vals = []

    # main loop
    print('Training DAD design policy...')
    for n in (bar := tqdm(range(n_steps))):
        
        # generate batched history
        theta_0 = prior.sample(n_outer)
        history_xi = []
        history_y = []
        design_network.reset_buffer(n_outer)
        next_xi = torch.zeros((n_outer,design_network.design_dim))
        next_y = torch.zeros((n_outer,design_network.y_dim))
        for t in range(T):
            next_xi = design_network(next_xi, next_y)
            next_y = likelihood.sample(next_xi, theta_0)
            history_xi.append(next_xi.unsqueeze(1))
            history_y.append(next_y.unsqueeze(1))
        history_xi = torch.cat(history_xi, dim=1)  # n_outer x T x design_dim
        history_y = torch.cat(history_y, dim=1)    # n_outer x T x y_dim

        # compute loss
        thetas = prior.sample(n_outer * L).reshape((n_outer,L,-1))
        thetas = torch.cat((theta_0.unsqueeze(1), thetas), dim=1)  # n_outer x (1+L) x theta_dim
        lp_history = likelihood_history(
            history_xi, 
            history_y, 
            thetas
        ) # n_outer x (1+L) 
        lp_nom = lp_history[:,0]
        lp_denom = torch.logsumexp(lp_history,axis=1) - torch.log(torch.tensor(1+L))
        gL = lp_nom - lp_denom
        loss = -gL.mean()

        # param update
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
        if n%print_every == 0:
            bar.set_description(f'loss (asc) = {-loss:.6f}')
        if verbose:
            vals.append(-loss.detach().cpu().numpy())
    print('Trained DAD design policy.')

    # verbose
    if verbose:
        plt.figure(figsize=(2, 4))
        plt.plot(vals)
        plt.xlabel('optim step')
        plt.ylabel('EIG')
        plt.show()