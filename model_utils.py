from asyncio import Condition
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
from data_utils import Data_2d,Data_3d

from tissue import Tissue
import matplotlib
from tqdm import tqdm
from itertools import product
from scipy.stats import bernoulli,multivariate_normal
from copy import deepcopy

from distributions import Distr, Conditional_distr

def fit_GP_2d(
        X,Y,designs,n_experimental_iters,
        slice_radius = 0.25
):
    # setup
    X_fragment_idx = []
    best_designs = []
    # kernel = Matern() + WhiteKernel()  # can choose any kernel here
    kernel = Matern()
    noise_variance = 1e-2
    n_candidate_designs = len(designs)
    eigs = np.zeros(n_candidate_designs) 


    # first slice
    for dd in range(n_candidate_designs):

        curr_design = designs[dd]
        curr_observed_idx = Data_2d.get_points_near_line(
            X=X, slope=curr_design[1], slice_radius=slice_radius, intercept=curr_design[0]
        )

        # Make predictions of expression
        cov = kernel(X[curr_observed_idx])

        # Compute EIG
        # noise_variance = np.exp(kernel.k2.theta[0])
        eigs[dd] = (
            0.5 * np.linalg.slogdet(1 / noise_variance * cov + np.eye(len(curr_observed_idx)))[1]
        )

    curr_best_design_idx = np.argmax(eigs)
    curr_best_design = designs[curr_best_design_idx]
    best_designs.append(curr_best_design)
    observed_idx = Data_2d.get_points_near_line(
        X=X, slope=curr_best_design[1], intercept=curr_best_design[0],
        slice_radius=slice_radius
    ).tolist()

    above_fragment_idx = np.where(
        X[:, 1] >= curr_best_design[0] + X[:, 0] * curr_best_design[1]
    )[0]
    below_fragment_idx = np.where(
        X[:, 1] <= curr_best_design[0] + X[:, 0] * curr_best_design[1]
    )[0]
    X_fragment_idx.append(above_fragment_idx)
    X_fragment_idx.append(below_fragment_idx)


    # iterate over slices other than the first
    for iternum in range(1, n_experimental_iters):

        # Fit GP on observed data
        gpr = GPR(kernel=kernel)
        gpr.fit(X[observed_idx], Y[observed_idx])

        best_eig = -np.inf
        best_design_idx, best_fragment_idx, best_observed_idx = None, None, None

        for ff in range(len(X_fragment_idx)):

            # Get data for this fragment
            curr_X = X[X_fragment_idx[ff]]

            for dd in range(len(designs)):

                # Get points that would be observed by this slice
                curr_design = designs[dd]

                above_fragment_idx = np.where(
                    curr_X[:, 1] >= curr_design[0] + curr_X[:, 0] * curr_design[1]
                )[0]
                if len(above_fragment_idx) in [
                    0,
                    1,
                    2,
                    len(curr_X),
                    len(curr_X) - 1,
                    len(curr_X) - 2,
                ]:
                    continue

                curr_observed_idx = Data_2d.get_points_near_line(
                    X=curr_X, slope=curr_design[1], intercept=curr_design[0],
                    slice_radius=slice_radius
                )
                if len(curr_observed_idx) == 0:
                    continue

                _, cov = gpr.predict(curr_X[curr_observed_idx], return_cov=True)

                # Compute EIG for each slice through this fragment
                # noise_variance = np.exp(gpr.kernel_.k2.theta[0])
                curr_eig = (
                    0.5 * np.linalg.slogdet(
                        1 / noise_variance * cov + np.eye(len(curr_observed_idx))
                    )[1]
                )

                if curr_eig > best_eig:
                    best_design_idx = dd
                    best_fragment_idx = ff
                    best_observed_idx = X_fragment_idx[ff][curr_observed_idx]
                    best_eig = curr_eig

        curr_best_design = designs[best_design_idx]
        print(f'{iternum}/{n_experimental_iters}: {best_eig}, {curr_best_design}')
        best_fragment_X = X[X_fragment_idx[best_fragment_idx]]

        above_fragment_idx = np.where(
            best_fragment_X[:, 1]
            >= curr_best_design[0] + best_fragment_X[:, 0] * curr_best_design[1]
        )[0]
        below_fragment_idx = np.where(
            best_fragment_X[:, 1]
            <= curr_best_design[0] + best_fragment_X[:, 0] * curr_best_design[1]
        )[0]

        above_idx = X_fragment_idx[best_fragment_idx][above_fragment_idx]
        below_idx = X_fragment_idx[best_fragment_idx][below_fragment_idx]
        X_fragment_idx.pop(best_fragment_idx)
        X_fragment_idx.append(above_idx)
        X_fragment_idx.append(below_idx)

        best_designs.append(curr_best_design)
        observed_idx.extend(best_observed_idx)

    return best_designs, X_fragment_idx


def fit_GP_3d(
        coords,outcome,designs,n_experimental_iters,CLOSE_DIST=0.5,
        noise_variance=1e-2, length_scale=10):
    
    # prepare data
    # X,Y = datamodule.get_data()
    # # length_scale = datamodule.length_scale
    # # noise_variance = datamodule.noise_variance
    # coords, outcome = X,Y

    tissue_fragments_idx = [np.arange(len(coords))]
    observed_idx = []
    chosen_designs = []
    
    r2_eig = np.zeros((n_experimental_iters))
    mse_eig = np.zeros((outcome.shape[0], n_experimental_iters))

    # define likelihood
    noise_variance = noise_variance
    length_scale = length_scale
    kernel = RBF(length_scale=length_scale)

    # main
    for experimental_iter in range(n_experimental_iters):
    
        
        best_eig = -np.inf
        best_design_idx, best_fragment_idx, best_observed_idx = None, None, None

        for ff in range(len(tissue_fragments_idx)):

            curr_coords = coords[tissue_fragments_idx[ff]]
            curr_outcome = outcome[tissue_fragments_idx[ff]]

            ## Loop over designs
            for dd, design in enumerate(designs):

                ## Get normal vector of plane
                normal_vector = design[:3] / np.linalg.norm(design[:3], ord=2)

                ## Find new observed points
                dists_signed = Data_3d.compute_point_to_plane_dists(
                    curr_coords, design, signed=True)
                dists = np.abs(dists_signed)
                curr_observed_idx = np.where(dists < CLOSE_DIST)[0]

                if len(curr_observed_idx) == 0:
                    continue

                if experimental_iter == 0:
                    cov = kernel(curr_coords[curr_observed_idx])
                else:

                    ## Compute EIG
                    K_XX = kernel(coords[observed_idx])
                    K_XtestXtest = kernel(curr_coords[curr_observed_idx])
                    K_XXtest = kernel(coords[observed_idx], curr_coords[curr_observed_idx])
                    cov = K_XtestXtest + noise_variance * np.eye(len(K_XtestXtest)) - K_XXtest.T @ np.linalg.solve(K_XX + noise_variance * np.eye(len(K_XX)), K_XXtest)

                curr_eig = Data_3d.compute_eig(cov, noise_variance)

                if curr_eig > best_eig:
                    best_design_idx = dd
                    best_fragment_idx = ff
                    best_observed_idx = tissue_fragments_idx[ff][curr_observed_idx]
                    best_eig = curr_eig

                assert len(np.intersect1d(observed_idx, tissue_fragments_idx[ff][curr_observed_idx])) == 0


        curr_best_design = designs[best_design_idx]
        best_fragment_coords = coords[tissue_fragments_idx[best_fragment_idx]]
        print(f'{experimental_iter+1}/{n_experimental_iters}: {best_eig}, {curr_best_design}')


        dists_signed = Data_3d.compute_point_to_plane_dists(
            best_fragment_coords, curr_best_design, signed=True)

        above_plane_idx = np.where(dists_signed > 0)[0]
        below_plane_idx = np.where(dists_signed <= 0)[0]

        above_idx = tissue_fragments_idx[best_fragment_idx][above_plane_idx]
        below_idx = tissue_fragments_idx[best_fragment_idx][below_plane_idx]
        above_idx = np.setdiff1d(above_idx, best_observed_idx)
        below_idx = np.setdiff1d(below_idx, best_observed_idx)

        tissue_fragments_idx.pop(best_fragment_idx)
        if len(above_idx) > 0:
            tissue_fragments_idx.append(above_idx)
        if len(below_idx) > 0:
            tissue_fragments_idx.append(below_idx)

        chosen_designs.append(curr_best_design)
        observed_idx.extend(best_observed_idx)

        unobserved_idx = np.setdiff1d(np.arange(len(coords)), observed_idx)

        if len(observed_idx) > 0.5 * len(coords):
            train_idx = np.random.choice(observed_idx, size=int(0.5 * len(coords)), replace=False)
            test_idx = np.setdiff1d(np.arange(len(coords)), train_idx)
        else:
            train_idx = observed_idx.copy()
            test_idx = unobserved_idx.copy()

        ## Fit GP, make predictions, and compute error
        gpr = GPR(kernel=RBF(length_scale=length_scale) + WhiteKernel(noise_variance, noise_level_bounds="fixed")) #, optimizer=None)
        gpr.fit(coords[train_idx], outcome[train_idx])
        preds = gpr.predict(coords[test_idx])

        r2_eig[experimental_iter] = r2_score(outcome[test_idx], preds)
        mse_eig[:, experimental_iter] = (outcome - gpr.predict(coords)) ** 2

    metrics = {
        'r2_eig':r2_eig,
        'mse_eig':mse_eig
    }
    return chosen_designs, observed_idx


def fit_2d(
        X,y, candidate_designs, n_experimental_iters,
        prior_distr,
        predictive_distr,
        eig_method,
        slice_radius = 0.25,
        torch_device = 'cpu',
        datamodule=None,
        **kwargs
):
    tissue = Tissue(X,y,slice_radius=slice_radius)
    n_candidate_designs = len(candidate_designs)

    # prepare logging
    best_eigs = []
    metrics = [roc_auc_score, f1_score]
    metric_is_binary = [False, True]
    metric_names = ['roc_auc','f1']
    metric_values = [[] for _ in range(len(metrics))]

    for iternum in range(n_experimental_iters):

        # find best slice
        n_fragments = len(tissue.X_fragment_idx)
        ff_best, dd_best, eig_best = None, None, -np.inf
        
        for ff in tqdm(range(n_fragments)):
            for dd in range(n_candidate_designs):

                curr_design = candidate_designs[dd]

                # check that valid fragment
                below_idx,curr_observed_idx,above_idx = \
                    tissue.compute_slice(curr_design, ff)
                if below_idx.shape[0] < 3 or above_idx.shape[0] < 3:
                    # print(f'Skip: {dd}')
                    continue

                # compute eig
                curr_observed_X = tissue.X[curr_observed_idx]
                curr_eig = eig_method(
                    curr_observed_X,
                    prior_distr, predictive_distr, **kwargs
                )
                # print(f'{ff},{dd},{curr_eig}')
                if curr_eig > eig_best:
                    eig_best = curr_eig
                    ff_best = ff
                    dd_best = dd
        design_best = candidate_designs[dd_best]
        best_eigs.append(eig_best)
        tissue.slice(design_best,ff_best)
        print(f'{iternum+1}/{n_experimental_iters}: eig = {eig_best}, design = {design_best}')
        
        # update prior model with new knowledge for next step
        x_obs,y_obs = tissue.get_all_observed_data()
        guide = deepcopy(prior_distr)
        prior_distr = variational_inference(
            x_obs, y_obs, prior_distr, predictive_distr, guide, device=torch_device,*kwargs)

        # set params to MAP estimates and do deterministic prediction on all data
        # WARNING: this might not be the correct posterior predictive
        prior_params = prior_distr.params['mu']
        thetas = np.concatenate([np.exp(prior_params[0]).reshape((1,)),prior_params[1:]])
        pred_probs = predictive_distr.predict(X, thetas=thetas).reshape((-1))
        preds_int = (pred_probs > 0.5).astype(int).reshape((-1))
        for i in range(len(metrics)):
            if metric_is_binary[i]:
                val = metrics[i](y,preds_int)
            else:
                val = metrics[i](y,pred_probs)
            metric_values[i].append(val)
        print(np.column_stack([y.reshape((-1,1)), pred_probs.reshape((-1,1))]))

        # debug
        if datamodule is not None:
            datamodule.plot_slices(tissue.designs, predictive_distr,metric_values=metric_values,metric_names=metric_names, thetas=thetas)
    
    # set params of predictive distribution to MAP
    predictive_distr.params['r'] = thetas[0]
    predictive_distr.params['c'] = thetas[1:]
    return tissue, predictive_distr, metric_values, metric_names


def eig_NMC(
        curr_observed_X, # nxd
        prior_distr:Distr, 
        predictive_distr:Conditional_distr,
        n_outer=100, 
        n_inner=50,
        **kwargs
):
    """
    Computes Expected Information Gain (EIG) with Nested Monte Carlo (NMC)
    which is unbiased
    """
    n_obs, d = curr_observed_X.shape

    # sample
    thetas = prior_distr.sample(n_outer * (1+n_inner)).reshape((n_outer, 1+n_inner, -1))
        # n_outer x 1+n_inner x p
    ys = predictive_distr.sample(curr_observed_X, thetas=thetas[:,0,:])
        # n_obs x n_outer
    
    # broadcast over dimensions
    thetas = np.tile(thetas[np.newaxis,:,:,:], (n_obs,1,1,1))
        # n_obs x n_outer x 1+n_inner x p
    ys = np.tile(ys[:,:,np.newaxis], (1,1,1+n_inner))
        # n_obs x n_outer x 1+n_inner
    xs = np.tile(curr_observed_X[:,np.newaxis,np.newaxis,:], (1,n_outer,1+n_inner,1)) 
        # n_obs x n_outer x 1+n_inner x d

    # calculate probabilities
    probs = predictive_distr.probs(
        ys = np.reshape(ys, (-1,1)),
        xs = np.reshape(xs, (-1,d)), 
        thetas = thetas.reshape((-1,thetas.shape[-1])))
    probs = probs.reshape((n_obs, n_outer, 1+n_inner))
        # n_obs x n_outer x 1+n_inner

    # calculate estimate of eig
    log_probs = np.log(probs).sum(axis=0) 
        # n_outer x 1+n_inner
    log_numerator = log_probs[:,0]
    log_denominator = -n_inner+ log_probs[:,1:].mean(axis=1)
    eig = np.mean(log_numerator - log_denominator)
    return eig

def eig_varNMC():
    """
    variational NMC is an upper bound on the EIG which is 
    asymptotically (for nsamples->infty) unbiased
    """

    # learn proposal distribution
    propsal = variational_inference()

    # final nmc estimator
    eig = None

    raise NotImplementedError()
    return eig

def eig_ACE():
    """
    the Adaptive Contrastive Estimator is a lower bound on the EIG 
    which is asymptotically (for nsamples->infty) unbiased
    """
    raise NotImplementedError()

def variational_inference(
        xs,
        ys,
        prior_distr:Distr,
        predictive_distr:Conditional_distr,
        guide:Distr,
        nsteps = 1000,
        nsamples_elbo = 30,
        print_every = 100,
        device = 'cpu'):
    """
    Standard variational inference to find posterior.

    Variational distribution is named 'guide' as in pyro.
    """

    assert guide.reparam_trick == True
    assert xs.shape[0] == ys.shape[0]
    n_obs = xs.shape[0]
    xs = torch.from_numpy(xs).tile((nsamples_elbo,1))
    ys = torch.from_numpy(ys.reshape(-1,1)).tile((nsamples_elbo,1))
    guide.set_torch(True,device=device)
    prior_distr.set_torch(True,device=device)
    predictive_distr.set_torch(True, device=device)

    # actual optimization
    params = guide.params.values()
    for p in params:
        p.requires_grad = True
    optim = torch.optim.Adam(params, lr=1e-2)
    for s in range(nsteps):

        # compute (negative) elbo
        thetas = guide.sample(nsamples_elbo)
            # nsamples x p
        log_denom = torch.log(guide.probs(thetas))
        log_prior = torch.log(prior_distr.probs(thetas))

        thetas = torch.repeat_interleave(thetas,repeats=n_obs,dim=0)  
        log_predictive = torch.log(predictive_distr.probs(ys,xs,thetas=thetas))
        log_predictive = log_predictive.reshape((nsamples_elbo,n_obs))
            # nsamples_elbo x n_obs
        log_predictive = log_predictive.sum(axis=1)
            # nsamples_elbo
        log_numerator = log_predictive + log_prior
            # nsamples_elbo
        neg_elbo = -torch.mean(log_numerator - log_denom) # negative s.t. minimization

        # torch step
        optim.zero_grad()
        neg_elbo.backward()
        optim.step()

        if s%print_every == 0:
            print(f'{s+1}/{nsteps}: elbo = {-neg_elbo:.6f}')

    guide.set_torch(False)
    prior_distr.set_torch(False)
    predictive_distr.set_torch(False)
    return guide