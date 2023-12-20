#####
# parts of the code are from https://github.com/andrewcharlesjones/spatial-experimental-design
#####

import numpy as np
import pandas as pd
import warnings
import torch
from itertools import product
from copy import deepcopy
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.metrics import r2_score, roc_auc_score, f1_score
from scipy.stats import multivariate_normal as mvn
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from src.distributions import Distr, Conditional_distr
from src.model_utils import variational_inference
from tqdm import tqdm

class Tissue():

    def __init__(self, spatial_locations, readouts, slice_radius):

        assert len(spatial_locations) == len(readouts)

        self.X = spatial_locations
        self.Y = readouts
        self.slice_radius = slice_radius
        self.n_total, self.p = self.X.shape

        # Start with one tissue fragment
        self.X_fragment_idx = [np.arange(self.n_total)]
        self.observed_idx = []
        self.designs = []

    def compute_slice(self,design,fragment_num):
        """
        computes indices above below and in a slice
        
        design: should be of the form [b0, b1] or [b0, b1, b2]
            b0 is the intercept and b1, b2 are slopes (depending on if
            2D or 3D)
        """

        curr_fragment_idx = self.X_fragment_idx[fragment_num]
        fragment_X = self.X[curr_fragment_idx]
        warnings.warn('Not Correct. Need to fix')

        plane_values = design[0] + np.dot(fragment_X[:, :-1], design[1:])
        above_fragment_idx = np.where(
            fragment_X[:, -1] >= plane_values + self.slice_radius
        )[0]
        below_fragment_idx = np.where(
            fragment_X[:, -1] <= plane_values - self.slice_radius
        )[0]
        in_slice_idx = np.where(
            np.abs(fragment_X[:, -1] - plane_values) < self.slice_radius
        )

        above_idx = curr_fragment_idx[above_fragment_idx]
        below_idx = curr_fragment_idx[below_fragment_idx]
        in_idx = curr_fragment_idx[in_slice_idx]
        return below_idx,in_idx,above_idx


    def slice(self, design, fragment_num):
        below_idx,in_idx,above_idx = self.compute_slice(design,fragment_num)
        self.X_fragment_idx.pop(fragment_num)
        self.X_fragment_idx.append(above_idx)
        self.X_fragment_idx.append(below_idx)
        self.observed_idx.extend(in_idx)
        self.designs.append(design)

    def get_X_idx_near_slice(self, design, fragment_num):
        below_idx,in_idx,above_idx = self.compute_slice(design,fragment_num)
        return in_idx

    def get_fragment(self,idx):
        return self.X[self.X_fragment_idx[idx]]
    
    def get_all_observed_data(self):
        return self.X[self.observed_idx,:], self.Y[self.observed_idx]
    


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

        # verbose
        if datamodule is not None:
            datamodule.plot_slices(
                tissue.designs, 
                predictive_distr,
                metric_values=metric_values,
                metric_names=metric_names, 
                thetas=thetas,
                observed_idx_naive=tissue.observed_idx)
    
    # set params of predictive distribution to MAP
    predictive_distr.params['r'] = thetas[0]
    predictive_distr.params['c'] = thetas[1:]
    return tissue, predictive_distr, metric_values, metric_names


def eig_NMC_old(
        curr_observed_X, # nxd
        prior_distr:Distr, 
        predictive_distr:Conditional_distr,
        n_outer=100, 
        n_inner=50,
        **kwargs
):
    """
    Computes Expected Information Gain (EIG) with Nested Monte Carlo (NMC)
    which is unbiased.

    Old design!
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
    log_prob = np.log(probs).sum(axis=0) 
        # n_outer x 1+n_inner
    log_numerator = log_prob[:,0]
    log_denominator = -n_inner+ log_prob[:,1:].mean(axis=1)
    eig = np.mean(log_numerator - log_denominator)
    return eig



class Data_2d():
    def __init__(self, radius=5):
        self.radius = radius

    def get_data():
        raise NotImplementedError()

    @staticmethod
    def get_points_near_line(slope, X, slice_radius, intercept=0):
        dists = np.abs(-slope * X[:, 0] + X[:, 1] - intercept) / np.sqrt(slope ** 2 + 1)
        observed_idx = np.where(dists <= slice_radius)[0]
        return observed_idx
    

class Simulated_GP_2d(Data_2d):

    def get_data(self,
            grid_size = 40,
            noise_variance = 1e-1):

        limits = [-self.radius, self.radius]
        x1s = np.linspace(*limits, num=grid_size)
        x2s = np.linspace(*limits, num=grid_size)
        X1, X2 = np.meshgrid(x1s, x2s)
        X = np.vstack([X1.ravel(), X2.ravel()]).T
        X += np.random.uniform(low=-0.5, high=0.5, size=X.shape)

        # Filter by radius
        norms = np.linalg.norm(X, ord=2, axis=1)
        X = X[norms <= self.radius]

        # Generate response
        Y = mvn.rvs(
            mean=np.zeros(X.shape[0]), 
            cov=Matern()(X) + noise_variance * np.eye(len(X)))

        self.X = X
        self.Y = Y
        return X,Y

    def designs_discrete(self,
            n_slope_discretizations = 30,
            n_intercept_discretizations = 30):
        slope_angles = np.linspace(0, np.pi, n_slope_discretizations)
        slopes = np.tan(slope_angles)
        intercepts = np.linspace(-self.radius, self.radius, n_intercept_discretizations)
        designs1, designs2 = np.meshgrid(intercepts, slopes)
        designs = np.vstack([designs1.ravel(), designs2.ravel()]).T
        return designs
    
    def plot_slices(self, X_fragment_idx, fancy=False):

        if fancy:
            circle = plt.Circle((0, 0), self.radius, fill=False)
            ax = plt.gca()
            ax.add_patch(circle)

        plt.scatter(self.X[:, 0], self.X[:, 1], color="gray", alpha=0.6)
        n_experimental_iters = len(X_fragment_idx) - 1

        patches = []
        colors = plt.cm.jet(np.linspace(0, 1, n_experimental_iters + 1))
        for ii in range(n_experimental_iters + 1):

            curr_X = self.X[X_fragment_idx[ii]]

            hull = ConvexHull(curr_X)

            polygon = Polygon(curr_X[hull.vertices], True)
            patches.append(polygon)

            plt.fill(
                curr_X[hull.vertices, 0],
                curr_X[hull.vertices, 1],
                alpha=0.3,
                color=colors[ii],
            )
        if fancy:
            plt.axis("off")


class Data_3d():
    def __init__(self, 
            xlimits = [-10, 10],
            ylimits = [-10, 10],
            length_scale = 5,
            noise_variance = 1e-2):
        self.xlimits = xlimits
        self.ylimits = ylimits
        self.length_scale = length_scale
        self.noise_variance = noise_variance
        self.obtained_data = False
        self.coords = None
        self.outcome= None

    def get_data(self, *args, **kwargs):
        raise NotImplementedError()
    
    def plot_slices(self, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def compute_point_to_plane_dists(points, plane, signed=False):
        dists_signed = (plane[0] * points[:, 0] + plane[1] * points[:, 1] + plane[2] * points[:, 2] + plane[3]) / np.sqrt(np.sum(plane[:3] ** 2))
        return dists_signed if signed else np.abs(dists_signed)

    @staticmethod
    def compute_eig(cov, noise_variance):
        return 0.5 * np.linalg.slogdet(1 / noise_variance * cov + np.eye(len(cov)))[1]
    
    def _meshgrid2(self,*arrs):
        arrs = tuple(reversed(arrs))  #edit
        lens = [len(x) for x in arrs] # map(len, arrs)
        dim = len(arrs)

        sz = 1
        for s in lens:
            sz*=s

        ans = []    
        for i, arr in enumerate(arrs):
            slc = [1]*dim
            slc[i] = lens[i]
            arr2 = np.asarray(arr).reshape(slc)
            for j, sz in enumerate(lens):
                if j!=i:
                    arr2 = arr2.repeat(sz, axis=j) 
            ans.append(arr2)

        return tuple(ans)


class Simulated_GP_3d(Data_3d):

    def get_data(self,grid_size = 10):
        if self.obtained_data:
            print('Return already simulated data.')
            return self.coords, self.outcome

        # generate data 
        x1s = np.linspace(*self.xlimits, num=grid_size)
        x2s = np.linspace(*self.ylimits, num=grid_size)
        x3s = np.linspace(*self.ylimits, num=grid_size)
        X1, X2, X3 = np.meshgrid(x1s, x2s, x3s)
        X = np.vstack([X1.ravel(), X2.ravel(), X3.ravel()]).T
        X += np.random.uniform(low=-1, high=1, size=X.shape)

        Y_full = mvn.rvs(
            mean=np.zeros(X.shape[0]), 
            cov=RBF(length_scale=self.length_scale)(X) + self.noise_variance * np.eye(len(X)))
        self.X = X
        self.Y = Y_full
        self.obtained_data = True

        return X, Y_full
    
    def designs_parallel(self):
        pass

    def designs_all(self, n_discretizations = 5):

        out = self._meshgrid2(
            np.linspace(self.xlimits[0] - 5, self.xlimits[1] + 5, n_discretizations),
            [-1],
            np.linspace(-1, 1, n_discretizations),
            [0], #np.linspace(-1, 1, grid_size), #np.linspace(-1, 1, 3), #[0],
        )
        designs = np.stack([np.ravel(x) for x in out], axis=1)
        designs = designs[(designs[:, :3] ** 2).sum(1) > 0]
        return designs
    
    def plot_slices(self, chosen_designs, observed_idx):
        coords = self.coords

        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes(projection='3d')
        ax.scatter3D(coords[:, 0], coords[:, 1], coords[:, 2])
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        plt.close()

        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes(projection='3d')

        CLOSE_DIST = 2

        ax.scatter3D(coords[:, 0], coords[:, 1], coords[:, 2], c=np.isin(np.arange(len(coords)), observed_idx), alpha=0.7) #, s=5)

        # xx, yy = np.meshgrid(range(np.max(coords[:, 0])), range(np.max(coords[:, 1])))
        xx, yy = np.meshgrid(
            np.linspace(np.min(coords[:, 0]), np.max(coords[:, 0]), 10),
            np.linspace(np.min(coords[:, 1]), np.max(coords[:, 1]), 10),
        )

        for P in chosen_designs[:5]: #[:6]:
        # for P in designs_serial: #[:6]:
            if P[2] == 0:
                z = np.zeros(xx.shape)
            else:
                z = (P[0] * xx + P[1] * yy + P[3]) / -P[2]
            ax.plot_surface(xx, yy, z, color="gray", alpha=0.3)
            
        # dists = compute_point_to_plane_dists(coords, P, signed=False)
        # ax.scatter3D(coords[:, 0], coords[:, 1], coords[:, 2], c=dists < CLOSE_DIST, alpha=0.7) #, s=5)

        ax.set_xlabel(r"$x$") #, rotation=90)
        ax.set_ylabel(r"$y$") #, rotation=90)
        ax.set_zlabel(r"$z$") #, rotation=90)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        plt.show()
 