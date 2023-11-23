import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.metrics import r2_score
import seaborn as sns
from scipy.stats import multivariate_normal as mvn
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull
from data_utils import Data_2d,Data_3d

import matplotlib

def fit_GP(
        X,Y,designs,n_experimental_iters,
        slice_radius = 0.25
):
    # setup
    X_fragment_idx = []
    best_designs = []
    kernel = Matern() + WhiteKernel()
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
        noise_variance = np.exp(kernel.k2.theta[0])
        eigs[dd] = (
            0.5
            * np.linalg.slogdet(1 / noise_variance * cov + np.eye(len(curr_observed_idx)))[
                1
            ]
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
                noise_variance = np.exp(gpr.kernel_.k2.theta[0])
                curr_eig = (
                    0.5
                    * np.linalg.slogdet(
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
        datamodule:Data_3d,designs,n_experimental_iters):
    
    # prepare data
    X,Y = datamodule.get_data()

    length_scale = datamodule.length_scale
    noise_variance = datamodule.noise_variance
    coords, outcome = X,Y

    tissue_fragments_idx = [np.arange(len(coords))]
    observed_idx = []
    chosen_designs = []
    observed_idx_no_fragmenting = []
    
    r2_eig = np.zeros((n_experimental_iters))
    mse_eig = np.zeros((outcome.shape[0], n_experimental_iters))

    # np.random.shuffle(serial_designs_idx)
    
    CLOSE_DIST = 0.5
    kernel = RBF(length_scale=10)

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

    return chosen_designs, observed_idx