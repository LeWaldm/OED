from importlib.metadata import distribution
from lib2to3.pytree import convert
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pkg_resources import NullProvider
# import scanpy as sc
from sklearn.metrics import pairwise_distances
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.metrics import r2_score
import seaborn as sns
from scipy.stats import multivariate_normal as mvn
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull
from brainrender.atlas_specific import GeneExpressionAPI
from sympy import product
from torch import Value, _nested_tensor_softmax_with_shape
import torch
from math import ceil,sqrt

from src.utils import get_data_dir
import os
import src.distributions as distr
from copy import deepcopy
import warnings
from itertools import product


# ----- 
# loading datasets code partly copied from https://github.com/andrewcharlesjones/spatial-experimental-design
# -----

class Allen_brain_3d():

    def get_data(self):
        if self.obtained_data:
            return self.coords, self.outcome
        print('Downloading data')

        # load data
        gene = "Pcp4"
        geapi = GeneExpressionAPI()
        expids = geapi.get_gene_experiments(gene)
        data = geapi.get_gene_data(gene=gene,  exp_id=expids[0])

        # process data
        data_dims = data.shape
        xs, zs, ys = [np.arange(data_dims[i]) for i in range(3)]
        coords = np.array(np.meshgrid(xs, ys, zs)).T.reshape(-1, 3)
        outcome = np.array([data[c[0], c[2], c[1]] for c in coords])
        observed_idx = np.where(outcome > 2)
        coords = coords[observed_idx]
        coords[:, 2] = np.max(coords[:, 2]) - coords[:, 2]
        outcome = outcome[observed_idx]
        coords = coords.astype(np.float64)
        coords -= coords.mean(0)

        self.coords, self.outcome = coords, outcome
        self.obtained_data = True

        return coords, outcome
    
    def designs_discrete(self, grid_size = 5, lims = [-40, 40]):
        coords, outcome = self.coords, self.outcome
        out = self._meshgrid2(
            np.linspace(np.min(coords[:, 1]), np.max(coords[:, 1]), grid_size),
            np.linspace(-3, 3, grid_size),
            np.linspace(lims[0], lims[1], grid_size),
            np.linspace(lims[0], lims[1], grid_size),
        )
        designs = np.stack([np.ravel(x) for x in out], axis=1)
        return designs
    
    def plot_full(self, coords=None, outcome=None, ax=None):
        if not coords or not outcome:
            coords,outcome = self.coords, self.outcome

        if not ax:
            plt.close()
            fig = plt.figure(figsize=(5, 5))
            ax = plt.axes(projection='3d')

        ax.scatter3D(coords[:, 0], coords[:, 1], coords[:, 2], c=outcome, alpha=0.3)
        ax.set_ylabel("Anterior <--> Posterior", rotation=90)
        ax.set_zlabel("Inferior <--> Superior", rotation=90)

        if not ax:
            plt.show()

    def plot_slices(self, chosen_designs, observed_idx, fancy=True):
        
        coords,outcome = self.coords, self.outcome
        if fancy:
            fig = plt.figure(figsize=(5,5))
            ax = plt.axes(projection='3d')

            self.plot_full(ax=ax)

            for P in chosen_designs:
                xx, yy = np.meshgrid(
                    np.linspace(np.min(coords[:, 0]), np.max(coords[:, 0]), 10),
                    np.linspace(np.min(coords[:, 1]), np.max(coords[:, 1]), 10),
                )
                z = (P[0] * xx + P[1] * yy + P[3]) / -P[2]
                ax.plot_surface(xx, yy, z, color="gray", alpha=0.3)

            plt.show()
        else: 

            plt.close()
            fig = plt.figure(figsize=(5, 5))
            ax = plt.axes(projection='3d')

            observed_idx_bool = np.full((outcome.shape[0]), False)
            observed_idx_bool[observed_idx] = True
            ax.scatter3D(coords[:, 0], coords[:, 1], coords[:, 2], c=observed_idx_bool, alpha=0.3) #, s=5)

            plt.show()

    def metric_full_atlas(self,
            n_neighbors = 10,
            n_repeats = 10,
            frac_drop = 0.5
    ):
        """
        compute R^2 coefficient of determination when dropping observations 
        and computing predictions from gaussian process fit on rest
        """
        coords,outcome = self.coords, self.outcome
        mse_full_atlas = np.zeros(n_repeats)
        for ii in range(n_repeats):
            
            # Randomly drop points
            test_idx = np.random.choice(np.arange(len(coords)), size=int(frac_drop * len(coords)), replace=False)
            train_idx = np.setdiff1d(np.arange(len(coords)), test_idx)
            coords_train, outcome_train = coords[train_idx], outcome[train_idx]
            coords_test, outcome_test = coords[test_idx], outcome[test_idx]
            
            # Fit GP
            gpr = GPR(kernel=RBF(length_scale=10) + WhiteKernel(1.)) #, optimizer=None)
            gpr.fit(coords_train, outcome_train)
        #     knn = KNeighborsRegressor(n_neighbors=n_neighbors)
        #     knn.fit(coords_train, outcome_train)
            
            # Make predictions for dropped points
            preds = gpr.predict(coords_test)
        #     preds = knn.predict(coords_test)
            
            # Compute MSE
            mse = np.mean((preds - outcome_test) ** 2)
        #     mse_full_atlas[ii] = mse
            mse_full_atlas[ii] = r2_score(outcome_test, preds)
            print(mse_full_atlas[ii])
        return mse_full_atlas
    

class Prostate_cancer_2d():
    """
    from https://www.10xgenomics.com/resources/datasets/human-prostate-cancer-adenocarcinoma-with-invasive-carcinoma-ffpe-1-standard-1-3-0 
    download 'Spatial imaging data' and 'clustering' into data folder
    """

    def get_data(self, name='invasive', processed=False):

        DATA_DIR = os.path.join(get_data_dir(),'Visium_FFPE_Human_Prostate_IF_analysis',name)
        if name == 'normal_section':
            tissue_positions_path = os.path.join(DATA_DIR, 'spatial','tissue_positions.csv')
            clusters_path = os.path.join(DATA_DIR,'analysis','clustering','gene_expression_kmeans_9_clusters','clusters.csv')
        elif name == 'invasive':
            tissue_positions_path = os.path.join(DATA_DIR, 'spatial','tissue_positions_list.csv')
            clusters_path = os.path.join(DATA_DIR,'analysis','clustering','kmeans_9_clusters','clusters.csv')
        else:
            raise ValueError()

        locations = pd.read_csv(tissue_positions_path,index_col=0)
        locations.columns = ["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
        locations = locations[locations.in_tissue == 1]
        clusters = pd.read_csv(clusters_path, header=0, index_col=0)

        data = pd.merge(locations, clusters, left_index=True, right_index=True)

        tumor_idx = np.where(data.Cluster.values == 1)[0]
        # tumor_idx = onp.where(data.Cluster.isin([1, 4]))[0]
        tumor_mask = np.zeros(len(data))
        tumor_mask[tumor_idx] = 1
        data["is_tumor"] = tumor_mask.astype(bool)
        self.data = data

        X = data[["array_col", "array_row"]].values
        y = data["is_tumor"].values.astype(int)
        X = torch.from_numpy(X)
        y = torch.from_numpy(y).int()
        if processed:
            means = X.float().mean(axis=0)
            std_both = X.float().std() # do not want to change ratio of axes
            X = (X - means) / std_both
            self.norm_params = {'means':means, 'std_both':std_both}
        self.X = X
        self.y = y
        self.processed = processed
        return X,y

    def plot_data(self):
        data = self.data

        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        sns.scatterplot(data=data, x="array_col", y="array_row", hue=data.Cluster.astype(str), marker="H", s=30)
        plt.gca().invert_yaxis()
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.legend([],[], frameon=False)
        # plt.axis("off")

        plt.subplot(122)
        sns.scatterplot(data=data, x="array_col", y="array_row", hue=data.is_tumor, marker="H", s=30)
        plt.gca().invert_yaxis()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.axis("off")
        plt.show()

    def params2designs_discrete(self,
            n_slope_discretizations = 10,
            n_intercept_discretizations = 10,):
        
        if self.processed:
            d_border = 0
        else:
            d_border = 5

        limits = [self.X.min(0).values[1], self.X.max.values(0)[1]]
        slope_angles = torch.linspace(0, torch.pi, n_slope_discretizations)
        slopes = torch.tan(slope_angles)
        intercepts = torch.linspace(limits[0] - d_border, limits[1] + d_border, n_intercept_discretizations)
        designs1, designs2 = torch.meshgrid(intercepts, slopes)
        candidate_designs = torch.vstack([designs1.ravel(), designs2.ravel()]).T
        return candidate_designs
    
    def plot_slices(self, 
                    designs, 
                    predictive_model:distr.Conditional_distr,
                    metric_values=None,
                    metric_names=None,
                    observed_idx_naive = None,
                    thetas=None):

        X,Y = self.X, self.y
        def abline(slope, intercept, label=None, **args):
            """Plot a line from slope and intercept"""
            axes = plt.gca()
            x_vals = np.array(axes.get_xlim())
            y_vals = intercept + slope * x_vals
            plt.plot(x_vals, y_vals, '--', label=label, **args)

        plt.figure(figsize=(12, 5))
        ncols = 2 if metric_values is None else 3

        # 1
        plt.subplot(1,ncols,1)
        if observed_idx_naive is not None:
            plt.scatter(X[:, 0], X[:, 1], s=20, c=np.isin(np.arange(len(X)), observed_idx_naive), marker="H")
        else:
            plt.scatter(X[:, 0], X[:, 1], s=20, color="gray", marker="H") #, c=onp.isin(onp.arange(len(X)), observed_idx), marker="H", s=20)
        plt.scatter(X[Y==1, 0], X[Y==1, 1], c="red", alpha=0.3, s=20, label="Invasive\ncarcinoma")
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        for i,dd in enumerate(designs):
            abline(dd[1], dd[0], label="Slice {}".format(i + 1), linewidth=3)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend(loc='center left', bbox_to_anchor=(-0.3, 0.5), fontsize=10)
        plt.gca().invert_yaxis()
        plt.title('Slices chosen by the algorithm')

        # 2
        plt.subplot(1,ncols,2)
        preds = predictive_model.predict(X,thetas=thetas)
        plt.scatter(X[:, 0], X[:, 1], s=20, c=preds, marker="H")
        plt.gca().invert_yaxis()
        plt.title('Predictions of the model')

        # 3
        if metric_values is not None:
            plt.subplot(1,ncols,3)
            assert len(metric_values) == len(metric_names)
            df = pd.DataFrame(metric_values, 
                              index=metric_names, 
                              columns=range(len(metric_values[0]))).transpose()
            df['number designs'] = df.index + 1
            for i in range(len(metric_values)):
                sns.lineplot(data=df, 
                             y=metric_names[i], 
                             x='number designs', 
                             label=metric_names[i],
                             markers=True)
            plt.title('Metrics of predictions')
        plt.show()

    def get_data_grid(self,smooth=True):
        """
        X: nx2 array of integer positions
        y: nx... array of corresponding outputs
        """
        # calculate grid
        X,y = self.get_data(processed=False)
        dim_outcome = 1 if len(y.shape)==1 else y.shape[1]
        X = X - X.min(axis=0).values
        xmax = X.max(axis=0).values
        grid = torch.zeros((1,dim_outcome,xmax[1]+1,xmax[0]+1)).int() 
        grid[0,:,X[:,1],X[:,0]] = y.int()

        # smooth grid
        if smooth:          
            x2 = torch.zeros((1,1,1,grid.shape[3]))
            sgrid = torch.concat([x2,grid,x2],axis=2)
            x3 = torch.zeros((1,1,sgrid.shape[2],1))
            sgrid = torch.concat([x3,sgrid,x3],axis=3)
            sgrid = (sgrid[:,:,1:-1,0:-2] + sgrid[:,:,1:-1,2:] \
                + sgrid[:,:,0:-2,1:-1] + sgrid[:,:,2:,1:-1]) / 4
            sgrid = (sgrid > 0.5).int()
            sgrid[0,:,X[:,1],X[:,0]] = y.int()
            grid = sgrid
        grid = grid.float()
        return grid


# ----- 
# abstract Experimenter class
# -----

class Experimenter:
    def __init__(self) -> None:
        self.obs_data = {'design':[], 'y':[], 'design_params':[]}
            # contains lists where each element corressponds to an executed design

    def params2design(self,design_params):
        """ get design in form that can be inputted into densities from design_params"""
        raise NotImplementedError()

    def execute_design(self,design_params):
        """
        Executes experiment identified by design_params
        """
        raise NotImplementedError()

    def get_initial_design_params(self):
        """ starting point for optimization 
        returns: design_params
        """
        raise NotImplementedError()

    def get_obs_data(self):
        """ Get all designs and their outcomes that have been observed so far"""
        return self.obs_data

    def get_eval_data(self,exclude_obs=False):
        """ Get data (design, outcome) for evaluation 
        returns: y,design
        """
        raise NotImplementedError()
    
    def verbose_designs(self):
        """
        Method to give some information about designs, e.g. plotting
        """
        raise NotImplementedError()

    def get_candidate_designs(self):
        """
        returns: list of tuples, each tuple specifies design, first coordinate 
        is design_params, second is design
        """
        raise NotImplementedError()

# ----- 
# specific Experimenters
# -----

class Tissue_discrete(Experimenter):

    
    def __init__(self, 
                 spatial_locations, 
                 readouts, 
                 slice_radius,
        ):
        super().__init__()

        self.X = spatial_locations
        self.y = readouts
        self.slice_radius = slice_radius

        # compute candidate designs
        n_slope_discretizations = 10
        n_intercept_discretizations = 10
        d_border = 0

        warnings.warn('Eventually need to adjust d_border.')
        limits = [self.X.min(0).values[1], self.X.max(0).values[1]]
        slope_angles = torch.linspace(0, torch.pi, n_slope_discretizations)
        slopes = torch.tan(slope_angles)
        intercepts = torch.linspace(
            limits[0]-d_border, limits[1]+d_border, n_intercept_discretizations)
        candidate_designs = []
        for slope,intercept in product(slopes,intercepts):
            design_params = {
                'slope': slope,
                'intercept': intercept
            }
            y,design = self.params2design(design_params)
            candidate_designs.append((design_params,design))  
        self.candidate_designs = candidate_designs

    def get_candidate_designs(self):
        return self.candidate_designs
    
    def get_eval_data(self, exclude_obs=False):
        if exclude_obs:
            raise NotImplementedError()
        return self.y, self.X
    
    def execute_design(self, design_params):
        y,design = self.params2design(design_params)
        self.obs_data['design'].append(design.detach().clone())
        self.obs_data['y'].append(y.detach().clone().int()) 
        self.obs_data['design_params'].append(
            {k:v.detach().clone() for k,v in design_params.items()})

    def params2design(self, design_params):

        intercept = design_params['intercept']
        slope = design_params['slope']

        normal_vector = torch.tensor([torch.nan, -1])
        normal_vector[0] = slope
        norm = torch.norm(normal_vector)
        normal_vector = normal_vector / norm
        dists = torch.abs(torch.matmul(self.X, normal_vector) + intercept/norm)
        in_idx = torch.where(dists <= self.slice_radius)[0]

        design = self.X[in_idx,:]
        y = self.y[in_idx]
        return y,design
    
    def verbose_designs(self,
            design_eval=None,
            pred_probs=None,
            metric_values=None,
            metric_names=None,
        ):

        X,Y = self.X, self.y
        def abline(slope, intercept, label=None, **args):
            """Plot a line from slope and intercept"""
            axes = plt.gca()
            x_vals = np.array(axes.get_xlim())
            y_vals = intercept + slope * x_vals
            plt.plot(x_vals, y_vals, '--', label=label, **args)

        plt.figure(figsize=(5,12))

        if metric_values is not None:
            nrows = 4
        elif design_eval is not None:
            nrows=3
        else:
            nrows=1

        plt.subplot(nrows,1,1)
        plt.scatter(X[:, 0], X[:, 1], s=20, color="gray", marker="H") #, c=onp.isin(onp.arange(len(X)), observed_idx), marker="H", s=20)
        plt.scatter(X[Y==1, 0], X[Y==1, 1], c="red", alpha=0.3, s=20, label="Invasive\ncarcinoma")
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        for i,dd in enumerate(self.obs_data['design_params']):
            intercept = dd['intercept'].detach().cpu().numpy()
            slope = dd['slope'].detach().cpu().numpy()
            abline(slope, intercept, label="Slice {}".format(i + 1), linewidth=3)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend(loc='center left', bbox_to_anchor=(-0.3, 0.5), fontsize=10)
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.title('Slices chosen by the algorithm')

        if nrows >= 3:
            plt.subplot(nrows,1,2)
            plt.scatter(X[:, 0], X[:, 1], s=20, c=pred_probs, marker="H")
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal')
            plt.colorbar()
            plt.title('Predictions of the model')

            plt.subplot(nrows,1,3)
            pred_int = (pred_probs > 0.5).astype(np.int8)
            if pred_int.mean() == 1.0:
                x1 = torch.cat((X[0,0].unsqueeze(0), X[:,0]))
                x2 = torch.cat((X[0,1].unsqueeze(0), X[:,1]))
                c = torch.cat((torch.tensor(0.0).unsqueeze(0), torch.from_numpy(pred_int)))         
                plt.scatter(x1,x2, s=5, c=c, marker="H")  
            else:
                plt.scatter(X[:,0],X[:,1], s=5, c=pred_int, marker="H")
            plt.gca().invert_yaxis()
            plt.title('Binary predictions of the model')
            plt.gca().set_aspect('equal')

        if nrows >= 4:
            plt.subplot(nrows,1,4)
            assert len(metric_values) == len(metric_names)
            df = pd.DataFrame(metric_values, 
                              index=metric_names, 
                              columns=range(len(metric_values[0]))).transpose()
            df['number designs'] = df.index + 1
            for i in range(len(metric_values)):
                sns.lineplot(data=df, 
                             y=metric_names[i], 
                             x='number designs', 
                             label=metric_names[i],
                             markers=True)
            plt.gca().set_aspect('equal')
            plt.title('Metrics of predictions')
        plt.show()

class Tissue_cont_indicator(Experimenter):
    """
    No fragmentation.
    Design is nx3, where first column is weight
    """
    def __init__(self, 
                 spatial_locations, 
                 readouts, 
                 slice_radius,
                 shelf_scale=0.5,
                 n_plateau=5,
        ):
        super().__init__()

        assert len(spatial_locations) == len(readouts)
        
        self.X = spatial_locations
        self.y = readouts
        self.slice_radius = slice_radius
        self.shelf_radius = self.slice_radius * shelf_scale
        self.n_total, self.p = self.X.shape
        self.n_plateau = n_plateau

        # plot plateau
        self.plateau_fct = lambda x: torch.exp(-(x/self.slice_radius)**(2*self.n_plateau))
        a = 2* self.slice_radius
        x = torch.linspace(-a,a,100)
        y = self.plateau_fct(x).numpy()
        plt.figure(figsize=(3,1))
        plt.plot(x,y)
        plt.plot([-self.slice_radius,-self.slice_radius],[0,1],'--',color='black')
        plt.plot([self.slice_radius,self.slice_radius],[0,1],'--',color='black')
        plt.plot([-(self.shelf_radius+self.slice_radius),-(self.shelf_radius+self.slice_radius)],[0,1],'--',color='gray')
        plt.plot([(self.shelf_radius+self.slice_radius),(self.shelf_radius+self.slice_radius)],[0,1],'--',color='gray')
        plt.title('Plateau function')
        plt.show()

        # setup data
        # self.X_fragment_idx = [np.arange(self.n_total)]
        self.obs_data = {
            'design_params': [],
            'design': [],
            'y': [],
        }

    def get_initial_design_params(self):
        return {
            'slope': torch.tensor(0.0),
            'intercept': torch.tensor(0.0)
        }

    def get_eval_data(self, exclude_obs=False):
        X = torch.cat((torch.ones((self.X.shape[0],1)).float(),self.X),dim=1)
        return self.y, X
    
    def execute_design(self, design_params):
        y,design,in_idx = self.params2design(design_params,return_in_idx=True)
        y = y[in_idx]
        design = design[in_idx,:]
        design[:,0] = 1.0
        self.obs_data['design'].append(design.detach().clone())
        self.obs_data['y'].append(y.detach().clone().int()) 
        self.obs_data['design_params'].append(
            {k:v.detach().clone() for k,v in design_params.items()})

    def params2design(self, design_params, return_in_idx=False):

        intercept = design_params['intercept']
        slope = design_params['slope']

        normal_vector = torch.tensor([torch.nan, -1])
        normal_vector[0] = slope
        norm = torch.norm(normal_vector)
        normal_vector = normal_vector / norm
        dists = torch.abs(torch.matmul(self.X, normal_vector) + intercept/norm)
        in_idx = torch.where(dists <= self.slice_radius)[0]
        shelf_idx = torch.where(torch.logical_and(
            self.slice_radius < dists, dists <= self.slice_radius + self.shelf_radius))[0]
        weights_all = self.plateau_fct(dists)
        idx = torch.cat([in_idx, shelf_idx])

        # design = torch.cat((weights_all[idx].reshape(-1,1),self.X[idx,:]),dim=1)
        # y = self.y[idx]
        # if return_in_idx:
        #     in_idx_rel = torch.tensor([1]*len(in_idx) + [0]*len(shelf_idx))
        #     return y,design,in_idx_rel
        design = torch.cat((weights_all.reshape(-1,1),self.X),dim=1)
        y = self.y
        if return_in_idx:
            return y,design, torch.arange(self.X.shape[0])
        return y,design
    
    def verbose_designs(self,
            design_eval=None,
            pred_probs=None,
            metric_values=None,
            metric_names=None,
        ):

        X,Y = self.X, self.y
        def abline(slope, intercept, label=None, **args):
            """Plot a line from slope and intercept"""
            axes = plt.gca()
            x_vals = np.array(axes.get_xlim())
            y_vals = intercept + slope * x_vals
            plt.plot(x_vals, y_vals, '--', label=label, **args)

        plt.figure(figsize=(4,7))

        if metric_values is not None:
            nrows = 4
        elif design_eval is not None:
            nrows=3
        else:
            nrows=1

        plt.subplot(nrows,1,1)
        plt.scatter(X[:, 0], X[:, 1], s=20, color="gray", marker="H") #, c=onp.isin(onp.arange(len(X)), observed_idx), marker="H", s=20)
        plt.scatter(X[Y==1, 0], X[Y==1, 1], c="red", alpha=0.3, s=20, label="Invasive\ncarcinoma")
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        for i,dd in enumerate(self.obs_data['design_params']):
            intercept = dd['intercept'].detach().cpu().numpy()
            slope = dd['slope'].detach().cpu().numpy()
            abline(slope, intercept, label="Slice {}".format(i + 1), linewidth=3)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend(loc='center left', bbox_to_anchor=(-0.3, 0.5), fontsize=10)
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.title('Slices chosen by the algorithm')

        if nrows >= 3:
            plt.subplot(nrows,1,2)
            plt.scatter(X[:, 0], X[:, 1], s=20, c=pred_probs, marker="H")
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal')
            plt.title('Predictions of the model')

            plt.subplot(nrows,1,3)
            pred_int = (pred_probs > 0.5).astype(np.int8)
            plt.scatter(X[:,0],X[:,1], s=5, c=pred_int, marker="H")
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal')

        if nrows >= 4:
            plt.subplot(nrows,1,4)
            assert len(metric_values) == len(metric_names)
            df = pd.DataFrame(metric_values, 
                              index=metric_names, 
                              columns=range(len(metric_values[0]))).transpose()
            df['number designs'] = df.index + 1
            for i in range(len(metric_values)):
                sns.lineplot(data=df, 
                             y=metric_names[i], 
                             x='number designs', 
                             label=metric_names[i],
                             markers=True)
            plt.gca().set_aspect('equal')
            plt.title('Metrics of predictions')
        plt.show()

class Tissue_continuous(Experimenter):
    """
    Tissue without different fragments
    design_params: dict with keys 'fragment_num' and 'slice'
    design: nx2 denoting location of points
    y: nx binary vector denoting presence/absence of cancer cells
    """

    def __init__(self, X:torch.tensor, y, slice_radius):
        """
        slice_radius: given in distance metric of the grid
        """
        super().__init__()
        
        # create grid
        assert X.dtype == torch.int64
        dim_outcome = 1 if len(y.shape)==1 else y.shape[1]
        X = X - X.min(axis=0).values
        xmax = X.max(axis=0).values
        grid = torch.zeros((1,dim_outcome,xmax[1]+1,xmax[0]+1)).int() 
        grid[0,:,X[:,1],X[:,0]] = y.int()

        # smooth grid
        x2 = torch.zeros((1,1,1,grid.shape[3]))
        sgrid = torch.concat([x2,grid,x2],axis=2)
        x3 = torch.zeros((1,1,sgrid.shape[2],1))
        sgrid = torch.concat([x3,sgrid,x3],axis=3)
        sgrid = (sgrid[:,:,1:-1,0:-2] + sgrid[:,:,1:-1,2:] \
            + sgrid[:,:,0:-2,1:-1] + sgrid[:,:,2:,1:-1]) / 4
        sgrid = (sgrid > 0.5).int()
        sgrid[0,:,X[:,1],X[:,0]] = y.int()
        grid = sgrid
        grid = grid.float()
        
        # set variables
        self.grid = grid
        self.x_means = X.float().mean(axis=0)
        self.x_std = X.float().std()
        self.X_actual = X
        self.X_normed = (X - self.x_means) / self.x_std
        self.X_normed_min = self.X_normed.min(axis=0).values
        self.X_normed_max = self.X_normed.max(axis=0).values
        self.y = y

        # calculate scaling and sample grid size
        self.slice_radius = slice_radius 
        self.slicing = {}
        pixel_short = ceil(2*slice_radius)
        pixel_long = ceil(sqrt( grid.shape[2]**2 + grid.shape[3]**2))
        # self.slicing['scale_x_in'] = torch.tensor(pixel_long / grid.shape[3])
        self.slicing['scale_x_in'] = torch.tensor(2.0) # don't know how to set to exactly diagonal length
        self.slicing['scale_y_in'] = torch.tensor(pixel_short / grid.shape[2])
        self.slicing['grid_size'] = (1,1,pixel_short,pixel_long)

        # adjust for uneven ratio on axes
        if grid.shape[2] > grid.shape[3]:
            self.slicing['scale_x_out'] = 1
            self.slicing['scale_y_out'] = grid.shape[3] / grid.shape[2]
        else:
            self.slicing['scale_x_out'] = grid.shape[2] / grid.shape[3]
            self.slicing['scale_y_out'] = 1

    def get_initial_design_params(self):
        return {
            'alpha': torch.tensor(0.0),
            'y_intercept': torch.tensor(0.0)
        }


    def execute_design(self, design_params):
        y,design = self.params2design(design_params)
        self.obs_data['design'].append(design.detach())
        self.obs_data['y'].append(y.detach().int())

    def get_eval_data(self, exclude_obs=False):
        if exclude_obs:
            raise NotImplementedError()
        else:
            return self.y,self.X_normed

    def params2design(self, design_params):

        # compute slice 
        alpha, y_intercept = self._extract_design(design_params)
        scale_x_in = self.slicing['scale_x_in']
        scale_y_in = self.slicing['scale_y_in']
        scale_x_out = self.slicing['scale_x_out']
        scale_y_out = self.slicing['scale_y_out']

        rotation_mat = torch.zeros((2,3))
        rotation_mat[0,0] = scale_x_out * scale_x_in * torch.cos(alpha)
        rotation_mat[0,1] = scale_x_out * scale_y_in * -torch.sin(alpha)
        rotation_mat[1,0] = scale_y_out * scale_x_in * torch.sin(alpha)
        rotation_mat[1,1] = scale_y_out * scale_y_in * torch.cos(alpha)
        rotation_mat[1,2] = scale_y_out * y_intercept
        rotation_mat = rotation_mat.unsqueeze(0)

        grid_loc = torch.nn.functional.affine_grid(
            rotation_mat, self.slicing['grid_size'])
        grid_out = torch.nn.functional.grid_sample(
            self.grid, grid_loc, padding_mode='zeros')

        # transform to design
        y = grid_out.reshape((-1))
        design_constrained = grid_loc.reshape((-1,2))
        design_normed = (1+design_constrained) / 2 * (self.X_normed_max - self.X_normed_min) + self.X_normed_min
        # design_actual = design_normed * self.x_std + self.x_means.reshape((1,-1))
        
        return y,design_normed
    
    def _extract_design(self,design_param):
        alpha = design_param['alpha']
        y_intercept = design_param['y_intercept']
        return alpha, y_intercept
    
    def verbose_designs(self,
            design_eval=None,
            pred_probs=None,
            metric_values=None,
            metric_names=None,
    ):

        if metric_values is not None:
            nrows=4
        elif design_eval is not None:
            nrows=3
        else:
            nrows=1
        plt.subplots(nrows=nrows,ncols=1, figsize=(4,7))
        X = self.X_normed

        plt.subplot(nrows,1,1)
        # grid = self.grid[0,0,:,:].int()
        # mesh_y,mesh_x = torch.meshgrid(
        #     torch.arange(grid.shape[0]),torch.arange(grid.shape[1]))
        # plt.scatter(mesh_x.reshape((-1)),mesh_y.reshape((-1)), s=5, c=grid.reshape((-1)),
        #             alpha=0.05)
        plt.scatter(X[:,0],X[:,1], s=5, c=self.y, alpha=0.05)
        for y,design in zip(self.obs_data['y'],self.obs_data['design']):
            coords = deepcopy(design.cpu().numpy())
            # coords[:,0] = (1+coords[:,0]) * grid.shape[1]/2
            # coords[:,1] = (1+coords[:,1]) * grid.shape[0]/2
            norm = (y.cpu().numpy() > 0.5).astype(np.int16)
            plt.scatter(coords[:,0],coords[:,1], s=5, c=norm)       
        plt.gca().set_aspect('equal')
        plt.title('Slices')

        if nrows >= 3:
            plt.subplot(nrows,1,2)
            # mesh_y,mesh_x = torch.meshgrid(
            #     torch.arange(grid.shape[0]),torch.arange(grid.shape[1])) 
            plt.scatter(design_eval[:,0],design_eval[:,1], s=5, c=pred_probs, marker="H")
            # plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal')
            plt.colorbar()
            plt.title('Posterior predictions')
            print(pred_probs.max())

            plt.subplot(nrows,1,3)
            # mesh_y,mesh_x = torch.meshgrid(
            #     torch.arange(grid.shape[0]),torch.arange(grid.shape[1])) 
            pred_int = (pred_probs > 0.5).astype(np.int8)
            plt.scatter(design_eval[:,0],design_eval[:,1], s=5, c=pred_int, marker="H")
            # plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal')
            plt.title('Posterior thresholded 0.5')

        if nrows == 4:
            plt.subplot(nrows,1,4)
            assert len(metric_values) == len(metric_names)
            df = pd.DataFrame(metric_values, 
                              index=metric_names, 
                              columns=range(len(metric_values[0]))).transpose()
            df['number designs'] = df.index + 1
            for i in range(len(metric_values)):
                sns.lineplot(data=df, 
                             y=metric_names[i], 
                             x='number designs', 
                             label=metric_names[i],
                             markers=True)
            plt.title('Metrics of predictions')
        plt.show()

# -----
# abstract design network class
# -----

class Design_Network(torch.nn.Module):
    def __init__(self, encoder, emitter) -> None:
        super().__init__()
        assert isinstance(encoder, torch.nn.Module)
        assert isinstance(emitter, torch.nn.Module)
        self.encoder = encoder  # (xi=Nxd, y=Nxk) -> Nxr
        self.emitter = emitter  # (repr=Nxr) -> Nxd
        self.design_dim = encoder.design_dim
        self.y_dim = encoder.y_dim
        self.repr_dim = encoder.repr_dim

    def forward(self, xi, y):
        """ 
            xi: N x design_dim, 
            y: N x y_dim
        """
        self.buffer_mean_representation = self.buffer_mean_representation + self.encoder(xi,y)
        return self.emitter(self.buffer_mean_representation)
    
    def reset_buffer(self, batch_size):
        self.buffer_mean_representation = torch.zeros((batch_size,self.repr_dim))