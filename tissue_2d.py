import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import product
import torch
from math import ceil,sqrt

from src.data_utils import get_data_dir
import os
import src.distributions as distr
from copy import deepcopy
from itertools import product

from src.data_utils import Experimenter
from src.distributions import Conditional_distr, Distr

EPSILON = 1e-8

# ----- 
# loading datasets code partly copied from https://github.com/andrewcharlesjones/spatial-experimental-design
# -----

class Prostate_cancer_2d():
    """
    from https://www.10xgenomics.com/resources/datasets/human-prostate-cancer-adenocarcinoma-with-invasive-carcinoma-ffpe-1-standard-1-3-0 
    download 'Spatial imaging data' and 'clustering' into data folder
    """

    def get_data(self, name='invasive', processed=False):
        assert name in ['normal_section','invasive']

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
# specific Experimenters
# -----

class Tissue_discrete(Experimenter):
    """
    2d Tissue without fragmentation after slicing, parametrization by intercept
    and slope, heavy overlap with Tissue_continuous
    """
    
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

        # warnings.warn('Eventually need to adjust d_border.')
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
    2d tissue without fragmentation.
    Design is nx3, where first column is weight.
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
        plt.title('Plateau function of tissue model')
        plt.show()

    def get_initial_design_params(self):
        return {
            'slope': torch.tensor(0.0),
            'intercept': torch.tensor(0.0)
        }

    def get_eval_data(self, exclude_obs=False):
        if exclude_obs:
            raise NotImplementedError()
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

        # define plot size
        if metric_values is not None:
            nrows = 4
        elif design_eval is not None:
            nrows=3
        else:
            nrows=1
        plt.figure(figsize=(4,7))

        # helpful fct
        X,Y = self.X, self.y
        def abline(slope, intercept, label=None, **args):
            """Plot a line from slope and intercept"""
            axes = plt.gca()
            x_vals = np.array(axes.get_xlim())
            y_vals = intercept + slope * x_vals
            plt.plot(x_vals, y_vals, '--', label=label, **args)

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
    Tissue without fragmentation.
    design_params: dict with keys '
    design: nx2 denoting location of points
    y: nx0 binary vector denoting presence/absence of cancer cells
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
# specfic distributions
# -----
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