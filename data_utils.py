from importlib.metadata import distribution
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

from utils import get_data_dir
import os
import distributions as distr

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
    
class Allen_brain_3d(Data_3d):

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
    

class Prostate_cancer_2d(Data_2d):
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
        if processed:
            X = (X - X.mean(axis=0)) / X.std(axis=0)
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

    def get_designs_discrete(self,
            n_slope_discretizations = 10,
            n_intercept_discretizations = 10,):
        
        if self.processed:
            d_border = 0
        else:
            d_border = 5

        limits = [self.X.min(0)[1], self.X.max(0)[1]]
        slope_angles = np.linspace(0, np.pi, n_slope_discretizations)
        slopes = np.tan(slope_angles)
        intercepts = np.linspace(limits[0] - d_border, limits[1] + d_border, n_intercept_discretizations)
        designs1, designs2 = np.meshgrid(intercepts, slopes)
        candidate_designs = np.vstack([designs1.ravel(), designs2.ravel()]).T
        return candidate_designs
    
    def plot_slices(self, 
                    designs, 
                    predictive_model:distr.Conditional_distr,
                    metric_values=None,
                    metric_names=None,
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

        plt.subplot(1,ncols,1)
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

        plt.subplot(1,ncols,2)
        preds = predictive_model.predict(X,thetas=thetas)
        plt.scatter(X[:, 0], X[:, 1], s=20, c=preds, marker="H")
        plt.gca().invert_yaxis()
        plt.title('Predictions of the model')

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