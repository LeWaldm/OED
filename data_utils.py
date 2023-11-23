import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

class Data_2d():
    def __init__(self, radius=5):
        self.radius = radius

    def get_data():
        raise NotImplementedError()

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
        self.simulated = False
        self.X = None
        self.Y = None

    def get_data():
        raise NotImplementedError()

    @staticmethod
    def compute_point_to_plane_dists(points, plane, signed=False):
        dists_signed = (plane[0] * points[:, 0] + plane[1] * points[:, 1] + plane[2] * points[:, 2] + plane[3]) / np.sqrt(np.sum(plane[:3] ** 2))
        return dists_signed if signed else np.abs(dists_signed)

    @staticmethod
    def compute_eig(cov, noise_variance):
        return 0.5 * np.linalg.slogdet(1 / noise_variance * cov + np.eye(len(cov)))[1]

    def plot_slices(self, chosen_designs, observed_idx):
        coords = self.X

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


class Simulated_GP_3d(Data_3d):

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

    def get_data(self,grid_size = 10):
        if self.simulated:
            print('Return already simulated data.')
            return self.X, self.Y

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
        self.simulated = True

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