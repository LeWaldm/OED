import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import r2_score
from brainrender.atlas_specific import GeneExpressionAPI


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
    
