# taken from https://github.com/andrewcharlesjones/spatial-experimental-design/blob/main/models/tissue.py
import numpy as np
import sys
sys.path.append("../util/")
import warnings

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