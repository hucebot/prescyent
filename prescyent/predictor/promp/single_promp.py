# for a more complete implementation : https://pypi.org/project/mp-pytorch/
from typing import Any, List

import math
import torch
import matplotlib.pyplot as plt
import numpy as np
import functools


class SinglePromp:
    # make from a list of trajectory
    def __init__(
        self,
        num_bf,
        ridge_factor,
        mean_w: torch.Tensor = None,
        std_w: torch.Tensor = None,
        data=None,
    ):
        # parameters
        self.num_bf = num_bf  # lower = smoother
        self.ridge_factor = ridge_factor  # default value
        self.std_bf = 1.0 / (self.num_bf**2)
        self.alpha = 1.0  # time-modulation - NOT supported yet

        self.data = data
        self.mean_w = mean_w
        self.std_w = std_w
        if mean_w is not None:
            assert std_w is not None
            self.s = self.data.size(1)
            self.phase = torch.linspace(
                0, 1, math.floor(self.s / self.alpha)
            )  # should be phase_vec
            self.psi = self.gen_basis_functions(self.phase)  # should psi_mtx

    def clone(self):
        return SinglePromp(
            self.num_bf,
            self.ridge_factor,
            self.mean_w.clone(),
            self.std_w.clone(),
            self.data.clone(),
        )

    # make all the sequences the same length and put them in a 2-dimensional tensor
    # in the tensor, each row is a demonstration
    # we use np.interp to interpolate (actually to resample the trajectory)
    def normalize(self, trajs: List[Any], length: int) -> torch.Tensor:
        data_normalized = torch.Tensor(len(trajs), length)
        new_x = np.linspace(0, 1, length)
        for i, d in enumerate(trajs):
            x = np.linspace(0, 1, d.size(0))
            new_y = np.interp(new_x, x, d)
            data_normalized[i, :] = torch.Tensor(new_y)
        return data_normalized

    # generate a psi matrix (RBF matrix) from a phase vector (a linear space)
    def gen_basis_functions(self, phase: torch.Tensor):
        # create vector of rbf centers
        centers = torch.linspace(
            0.0 - 2 * self.std_bf, 1.0 + 2 * self.std_bf, self.num_bf
        )

        # create a matrix with n_rbf rows x phase.size() cols.
        # each row is the phase
        phase_bf_mtx = phase.repeat(self.num_bf, 1)
        assert phase_bf_mtx.size(0) == self.num_bf
        assert phase_bf_mtx.size(1) == phase.size(0)

        # create a matrix with n_rbf rows x phase.size() cols.
        # each col is centers
        c_mtx = centers.repeat(phase.size(0), 1).T
        assert c_mtx.size(0) == self.num_bf
        assert c_mtx.size(1) == phase.size(0)

        # compute t-c(i)
        phase_diff = phase_bf_mtx - c_mtx

        # compute RBF matrix
        psi = np.exp(-0.5 * phase_diff**2 / self.std_bf)

        # normalize psi colwise / TODO : why?
        # print('psi sum:', psi.sum(dim=0))
        psi = psi / psi.sum(dim=0)

        return psi

    # for checks / debugging
    def plot(self, prefix=""):
        # the data
        fig, ax = plt.subplots()
        for i in range(self.data.size(0)):
            x = np.arange(0, self.data.size(1))
            ax.plot(x, self.data[i, :], label=str(i))
        fig.legend()
        fig.savefig(prefix + "promp_data.pdf")

        # the RBF
        fig, ax = plt.subplots()
        for i in range(self.psi.size(0)):
            x = np.arange(0, self.psi.size(1))
            ax.plot(x, self.psi[i, :], label=str(i))  # each row is a RBF
        fig.legend()
        fig.savefig(prefix + "promp_bases.pdf")

        # the mean
        fig, ax = plt.subplots()
        for i in range(self.data.size(0)):
            x = np.arange(0, self.data.size(1))
            ax.plot(x, self.data[i, :], label=str(i), alpha=0.5)
        x = np.arange(0, self.psi.size(1))
        m = self.mean()
        s = self.std()
        ax.plot(x, m, lw=5)
        ax.fill_between(x, m - s, m + s, alpha=0.2)
        fig.savefig(prefix + "mean.pdf")
        plt.close()

    def train(self, trajs: List[torch.Tensor]):
        # data pre-processing
        mean_length = math.floor(
            functools.reduce(lambda x, y: x + y.size(0), trajs, 0) / float(len(trajs))
        )
        # Put all the trajectories with the same length (interpolate when needed)
        self.data = self.normalize(trajs, mean_length)
        assert self.data.size(0) == len(trajs)
        self.s = mean_length  # number of timesteps in the trajectory / we take the mean from the data

        # create the basis functions
        self.phase = torch.linspace(
            0, 1, math.floor(self.s / self.alpha)
        )  # should be phase_vec
        self.psi = self.gen_basis_functions(self.phase)  # should psi_mtx

        # compute the ridge pseudo inverse, that is, fit the weights of the RBF to the data
        id = torch.eye(self.psi.size(0), self.psi.size(0))
        psi_t_inv = (
            self.psi @ self.psi.T + self.ridge_factor * id
        ).inverse() @ self.psi

        # put demonstrations weights matrices into a single matrix
        # linear ridge regression for each demonstration
        # each column is a demonstration (should be rows to be consistent with data?)
        w_mtx = torch.Tensor(self.num_bf, self.data.size(0))
        for i in range(0, self.data.size(0)):
            w_mtx[:, i] = (
                psi_t_inv @ self.data[i, :]
            )  # is this type converted to a float?

        # mean of each row
        self.mean_w = w_mtx.mean(dim=1)

        # variance of each row
        self.std_w = w_mtx.std(dim=1)

    # TODO: we should have a simpler version that by adding a single point
    # this would be faster for multiple/continuous predictions
    def condition(self, traj: torch.Tensor, std: torch.Tensor = None):
        if std is None:
            std = torch.zeros_like(traj) + 0.0000001
        mean_w = self.mean_w.clone()
        std_w = self.std_w.clone()
        # this  will bug if traj.size() > self.s
        for i in range(0, traj.size(0)):
            # get a single "phase" from time (here time-step)
            # and make a single basis function
            psi_obs = self.gen_basis_functions(torch.Tensor([i / float(self.s)]))
            obs = torch.Tensor([traj[i]])
            obs_std = torch.Tensor([std[i]])
            # we make a simple diagonal covariance matrix ; not in C++ code
            cov = std_w * torch.eye(self.std_w.size(0), self.std_w.size(0))
            l_ = cov @ psi_obs @ (obs_std + psi_obs.T @ cov @ psi_obs + 0.01).inverse()
            mean_w = mean_w + l_ @ (obs - psi_obs.T @ mean_w)
            std_w = std_w - l_ @ psi_obs.T @ std_w

        # self.mean_w = mean_w
        # self.std_w = std_w
        return SinglePromp(
            self.num_bf, self.ridge_factor, mean_w, std_w, self.data.clone()
        )

    def mean(self):
        return self.psi.T @ self.mean_w

    def std(self):
        return self.psi.T @ self.std_w
