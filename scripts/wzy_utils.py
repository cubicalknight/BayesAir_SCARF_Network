import numpy as np
import ot
import torch

import pyro.distributions as dist
import zuko
from zuko.flows.core import LazyDistribution
from flowgmm.flow_ssl.realnvp.realnvp import RealNVPTabular


import torch.nn as nn
from enum import Enum, auto
import abc


# starting to add the required components for inference on our more complicated graphical model

# w -> c (for now, let's just use a classifier -- in general it will be harder...)
    
class MLPClassifierW2C(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, n_layers=1, n_classes=2, act_layer=nn.Softplus()):
        super().__init__()

        modules = []
        if n_layers < 2:
            modules.append(nn.Linear(input_dim, n_classes))
        else:
            modules.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(n_layers-2):
                modules.append(nn.Linear(input_dim, hidden_dim))
                modules.append(nn.Softplus())
            modules.append(nn.Linear(hidden_dim, n_classes))

        self.network = nn.Sequential(*modules)
        self.input_dim = input_dim
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        input = x.reshape(-1, self.input_dim)
        return self.network(input)
    
    def predict(self, x):
        weights = self.forward(x)
        _, predicted = torch.max(weights, 1)
        return predicted
    

    
# TODO: flowgmm??
# TODO: alternatively: do w->z in a single step????

# example:

# # https://github.com/tonyduan/mixture-density-network/blob/master/

# class NoiseType(Enum):
#     DIAGONAL = auto()
#     ISOTROPIC = auto()
#     ISOTROPIC_ACROSS_CLUSTERS = auto()
#     FIXED = auto()

# class MixtureDensityNetwork(nn.Module):
#     """
#     Mixture density network.

#     [ Bishop, 1994 ]

#     Parameters
#     ----------
#     dim_in: int; dimensionality of the covariates
#     dim_out: int; dimensionality of the response variable
#     n_components: int; number of components in the mixture model
#     """
#     def __init__(self, dim_in, dim_out, n_components, hidden_dim, noise_type=NoiseType.DIAGONAL, fixed_noise_level=None):
#         super().__init__()
#         assert (fixed_noise_level is not None) == (noise_type is NoiseType.FIXED)
#         num_sigma_channels = {
#             NoiseType.DIAGONAL: dim_out * n_components,
#             NoiseType.ISOTROPIC: n_components,
#             NoiseType.ISOTROPIC_ACROSS_CLUSTERS: 1,
#             NoiseType.FIXED: 0,
#         }[noise_type]
#         self.dim_in, self.dim_out, self.n_components = dim_in, dim_out, n_components
#         self.noise_type, self.fixed_noise_level = noise_type, fixed_noise_level
#         self.pi_network = nn.Sequential(
#             nn.Linear(dim_in, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, n_components),
#         )
#         self.normal_network = nn.Sequential(
#             nn.Linear(dim_in, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, dim_out * n_components + num_sigma_channels)
#         )

#     def forward(self, x, eps=1e-6):
#         #
#         # Returns
#         # -------
#         # log_pi: (bsz, n_components)
#         # mu: (bsz, n_components, dim_out)
#         # sigma: (bsz, n_components, dim_out)
#         #
#         log_pi = torch.log_softmax(self.pi_network(x), dim=-1)
#         normal_params = self.normal_network(x)
#         mu = normal_params[..., :self.dim_out * self.n_components]
#         sigma = normal_params[..., self.dim_out * self.n_components:]
#         if self.noise_type is NoiseType.DIAGONAL:
#             sigma = torch.exp(sigma + eps)
#         if self.noise_type is NoiseType.ISOTROPIC:
#             sigma = torch.exp(sigma + eps).repeat(1, self.dim_out)
#         if self.noise_type is NoiseType.ISOTROPIC_ACROSS_CLUSTERS:
#             sigma = torch.exp(sigma + eps).repeat(1, self.n_components * self.dim_out)
#         if self.noise_type is NoiseType.FIXED:
#             sigma = torch.full_like(mu, fill_value=self.fixed_noise_level)
#         mu = mu.reshape(-1, self.n_components, self.dim_out)
#         sigma = sigma.reshape(-1, self.n_components, self.dim_out)
#         return log_pi, mu, sigma

#     def loss(self, x, y):
#         log_pi, mu, sigma = self.forward(x)
#         z_score = (y.unsqueeze(1) - mu) / sigma
#         normal_loglik = (
#             -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
#             -torch.sum(torch.log(sigma), dim=-1)
#         )
#         loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
#         return -loglik

#     def sample(self, x):
#         log_pi, mu, sigma = self.forward(x)
#         cum_pi = torch.cumsum(torch.exp(log_pi), dim=-1)
#         rvs = torch.rand(len(x), 1).to(x)
#         rand_pi = torch.searchsorted(cum_pi, rvs)
#         rand_normal = torch.randn_like(mu) * sigma + mu
#         samples = torch.take_along_dim(rand_normal, indices=rand_pi.unsqueeze(-1), dim=1).squeeze(dim=1)
#         return samples


# TODO: define a framework:

class WZY(object):
    def __init__(self):
        pass
        






if __name__ == "__main__":
    from tqdm import tqdm

    torch.manual_seed(0)

    n = 10
    train_n = n-1
    test_n = 1
    dataset = []
    for _ in range(train_n + test_n):
        x = torch.rand(6)
        y = (x > .5).long()
        dataset.append((x, y))

    clsf = MLPClassifierW2C(1, 10, 2)

    clsf_optim = torch.optim.Adam(
        clsf.parameters(),
        lr=1e-3,
    )

    epbar = tqdm(range(5000))
    for epoch in epbar:  # loop over the dataset multiple times
        pbar = tqdm(range(test_n), leave=False)
        for i in pbar:
            # get the inputs; dataset is a list of [inputs, labels]
            x, y = dataset[i]
            # zero the parameter gradients
            clsf_optim.zero_grad()

            # forward + backward + optimize
            out = clsf.forward(x)
            loss = clsf.loss(out, y)
            loss.backward()
            clsf_optim.step()

            x, y = dataset[train_n]
            out = clsf.forward(x)
            val_loss = clsf.loss(out, y)
            pbar.set_description(f"train and val loss: {loss:.2f} | {val_loss:.2f}")
            if i == test_n - 1:
                epbar.set_description(f"train and val loss: {loss:.2f} | {val_loss:.2f}")

    x = torch.tensor([1,2,3,4,5,6,7,8,9,10], dtype=torch.float32) / 10.0
    y = torch.tensor([0,0,0,0,0,1,1,1,1,1], dtype=torch.long)
    print(clsf.predict(x))