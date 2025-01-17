"""Define useful functions"""
from typing import Any

import numpy as np
import ot
import torch
import pyro.distributions as dist
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
    BinaryROC,
)
from zuko.flows.core import LazyDistribution

import torch.nn as nn

from enum import Enum, auto

import abc




def kl_divergence(p_dist, q_dist, num_particles=10):
    """KL divergence between two distributions."""
    p_samples, p_logprobs = p_dist.rsample_and_log_prob((num_particles,))
    q_logprobs = q_dist.log_prob(p_samples)

    kl_divergence = (p_logprobs - q_logprobs).mean(dim=0)

    return kl_divergence


def cross_entropy(p_samples, q_dist):
    """Cross entropy between two distributions."""
    q_logprobs = q_dist.log_prob(p_samples)

    cross_entropy = -q_logprobs.mean(dim=0)

    return cross_entropy


def f_score(nominal_samples, failure_samples, nominal_dist, failure_dist):
    """Compute the f score of a likelihood ratio test for failure."""
    # Concatenate the samples
    samples = torch.cat([nominal_samples, failure_samples], dim=0)
    true_labels = torch.cat(
        [
            torch.zeros(nominal_samples.shape[0]),
            torch.ones(failure_samples.shape[0]),
        ],
        dim=0,
    ).to(samples.device)

    # Compute the likelihood ratio and classify as a failure if is >= 1 (more likely
    # to be from the failure distribution than the nominal distribution)
    likelihood_ratio = failure_dist.log_prob(samples) - nominal_dist.log_prob(samples)
    predicted_labels = (likelihood_ratio >= 0).float()

    # Get the f score
    f1_loss = BinaryF1Score().to(samples.device)
    f_score = f1_loss(predicted_labels, true_labels)
    return f_score


def sinkhorn(p_samples, q_samples, epsilon=1.0):
    # Uniform weights on the samples
    n = p_samples.shape[0]
    a, b = (np.ones((n,)) / n, np.ones((n,)) / n)
    # EMD loss matrix
    M = ot.dist(
        p_samples.detach().cpu().numpy(),
        q_samples.detach().cpu().numpy(),
    )
    # Solve regularized EMD (sinkhorn)
    sinkhorn_dist = ot.sinkhorn2(a, b, M, epsilon, method="sinkhorn_log")
    return sinkhorn_dist


def anomaly_classifier_metrics(scores, labels):
    """Compute classification metrics (e.g. AUCROC and AUCPR)."""
    aucroc = BinaryAUROC().to(scores.device)(scores, labels)
    aucpr = BinaryAveragePrecision().to(scores.device)(scores, labels)

    # Compute the optimal decision threshold using the receiver operating curve
    fpr, tpr, thresholds = BinaryROC().to(scores.device)(scores, labels)
    optimal_idx = torch.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    precision = BinaryPrecision(threshold=optimal_threshold.item()).to(scores.device)(
        scores, labels
    )
    recall = BinaryRecall(threshold=optimal_threshold.item()).to(scores.device)(
        scores, labels
    )

    return aucroc, aucpr, precision, recall


class RBF(torch.nn.Module):
    # Implementation from Yiftach Beer on GitHub
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (
            torch.arange(n_kernels) - n_kernels // 2
        )
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples**2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(
            -L2_distances[None, ...]
            / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[
                :, None, None
            ]
        ).sum(dim=0)


class MMDLoss(torch.nn.Module):
    # Implementation from Yiftach Beer on GitHub
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


def simple_mmd(X, Y):
    Z = torch.vstack([X, Y])
    L2_distances = torch.cdist(Z, Z) ** 2
    K = torch.exp(-0.5 * L2_distances)
    X_size = X.shape[0]
    XX = K[:X_size, :X_size].mean()
    XY = K[:X_size, X_size:].mean()
    YY = K[X_size:, X_size:].mean()
    return XX - 2 * XY + YY


class ContextFreeBase(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, _):
        return self.base()


class MixtureOfDiagNormals(dist.MixtureOfDiagNormals):
    def rsample_and_log_prob(self, sample_shape=torch.Size()):
        """
        Returns a sample and the log probability of the sample.

        Arguments:
            sample_shape: Shape of the sample.

        Returns:
            A sample and the log probability of the sample.
        """
        sample = self.rsample(sample_shape)
        log_prob = self.log_prob(sample)

        return sample, log_prob


class ConditionalGaussianMixture(LazyDistribution):
    def __init__(self, n_context: int, n_features: int):
        super().__init__()
        self.n_context = n_context
        self.n_features = n_features

        self.means = torch.nn.Parameter(torch.randn(n_context + 1, n_features))
        self.log_vars = torch.nn.Parameter(torch.randn(n_context + 1, n_features))

    def forward(self, c: Any = None) -> torch.distributions.Distribution:
        r"""
        Arguments:
            c: A context :math:`c`.

        Returns:
            A distribution :math:`p(X | c)`.
        """
        if c is None:
            c = torch.zero(self.n_context).to(self.means.device)

        # Add a leading zero to the context
        c = torch.cat([torch.zeros(*c.shape[:-1], 1).to(c.device), c], dim=-1)

        # Wherever the row is all zero, make just the first element 1
        zero_rows = c.sum(dim=-1) == 0
        c[zero_rows, 0] = 1.0

        return MixtureOfDiagNormals(
            locs=self.means,
            coord_scale=torch.exp(self.log_vars).sqrt(),
            component_logits=c,
        )


class FlowEnsembleDistribution(torch.distributions.Distribution):
    has_rsample = True

    def __init__(self, flow, contexts, weights, temperature):
        """Initialize the ensemble distribution.

        Args:
            flow (zuko.core.LazyDistribution): Flow.
            contexts (torch.Tensor): Contexts for the flow for each mixture component.
            weights (torch.Tensor): Weights for the mixture distribution.
            temperature (float): Temperature parameter for temp annealed sampling.
        """
        self.flow = flow
        self.contexts = contexts
        self.weights = weights
        if contexts.shape[0] != weights.shape[0]:
            raise ValueError("Number of contexts must match the number of weights.")

        self.temperature = temperature

    def rsample_and_log_prob(self, sample_shape=torch.Size()):
        """Sample from the ensemble distribution and return the log probability."""
        # Get a relaxed sample from the mixture distribution
        relaxed_sample = torch.distributions.RelaxedOneHotCategorical(
            self.temperature, self.weights
        )
        relaxed_sample = relaxed_sample.rsample(sample_shape)

        # Get the samples from the flows
        component_distribution = self.flow(self.contexts)
        samples, logprobs = component_distribution.rsample_and_log_prob(sample_shape)
        # samples will be [*sample_shape, self.contexts.shape[0], flow.event_shape]
        # logprobs will be [*sample_shape, self.contexts.shape[0]]

        # Combine the samples according to the mixture weights
        samples = torch.einsum("...i, ...ij -> ...j", relaxed_sample, samples)

        # Compute the probability (have to exponentiate, then sum, then log)
        logprobs = torch.logsumexp(logprobs + torch.log(self.weights), dim=-1)

        return samples, logprobs

    def log_prob(self, value):
        """Compute the log probability of a value."""
        # Add a batch dimension if necessary
        if value.dim() == 1:
            value = value.unsqueeze(0)

        component_distribution = self.flow(self.contexts)
        logprobs = component_distribution.log_prob(
            value.reshape(value.shape[0], 1, *value.shape[1:])
        )
        # logprobs will be [value.shape[0], self.contexts.shape[0]]

        # Compute the probability (have to sum, then log)
        logprobs = torch.logsumexp(logprobs + torch.log(self.weights), dim=-1)

        return logprobs

    def rsample(self, sample_shape=torch.Size()):
        """Sample from the distribution."""
        return self.rsample_and_log_prob(sample_shape)[0]


# if __name__ == "__main__":
#     # Test sinkhorn and mmd
#     import matplotlib.pyplot as plt
#     from tqdm import tqdm

#     sinkhorns = [
#         sinkhorn(
#             torch.randn(100, 2),
#             torch.randn(100, 2) + torch.tensor([i, 0.0]),
#         )
#         for i in tqdm(torch.linspace(-2, 2, 100))
#     ]

#     mmd = [
#         MMDLoss()(
#             torch.randn(1000, 2),
#             torch.randn(1000, 2) + torch.tensor([i, 0.0]),
#         )
#         for i in tqdm(torch.linspace(-2, 2, 100))
#     ]

#     plt.plot(torch.linspace(-2, 2, 100), sinkhorns, label="Sinkhorn")
#     plt.plot(torch.linspace(-2, 2, 100), mmd, label="MMD")
#     plt.legend()
#     plt.show()





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
