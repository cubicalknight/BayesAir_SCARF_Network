import torch
import pyro.distributions as dist


def _gamma_dist_from_mean_std(mean, std):
    # std**2 = shape/rate**2
    # mean = shape/rate
    shape = (mean/std)**2
    rate = mean/std**2
    return dist.Gamma(
        torch.tensor(shape, device=device),
        torch.tensor(rate, device=device)
    )

def _beta_dist_from_mean_std(mean, std):
    # α = μν, β = (1 − μ)ν
    alpha = mean * std**2
    beta = (1-mean) * std**2
    return dist.Beta(
        torch.tensor(alpha, device=device),
        torch.tensor(beta, device=device)
    )