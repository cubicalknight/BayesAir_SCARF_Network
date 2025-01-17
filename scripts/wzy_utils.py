import numpy as np
import ot
import torch
import functools

import pyro.distributions as dist
import pyro
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



# TODO: define a framework:

class WZY(object):
    """
        our model: w -> c -> z -> y

        w -> c: "classifier", essentially determines which weather 
                based prior we will use for use with the observation
                we refer to this as (f_encoder) here

        c -> z: "weather informed priors", basically the compoments
                that make up the w -> z if we model it as a mixture
                we refer to this as (g_decoder) here

        z -> y: our simulation with latent variables (e.g. in pyro),
                but we just need any way to get likelihoods from it
                we refer to this as (p_model) here
        
        z|w,y : approximated with q_guide, see L_q in the paper
        w|y   : approximated with r_guide, see L_r in the paper

        we need to be able to compute:

        y|z
        z|c
        c|w

        
    """


    def __init__(
        self,
        w_observations,
        y_observations,
        # here
        p_model, # for y|z
        p_model_map_to_sample_sites_fn,
        # guides
        f_encoder, # for c|w, "classifier"
        g_decoder, # for z|c, "weather-informed priors"
        # variational distributions
        q_guide, # for z|w,y
        r_guide, # for w|y

        # log_h_estimate, # y|w, using our biased estimator for q
        # log_r_estimate, # w|y, using our biased estimator for q

        n_classes,

        # utils
        n_divergence_particles=10,
        lr=1e-3,
        weight_decay=0.0,
        lr_steps=None,
        lr_gamma=None,
        # TODO: make these sane
    ):
        self.w_observations = w_observations
        self.y_observations = y_observations

        self.p_model = p_model
        self.p_model_map_to_sample_sites = p_model_map_to_sample_sites_fn

        self.f_encoder = functools.partial(f_encoder, n_classes=n_classes)
        self.g_decoder = functools.partial(g_decoder, n_classes=n_classes)
        self.n_classes = n_classes

        self.q_guide = q_guide
        self.r_guide = r_guide

        self.q_optimizer = torch.optim.Adam(
            q_guide.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.q_scheduler = torch.optim.lr_scheduler.StepLR(
            self.q_optimizer, step_size=lr_steps, gamma=lr_gamma
        )

        self.r_optimizer = torch.optim.Adam(
            r_guide.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.r_scheduler = torch.optim.lr_scheduler.StepLR(
            self.r_optimizer, step_size=lr_steps, gamma=lr_gamma
        )

        self.g_optimizer = {}
        for i in range(n_classes):
            self.g_optimizer[i] = torch.optim.Adam(
                self.g_decoder.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
            # TODO: decide on how the f g based on label stuff will work...

        


    # TODO: consturction fo objective, e.g. compare VAE,IWAE, etc.

    # some utils

    def kl_divergence(self, p_dist, q_dist, num_particles):
        """KL divergence between two distributions."""
        p_samples, p_logprobs = p_dist.rsample_and_log_prob((num_particles,))
        q_logprobs = q_dist.log_prob(p_samples)
        kl_divergence = (p_logprobs - q_logprobs).mean(dim=0)
        return kl_divergence
    

    # TODO: some simple funcs for expressing each conditional explicitly
    # like a y_given_z , etc. just for some clarity i guess

    
    # let's figure out what we need for each part of the loss



    # TODO: this actually needs to be specified with the model...
    # def p_model_map_to_sample_sites(self, q_sample):
    #     """
    #     see the charles implementation for what this is supposed to do
    #     """
    #     # blah
    #     conditioning_dict = {} # TODO: !!!
    #     return conditioning_dict
    
    # TODO: we can "demote" latent z to parameter to compute just p(y|z)?
    # because more accurately this is y,z|w rn i think
    def p_model_logprob(self, q_sample):
        """
        E_q [ log p(y | z) * p(z) ] ?? can we omit z prior in model? or specify as input?
        """
        conditioning_dict = self.p_model_map_to_sample_sites(q_sample)
        model_trace = pyro.poutine.trace(
            pyro.poutine.condition(self.p_model, data=conditioning_dict)
        ).get_trace(
            # TODO: model inputs???
            # states=states,
            # delta_t=dt,
            # device=device,
            # include_cancellations=include_cancellations,
        )
        model_logprob = model_trace.log_prob_sum()
        return model_logprob

    def q_elbo_objective(self, y_obs, w_obs):
        """
        E_q [ log p(y|z) + log g(z;f(w)) - log q(z|y;f(w)) ]
        """
        c_labels = self.f_encoder.predict(w_obs)
        # TODO: use self.g_decoder to obtain priors for z
        # TODO: any point to intra class regularization term?

        q_sample, q_logprob = self.q_guide.rsample_and_log_prob()

        # the y_obs will be baked into this part, as intput
        # basically model needs to accept (1) context stuff/parameters,
        #   (2) observations (like the whole states dict thing), 
        #   (3) priors for the latent z, specified by w 
        p_logprob = self.p_model_logprob(q_sample)
        # TODO: is this enough? if the prior can be specified in the model inptu then wwe'r good
        # TODO: see if this is all???
        return p_logprob - q_logprob # we'll want to maximize this?
    

    # for next part

    def log_h_estimate(self, y, w):
        """
        estimator for log p(y|w), using our approximation for q
        biased downward by kl divergence D(q||p)
        E_q [ p(y,z|w) / q(z|y,w) ]
        """
        # q_sample, q_logprob = self.q_guide.rsample_and_log_prob()
        # p_logprob = self.p_model_logprob(q_sample)
        # # actually this is exaclty the same as the ELBO_q ????
        return self.q_elbo_objective(y, w)

    def log_r_estimate(self):
        """
        estimator for log p(w|y), using our approximation for q
        biased downward by kl divergence D(p||q)
        E_q [ p(w,z|y) / q(z|y,w) ]
        unfortunately this does require you to (cheaply) compute
            E_q [ p(z|y) * p(w|z) ]
        """
        pass

    # for r estimate using more variational inference
    def r_elbo_objective(self, y_obs):
        """
        E_r [ E_q [ log p(y|z) + log g(z;f(w)) - log q(z|y;f(w)) ] - log r(w|y) ]
        """
        pass






    
def simple_classifier_test():
    """
    i don't know how to use torch lol
    """
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


if __name__ == "__main__":
    simple_classifier_test()








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