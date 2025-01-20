import numpy as np
import ot
import torch
import functools

import pyro.distributions as dist
import pyro
import zuko
from zuko.flows.core import LazyDistribution

from flowgmm.flow_ssl.realnvp.realnvp import RealNVPTabular
from flowgmm.flow_ssl.distributions import SSLGaussMixture
from flowgmm.flow_ssl import FlowLoss

import torch
import torch.nn as nn
from enum import Enum, auto
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from scripts.utils import ConditionalGaussianMixture

from copy import deepcopy

MixtureLabel = torch.Tensor
# Observation = torch.Tensor
Observation = Any

@dataclass
class RegimeData:

    label: MixtureLabel
    weight: torch.Tensor # like how much it takes up?
    y_subsample: Optional[list[Observation]] = None
    z_subsample: Optional[list[Observation]] = None
    w_subsample: Optional[list[Observation]] = None
    # x_states: Optional[list[Any]] = None
    name: Optional[str] = None
        
    
# TODO: inheritance?
class ZW(ABC):
    
    @abstractmethod
    def z_given_w_log_prob_regime(self, regime):
        """
        p(z; r) * r.weight, e.g. r = f(w)
        """
        raise NotImplementedError
    
    # @abstractmethod
    # def z_given_w_log_prob(self, z_sample, w_sample):
    #     """
    #     p(z|w)
    #     TODO: label?
    #     """
    #     raise NotImplementedError


class GaussianMixtureTwoMoonsZW(ZW):

    def __init__(self, device):
        self.dist = ConditionalGaussianMixture(
            n_context=1, n_features=2
        ).to(device)

    def z_given_w_log_prob_regime(self, regime):
        return self.dist(regime.label).log_prob(regime.z_subsample)


class YZ(ABC):

    @abstractmethod
    def y_given_z_log_prob_regime(self, regime):
        """
        f(y|z; regime), e.g. regime = f(w)
        """
        raise NotImplementedError
    
    # @abstractmethod
    # def y_given_z_log_prob(self, y_sample, z_sample):
    #     """
    #     f(y|z)
    #     """
    #     raise NotImplementedError


class PyroStatesModelYZ(YZ):

    def __init__(self, model):
        self.model = model
    
    @abstractmethod
    def map_to_sample_sites(self, z_sample):
        raise NotImplementedError

    @abstractmethod
    def y_given_z_log_prob_regime(self, regime: RegimeData) -> torch.Tensor:
        raise NotImplementedError
    
    
class PyroTwoMoonsYZ(PyroStatesModelYZ):

    def __init__(self, model, device):
        super().__init__(model)
        self.device = device
 
    def map_to_sample_sites(self, sample):

        single_sample = len(sample.shape) == 1
        if single_sample:
            sample = sample.unsqueeze(0)
        assert sample.shape[-1] == 2

        conditioning_dict = {'z': sample}
        return conditioning_dict

    def y_given_z_log_prob_regime(self, regime: RegimeData) -> torch.Tensor:

        conditioning_dict = self.map_to_sample_sites(regime.z_subsample)
        model_trace = pyro.poutine.trace(
            pyro.poutine.condition(self.model, data=conditioning_dict)
        ).get_trace(
            states=regime.y_subsample # here y_subsample should have states?
        )

        model_logprob = model_trace.log_prob_sum()
        return model_logprob



class RegimeAssigner(ABC):

    # @property
    # @abstractmethod
    # def device(self):
    #     raise NotImplementedError

    @abstractmethod
    def assign_label(self, w_subsample):
        """
        the main thing implementations of this class need to do?
        i.e. take subsample and assign mixture label
        """
        raise NotImplementedError

    def make_regime(self, y_subsample, w_subsample, weight):
        label = self.assign_label(w_subsample)
        return RegimeData(
            label=label,
            weight=weight,
            y_subsample=y_subsample,
            w_subsample=w_subsample
        )
    
    def make_regimes(self, y_subsample_list, w_subsample_list, weight_list=None):
        if weight_list is None:
            weight_list = [
            torch.tensor(1.0 / len(y_subsample_list)) 
            for _ in range(len(y_subsample_list))
        ]
        yww_list = zip(y_subsample_list, w_subsample_list, weight_list)
        regimes = []
        for y_subsample, w_subsample, weight in yww_list:
            regimes.append(
                self.make_regime(
                    y_subsample, w_subsample, weight
                )
            )
        return regimes
    
    def make_regimes_per_point(self, y_samples, w_samples):
        weight_list = [
            torch.tensor(1.0 / len(y_samples)) 
            for _ in range(len(y_samples))
        ]
        y_subsample_list = [
            ys for ys in y_samples
        ]
        w_subsample_list = [
            ws for ws in w_samples
        ]

        return self.build_regimes(
            y_subsample_list, w_subsample_list, weight_list
        )
    
    def make_regimes_no_subsample(self, y_samples, w_samples):
        # TODO: device
        weight_list = [torch.tensor(1.0)]
        y_subsample_list = [y_samples]
        w_subsample_list = [w_samples]

        return self.build_regimes(
            y_subsample_list, w_subsample_list, weight_list
        )
    

class ThresholdTwoMoonsRA(RegimeAssigner):

    def __init__(self, device):
        self.device = device

    def assign_label(self, w_subsample):
        if w_subsample > .5:
            return torch.tensor([1.0]).to(self.device)
        else:
            return torch.tensor([0.0]).to(self.device)


# TODO: define a framework:

class WZY(ABC):
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

    p(z|w) = p(z;f(w))
    p(y,z|w) = p(y|z) * p(z|w)

    possibly:
    p(z) = \sum_c p(z;c) * p(f(w)=c) -> tractable sum
    p(w|z) = p(z|w) * p(w) / p(z)
    p(z|y) = \sum_c q(z|y;c) * p(f(w)=c) (approximately)
    # TODO: also bound the validity of this estimator for z|y ??
        
    """
    zw: ZW
    yz: YZ
    ra: RegimeAssigner
    
    # z_given_wy_guide = None # q guide
    # w_given_y_guide = None # r guide
    q_guide = None
    r_guide = None

    #
    # w_subsample_list: list[Any]
    # y_subsample_list: list[Any]

    yw_regimes: list[RegimeData]
    
    # make a dataclass for all these kinds of params
    device = None
    
    def __init__(self, zw, yz, ra, q_guide, r_guide, w_subsample_list, y_subsample_list):

        self.zw = zw
        self.yz = yz
        self.ra = ra
        self.q_guide = q_guide
        self.r_guide = r_guide

        self.yw_regimes = \
            self.ra.make_regimes(y_subsample_list, w_subsample_list)
        self.y_regimes = deepcopy(self.yw_regimes)
        for regime in self.y_regimes:
            regime.w_subsample = None # unnecssary though

    def q_elbo_objective(self, regimes=None, **kwargs):
        """
        E_q [ log p(y,z|w) - log q(z|y,w) ]
        log p(y,z|w) = log p(y|z) + log p(z|w)
        """
        if regimes is None:
            regimes = self.yw_regimes
        elbo = torch.tensor(0.0).to(self.device)
        for regime in regimes:
            # get z samples from guide for approximation of expectation
            z_sample, z_logprob = self.q_guide(regime.label).rsample_and_log_prob()
            regime.z_subsample = z_sample
            elbo += (
                self.yz.y_given_z_log_prob_regime(regime, **kwargs) + 
                self.zw.z_given_w_log_prob_regime(regime, **kwargs) - 
                z_logprob
            ) * regime.weight # weight can be equal, and maybe divide by flights ??
        return elbo
            
    def r_elbo_objective(self, regimes=None, **kwargs):
        """
        E_r [ p^ (y|w) - log r(w|y) ]
        p^ (y|w) = E_q [ log p(y,z|w) - log q(z|y,w) ] = ELBO_q (y, w)
        """
        if regimes is None:
            regimes = self.y_regimes
        elbo = torch.tensor(0.0).to(self.device)
        for regime in regimes:
            w_sample, w_logprob = self.r_guide(regime.label).rsample_and_log_prob()
            regime.w_subsample = w_sample
            # regimes should contain the y value and appropriate label corresponding to w ?
            elbo += (
                self.q_elbo_objective(regimes, **kwargs) - w_logprob
            )
        return elbo
    
    def q_elbo_loss(self, **kwargs):
        return -self.q_elbo_objective(**kwargs)
    
    def r_elbo_loss(self, **kwargs):
        return -self.r_elbo_objective(**kwargs)


    




# starting to add the required components for inference on our more complicated graphical model

# w -> c (for now, let's just use a classifier -- in general it will be harder...)
    
class EncoderMLP(nn.Module):
    """
    simple multi-layer perceptron for the w -> c mapping
    building block for a mixture of densities network
    here we consider the 
    """
    def __init__(self, input_dim, hidden_dim=100, n_layers=1, n_classes=2, act_layer=nn.Softplus()):
        super().__init__()

        modules = []
        if n_layers < 2:
            modules.append(nn.Linear(input_dim, n_classes))
        else:
            modules.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(n_layers-2):
                modules.append(nn.Linear(input_dim, hidden_dim))
                modules.append(act_layer)
            modules.append(nn.Linear(hidden_dim, n_classes))

        self.network = nn.Sequential(*modules)
        self.input_dim = input_dim
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        input = x.reshape(-1, self.input_dim)
        logits = self.network(input) # is this true???
        return logits
    
    def get_mixture_label(self, x):
        logits = self.forward(x)
        mixture_label = nn.functional.gumbel_softmax(logits, tau=1, hard=True)
        return mixture_label
    
    def classify(self, x):
        mixture_label = self.get_mixture_label(x)
        return torch.max(mixture_label, 1)[1]

    
    
# TODO: flowgmm??
class EncoderFlowGMM(RealNVPTabular):
    def __init__(
        self, in_dim=2, num_coupling_layers=6, hidden_dim=256, 
                 num_layers=2, init_zeros=False, dropout=False,
        prior=None,
        lr_init = 1e-4,
        weight_decay=1e-2,
    ):
        super().__init__(in_dim, num_coupling_layers, hidden_dim, 
                 num_layers, init_zeros, dropout)
        
        if prior is None:
            self.prior = SSLGaussMixture(
                means=3.5*torch.tensor([[-1.0,-1.0],[1.0,1.0]])
            ) # TODO: fix this

        self.classify = self.prior.classify
        
        self.loss_fn = FlowLoss(self.prior)

        self.optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad==True], 
            lr=lr_init, 
            weight_decay=1e-2
        )

    def loss(self, z, y):
        # x, y = torch.from_numpy(x), torch.from_numpy(y)
        sldj = self.logdet()
        loss = self.loss_fn(z, sldj, y)
        return loss
    
    # def get_mixture_label(self, x):
    #     logits = self.forward(x)
    #     mixture_label = nn.functional.gumbel_softmax(logits, tau=1, hard=True)
    #     return mixture_label

    # def predict(self, x):
    #     mixture_label = self.get_mixture_label(x)
    #     return torch.max(mixture_label, 1)[1]



class TestWZY(object):

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
        x = torch.rand(10)
        y = (x > .5).long()
        x = x.reshape(-1,1)
        dataset.append((x, y))

    clsf = EncoderMLP(1, 10, 1)
    clsf_optim = torch.optim.Adam(
        clsf.parameters(),
        lr=1e-3,
    )

    # clsf = EncoderFlowGMM(
    #     num_coupling_layers=5, in_dim=1, num_layers=2, hidden_dim=512
    # )
    # clsf_optim = clsf.optimizer

    epbar = tqdm(range(1000))
    for epoch in epbar:  # loop over the dataset multiple times
        # pbar = tqdm(range(train_n), leave=False)
        pbar = range(train_n)
        loss, val_loss = None, None
        for i in pbar:
            # get the inputs; dataset is a list of [inputs, labels]
            x, y = dataset[i]
            # zero the parameter gradients
            clsf_optim.zero_grad()

            # forward + backward + optimize
            z = clsf.forward(x)
            loss = clsf.loss(z, y)
            loss.backward()
            clsf_optim.step()

            x, y = dataset[train_n]
            out = clsf.forward(x)
            val_loss = clsf.loss(out, y)
            # if i % 100 == 0:
            #     pbar.set_description(f"train | test : {loss:.2f} | {val_loss:.2f}")
        if epoch % 10 == 0:
            epbar.set_description(f"train | test : {loss:.2f} | {val_loss:.2f}")

    x = torch.tensor([1,2,3,4,5,6,7,8,9,10], dtype=torch.float32) / 10.0
    y = torch.tensor([0,0,0,0,0,1,1,1,1,1], dtype=torch.long)
    x = x.reshape(-1, 1)
    print(clsf.classify(x))


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