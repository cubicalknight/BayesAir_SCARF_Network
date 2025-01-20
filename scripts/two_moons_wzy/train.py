"""Implement CalVI training for the two moons toy problem."""

import os

import matplotlib
import matplotlib.pyplot as plt
import pyro
import torch
import zuko
from click import command, option
import functools

import wandb
from scripts.training import train
from scripts.two_moons_wzy.model import *
from scripts.utils import kl_divergence, ConditionalGaussianMixture

from pyro.infer.autoguide import AutoIAFNormal
import pyro
from tqdm import tqdm

from scripts.wzy.wzy_core import *


@command()
@option("--n-nominal", default=1000, help="# of nominal examples")
@option("--n-failure", default=20, help="# of failure examples for training")
@option("--n-failure-eval", default=1000, help="# of failure examples for evaluation")
@option("--no-calibrate", is_flag=True, help="Don't use calibration")
@option("--balance", is_flag=True, help="Balance CalNF")
@option("--bagged", is_flag=True, help="Bootstrap aggregation")
@option("--regularize", is_flag=True, help="Regularize failure using KL wrt nominal")
@option("--wasserstein", is_flag=True, help="Regularize failure using W2 wrt nominal")
@option("--gmm", is_flag=True, help="Use GMM instead of NF")
@option("--seed", default=0, help="Random seed")
@option("--n-steps", default=200, type=int, help="# of steps")
@option("--lr", default=1e-3, type=float, help="Learning rate")
@option("--lr-gamma", default=1.0, type=float, help="Learning rate decay")
@option("--lr-steps", default=1000, type=int, help="Steps per learning rate decay")
@option("--grad-clip", default=100, type=float, help="Gradient clipping value")
@option("--weight-decay", default=0.0, type=float, help="Weight decay rate")
@option("--run-prefix", default="", help="Prefix for run name")
@option("--project-suffix", default="benchmark", help="Suffix for project name")
@option(
    "--n-elbo-particles",
    default=1,
    type=int,
    help="# of particles for ELBO estimation",
)
@option(
    "--n-calibration-particles",
    default=50,
    type=int,
    help="# of particles for calibration",
)
@option(
    "--n-calibration-permutations",
    default=5,
    type=int,
    help="# of permutations for calibration",
)
@option(
    "--n-divergence-particles",
    default=100,
    type=int,
    help="# of particles for divergence estimation",
)
@option(
    "--calibration-weight",
    default=1e0,
    type=float,
    help="weight applied to calibration loss",
)
@option(
    "--regularization-weight",
    default=1e0,
    type=float,
    help="weight applied to nominal divergence loss",
)
@option("--elbo-weight", default=1e0, type=float, help="weight applied to ELBO loss")
@option(
    "--calibration-ub", default=5e1, type=float, help="KL upper bound for calibration"
)
@option("--calibration-lr", default=1e-3, type=float, help="LR for calibration")
@option("--calibration-substeps", default=1, type=int, help="# of calibration substeps")
@option(
    "--exclude-nominal",
    is_flag=True,
    help="If True, don't learn the nominal distribution",
)
def run(
    n_nominal,
    n_failure,
    n_failure_eval,
    no_calibrate,
    balance,
    bagged,
    regularize,
    wasserstein,
    gmm,
    seed,
    n_steps,
    lr,
    lr_gamma,
    lr_steps,
    grad_clip,
    weight_decay,
    run_prefix,
    project_suffix,
    n_elbo_particles,
    n_calibration_particles,
    n_calibration_permutations,
    n_divergence_particles,
    calibration_weight,
    regularization_weight,
    elbo_weight,
    calibration_ub,
    calibration_lr,
    calibration_substeps,
    exclude_nominal,
):
    """Generate data and train the SWI model."""
    # matplotlib.use("Agg")
    # matplotlib.rcParams["figure.dpi"] = 300

    # Generate data (use consistent seed for all runs to make data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    pyro.set_rng_seed(0)

    # Generate training data
    # TODO: this

    # Change seed for training
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    # def plot_things(samples, labels, title=None):
    #     plt.figure(figsize=(4, 4))
    #     plt.scatter(*samples.T, s=1, c=labels, cmap="bwr")
    #     # Turn off axis ticks
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.axis("off")
    #     plt.ylim([-1.1, 1.1])
    #     plt.xlim([-1.7, 1.7])
    #     if title is not None:
    #         plt.title(title)
    #     # Equal aspect
    #     plt.gca().set_aspect("equal")
    #     plt.show()

    # def sample_y_given_something(k, w_obs=None, failure_obs=None, theta_obs=None):
    #     with pyro.plate("samples", k, dim=-2):
    #         samples = w_z_y_model(w_obs=w_obs, failure_obs=failure_obs, theta_obs=theta_obs)
    #     # print(samples)
    #     y = samples.reshape(-1,2).detach().cpu()
    #     return y

    def plot_stuff(y_list, w_list):
        plt.figure(figsize=(4, 4))
        for y, w in zip(y_list, w_list):
            c = 'r' if w > .5 else 'b'
            samples = y
            plt.scatter(*samples.T, s=1, c=c, cmap="bwr")
        # Turn off axis ticks
        plt.xticks([])
        plt.yticks([])
        plt.axis("off")
        plt.ylim([-1.1, 1.1])
        plt.xlim([-1.7, 1.7])
        # Equal aspect
        plt.gca().set_aspect("equal")
        plt.show()

    n_days = 100
    m_per_day = 10

    # y_obs, w_obs = generate_two_moons_data_hierarchical(n_days, device)
    # y_obs, w_obs, states = generate_two_moons_data_using_model(n_days, device, return_states=True)
    # plot_things(y_obs, w_obs > .5, "training data")

    y_list_failure, w_list_failure, states_list_failure, z_list_failure = \
        generate_two_moons_data_using_model(
            n_days, m_per_day, device, 
            failure_only=True, return_states=True, return_z=True
        )
    y_list_nominal, w_list_nominal, states_list_nominal, z_list_nominal = \
        generate_two_moons_data_using_model(
            n_days, m_per_day, device, 
            nominal_only=True, return_states=True, return_z=True
        )
    
    # plot_stuff(y_list_failure, w_list_failure)
    # plot_stuff(y_list_nominal, w_list_nominal)

    nominal_regimes = [
        RegimeData(
            label=torch.tensor([0.0], device=device), 
            weight=torch.tensor(0.5 / n_days, device=device),
            y_subsample=states_list_nominal[i],
            z_subsample=z_list_nominal[i],
            w_subsample=w_list_nominal[i],
        )
        for i in range(n_days)
    ]

    failure_regimes = [
        RegimeData(
            label=torch.tensor([1.0], device=device), 
            weight=torch.tensor(0.5 / n_days, device=device),
            y_subsample=states_list_failure[i],
            z_subsample=z_list_failure[i],
            w_subsample=w_list_failure[i],
        )
        for i in range(n_days)
    ]

    wzy_model = functools.partial(
        two_moons_wzy_model,
        device=device,
    )

    q_guide = zuko.flows.NSF(
        features=2,
        context=1,
        hidden_features=(64, 64),
    ).to(device)
    # print(states)
    # label = torch.tensor([1.0], device=device)
    # print(q_guide(label).rsample_and_log_prob())
    # label = torch.tensor([0.0], device=device)
    # print(q_guide(label).rsample_and_log_prob())
    # return

    yz = PyroTwoMoonsYZ(wzy_model, device)

    # print("\nfailure z in failure regime")
    # l = torch.tensor(0.0).to(device)
    # for i in range(n_days):
    #     failure_regimes[i].z_subsample = z_list_failure[i]
    #     l += yz.y_given_z_log_prob_regime(failure_regimes[i]) * failure_regimes[i].weight
    # print(l)

    # print("\nnominal z in failure regime")
    # l = torch.tensor(0.0).to(device)
    # for i in range(n_days):
    #     failure_regimes[i].z_subsample = z_list_nominal[i]
    #     l += yz.y_given_z_log_prob_regime(failure_regimes[i]) * failure_regimes[i].weight
    # print(l)

    ra = ThresholdTwoMoonsRA(device)
    zw = GaussianMixtureTwoMoonsZW(device)

    wzy = WZY(
        zw=zw,
        yz=yz,
        ra=ra,
        q_guide=q_guide,
        r_guide=None,
        w_subsample_list=w_list_failure+w_list_nominal,
        y_subsample_list=y_list_failure+y_list_nominal,
    )

    print(wzy.q_elbo_loss())


if __name__ == "__main__":
    run()
