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

    def plot_things(samples, labels, title=None, no_axlim=False):
        plt.figure(figsize=(4, 4))
        plt.scatter(*samples.T, s=1, c=labels, cmap="bwr")
        # # Turn off axis ticks
        # plt.xticks([])
        # plt.yticks([])
        # plt.axis("off")
        if not no_axlim:
            plt.ylim([-2.0, 2.0])
            plt.xlim([-2.0, 2.0])
        if title is not None:
            plt.title(title)
        # Equal aspect
        plt.gca().set_aspect("equal")
        plt.show()

    # def sample_y_given_something(k, w_obs=None, failure_obs=None, theta_obs=None):
    #     with pyro.plate("samples", k, dim=-2):
    #         samples = w_z_y_model(w_obs=w_obs, failure_obs=failure_obs, theta_obs=theta_obs)
    #     # print(samples)
    #     y = samples.reshape(-1,2).detach().cpu()
    #     return y

    def plot_stuff(y_list, w_list, c=None):
        plt.figure(figsize=(4, 4))
        for y, w in zip(y_list, w_list):
            if c is None:
                c = 'r' if w > .5 else 'b'
            samples = y
            plt.scatter(*samples.T, s=1, c=c, cmap="bwr")
        # # Turn off axis ticks
        # plt.xticks([])
        # plt.yticks([])
        # plt.axis("off")
        # plt.ylim([-1.1, 1.1])
        # plt.xlim([-1.7, 1.7])
        plt.ylim([-2.0, 2.0])
        plt.xlim([-2.0, 2.0])
        # Equal aspect
        plt.gca().set_aspect("equal")
        plt.show()

    # n_days = 100
    # m_per_day = 10

    # y_list_failure, w_list_failure, states_list_failure, z_list_failure = \
    #     generate_two_moons_data_using_model(
    #         n_days, m_per_day, device, 
    #         failure_only=True, return_states=True, return_z=True, 
    #         # even_simpler=True,
    #     )
    # y_list_nominal, w_list_nominal, states_list_nominal, z_list_nominal = \
    #     generate_two_moons_data_using_model(
    #         n_days, m_per_day, device, 
    #         nominal_only=True, return_states=True, return_z=True,
    #         # even_simpler=True,
    #     )
    
    # plot_stuff(y_list_failure, w_list_failure)
    # plot_stuff(y_list_nominal, w_list_nominal)

    n = 5
    m = 20
    y_failure_list, w_failure_list, z_failure_list = [], [], []
    for _ in range(m):
        y_failure, w_failure, z_failure = \
            generate_two_moons_data_using_model_plated(
                n, device, 
                failure_only=True, return_z=True, 
                # even_simpler=True,
            )
        y_failure_list.append(y_failure)
        w_failure_list.append(w_failure)
        z_failure_list.append(z_failure)

    y_nominal_list, w_nominal_list, z_nominal_list = [], [], []
    for _ in range(m):
        y_nominal, w_nominal, z_nominal = \
            generate_two_moons_data_using_model_plated(
                n, device, 
                nominal_only=True, return_z=True,
                # even_simpler=True,
            )
        y_nominal_list.append(y_nominal)
        w_nominal_list.append(w_nominal)
        z_nominal_list.append(z_nominal)
    
    # plot_things(y_failure, 'r')
    # plot_things(y_nominal, 'b')
    plot_stuff(y_failure_list, w_failure_list, c='r')
    plot_stuff(y_nominal_list, w_nominal_list, c='b')
    
    # nominal_regimes = [
    #     RegimeData(
    #         label=torch.tensor([0.0], device=device), 
    #         weight=torch.tensor(0.5 / n_days, device=device),
    #         y_subsample=states_list_nominal[i],
    #         z_subsample=z_list_nominal[i],
    #         w_subsample=w_list_nominal[i],
    #     )
    #     for i in range(n_days)
    # ]

    # failure_regimes = [
    #     RegimeData(
    #         label=torch.tensor([1.0], device=device), 
    #         weight=torch.tensor(0.5 / n_days, device=device),
    #         y_subsample=states_list_failure[i],
    #         z_subsample=z_list_failure[i],
    #         w_subsample=w_list_failure[i],
    #     )
    #     for i in range(n_days)
    # ]

    # nominal_regime = RegimeData(
    #     label=torch.tensor([0.0], device=device), 
    #     weight=torch.tensor(0.5 / n, device=device),
    #     y_subsample=y_nominal,
    #     z_subsample=z_nominal,
    #     w_subsample=w_nominal,
    # )

    # failure_regime = RegimeData(
    #     label=torch.tensor([1.0], device=device), 
    #     weight=torch.tensor(0.5 / n, device=device),
    #     y_subsample=y_failure,
    #     z_subsample=z_failure,
    #     w_subsample=w_failure,
    # )

    # wzy_model = functools.partial(
    #     two_moons_wzy_model,
    #     device=device,
    # )
    # wzy_model = functools.partial(
    #     two_moons_wzy_model_even_simpler,
    #     device=device,
    # )

    wzy_model = functools.partial(
        two_moons_wzy_model_plated,
        n=n,
        device=device,
    )

    q_guide = zuko.flows.NSF(
        features=2,
        context=1,
        hidden_features=(64, 64),
    ).to(device)
    # q_guide = ConditionalGaussianMixture(
    #     n_features=2,
    #     n_context=1,
    # )

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
    # ra = MLPTwoMoonsRA(1, 4, 1).to(device)
    # zw = NSFTwoMoonsZW(device)
    zw = GaussianMixtureTwoMoonsZW(device)
    zw.dist.means = torch.nn.Parameter(
        torch.tensor(
            [[-1.0, -1.0], [1.0, 1.0]], device=device
        )
    )
    zw.dist.log_vars = torch.nn.Parameter(
        torch.tensor(
            [[-1.0, -1.0], [-1.0, -1.0]], device=device
        )
    )

    wzy = WZY(
        zw=zw,
        yz=yz,
        ra=ra,
        q_guide=q_guide,
        r_guide=None,
        # w_subsample_list=w_list_failure+w_list_nominal,
        # y_subsample_list=states_list_failure+states_list_nominal,
        w_subsample_list=w_failure_list+w_nominal_list,
        y_subsample_list=y_failure_list+y_nominal_list,
    )

    # label = torch.tensor([1.0], device=device)
    # samples = wzy.zw.dist(label).sample((1000,))
    # plot_things(samples, 'r')

    # label = torch.tensor([0.0], device=device)
    # samples = wzy.zw.dist(label).sample((1000,))
    # plot_things(samples, 'b')


    # print(wzy.yw_regimes)

    q_optimizer = torch.optim.Adam(
        wzy.q_guide.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    q_scheduler = torch.optim.lr_scheduler.StepLR(
        q_optimizer, step_size=lr_steps, gamma=lr_gamma
    )


    ra_optimizer = torch.optim.Adam(
        wzy.ra.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    ra_scheduler = torch.optim.lr_scheduler.StepLR(
        ra_optimizer, step_size=lr_steps, gamma=lr_gamma
    )

    losses = []
    thresholds = []
    pbar = tqdm(range(301))
    for i in pbar:
        q_optimizer.zero_grad()
        # ra_optimizer.zero_grad()

        loss = wzy.q_elbo_loss()
        loss.backward()
        losses.append(loss.detach())

        q_optimizer.step()
        q_scheduler.step()

        # ra_optimizer.step()
        # ra_scheduler.step()

        # pbar.set_description(f'q_elbo_loss: {loss:.3f}')
        pbar.set_description(f'q_elbo_loss: {loss:.3f}, w threshold: {wzy.ra.threshold.item():.3f}')
        # wzy.ra.threshold.item()
        thresholds.append(wzy.ra.threshold.item())

        if i % 50 == 0:
            label = torch.tensor([1.0], device=device)
            samples = q_guide(label).sample((1000,))
            # print(samples)
            # plot_things(samples, 'r', no_axlim=True)
            plot_things(samples, 'r', no_axlim=False, 
                        title=f'[failure] iter {i}, learned threshold={wzy.ra.threshold.item():.3f}')

            label = torch.tensor([0.0], device=device)
            samples = q_guide(label).sample((1000,))
            # plot_things(samples, 'b', no_axlim=True)
            plot_things(samples, 'b', no_axlim=False,
                        title=f'[nominal] iter {i}, learned threshold={wzy.ra.threshold.item():.3f}')

    plt.figure()
    plt.plot(losses, label="loss")
    plt.plot(thresholds, label="thresholds")
    plt.legend()
    plt.show()


    label = torch.tensor([1.0], device=device)
    samples = q_guide(label).sample((1000,))
    print(samples)
    # plot_things(samples, 'r', no_axlim=True)
    plot_things(samples, 'r', no_axlim=False)

    label = torch.tensor([0.0], device=device)
    samples = q_guide(label).sample((1000,))
    # plot_things(samples, 'b', no_axlim=True)
    plot_things(samples, 'b', no_axlim=False)

    x = torch.tensor([1,2,3,4,5,6,7,8,9,10], dtype=torch.float32) / 10.0
    # y = torch.tensor([0,0,0,0,0,1,1,1,1,1], dtype=torch.long)
    # x = x.reshape(-1, 1)
    for i in range(len(x)):
        print(f'{x[i].item():.3f} -> {wzy.ra.assign_label(x[i]).item():.3f}')
    # print(f'ra threshold = {wzy.ra.threshold}')



if __name__ == "__main__":
    run()
