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

    def plot_things(samples, labels, title=None):
        plt.figure(figsize=(4, 4))
        plt.scatter(*samples.T, s=1, c=labels, cmap="bwr")
        # Turn off axis ticks
        plt.xticks([])
        plt.yticks([])
        plt.axis("off")
        plt.ylim([-1.1, 1.1])
        plt.xlim([-1.7, 1.7])
        if title is not None:
            plt.title(title)
        # Equal aspect
        plt.gca().set_aspect("equal")
        plt.show()

    def sample_y_given_something(k, w_obs=None, failure_obs=None, theta_obs=None):
        with pyro.plate("samples", k, dim=-2):
            samples = w_z_y_model(w_obs=w_obs, failure_obs=failure_obs, theta_obs=theta_obs)
        # print(samples)
        y = samples.reshape(-1,2).detach().cpu()
        return y

    n_days = 100

    y_obs, w_obs = generate_two_moons_data_hierarchical(n_days, device)
    # plot_things(y_obs, w_obs > .5, "training data")

    w_z_y_model = functools.partial(
        two_moons_w_z_y_model,
        n=n_days,
        device=device,
    )

    f = 1 - torch.randn(n_days) * .01
    # y = sample_y_given_something(1, failure_obs=f)
    y = sample_y_given_something(1, w_obs=w_obs)
    # plot_things(y, torch.ones(n_days), "sample test")

    # with pyro.plate("samples", 1, dim=-2):
    #     samples = w_z_y_model(w_obs=w_obs)
    # print(samples)
    # y = samples.reshape(-1,2).detach()
    # print(y)
    # plot_things(y, w_obs > .5)

    w_z_y_guide = AutoIAFNormal(w_z_y_model)

    # TODO: figure out how to specify a custom loss
    def simple_elbo(model, guide, *args, **kwargs):
        # run the guide and trace its execution
        guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
        # run the model and replay it against the samples from the guide
        model_trace = pyro.poutine.trace(
            pyro.poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)
        # construct the elbo loss function
        return -1*(model_trace.log_prob_sum() - guide_trace.log_prob_sum())
    
    l = simple_elbo(w_z_y_model, w_z_y_guide, w_obs=w_obs, y_obs=y_obs)
    # print(l)

    # Train the model
    # set up the optimizer
    n_steps = n_steps
    gamma = 0.1  # final learning rate will be gamma * initial_lr
    lrd = gamma ** (1 / n_steps)
    optim = pyro.optim.ClippedAdam({"lr": lr, "lrd": lrd})
    elbo = pyro.infer.Trace_ELBO(num_particles=1)

    # setup the inference algorithm
    svi = pyro.infer.SVI(
        w_z_y_model, 
        w_z_y_guide, 
        optim, 
        loss=elbo,
    )

    # do gradient steps
    pbar = tqdm(range(n_steps))
    losses = []
    for step in pbar:
        loss = svi.step(w_obs=w_obs, y_obs=y_obs)
        losses.append(loss)
        if step % 10 == 0:
            pbar.set_description(f"ELBO loss: {loss:.2f}")

    # plt.figure(figsize=(8,8))
    # plt.plot(losses)
    # plt.show()

    k = 10

    y_obs, w_obs = generate_two_moons_data_hierarchical(n_days*k, device, nominal=True)
    plot_things(y_obs, w_obs > .5, "test data")
    # print(w_obs)

    # predictive = pyro.infer.Predictive(
    #     w_z_y_model, 
    #     guide=w_z_y_guide, 
    #     num_samples=100,
    # )
    # with pyro.plate("samples", n_samples, dim=-1):
    #     posterior_samples = auto_guide(states, dt)
    with pyro.plate("samples", k, dim=-2):
        samples = w_z_y_guide(w_obs=w_obs, y_obs=y_obs)

    # print(samples)

    # w_post = samples['w'].flatten()
    # z_post = samples['z'].reshape(-1, 2)
    # t_post = samples['theta'].flatten()
    # f_post = samples['failure'].flatten()
    # w_post = samples['w'].mean(dim=0)
    # z_post = samples['z'].mean(dim=0).detach()
    t_post = samples['theta']
    f_post = samples['failure']

    t_post_plt = t_post.reshape(-1).detach().cpu()
    f_post_plt = f_post.reshape(-1).detach().cpu()

    print(t_post_plt.min(), t_post_plt.max())

    # print(w_post)

    plt.figure(figsize=(4, 4))
    plt.hist(f_post_plt, density=True, bins=64, label='failure',alpha=.5)
    # plt.hist(t_post_plt, density=True, bins=64, label='theta',alpha=.5)
    plt.legend()
    plt.show()

    y_obs = sample_y_given_something(k, failure_obs=f_post, theta_obs=t_post)
    plot_things(y_obs, f_post_plt, 'using failure/theta post')

    # y_obs = sample_y_given_something(k, failure_obs=f_post)
    # plot_things(y_obs, f_post_plt, 'using failure post')
    

if __name__ == "__main__":
    run()
