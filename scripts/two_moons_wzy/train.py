"""Implement CalVI training for the two moons toy problem."""

import os

import matplotlib
import matplotlib.pyplot as plt
import pyro
import torch
import zuko
from click import command, option

import wandb
from scripts.training import train
from scripts.two_moons_wzy.model import (
    generate_two_moons_data_hierarchical,
)
from scripts.utils import kl_divergence, ConditionalGaussianMixture


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
    matplotlib.use("Agg")
    matplotlib.rcParams["figure.dpi"] = 300

    # Generate data (use consistent seed for all runs to make data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    pyro.set_rng_seed(0)

    # Generate training data
    # TODO: this

    # Change seed for training
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    # TODO: figure out how to specify a custom loss
    def simple_elbo(model, guide, *args, **kwargs):
        # run the guide and trace its execution
        guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
        # run the model and replay it against the samples from the guide
        model_trace = pyro.poutine.trace(
            pyro.poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)
        # construct the elbo loss function
        return -1*(model_trace.log_prob_sum() - guide_trace.log_prob_sum())

    # Make the objective and divergence closures
    def objective_fn(guide_dist, n, obs):
        """Compute the data likelihood."""
        data_likelihood = guide_dist.log_prob(obs).mean()

        return -data_likelihood  # negative because we want to maximize

    def divergence_fn(p, q):
        """Compute the KL divergence"""
        return kl_divergence(p, q, n_divergence_particles)

    # Also make a closure for classifying anomalies
    def score_fn(nominal_guide_dist, failure_guide_dist, n, obs):
        # Score function is the log likelihood ratio
        failure_likelihood = failure_guide_dist.log_prob(obs)
        nominal_likelihood = nominal_guide_dist.log_prob(obs)
        scores = failure_likelihood - nominal_likelihood
        return scores

    # Initialize the models
    if wasserstein:
        failure_guide = zuko.flows.CNF(
            features=2, context=n_calibration_permutations, hidden_features=(128, 128)
        ).to(device)
    elif gmm:
        failure_guide = ConditionalGaussianMixture(
            n_context=n_calibration_permutations, n_features=2
        ).to(device)
    else:
        failure_guide = zuko.flows.NSF(
            features=2, context=n_calibration_permutations, hidden_features=(64, 64)
        ).to(device)

    # Train the model
    # TODO:

    wandb.finish()


if __name__ == "__main__":
    run()
