"""Run the simulation for a LGA focused augmented network"""
import os
from itertools import combinations

import click
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyro
import seaborn as sns
import torch
from math import ceil, floor
import functools
import dill
import sys
from pathlib import Path

import bayes_air.utils.dataloader as ba_dataloader
import wandb
from bayes_air.model import augmented_air_traffic_network_model_simplified
from bayes_air.network import NetworkState, AugmentedNetworkState
from bayes_air.schedule import split_and_parse_full_schedule

from tqdm import tqdm

import pyro.distributions as dist
from scripts.utils import (
    _affine_beta_dist_from_alpha_beta,
    _affine_beta_dist_from_mean_std,
    _beta_dist_from_alpha_beta,
    _beta_dist_from_mean_std,
    _gamma_dist_from_mean_std,
    _gamma_dist_from_shape_rate,
    _shifted_gamma_dist_from_mean_std,
    _shifted_gamma_dist_from_shape_rate,
    ConditionalGaussianMixture
)
import zuko

from pbar_pool import PbarPool, Pbar

from scripts.lga.lga_network import (
    make_travel_times_dict_and_observation_df,
    make_states,
    plot_time_indexed_network_var,
    plot_hourly_delays,
    get_hourly_delays,
)


plot_service_times = functools.partial(
    plot_time_indexed_network_var,
    "mean_service_time",
    plots_per_row=1,
    transform=torch.exp,
    xlim=(.005, .030) # new
)


def single_particle_model_log_prob(model, states):

    # TODO: if we had more time should vectorize the model but ugh
    # with pyro.plate("samples", 10, dim=-1):
    model_trace = pyro.poutine.trace(model).get_trace(states)
    model_logprob = model_trace.log_prob_sum()

    return model_logprob

def model_log_prob(model, states, device, num_particles=1):
    """
    this is like p(y|z;c) i think
    """

    model_log_prob = torch.tensor(0.0, device=device)

    for _ in range(num_particles):
        model_log_prob += single_particle_model_log_prob(model, states) 
    model_log_prob /= num_particles

    return model_log_prob

def map_to_sample_sites_exp(sample):
    return {'LGA_0_mean_service_time': torch.exp(sample)}

def map_to_sample_sites_identity(sample):
    return {'LGA_0_mean_service_time': sample}


def single_particle_elbo(model, guide_dist, states):
    posterior_sample, posterior_logprob = guide_dist.rsample_and_log_prob()

    conditioning_dict = map_to_sample_sites_exp(posterior_sample)
    model_trace = pyro.poutine.trace(
        pyro.poutine.condition(model, data=conditioning_dict)
    ).get_trace(
        states=states
    )

    model_logprob = model_trace.log_prob_sum()
    return model_logprob - posterior_logprob

def objective_fn(model, guide_dist, states, device, n_elbo_particles=1):
    """ELBO loss for the air traffic problem."""
    elbo = torch.tensor(0.0).to(device)
    for _ in range(n_elbo_particles):
        elbo += single_particle_elbo(model, guide_dist, states) / n_elbo_particles
    # we already scale in the model?
    return -elbo




def single_particle_objective(model, guide_dist, prior_dist):
    """
    log p(y|z) + log p(z;c) - log q(z|y;c)
    """
    posterior_sample, posterior_logprob = guide_dist.rsample_and_log_prob()

    conditioning_dict = map_to_sample_sites_identity(
        torch.clamp(torch.exp(posterior_sample)/1000, max=.05)
        # torch.clamp(posterior_sample, min=1e-5, max=.05)
    )

    model_trace = pyro.poutine.trace(
        pyro.poutine.condition(model, data=conditioning_dict)
    ).get_trace()
    model_logprob = model_trace.log_prob_sum()

    prior_logprob = prior_dist.log_prob(torch.exp(posterior_sample)/1000)

    return (model_logprob + prior_logprob - posterior_logprob).squeeze()


def objective(model, guide_dist, prior_dist, device, n_particles=1):
    """ELBO loss for the air traffic problem."""
    elbo = torch.tensor(0.0).to(device)
    for _ in range(n_particles):
        elbo += single_particle_objective(model, guide_dist, prior_dist) / n_particles
    # we already scale in the model?
    return -elbo


def single_particle_y_given_z(model, sample):
    """
    log p(y|z)
    """
    conditioning_dict = map_to_sample_sites_identity(sample)

    model_trace = pyro.poutine.trace(
        pyro.poutine.condition(model, data=conditioning_dict)
    ).get_trace()
    model_logprob = model_trace.log_prob_sum()

    return (model_logprob).squeeze()



def train(
    project,
    network_airport_codes, 
    svi_steps, 
    n_samples, 
    svi_lr, 
    gamma,
    dt,
    n_elbo_particles,
    plot_every,
    rng_seed,
    rem_args,
    use_gpu=False
):
    pyro.clear_param_store()  # avoid leaking parameters across runs
    pyro.enable_validation(True)
    pyro.set_rng_seed(int(rng_seed))

    day_strs_list, posterior_guide = rem_args

    if use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device("cpu")

    # Avoid plotting error . test
    matplotlib.use("Agg")

    dir_path = Path(__file__).parent

    processed_visibility = pd.read_csv(dir_path / 'processed_visibility.csv')
    visibility_dict = dict(processed_visibility.values)
    processed_ceiling = pd.read_csv(dir_path / 'processed_ceiling.csv')
    ceiling_dict = dict(processed_ceiling.values)

    checkpoints_dir = dir_path / "bayes-air-atrds-attempt-7/checkpoints/LGA/"
    model_logprobs_dir = dir_path / "model_logprobs"

    # SETTING UP THE MODEL AND STUFF

    # Hyperparameters
    initial_aircraft = 50.0 # not used!
    mst_effective_hrs = 24 # not used!
    mst_split = 1 # not really used

    subsamples = {}

    for day_strs in day_strs_list:

        # gather data
        days = pd.to_datetime(day_strs)
        data = ba_dataloader.load_remapped_data_bts(days)
        name = ", ".join(day_strs)

        num_days = len(days)
        num_flights = sum([len(df) for df in data.values()])
        print(f"{name} -> days: {num_days}, flights: {num_flights}")

        # make things with the data
        travel_times_dict, observations_df = \
            make_travel_times_dict_and_observation_df(
                data, network_airport_codes
            ) 
        states = make_states(data, network_airport_codes)

        # set up common model arguments
        model = functools.partial(
            augmented_air_traffic_network_model_simplified,

            states=states,

            travel_times_dict=travel_times_dict,
            initial_aircraft=initial_aircraft,

            # include_cancellations=True,
            include_cancellations=False,
            mean_service_time_effective_hrs=mst_effective_hrs,
            delta_t=dt,

            source_use_actual_departure_time=True,
            # source_use_actual_late_aircraft_delay=True,
            # source_use_actual_carrier_delay=True,
            # source_use_actual_security_delay=True,

            # source_use_actual_cancelled=True,
            source_use_actual_cancelled=False,
            mst_prior_weight=1e-12, # effectively zero
        )

        # by default, scale ELBO down by num flights
        model_scale = 1.0 / (num_flights)
        model = pyro.poutine.scale(model, scale=model_scale)

        with open(checkpoints_dir / f'{name}/empty_0.00_gaussian/final/output_dict.pkl', 'rb') as f:
            s_guide_output_dict = dill.load(f)
        
        with open(model_logprobs_dir / f'{"_".join(name.split("-")[:2])}_output_dict.pkl') as f:
            model_logprobs_output_dict = dill.load(f)

        print(s_guide_output_dict.keys(), s_guide_output_dict)
        print(model_logprobs_output_dict.keys(), model_logprobs_output_dict)

        return

        subsamples[name] = {
            # "states": states,
            # "travel_times_dict": travel_times_dict,
            # "observations_df": observations_df,
            "num_flights": num_flights,
            "model": model,
            "visibility": visibility_dict[name],
            "ceiling": ceiling_dict[name],
            
            "s_guide_dist": None, # TODO: !!!
            "model_logprobs": None, # TODO: !!!
        }


    def y_given_c_log_prob(model_logprobs, prior_log_prob_fn):
        # index i corresonds to .001*(1+i)
        start_idx = 10
        start_val = start_idx / 1000.0
        samples = torch.arange(start_val, 0.040, 0.001).to(device)
        model_logprobs = model_logprobs[start_idx:]
        prior_logprobs = prior_log_prob_fn(samples)
        logprob = torch.logsumexp(model_logprobs + prior_logprobs)
        return logprob # i think technically this is  abit off but whatever just approx for now.
    
    # now define guide
    n_context = 1
    hidden_dim = 2
    if posterior_guide == "nsf":
       guide = zuko.flows.NSF(
            features=mst_split,
            context=n_context,
            hidden_features=(hidden_dim, hidden_dim),
        ).to(device)
    elif posterior_guide == "cnf":
        guide = zuko.flows.CNF(
            features=mst_split,
            context=n_context,
            hidden_features=(hidden_dim, hidden_dim),
        ).to(device)
    elif posterior_guide == "gmm":
        guide = ConditionalGaussianMixture(
            n_context=n_context, 
            n_features=mst_split,
        )
    else:
        raise ValueError

    # here, we make the nominal, failure, and uniform (not used) priors...
    # TODO: shifted gamma doesn't work. i have no idea why
    # mst_prior_nominal = _affine_beta_dist_from_mean_std(.0125, .001, .010, .020, device)
    # mst_prior_nominal = _gamma_dist_from_mean_std(.0125, .0004, device)
    mst_prior_nominal = dist.Normal(
        torch.tensor(.0125).to(device), 
        torch.tensor(.0004).to(device)
    )

    # mst_prior_failure = _affine_beta_dist_from_mean_std(.0200, .002, .010, .030, device)
    # mst_prior_failure = _gamma_dist_from_mean_std(.0250, .0012, device)
    mst_prior_failure = dist.Normal(
        torch.tensor(.0190).to(device), 
        torch.tensor(.0012).to(device)
    )

    fig = plt.figure()
    s = mst_prior_nominal.sample((10000,)).detach().cpu()
    sns.histplot(s, color='b', alpha=.33, fill=True, label="nominal", kde=True, binwidth=.0001, edgecolor='k', linewidth=0)
    s = mst_prior_failure.sample((10000,)).detach().cpu()
    sns.histplot(s, color='r', alpha=.33, fill=True, label="failure", kde=True, binwidth=.0001, edgecolor='k', linewidth=0)
    plt.savefig('ab_test.png')
    plt.legend()
    plt.close(fig)

    return
    
    
    # now define prior
    # TODO!!:
    prior_dists = (
        mst_prior_nominal,
        mst_prior_failure,
    )
    failure_prior = mst_prior_failure
    nominal_prior = mst_prior_nominal

    class PriorMixture(object):
        def __init__(self, failure_prior, nominal_prior):
            self.failure_prior = failure_prior
            self.nominal_prior = nominal_prior
            return
        def log_prob(self, sample, label):
            return (
                    (label) * self.failure_prior.log_prob(sample) 
                + (1-label) * self.nominal_prior.log_prob(sample)
            )
        
    class WeatherThreshold(torch.nn.Module):

        def __init__(self, a):
            super().__init__()
            self.device = device
            self.visibility_threshold = torch.nn.Parameter(
                torch.tensor(2.5).to(device),
                requires_grad=True
            )
            self.ceiling_threshold = torch.nn.Parameter(
                torch.tensor(1.0).to(device), # scale by 1k
                requires_grad=True
            )
            # self.ceiling_threshold = torch.nn.Parameter()
            self.a = torch.tensor(a).to(device)

        def assign_label(self, visibility, ceiling):
            return torch.nn.functional.sigmoid(
                self.a * (self.visibility_threshold - visibility)
            ) * torch.nn.functional.sigmoid(
                self.a * (self.ceiling_threshold - ceiling/1000.0)
            )
        

    class ClusterThreshold(torch.nn.Module):

        def __init__(self, a):
            super().__init__()
            self.device = device
            self.visibility_threshold = torch.nn.Parameter(
                torch.tensor(3.5).to(device),
                requires_grad=True
            )
            self.ceiling_threshold = torch.nn.Parameter(
                torch.tensor(2.0).to(device), # scale by 1k
                requires_grad=True
            )
            self.num_flights_threshold = torch.nn.Parameter(
                torch.tensor(0.9).to(device), # scale by 1k
                requires_grad=True
            )
            # self.ceiling_threshold = torch.nn.Parameter()
            self.a = torch.tensor(a).to(device)

        def assign_label(self, visibility, ceiling, num_flights):
            return torch.nn.functional.sigmoid(
                self.a * (self.visibility_threshold - visibility)
            ) * torch.nn.functional.sigmoid(
                self.a * (self.ceiling_threshold - ceiling/1000)
            ) * torch.nn.functional.sigmoid(
                self.a * (num_flights/1000 - self.num_flights_threshold)
            )
        
    weather_threshold = WeatherThreshold(50.0)
    cluster_threshold = ClusterThreshold(50.0)

    prior_mixture = PriorMixture(failure_prior, nominal_prior)

    def objective_fn(subsample, n=1):

        visibility = subsample["visibility"]
        ceiling = subsample["ceiling"]
        num_flights = subsample["num_flights"]
        s_guide_dist = subsample["s_guide_dist"]
        model_logprobs = subsample["model_logprobs"]

        prior_label = weather_threshold.assign_label(visibility, ceiling)
        posterior_label = cluster_threshold.assign_label(visibility, ceiling, num_flights)

        posterior_samples = guide(posterior_label).rsample((n,))
        posterior_logprobs = guide(posterior_label).log_prob(posterior_samples)

        s_logprobs = s_guide_dist.log_prob(posterior_samples)
        prior_logprobs = prior_mixture.log_prob(posterior_samples, prior_label)

        y_given_c_logprobs = y_given_c_log_prob(
            model_logprobs, 
            functools.partial(prior_mixture.log_prob, label=prior_label)
        )

        objective = (
            s_logprobs + prior_logprobs - posterior_logprobs
        ).mean() - y_given_c_logprobs

        return objective
    
    def total_objective_fn(subsamples, n=1):
        loss = torch.tensor(0.0).to(device)
        for name, subsample in subsamples.items():
            loss += objective_fn(subsample, n)
        return loss

    # Set up SVI
    gamma = gamma  # final learning rate will be gamma * initial_lr
    lrd = gamma ** (1 / svi_steps)

    guide_optimizer = torch.optim.Adam(
        guide.parameters(),
        lr=svi_lr,
        weight_decay=0.0, # fix this
    )
    guide_scheduler = torch.optim.lr_scheduler.StepLR(
        guide_optimizer, step_size=svi_steps, gamma=gamma
    )

    wt_optimizer = torch.optim.Adam(
        weather_threshold.parameters(),
        lr=svi_lr,
        weight_decay=0.0, # fix this
    )
    wt_scheduler = torch.optim.lr_scheduler.StepLR(
        wt_optimizer, step_size=svi_steps, gamma=gamma
    )

    ct_optimizer = torch.optim.Adam(
        cluster_threshold.parameters(),
        lr=svi_lr,
        weight_decay=0.0, # fix this
    )
    ct_scheduler = torch.optim.lr_scheduler.StepLR(
        ct_optimizer, step_size=svi_steps, gamma=gamma
    )

    run_name = f"[{','.join(network_airport_codes)}]_"
    run_name += f"[{posterior_guide}]_"
    run_name += f"[{day_strs_list[0][0]}_{day_strs_list[-1][0]}]"
    # print(run_name)
    group_name = f"{posterior_guide}"
    # print(group_name)
    # TODO: fix this for non-day-by-day???
    sub_dir = f"{project}/checkpoints/{'_'.join(network_airport_codes)}/{day_strs_list[0][0]}_{day_strs_list[-1][0]}/{posterior_guide}/"

    wandb_init_config_dict = {
        # "starting_aircraft": starting_aircraft,
        "days": days,
        "num_flights": num_flights,
        "num_days": num_days,

        "dt": dt,
        "svi_lr": svi_lr,
        "svi_steps": svi_steps,
        "gamma": gamma,
        "n_samples": n_samples,
        "n_elbo_particles": n_elbo_particles,

        # "prior_type": prior_type,
        # "prior_scale": prior_scale,
        "posterior_guide": posterior_guide,
    }
    
    wandb.init(
        project=project,
        name=run_name,
        group=group_name,
        config=wandb_init_config_dict,
    )

    losses = []

    pbar = tqdm(range(svi_steps))

    for i in pbar:
        guide_optimizer.zero_grad()
        wt_optimizer.zero_grad()
        ct_optimizer.zero_grad()

        loss = total_objective_fn()
        loss.backward()
        guide_grad_norm = torch.nn.utils.clip_grad_norm_(
            guide.parameters(), 100.0 # TODO :grad clip param
        )
        wt_grad_norm = torch.nn.utils.clip_grad_norm_(
            weather_threshold.parameters(), 100.0 # TODO :grad clip param
        )
        ct_grad_norm = torch.nn.utils.clip_grad_norm_(
            cluster_threshold.parameters(), 100.0 # TODO :grad clip param
        )

        guide_optimizer.step()
        guide_scheduler.step()
        wt_optimizer.step()
        wt_scheduler.step()
        ct_optimizer.step()
        ct_scheduler.step()

        loss = loss.detach()
        losses.append(loss)

        # posterior_samples = torch.exp(guide.sample((n_samples,)))

        # z_samples = [posterior_samples[:,t_idx] for t_idx in range(mst_split)]
        # mst_mles = [z_samples[t_idx].mean().item() for t_idx in range(mst_split)]

        desc = f"ELBO loss: {loss:.2f}"
                
        if i % plot_every == 0 or i == svi_steps - 1:
            labels = [torch.tensor([a, b]).to(device) for a in (0.0, 1.0) for b in (0.0, 1.0)]
            for label in labels:
                fig = plt.figure()
                samples = torch.exp(guide(label).sample((n_samples,)))/1000
                sns.histplot(
                    samples, 
                    color='b', 
                    alpha=.5, 
                    fill=True, 
                    edgecolor='k', 
                    linewidth=0
                )
                plt.xlim(0,.050)
                wandb.log({f"posteriors/{label.numpy()}": wandb.Image(fig)}, commit=False)
                plt.close(fig)
            

        #     # hourly_delays = get_hourly_delays(
        #     #     model, guide, states, observations_df, 1
        #     # )

        #     # fig = plot_hourly_delays(hourly_delays)
        #     # wandb.log({"Hourly delays": wandb.Image(fig)}, commit=False)
        #     # plt.close(fig)

        #     # plotting_dict = {
        #     #     "mean service times": plot_service_times,
        #     # }
        
        #     # for name, plot_func in plotting_dict.items():
        #     #     # for now require this common signature
        #     #     fig = plot_func(guide, states, n_samples)
        #     #     wandb.log({name: wandb.Image(fig)}, commit=False)
        #     #     plt.close(fig)

        #     # Save the params and autoguide
        #     dir_path = os.path.dirname(__file__)
        #     # save_path = os.path.join(dir_path, "checkpoints_final", run_name, f"{i}")
        #     save_path = os.path.join(dir_path, sub_dir, f"{i}")
        #     os.makedirs(save_path, exist_ok=True)
        #     pyro.get_param_store().save(os.path.join(save_path, "params.pth"))
        #     torch.save(guide.state_dict(), os.path.join(save_path, "guide.pth"))

        # wandb.log({"ELBO": loss})
        log_dict = {
            "ELBO/loss (units = nats per dim)": loss, 
        }
        # for i in range(len(mst_mles)):
        #     log_dict[f"mean service time mle/{i} (hours)"] = mst_mles[i]
        wandb.log(log_dict)

    wandb.save(f"checkpoints/{run_name}/checkpoint_{svi_steps - 1}.pt")

    output_dict = {
        'model': model,
        'guide': guide,
        'states': states,
        'dt': dt,
        # 'set_model': functools.partial(model, states),
        # 'set_guide': functools.partial(guide, states),
        'run_name': run_name,
        'group_name': group_name,
        'config': wandb_init_config_dict
    }
    # TODO: this is like not the best way of handling it but whatever
    # also maybe redundant but just in case i guess

    dir_path = os.path.dirname(__file__)
    # save_path = os.path.join(dir_path, "checkpoints_final", run_name, "final")
    save_path = os.path.join(dir_path, sub_dir, "final")
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "output_dict.pkl"), 'wb+') as handle:
        dill.dump(output_dict, handle)

    wandb.finish(0)

    return loss


from multiprocessing import Process, Pool, cpu_count
# https://stackoverflow.com/questions/7207309/how-to-run-functions-in-parallel
# trying something
def run_cpu_tasks_in_parallel(tasks):
    running_tasks = [Process(target=task) for task in tasks]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()

import warnings

# TODO: add functionality to pick days
@click.command()
@click.option("--project", default="bayes-air-atrds-attempt-5")
@click.option("--network-airport-codes", default="LGA", help="airport codes")

@click.option("--svi-steps", default=500, help="Number of SVI steps to run")
@click.option("--n-samples", default=5000, help="Number of posterior samples to draw")
@click.option("--svi-lr", default=5e-3, help="Learning rate for SVI")
@click.option("--plot-every", default=50, help="Plot every N steps")
@click.option("--rng-seed", default=1, type=int)
@click.option("--gamma", default=.4) # was .1
@click.option("--dt", default=.1)
@click.option("--n-elbo-particles", default=1)

# @click.option("--prior-type", default="empty", help="nominal/failure/empty")
@click.option("--posterior-guide", default="gaussian", help="gaussian/iafnormal/delta/laplace") 
# # delta and laplace break plots rn 
# @click.option("--prior-scale", default=0.0, type=float)

@click.option("--day-strs", default=None)
# @click.option("--day-strs-path", default=None) # TODO: make this!!
@click.option("--year", default=None, type=int)
@click.option("--month", default=None, type=int)
@click.option("--start-day", default=None)
@click.option("--end-day", default=None)

# @click.option("--learn-together", is_flag=True)
# @click.option("--all-combos", is_flag=True)

# @click.option("--multiprocess/--no-multiprocess", default=True)
# @click.option("--processes", default=None)
# @click.option("--wandb-silent", is_flag=True)


def train_cmd(
    project, network_airport_codes, 
    svi_steps, n_samples, svi_lr, 
    plot_every, rng_seed, gamma, dt, n_elbo_particles,
    # prior_type, prior_scale, 
    posterior_guide, 
    day_strs, year, month, start_day, end_day,
    # learn_together, 
    # all_combos, 
    # multiprocess, processes, wandb_silent
):
    # TODO: make this better

    # nominal_days = [
    #     1,2,3,4,5,7,9,
    #     10,12,13,14,15,16,
    #     20,24,25,26,27,28,29
    # ]

    # failure_days = [
    #     d for d in range(1,32)
    #     if d not in nominal_days
    # ] 

    network_airport_codes = network_airport_codes.split(',')
    if day_strs is not None:
        day_strs = day_strs.split(',')
    elif year is not None and month is not None:
        start_day = f'{year}-{month}-1'
        days_in_month = pd.Period(start_day).days_in_month
        end_day = f'{year}-{month}-{days_in_month}'
        day_strs = pd.date_range(start=start_day, end=end_day, freq='D').strftime('%Y-%m-%d').to_list()
    elif start_day is not None:
        if end_day is None:
            end_day = start_day
        day_strs = pd.date_range(start=start_day, end=end_day, freq='D').strftime('%Y-%m-%d').to_list()
    else:
        raise ValueError
    
    day_strs_list = [
        [day_str] for day_str in day_strs
    ]
    

    train(
        project,
        network_airport_codes,
        svi_steps,
        n_samples,
        svi_lr,
        gamma,
        dt,
        n_elbo_particles,
        plot_every,
        rng_seed,
        (
            day_strs_list,
            posterior_guide
        )
    )


if __name__ == "__main__":
    train_cmd()