"""Run the simulation for a LGA focused augmented network"""
import os
# from itertools import combinations

import click
import matplotlib
# import matplotlib.pyplot as plt
import pandas as pd
# import numpy as np
import pyro
# import seaborn as sns
import torch
# from math import ceil, floor
import functools
import dill
# import sys
from pathlib import Path

import bayes_air.utils.dataloader as ba_dataloader
import wandb
from bayes_air.model import augmented_air_traffic_network_model_simplified
# from bayes_air.network import NetworkState, AugmentedNetworkState
# from bayes_air.schedule import split_and_parse_full_schedule

from bayes_air.types.util import CoreAirports

from tqdm import tqdm
import tqdm

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
# import zuko

# from pbar_pool import PbarPool, Pbar

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



from multiprocessing import Pool, cpu_count

def train(
        day_strs_list,
        year, month,
        use_gpu=False
    ):
    pyro.clear_param_store()  # avoid leaking parameters across runs
    pyro.enable_validation(True)
    rng_seed = 1
    pyro.set_rng_seed(int(rng_seed))

    # network_airport_codes = ['LGA']
    # network_airport_codes = ['JFK', 'LGA', 'EWR']
    network_airport_codes = ['JFK']
    dt = .1

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

    # processed_visibility = pd.read_csv(dir_path / 'processed_visibility.csv')
    # visibility_dict = dict(processed_visibility.values)
    # processed_ceiling = pd.read_csv(dir_path / 'processed_ceiling.csv')
    # ceiling_dict = dict(processed_ceiling.values)

    # SETTING UP THE MODEL AND STUFF

    # Hyperparameters
    initial_aircraft = 50.0 # not used!
    mst_effective_hrs = 24 # not used!
    mst_split = 1 # not really used

    subsamples = {}

    for day_strs in day_strs_list:

        # gather data
        days = pd.to_datetime(day_strs)
        data = ba_dataloader.load_remapped_data_bts(days, network_airport_codes[0])
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

        subsamples[name] = {
            # "states": states,
            # "travel_times_dict": travel_times_dict,
            # "observations_df": observations_df,
            "num_flights": num_flights,
            "model": model,
            # "visibility": visibility_dict[name]failure_prior, nominal_prior, 
            "name": name,
        }


    # here, we make the nominal, failure, and uniform (not used) priors...
    # TODO: shifted gamma doesn't work. i have no idea why
    # mst_prior_nominal = _affine_beta_dist_from_mean_std(.0125, .001, .010, .020, device)
    mst_prior_nominal = _gamma_dist_from_mean_std(.0125, .0004, device)

    # mst_prior_failure = _affine_beta_dist_from_mean_std(.0200, .002, .010, .030, device)
    mst_prior_failure = _gamma_dist_from_mean_std(.0250, .0012, device)

    # mst_prior_default = _affine_beta_dist_from_alpha_beta(1.0, 1.0, .005, .030, device)
    mst_prior_default = _gamma_dist_from_mean_std(.0135, .0050, device)

    # fig = plt.figure()
    # s = mst_prior_nominal.sample((10000,)).detach().cpu()
    # sns.histplot(s, color='b', alpha=.33, fill=True, label="nominal", kde=True, binwidth=.0001, edgecolor='k', linewidth=0)
    # s = mst_prior_failure.sample((10000,)).detach().cpu()
    # sns.histplot(s, color='r', alpha=.33, fill=True, label="failure", kde=True, binwidth=.0001, edgecolor='k', linewidth=0)
    # plt.savefig('ab_test.png')
    # s = mst_prior_default.sample((10000,)).detach().cpu()
    # sns.histplot(s, color='purple', alpha=.33, fill=True, label="empty", kde=True, binwidth=.0001, edgecolor='k', linewidth=0)
    # plt.legend()
    # plt.title("prior distributions example")
    # plt.savefig('abc_test.png')
    # plt.close(fig)
    
    failure_prior = mst_prior_failure
    nominal_prior = mst_prior_nominal

    output_dict = {}

    @torch.no_grad
    def process_subsamples(subsample):
            
        model = subsample["model"]
        # output_dict[subsample["name"]] = torch.zeros((30,), requires_grad=False)
        output_dict[subsample["name"]] = torch.zeros((300,), requires_grad=False)

        # pbar = tqdm.tqdm(np.arange(.001, .041, .001), leave=False)
        # for zi in tqdm.tqdm(range(10, 40)):
        for zi in tqdm.tqdm(range(100, 400)):
            z = zi / 10000.0
            tz = torch.tensor(z, requires_grad=False).to(device)
            l = single_particle_y_given_z(model, tz)
            pf = failure_prior.log_prob(tz)
            pn = nominal_prior.log_prob(tz)
            # print(f'\n{subsample["name"]} {z:.3f} {l.item():.3f} {pf.item():.3f} {pn.item():.3f}')
            print(f'\n{subsample["name"]} {z:.4f} {l.item():.3f} {pf.item():.3f} {pn.item():.3f}')
            # output_dict[subsample["name"]][zi-10] = l
            output_dict[subsample["name"]][zi-100] = l
        
        print(output_dict[subsample["name"]])

    for name, subsample in tqdm.tqdm(subsamples.items()):
        process_subsamples(subsample)

    dir_path = os.path.dirname(__file__)
    # save_path = os.path.join(dir_path, "model_logprobs")
    save_path = os.path.join(dir_path, f"{network_airport_codes[0]}_model_logprobs_finer_testing")
    os.makedirs(save_path, exist_ok=True)
    fname = (
        f"{year:04d}_{month:02d}_output_dict.pkl" 
        if year is not None and month is not None else
        f"output_dict.pkl" 
    )
    with open(os.path.join(save_path, fname), 'wb+') as handle:
        dill.dump(output_dict, handle)

    list_dict = {
        k: v.tolist()
        for k, v in output_dict.items()
    }

    df = pd.DataFrame.from_dict(list_dict,orient='index').transpose()
    print(df)
    fname = (
        f"{year:04d}_{month:02d}_output.csv" 
        if year is not None and month is not None else
        f"output.csv" 
    )
    df.to_csv(os.path.join(save_path, fname), index=False)
    fname = (
        f"{year:04d}_{month:02d}_output.parquet" 
        if year is not None and month is not None else
        f"output.parquet"
    )
    df.to_parquet(os.path.join(save_path, fname))



    # pool = Pool(processes=cpu_count())
    # for _ in tqdm.tqdm(pool.imap_unordered(process_subsamples, subsamples.values()), total=len(subsamples)):
    #     pass


@click.command()
@click.option("--day-strs", default="2019-07-15")
# @click.option("--day-strs-path", default=None) # TODO: make this!!
@click.option("--year", default=None, type=int)
@click.option("--month", default=None, type=int)
@click.option("--start-day", default=None)
@click.option("--end-day", default=None)

def train_cmd(day_strs, year, month, start_day, end_day):
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

    if year is None:
        year = pd.to_datetime(day_strs[0]).year
    if month is None:
        month = pd.to_datetime(day_strs[0]).month
    
    train(day_strs_list, year, month)


if __name__ == "__main__":
    train_cmd()