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
)


def plot_travel_times(
    auto_guide, states, dt, n_samples, empirical_travel_times, wandb=True
):
    """Plot posterior samples of travel times."""
    # Sample nominal travel time estimates from the posterior
    with pyro.plate("samples", n_samples, dim=-1):
        posterior_samples = auto_guide(states, dt)

    # Plot training curve and travel time posterior
    # airport_codes = states[0].airports.keys()
    # pairs = list(combinations(airport_codes, 2))
    pairs = list(
        empirical_travel_times[
            ["origin_airport", "destination_airport"]
        ].itertuples(index=False, name=None)
    )

    # Make subplots for the learning curve and each travel time pair
    n_pairs = len(pairs)
    # max_rows = 30
    # max_pairs_per_row = n_pairs // max_rows + 1
    max_pairs_per_row = 4
    max_rows = ceil(n_pairs / max_pairs_per_row)
    subplot_spec = []
    for i in range(max_rows):
        subplot_spec.append(
            [f"{i * max_pairs_per_row +j}" for j in range(max_pairs_per_row)]
        )

    fig = plt.figure(layout="constrained", figsize=(12, 3 * max_rows))
    axs = fig.subplot_mosaic(subplot_spec)

    shared_axs = []

    for i, pair in enumerate(pairs):
        # Put all of the data into a DataFrame to plot it
        tmp = (
            posterior_samples[
                f"travel_time_{pair[0]}_{pair[1]}"
            ]
            .detach()
            .cpu()
            .numpy()
        )
        plotting_df = pd.DataFrame(
            {
                f"{pair[0]}->{pair[1]}": tmp,
                f"{pair[1]}->{pair[0]}": np.zeros(len(tmp)),
                "type": "Posterior",
            }
        )

        sns.histplot(
            x=f"{pair[0]}->{pair[1]}",
            hue="type",
            ax=axs[f"{i}"],
            data=plotting_df,
            color="blue",
            kde=True,
        )

        tmp = empirical_travel_times.loc[
            (empirical_travel_times.origin_airport == pair[0])
            & (empirical_travel_times.destination_airport == pair[1]),
            "travel_time",
        ].mean()

        axs[f"{i}"].axvline(tmp, color="red",zorder=9, label="scheduled mean")

        # axs[f"{i}"].set_xlabel(f"{pair[0]} -> {pair[1]}")
        # axs[f"{i}"].set_ylabel(f"{pair[1]} -> {pair[0]}")
        # axs[f"{i}"].set_xlim(0, 6)
        # axs[f"{i}"].set_ylim(0, 5)

        if i == 0:
            axs[f"{i}"].legend()
        else:
            axs[f"{i}"].legend([], [], frameon=False)

        shared_axs.append(axs[f"{i}"])

    handle_shared_ax_lims(shared_axs)

    return fig


def handle_shared_ax_lims(
        shared_axs,
        xlim=None,
        ylim=None,
    ):

    if xlim == None:
        xmins, xmaxs = [], []
        for ax in shared_axs:
            xmin, xmax = ax.get_xlim()
            xmins.append(xmin)
            xmaxs.append(xmax)
        shared_xmin = min(xmins)
        shared_xmax = max(xmaxs)
    else:
        shared_xmin, shared_xmax = xlim

    if ylim == None:
        ymins, ymaxs = [], []
        for ax in shared_axs:
            ymin, ymax = ax.get_ylim()
            ymins.append(ymin)
            ymaxs.append(ymax)
        shared_ymin = min(ymins)
        shared_ymax = max(ymaxs)
    else:
        shared_ymin, shared_ymax = ylim

    for ax in shared_axs:
        ax.set_xlim(shared_xmin, shared_xmax)
        ax.set_ylim(shared_ymin, shared_ymax)


def plot_time_indexed_network_var(
        base_var_name,
        auto_guide, states, dt, n_samples,
        transform=(lambda x: x),
        plots_per_row=3,
        width=None,
        ignore_time_index=False,
        xlim=None,
        ylim=None,
    ):

    if width is None:
        default_height = 5
        width = plots_per_row * default_height

    # Sample mean service time estimates from the posterior
    with pyro.plate("samples", n_samples, dim=-1):
        posterior_samples = auto_guide(states, dt)

    # Make subplots for each airport
    airport_codes = states[0].network_state.airports.keys()
    n_pairs = len(airport_codes)

    # Make subplots for each airport
    n_pairs = len(airport_codes)

    max_plots_per_row = plots_per_row
    max_rows = ceil(n_pairs / max_plots_per_row)
    subplot_spec = []
    for i in range(max_rows):
        subplot_spec.append(
            [f"{i * max_plots_per_row +j}" for j in range(max_plots_per_row)]
        )

    row_height = width / max_plots_per_row
    fig = plt.figure(layout="constrained", figsize=(width, row_height * max_rows))
    axs = fig.subplot_mosaic(subplot_spec)

    for i, code in enumerate(airport_codes):
        # Put all of the data into a DataFrame to plot it
        shared_axs = []
        t_idx = 0
        while True:
            prefix = (
                f"{code}_{t_idx}" 
                if not ignore_time_index
                else f"{code}"
            )
            name = f"{prefix}_{base_var_name}"
            if name not in posterior_samples:
                break

            plot_idx = i*max_plots_per_row + t_idx
            t_idx += 1
            
            plotting_df = pd.DataFrame(
                {
                    prefix: transform(posterior_samples[name])
                    .detach()
                    .cpu()
                    .numpy(),
                    "type": "Posterior",
                }
            )
            
            ax = axs[f"{plot_idx}"]

            sns.histplot(
                x=prefix,
                hue="type",
                ax=ax,
                data=plotting_df,
                color="blue",
                kde=True,
                edgecolor='k',
                linewidth=0,
                bins=ceil(np.sqrt(n_samples)),
                # binwidth=.0001 # this is bit of a hack
            )
            mle = plotting_df[prefix].mean()
            ax.axvline(mle,linestyle=":")
            ax.title.set_text(f'{name} mle = {mle:.5f}')

            shared_axs.append(ax)
            
            if ignore_time_index:
                break

        handle_shared_ax_lims(shared_axs, xlim, ylim)

    return fig


plot_service_times = functools.partial(
    plot_time_indexed_network_var,
    "mean_service_time",
    plots_per_row=1,
    xlim=(.005, .030) # new
)

plot_turnaround_times = functools.partial(
    plot_time_indexed_network_var,
    "mean_turnaround_time",
    plots_per_row=1,
)

plot_base_cancel_prob = functools.partial(
    plot_time_indexed_network_var,
    "base_cancel_neg_logprob",
    transform=lambda x: torch.exp(-x),
    plots_per_row=1,
)

plot_starting_aircraft = functools.partial(
    plot_time_indexed_network_var,
    "log_initial_available_aircraft",
    transform=torch.exp,
    plots_per_row=1,
    ignore_time_index=True,
)

plot_soft_max_holding_time = functools.partial(
    plot_time_indexed_network_var,
    "soft_max_holding_time",
    plots_per_row=1,
    ignore_time_index=True,
)


def plot_elbo_losses(losses):

    fig = plt.figure(figsize=(8, 8))
    plt.plot(losses, label="ELBO loss")
    plt.title("ELBO loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()

    return fig

def plot_rmses(arr_rmses, dep_rmses, arr_rmses_adj, dep_rmses_adj, rmse_idxs):

    fig = plt.figure(figsize=(8, 8))
    plt.plot(rmse_idxs, arr_rmses, "-b", label="arrivals RMSE")
    plt.plot(rmse_idxs, dep_rmses, "-r", label="departures RMSE")
    plt.plot(rmse_idxs, arr_rmses_adj, ":b", label="arrivals RMSE (excluding worst 2.5%)")
    plt.plot(rmse_idxs, dep_rmses_adj, ":r", label="departures RMSE (excluding worst 2.5%)")
    plt.title("RMSE of arrival and departure times for flights at LGA")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()

    return fig


def get_arrival_departures_rmses(
    model, auto_guide, states, dt, 
    observations_df, n_samples, wandb=True
):
    
    with pyro.plate("samples", n_samples, dim=-1):
        posterior_samples = auto_guide(states, dt)

    predictive = pyro.infer.Predictive(
        model=model,
        posterior_samples=posterior_samples,
    )

    samples = predictive(
        states, dt, 
        obs_none=True,
    )

    # trace_pred = pyro.infer.TracePredictive(
    #     model, svi, num_samples=n_samples
    # ).run(states, dt)
    
    # samples = trace_pred()

    # print(samples)

    day_str_list = []
    flight_number_list = []
    origin_airport_list = []
    destination_airport_list = []
    sample_arrival_time_list = []
    sample_departure_time_list = []
    sample_cancelled_list = []

    for key, sample in samples.items():
        split_key = key.split('_')
        if split_key[-1] != 'time' and split_key[-1] != 'cancelled':
            continue

        arr_sample = None
        dep_sample = None
        # ccl_sample = None

        if split_key[-1] == 'cancelled':
            # ccl_sample = sample.mean().item() # each 1 or 0
            continue
        elif split_key[-2] == 'arrival':
            arr_sample = sample.mean().item()
        elif split_key[-2] == 'departure':
            dep_sample = sample.mean().item()
        else:
            continue

        # print(split_key, sample)

        day_str_list.append(split_key[0])
        flight_number_list.append(
            split_key[1]
            # f'{split_key[1]}_{split_key[2]}_{split_key[3]}'
        )
        origin_airport_list.append(split_key[2])
        destination_airport_list.append(split_key[3])
        sample_arrival_time_list.append(arr_sample)
        sample_departure_time_list.append(dep_sample)
        # sample_cancelled_list.append(ccl_sample)

    samples_df = pd.DataFrame(
        {   
            "date": pd.to_datetime(day_str_list),
            "flight_number": flight_number_list,
            "origin_airport": origin_airport_list,
            "destination_airport": destination_airport_list,
            "sample_arrival_time": sample_arrival_time_list,
            "sample_departure_time": sample_departure_time_list,
            # "sample_cancelled": sample_cancelled_list,
        }
    )

    # print(samples_df)
    # print(observations_df)

    merged_df = pd.merge(
        samples_df, 
        observations_df,
        on=[
            "date",
            "flight_number",
            "origin_airport",
            "destination_airport"
        ],
        how='inner',
    )

    dep_mask_successful = (
        (merged_df["origin_airport"] == "LGA")
        & (merged_df["actual_departure_time"] != 0)
        & (merged_df["sample_departure_time"] != 0)
    )
    arr_mask_successful = (
        (merged_df["destination_airport"] == "LGA")
        & (merged_df["actual_arrival_time"] != 0)
        & (merged_df["sample_arrival_time"] != 0)
    )

    dep_mask_all = (merged_df["origin_airport"] == "LGA")
    arr_mask_all = (merged_df["destination_airport"] == "LGA")

    split_delay_cols = [
        "carrier_delay", 
        # "weather_delay", 
        "nas_delay", 
        # "security_delay", 
        "late_aircraft_delay",
    ]

    arrivals_df = merged_df.loc[
        arr_mask_successful,
        ["date", "flight_number",
         "sample_arrival_time", 
         "actual_arrival_time"]
         + split_delay_cols
    ]

    departures_df = merged_df.loc[
        dep_mask_successful,
        ["date", "flight_number",
         "sample_departure_time", 
         "actual_departure_time"]
         + split_delay_cols
    ]

    arrivals_df["squared_dist"] = (
        arrivals_df["sample_arrival_time"] - 
        arrivals_df["actual_arrival_time"]
    )**2

    departures_df["squared_dist"] = (
        departures_df["sample_departure_time"] - 
        departures_df["actual_departure_time"]
    )**2

    arrivals_mse = arrivals_df["squared_dist"].mean()
    arrivals_rmse = np.sqrt(arrivals_mse)

    departures_mse = departures_df["squared_dist"].mean()
    departures_rmse = np.sqrt(departures_mse)

    # arr_n_ignore = int(len(arrivals_df) * 0.025)
    # arrivals_mse_adj = (
    #     arrivals_df
    #     .drop(arrivals_df.nlargest(arr_n_ignore, "squared_dist").index)
    #     ["squared_dist"]
    #     .mean()
    # )
    # arrivals_rmse_adj = np.sqrt(arrivals_mse_adj)
    arrivals_mse_adj = 0.0
    arrivals_rmse_adj = 0.0

    dep_n_ignore = int(len(departures_df) * 0.025)
    departures_mse_adj = (
        departures_df
        .drop(departures_df.nlargest(dep_n_ignore, "squared_dist").index)
        ["squared_dist"]
        .mean()
    )
    departures_rmse_adj = np.sqrt(departures_mse_adj)

    # # print(arrivals_df)
    # # print(departures_df)
    # print(arrivals_df.nlargest(100, columns=['squared_dist']).drop(["date"], axis=1).to_string(index=False))
    # print("")
    # print(departures_df.nlargest(100, columns=['squared_dist']).drop(["date"], axis=1).to_string(index=False))

    # print("")
    # # print(arrivals_mse, departures_mse)
    # print(arrivals_rmse, departures_rmse)
    # print("")
    # # print(arrivals_mse_adj, departures_mse_adj)
    # print(arrivals_rmse_adj, departures_rmse_adj)


    # hr_split = 10 # about to do some heresy
    hr_lim = 25

    arrival_delays_df = merged_df.loc[
        arr_mask_successful,
        ["date", "flight_number",
         "sample_arrival_time", 
         "actual_arrival_time",
         "scheduled_arrival_time"]
    ]

    arrival_delays_df["sample_arrival_delay"] = (
        arrival_delays_df["sample_arrival_time"] 
        - arrival_delays_df["scheduled_arrival_time"]
    )
    arrival_delays_df["actual_arrival_delay"] = (
        arrival_delays_df["actual_arrival_time"] 
        - arrival_delays_df["scheduled_arrival_time"]
    )

    actual_arrival_hour = (
        np.floor(arrival_delays_df.actual_arrival_time).astype(int)
    )
    sample_arrival_hour = (
        np.floor(arrival_delays_df.sample_arrival_time).astype(int)
    )

    hourly_sample_arrival_delay = (
        arrival_delays_df
        .groupby(sample_arrival_hour)
        ["sample_arrival_delay"]
        .mean()
        .loc[:hr_lim]
    )
    hourly_actual_arrival_delay = (
        arrival_delays_df
        .groupby(actual_arrival_hour)
        ["actual_arrival_delay"]
        .mean()
        .loc[:hr_lim]
    )

    departure_delays_df = merged_df.loc[
        dep_mask_successful,
        ["date", "flight_number",
         "sample_departure_time", 
         "actual_departure_time",
         "scheduled_departure_time"]
    ]

    departure_delays_df["sample_departure_delay"] = (
        departure_delays_df["sample_departure_time"] 
        - departure_delays_df["scheduled_departure_time"]
    )
    departure_delays_df["actual_departure_delay"] = (
        departure_delays_df["actual_departure_time"] 
        - departure_delays_df["scheduled_departure_time"]
    )

    actual_departure_hour = (
        np.floor(departure_delays_df.actual_departure_time).astype(int)
    )
    sample_departure_hour = (
        np.floor(departure_delays_df.sample_departure_time).astype(int)
    )

    hourly_sample_departure_delay = (
        departure_delays_df
        .groupby(sample_departure_hour)
        ["sample_departure_delay"]
        .mean()
        .loc[:hr_lim]
    )
    hourly_actual_departure_delay = (
        departure_delays_df
        .groupby(actual_departure_hour)
        ["actual_departure_delay"]
        .mean()
        .loc[:hr_lim]
    )

    combined_hourly_delays_df = pd.concat(
        [
            hourly_sample_arrival_delay,
            hourly_actual_arrival_delay,
            hourly_sample_departure_delay,
            hourly_actual_departure_delay,
        ],
        axis=1,
    ).sort_index()

    # print(combined_hourly_delays_df)

    # TODO: cancellation
    # arrival_cancel_df = merged_df.loc[
    #     arr_mask_all,
    #     ["date", "flight_number",
    #      "sample_arrival_time", 
    #      "actual_arrival_time",
    #      "sample_cancelled",
    #      "cancelled"]
    # ]
    # departure_cancel_df = merged_df.loc[
    #     dep_mask_all,
    #     ["date", "flight_number",
    #      "sample_departure_time", 
    #      "actual_departure_time",
    #      "sample_cancelled",
    #      "cancelled"]
    # ]

    # hourly_arrival_delays_rmse = np.sqrt(
    #     (hourly_sample_arrival_delay - hourly_actual_arrival_delay)**2
    #     .mean()
    # )
    # hourly_departure_delays_rmse = np.sqrt(
    #     (hourly_sample_departure_delay - hourly_actual_departure_delay)**2
    #     .mean()
    # )

    # print(hourly_arrival_delays_rmse, hourly_departure_delays_rmse)

    return (
        arrivals_rmse, departures_rmse, 
        arrivals_rmse_adj, departures_rmse_adj,
        combined_hourly_delays_df
    )




def get_hourly_delays(
    model, auto_guide, states, dt, 
    observations_df, n_samples, wandb=True
):
    
    with pyro.plate("samples", n_samples, dim=-1):
        posterior_samples = auto_guide(states, dt)

    predictive = pyro.infer.Predictive(
        model=model,
        posterior_samples=posterior_samples,
    )

    samples = predictive(
        states, dt, 
        obs_none=True,
    )

    # conditioning_dict = {
    #     'LGA_0_mean_service_time': posterior_samples['LGA_0_mean_service_time']
    # }

    # model_trace = pyro.poutine.trace(
    #     pyro.poutine.condition(model, data=conditioning_dict)
    # ).get_trace(
    #     states=states,
    #     delta_t=dt,
    # )

    # trace_pred = pyro.infer.TracePredictive(
    #     model, svi, num_samples=n_samples
    # ).run(states, dt)
    
    # samples = trace_pred()

    # print(samples)

    day_str_list = []
    flight_number_list = []
    origin_airport_list = []
    destination_airport_list = []
    sample_arrival_time_list = []
    sample_departure_time_list = []
    sample_cancelled_list = []

    for key, sample in samples.items():
        split_key = key.split('_')
        if split_key[-1] != 'time' and split_key[-1] != 'cancelled':
            continue

        arr_sample = None
        dep_sample = None
        # ccl_sample = None

        if split_key[-1] == 'cancelled':
            # ccl_sample = sample.mean().item() # each 1 or 0
            continue
        elif split_key[-2] == 'arrival':
            arr_sample = sample.mean().item()
        elif split_key[-2] == 'departure':
            dep_sample = sample.mean().item()
        else:
            continue

        # print(split_key, sample)

        day_str_list.append(split_key[0])
        flight_number_list.append(
            split_key[1]
            # f'{split_key[1]}_{split_key[2]}_{split_key[3]}'
        )
        origin_airport_list.append(split_key[2])
        destination_airport_list.append(split_key[3])
        sample_arrival_time_list.append(arr_sample)
        sample_departure_time_list.append(dep_sample)
        # sample_cancelled_list.append(ccl_sample)

    samples_df = pd.DataFrame(
        {   
            "date": pd.to_datetime(day_str_list),
            "flight_number": flight_number_list,
            "origin_airport": origin_airport_list,
            "destination_airport": destination_airport_list,
            "sample_arrival_time": sample_arrival_time_list,
            "sample_departure_time": sample_departure_time_list,
            # "sample_cancelled": sample_cancelled_list,
        }
    )

    # print(samples_df)
    # print(observations_df)

    merged_df = pd.merge(
        samples_df, 
        observations_df,
        on=[
            "date",
            "flight_number",
            "origin_airport",
            "destination_airport"
        ],
        how='inner',
    )

    dep_mask_successful = (
        (merged_df["origin_airport"] == "LGA")
        & (merged_df["actual_departure_time"] != 0)
        & (merged_df["sample_departure_time"] != 0)
    )
    arr_mask_successful = (
        (merged_df["destination_airport"] == "LGA")
        & (merged_df["actual_arrival_time"] != 0)
        & (merged_df["sample_arrival_time"] != 0)
    )

    hr_lim = 25

    arrival_delays_df = merged_df.loc[
        arr_mask_successful,
        ["date", "flight_number",
         "sample_arrival_time", 
         "actual_arrival_time",
         "scheduled_arrival_time"]
    ]

    arrival_delays_df["sample_arrival_delay"] = (
        arrival_delays_df["sample_arrival_time"] 
        - arrival_delays_df["scheduled_arrival_time"]
    )
    arrival_delays_df["actual_arrival_delay"] = (
        arrival_delays_df["actual_arrival_time"] 
        - arrival_delays_df["scheduled_arrival_time"]
    )

    actual_arrival_hour = (
        np.floor(arrival_delays_df.actual_arrival_time).astype(int)
    )
    sample_arrival_hour = (
        np.floor(arrival_delays_df.sample_arrival_time).astype(int)
    )

    hourly_sample_arrival_delay = (
        arrival_delays_df
        .groupby(sample_arrival_hour)
        ["sample_arrival_delay"]
        .mean()
        .loc[:hr_lim]
    )
    hourly_actual_arrival_delay = (
        arrival_delays_df
        .groupby(actual_arrival_hour)
        ["actual_arrival_delay"]
        .mean()
        .loc[:hr_lim]
    )

    departure_delays_df = merged_df.loc[
        dep_mask_successful,
        ["date", "flight_number",
         "sample_departure_time", 
         "actual_departure_time",
         "scheduled_departure_time"]
    ]

    departure_delays_df["sample_departure_delay"] = (
        departure_delays_df["sample_departure_time"] 
        - departure_delays_df["scheduled_departure_time"]
    )
    departure_delays_df["actual_departure_delay"] = (
        departure_delays_df["actual_departure_time"] 
        - departure_delays_df["scheduled_departure_time"]
    )

    actual_departure_hour = (
        np.floor(departure_delays_df.actual_departure_time).astype(int)
    )
    sample_departure_hour = (
        np.floor(departure_delays_df.sample_departure_time).astype(int)
    )

    hourly_sample_departure_delay = (
        departure_delays_df
        .groupby(sample_departure_hour)
        ["sample_departure_delay"]
        .mean()
        .loc[:hr_lim]
    )
    hourly_actual_departure_delay = (
        departure_delays_df
        .groupby(actual_departure_hour)
        ["actual_departure_delay"]
        .mean()
        .loc[:hr_lim]
    )

    combined_hourly_delays_df = pd.concat(
        [
            hourly_sample_arrival_delay,
            hourly_actual_arrival_delay,
            hourly_sample_departure_delay,
            hourly_actual_departure_delay,
        ],
        axis=1,
    ).sort_index()

    # print(combined_hourly_delays_df)

    # TODO: cancellation
    # arrival_cancel_df = merged_df.loc[
    #     arr_mask_all,
    #     ["date", "flight_number",
    #      "sample_arrival_time", 
    #      "actual_arrival_time",
    #      "sample_cancelled",
    #      "cancelled"]
    # ]
    # departure_cancel_df = merged_df.loc[
    #     dep_mask_all,
    #     ["date", "flight_number",
    #      "sample_departure_time", 
    #      "actual_departure_time",
    #      "sample_cancelled",
    #      "cancelled"]
    # ]

    # print(hourly_arrival_delays_rmse, hourly_departure_delays_rmse)

    return combined_hourly_delays_df





def plot_hourly_delays(df):

    fig = plt.figure(figsize=(16, 8))

    plt.plot(df.index, df.sample_arrival_delay, ":b", label="sample arrival")
    plt.plot(df.index, df.sample_departure_delay, ":r", label="sample departure")
    plt.plot(df.index, df.actual_arrival_delay, "-b", label="actual arrival")
    plt.plot(df.index, df.actual_departure_delay, "-r", label="actual departure")

    plt.title("hourly mean delay (excluding cancellations?)")
    plt.xlabel("hour of day")
    plt.ylabel("delay (hrs)")
    plt.legend()

    plt.xlim(4.0,26.0)
    plt.ylim(-.5, 5.0)

    return fig


def make_travel_times_dict_and_observation_df(data, network_airport_codes):
    # Estimate travel times between each pair
    all_data_df = pd.concat(data.values())
    all_data_df["actual_travel_time"] = (
        all_data_df["actual_arrival_time"] 
        - all_data_df["actual_departure_time"]
    )
    all_data_df["scheduled_travel_time"] = (
        all_data_df["scheduled_arrival_time"] 
        - all_data_df["scheduled_departure_time"]
    )
    all_data_df["travel_time"] = np.where(
        all_data_df["actual_travel_time"] == 0,
        all_data_df["scheduled_travel_time"], 
        all_data_df["actual_travel_time"]
    )
    # use scheduled as fallback if only actuals are zero
    all_data_df["travel_time"] = \
        all_data_df["actual_travel_time"].where(
            all_data_df["actual_travel_time"] > 0,
            all_data_df["scheduled_travel_time"].values
        )
    
    # tt_dict_col = "travel_time"
    # or i guess always use scheduled?
    tt_dict_col = "scheduled_travel_time"

    travel_times = (
        all_data_df
        .groupby(["origin_airport", "destination_airport"])[tt_dict_col]
        .mean()
        .reset_index()
        .rename(columns={tt_dict_col:"travel_time"})
    )
    ignore_routes_mask = (
        ~travel_times["destination_airport"]
        .isin(network_airport_codes)
    )
    travel_times = (
        travel_times
        .drop(travel_times[ignore_routes_mask].index)
        .reset_index(drop=True)
    )
    travel_times_dict = (
        travel_times
        .set_index(["origin_airport","destination_airport"])
        ["travel_time"]
        .to_dict()
    )

    # print(travel_times_dict)
    # exit()

    observations_df = all_data_df.loc[
        ~all_data_df["cancelled"],
        [
            "date", "flight_number", 
            "origin_airport", "destination_airport", 
            "actual_arrival_time", "actual_departure_time", 
            "scheduled_arrival_time", "scheduled_departure_time",
            "carrier_delay", "weather_delay", "nas_delay", 
            "security_delay", "late_aircraft_delay",
        ]
    ]

    return travel_times_dict, observations_df


def make_states(data, network_airport_codes):
    # Convert each day into a schedule
    states = []
    for day_str, schedule_df in data.items():
        (
            network_flights, network_airports,
            incoming_flights, source_supernode,
        ) = \
            split_and_parse_full_schedule(
                schedule_df, 
                network_airport_codes,
            )   
        
        # print([f for f in incoming_flights if f.flight_number == 'DL:2358'])
        # print([f for f in network_flights if f.flight_number == 'DL:2358'])

        network_state = NetworkState(
            airports={airport.code: airport for airport in network_airports},
            pending_flights=network_flights
        )
        state = AugmentedNetworkState(
            day_str=day_str,
            network_state=network_state,
            source_supernode=source_supernode,
            pending_incoming_flights=incoming_flights,
        )
        states.append(state)

    return states

# TODO: deal with all of the above

def train(
    network_airport_codes, 
    svi_steps, 
    n_samples, 
    svi_lr, 
    gamma,
    dt,
    n_elbo_particles,
    plot_every,
    rng_seed,
    day_strs,
    prior_type,
    prior_scale,
    posterior_guide,
):
    pyro.clear_param_store()  # avoid leaking parameters across runs
    pyro.enable_validation(True)
    pyro.set_rng_seed(int(rng_seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Avoid plotting error
    matplotlib.use("Agg")

    # Hyperparameters
    initial_aircraft = 50.0 # not used!
    mst_effective_hrs = 24 # not used!
    mst_split = 1 # not really used
    mst_prior_scale = prior_scale # .1 # 1.0 = scaled to be as much as rest of the model?

    # gather data
    days = pd.to_datetime(day_strs)
    data = ba_dataloader.load_remapped_data_bts(days)

    num_days = len(days)
    num_flights = sum([len(df) for df in data.values()])
    print(f"Number of days: {num_days}")
    print(f"Number of flights: {num_flights}")

    # make things with the data
    travel_times_dict, observations_df = \
        make_travel_times_dict_and_observation_df(
            data, network_airport_codes
        ) 
    states = make_states(data, network_airport_codes)


    # here, we make the nominal, failure, and uniform (not used) priors...
    fig = plt.figure()

    # TODO: shifted gamma doesn't work. i have no idea why
    mst_prior_nominal = _affine_beta_dist_from_mean_std(.0125, .001, .010, .020, device)
    # mst_prior_nominal = _shifted_gamma_dist_from_mean_std(.0125, .001, .01, device)
    s = mst_prior_nominal.sample((10000,))
    sns.histplot(s, color='b', alpha=.33, fill=True, label="nominal", kde=True, binwidth=.0001, edgecolor='k', linewidth=0)

    mst_prior_failure = _affine_beta_dist_from_mean_std(.0200, .002, .010, .030, device)
    # mst_prior_failure = _shifted_gamma_dist_from_mean_std(.0200, .002, .01, device)
    s = mst_prior_failure.sample((10000,))
    sns.histplot(s, color='r', alpha=.33, fill=True, label="failure", kde=True, binwidth=.0001, edgecolor='k', linewidth=0)

    plt.savefig('ab_test.png')

    mst_prior_default = _affine_beta_dist_from_alpha_beta(1.0, 1.0, .005, .030, device)
    s = mst_prior_default.sample((10000,))
    sns.histplot(s, color='purple', alpha=.33, fill=True, label="empty", kde=True, binwidth=.0001, edgecolor='k', linewidth=0)
    
    plt.legend()
    plt.title("prior distributions example")
    plt.savefig('abc_test.png')
    plt.close(fig)
    # return


    # by default, scale ELBO down by num flights
    model_scale = 1.0 / (num_flights)
    mst_prior_weight = mst_prior_scale / model_scale # equals mst_prior_scale * num_flights ?

    do_mle = False
    if prior_type == "nominal":
        mst_prior = mst_prior_nominal
    elif prior_type == "failure":
        mst_prior = mst_prior_failure
    elif prior_type == "empty":
        mst_prior = mst_prior_default
        # mst_prior_weight = 1e-12 # basically zero lol
        do_mle = True
    else:
        raise ValueError 


    # set up common model arguments
    model = functools.partial(
        augmented_air_traffic_network_model_simplified,

        travel_times_dict=travel_times_dict,
        initial_aircraft=initial_aircraft,

        # include_cancellations=True,
        include_cancellations=False,
        mean_service_time_effective_hrs=mst_effective_hrs,

        source_use_actual_departure_time=True,
        # source_use_actual_late_aircraft_delay=True,
        # source_use_actual_carrier_delay=True,
        # source_use_actual_security_delay=True,

        # source_use_actual_cancelled=True,
        source_use_actual_cancelled=False,

        mst_prior=mst_prior,
        mst_prior_weight=mst_prior_weight,
        do_mle=do_mle,
    )

    # re-scale ELBO
    model = pyro.poutine.scale(model, scale=model_scale)


    # Create an autoguide for the model
    init_loc_fn = pyro.infer.autoguide.initialization.init_to_value(
        values={
            f'LGA_{t_idx}_mean_service_time': torch.tensor(.015).to(device)
            for t_idx in range(3)
        },
        fallback=pyro.infer.autoguide.initialization.init_to_median
    )

    if posterior_guide == "gaussian":
        guide = pyro.infer.autoguide.AutoMultivariateNormal(model, init_loc_fn=init_loc_fn)
    elif posterior_guide == "iafnormal":
        guide = pyro.infer.autoguide.AutoIAFNormal(
            model, num_transforms=3, hidden_dim=[3,3]
        )
    elif posterior_guide == "delta":
        guide = pyro.infer.autoguide.AutoDelta(model, init_loc_fn=init_loc_fn)
    elif posterior_guide == "laplace":
        guide = pyro.infer.autoguide.AutoLaplaceApproximation(model, init_loc_fn=init_loc_fn)
    else:
        raise ValueError
    # print(guide(states, dt))


    # Set up SVI
    gamma = gamma  # final learning rate will be gamma * initial_lr
    lrd = gamma ** (1 / svi_steps)
    optim = pyro.optim.ClippedAdam({"lr": svi_lr, "lrd": lrd})
    elbo = pyro.infer.Trace_ELBO(num_particles=n_elbo_particles)
    svi = pyro.infer.SVI(model, guide, optim, elbo)

    run_name = f"[{','.join(network_airport_codes)}]_"
    run_name += f"[{prior_type},{prior_scale:.2f},{posterior_guide}]_"
    run_name += f"[{','.join(days.strftime('%Y-%m-%d').to_list())}]"
    # print(run_name)
    group_name = f"{prior_type}-{prior_scale:.2f}-{posterior_guide}"
    # print(group_name)
    # TODO: fix this for non-day-by-day???
    sub_dir = f"checkpoints_{'_'.join(network_airport_codes)}/{'_'.join(days.strftime('%Y-%m-%d').to_list())}/{prior_type}_{prior_scale:.2f}_{posterior_guide}/"

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
        "do_mle": do_mle,
        "n_elbo_particles": n_elbo_particles,

        "prior_type": prior_type,
        "prior_scale": prior_scale,
        "posterior_guide": posterior_guide,
    }
    
    wandb.init(
        project="bayes-air-atrds-attempt-1",
        name=run_name,
        group=group_name,
        config=wandb_init_config_dict,
    )

    losses = []
    losses_from_prior = []

    pbar = tqdm(range(svi_steps))
    for i in pbar:
        loss = svi.step(states, dt)
        losses.append(loss)

        with pyro.plate("samples", n_samples, dim=-1):
            posterior_samples = guide(states, dt)
        z_samples = [posterior_samples[f'LGA_{t_idx}_mean_service_time'] for t_idx in range(mst_split)]
        mst_mles = [z_samples[t_idx].mean().item() for t_idx in range(mst_split)]

        elbo_from_prior = (
            sum([
                -mst_prior.log_prob(z_sample)
                .mean().item() 
                for z_sample in z_samples
            ]) #/ len(z_samples)
            * mst_prior_weight
        )
        losses_from_prior.append(elbo_from_prior)

        desc = f"ELBO loss: {loss:.2f}, mst mles: "
        desc += ", ".join([f'{mst_mle:.5f}' for mst_mle in mst_mles])
        pbar.set_description(desc)
                
        if i % plot_every == 0 or i == svi_steps - 1:

            hourly_delays = get_hourly_delays(
                model, guide, states, dt, observations_df, 1
            )

            fig = plot_hourly_delays(hourly_delays)
            wandb.log({"Hourly delays": wandb.Image(fig)}, commit=False)
            plt.close(fig)

            plotting_dict = {
                "mean service times": plot_service_times,
            }
        
            for name, plot_func in plotting_dict.items():
                # for now require this common signature
                fig = plot_func(guide, states, dt, n_samples)
                wandb.log({name: wandb.Image(fig)}, commit=False)
                plt.close(fig)

            # Save the params and autoguide
            dir_path = os.path.dirname(__file__)
            # save_path = os.path.join(dir_path, "checkpoints_final", run_name, f"{i}")
            save_path = os.path.join(dir_path, sub_dir, f"{i}")
            os.makedirs(save_path, exist_ok=True)
            pyro.get_param_store().save(os.path.join(save_path, "params.pth"))
            torch.save(guide.state_dict(), os.path.join(save_path, "guide.pth"))

        # wandb.log({"ELBO": loss})
        log_dict = {
            "ELBO/loss (units = nats per dim)": loss, 
            "ELBO/loss (from prior component)": elbo_from_prior,
            "ELBO/loss (from other component)": loss - elbo_from_prior,
        }
        for i in range(len(mst_mles)):
            log_dict[f"mean service time mle/{i} (hours)"] = mst_mles[i]
        wandb.log(log_dict)

    wandb.save(f"checkpoints/{run_name}/checkpoint_{svi_steps - 1}.pt")

    output_dict = {
        'model': model,
        'guide': guide,
        'states': states,
        'dt': dt,
        # 'set_model': functools.partial(model, states, dt),
        # 'set_guide': functools.partial(guide, states, dt),
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

    return loss


# TODO: add functionality to pick days
@click.command()
@click.option("--network-airport-codes", default="LGA", help="airport codes")
# @click.option("--failure", is_flag=True, help="Use failure prior")
@click.option("--svi-steps", default=500, help="Number of SVI steps to run")
@click.option("--n-samples", default=5000, help="Number of posterior samples to draw")
@click.option("--svi-lr", default=5e-3, help="Learning rate for SVI")
@click.option("--plot-every", default=50, help="Plot every N steps")
@click.option("--rng-seed", default=1, type=int)
@click.option("--gamma", default=.4) # was .1
@click.option("--dt", default=.1)
@click.option("--n-elbo-particles", default=1)



@click.option("--prior-type", default="empty", help="nominal/failure/empty")
@click.option("--posterior-guide", default="gaussian", help="gaussian/iafnormal/delta/laplace") # delta and laplace break plots rn 
@click.option("--prior-scale", default=0.0, type=float)
# empty: 2 (each guide)
# nominal: 4 (each guide, two scale levels)
# failure: 4 (each guide, two scale levels)
# for scale levels, let's try .1 and .25 first maybe?

@click.option("--day-strs", default=None)
@click.option("--year", default=None)
@click.option("--month", default=None)
@click.option("--start-day", default=None)
@click.option("--end-day", default=None)

@click.option("--learn-together", is_flag=True)
@click.option("--all-combos", is_flag=True)



def train_cmd(
    network_airport_codes, svi_steps, n_samples, svi_lr, 
    plot_every, rng_seed, gamma, dt, n_elbo_particles,
    prior_type, prior_scale, posterior_guide, 
    day_strs, year, month, start_day, end_day,
    learn_together, all_combos,
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

    default_zero_scale = 1e-12
    default_low_scale = .02
    default_high_scale = .1

    if prior_scale <= 0.0:
        prior_scale = default_zero_scale # can't actually be 0
        if prior_type != "empty":
            print("warning: zero prior_scale being used with non-empty prior type")

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
    
    if learn_together:
        day_strs_list = [day_strs]
    else:
        day_strs_list = [
            [day_str] for day_str in day_strs
        ]

    if not all_combos:
        ppp_params = [(prior_type, prior_scale, posterior_guide,)]
    else:
        ppp_params = [
            (ptype, pscale, pguide)
            for ptype in ("failure", "nominal")
            for pscale in (default_low_scale, default_high_scale)
            for pguide in ("gaussian", "iafnormal")
        ] + [
            ("empty", default_zero_scale, pguide)
            for pguide in ("gaussian", "iafnormal")
        ]

    pbar = tqdm(range(len(day_strs_list)))
    pbar.set_description('day')
    for i in pbar:
        pppbar = tqdm(range(len(ppp_params)), leave=False)
        pppbar.set_description('param combo')
        for j in pppbar:
            train(
                network_airport_codes,
                svi_steps,
                n_samples,
                svi_lr,
                gamma,
                dt,
                n_elbo_particles,
                plot_every,
                rng_seed,
                day_strs_list[i],
                # prior_type,
                # prior_scale,
                # posterior_guide,
                *(ppp_params[j])
            )
            # print(day_strs_list[i], *(ppp_params[j]))


if __name__ == "__main__":
    train_cmd()