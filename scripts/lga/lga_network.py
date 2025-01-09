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
import tqdm
from math import ceil, floor
import functools
import dill

import bayes_air.utils.dataloader as ba_dataloader
import wandb
from bayes_air.model import augmented_air_traffic_network_model
from bayes_air.network import NetworkState, AugmentedNetworkState
from bayes_air.schedule import split_and_parse_full_schedule


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
        width=12,
        ignore_time_index=False,
    ):

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
            )

            shared_axs.append(ax)
            
            if ignore_time_index:
                break

        handle_shared_ax_lims(shared_axs)

    return fig


plot_service_times = functools.partial(
    plot_time_indexed_network_var,
    "mean_service_time",
)

plot_turnaround_times = functools.partial(
    plot_time_indexed_network_var,
    "mean_turnaround_time",
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

    arr_n_ignore = int(len(arrivals_df) * 0.025)
    arrivals_mse_adj = (
        arrivals_df
        .drop(arrivals_df.nlargest(arr_n_ignore, "squared_dist").index)
        ["squared_dist"]
        .mean()
    )
    arrivals_rmse_adj = np.sqrt(arrivals_mse_adj)

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


def plot_hourly_delays(df):

    fig = plt.figure(figsize=(8, 8))

    plt.plot(df.index, df.sample_arrival_delay, ":b", label="sample arrival")
    plt.plot(df.index, df.sample_departure_delay, ":r", label="sample departure")
    plt.plot(df.index, df.actual_arrival_delay, "-b", label="actual arrival")
    plt.plot(df.index, df.actual_departure_delay, "-r", label="actual departure")

    plt.title("hourly mean delay (excluding cancellations?)")
    plt.xlabel("hour of day")
    plt.ylabel("delay (hrs)")
    plt.legend()

    plt.xlim(4.0,26.0)
    plt.ylim(-.5, 4.0)

    return fig


# TODO: deal with all of the above

def train(
    network_airport_codes, 
    # nominal_days, 
    # failure_days,
    svi_steps, 
    n_samples, 
    svi_lr, 
    plot_every, 
    nominal=True,
    day_nums=[1],
):
    pyro.clear_param_store()  # avoid leaking parameters across runs
    pyro.enable_validation(True)
    pyro.set_rng_seed(1)

    # Avoid plotting error
    matplotlib.use("Agg")

    # # Set the number of starting aircraft at each airport
    # starting_aircraft = 50

    # Hyperparameters
    dt = 0.1 # .2

    days_str = [
        f"2019-07-{day:02d}"
        for day in day_nums
    ]
    days = pd.to_datetime(days_str)
    num_days = len(days)

    data = ba_dataloader.load_remapped_data_bts(days)

    num_flights = sum([len(df) for df in data.values()])
    print(f"Number of flights: {num_flights}")

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

    # Re-scale the ELBO by the number of days
    model = functools.partial(
        augmented_air_traffic_network_model,

        travel_times_dict=travel_times_dict,
        include_cancellations=True,

        # source_use_actual_departure_time=True,
        source_use_actual_late_aircraft_delay=True,
        source_use_actual_carrier_delay=True,
        # source_use_actual_security_delay=True,

        mean_service_time_effective_hrs=24,
        mean_turnaround_time_effective_hrs=24,

        use_nominal_prior=nominal,
        use_failure_prior=not nominal,

        # max_holding_time=1.0,
        # soft_max_holding_time=None,
        # max_waiting_time=5.0,
    )
    model = pyro.poutine.scale(model, scale=1.0 / num_days)

    # Create an autoguide for the model
    # auto_guide = pyro.infer.autoguide.AutoDelta(model)
    auto_guide = pyro.infer.autoguide.AutoMultivariateNormal(model)
    # auto_guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(model)


    # Set up SVI
    gamma = 0.1  # final learning rate will be gamma * initial_lr
    lrd = gamma ** (1 / svi_steps)
    optim = pyro.optim.ClippedAdam({"lr": svi_lr, "lrd": lrd})
    elbo = pyro.infer.Trace_ELBO(num_particles=1)
    svi = pyro.infer.SVI(model, auto_guide, optim, elbo)

    run_name = f"[{','.join(network_airport_codes)}]_"
    run_name += f"{'nominal' if nominal else 'failure'}_"
    run_name += f"[{','.join(days.strftime('%Y-%m-%d').to_list())}]"
    
    wandb.init(
        project="bayes-air_july-2019_test-1",
        name=run_name,
        group="nominal" if nominal else "failure",
        config={
            "type": "nominal" if nominal else "failure",
            # "starting_aircraft": starting_aircraft,
            "dt": dt,
            "days": days,
            "svi_lr": svi_lr,
            "svi_steps": svi_steps,
            "n_samples": n_samples,
        },
    )

    losses = []
    arr_rmses = []
    dep_rmses = []
    arr_rmses_adj = []
    dep_rmses_adj = []
    rmse_idxs = []
    rmses_record_every = 10

    pbar = tqdm.tqdm(range(svi_steps))
    for i in pbar:
        loss = svi.step(states, dt)
        losses.append(loss)

        pbar.set_description(f"ELBO loss: {loss:.2f}")

        if i % rmses_record_every == 0 or i == svi_steps - 1:
            arr_rmse, dep_rmse, arr_rmse_adj, dep_rmse_adj, hourly_delays = \
            get_arrival_departures_rmses(
                model, auto_guide, states, dt, observations_df, 1
            )
            arr_rmses.append(arr_rmse)
            dep_rmses.append(dep_rmse)
            arr_rmses_adj.append(arr_rmse_adj)
            dep_rmses_adj.append(dep_rmse_adj)
            rmse_idxs.append(i)
            if i % plot_every == 0 or i == svi_steps - 1:
                fig = plot_hourly_delays(hourly_delays)
                wandb.log({"Hourly delays": wandb.Image(fig)}, commit=False)
                plt.close(fig)

        if i % plot_every == 0 or i == svi_steps - 1:

            fig = plot_elbo_losses(losses)
            wandb.log({"ELBO loss": wandb.Image(fig)}, commit=False)
            plt.close(fig)

            # print(arr_rmses, dep_rmses, rmse_idxs)
            fig = plot_rmses(arr_rmses, dep_rmses, arr_rmses_adj, dep_rmses_adj, rmse_idxs)
            wandb.log({"Flight time RMSEs": wandb.Image(fig)}, commit=False)
            plt.close(fig)

            fig = plot_travel_times(auto_guide, states, dt, n_samples, travel_times)
            wandb.log({"Travel times": wandb.Image(fig)}, commit=False)
            plt.close(fig)

            plotting_dict = {
                "mean service times": plot_service_times,
                "mean turnaround_times": plot_turnaround_times,
                "starting aircraft": plot_starting_aircraft,
                "baseline cancel probability": plot_base_cancel_prob,
                # "soft max holding time": plot_soft_max_holding_time,
            }
        
            for name, plot_func in plotting_dict.items():
                # for now require this common signature
                fig = plot_func(auto_guide, states, dt, n_samples)
                wandb.log({name: wandb.Image(fig)}, commit=False)
                plt.close(fig)

            # Save the params and autoguide
            dir_path = os.path.dirname(__file__)
            save_path = os.path.join(dir_path, "checkpoints", run_name, f"{i}")
            os.makedirs(save_path, exist_ok=True)
            pyro.get_param_store().save(os.path.join(save_path, "params.pth"))
            torch.save(auto_guide.state_dict(), os.path.join(save_path, "guide.pth"))

        wandb.log({"ELBO": loss})

    wandb.save(f"checkpoints/{run_name}/checkpoint_{svi_steps - 1}.pt")

    output_dict = {
        'model': model,
        'guide': auto_guide,
        'states': states,
        'dt': dt,
    }
    # TODO: this is like not the best way of handling it but whatever
    # also maybe redundant but just in case i guess

    dir_path = os.path.dirname(__file__)
    save_path = os.path.join(dir_path, "checkpoints", run_name, "final")
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "output_dict.pkl"), 'wb+') as handle:
        dill.dump(output_dict, handle)

    return loss


# TODO: add functionality to pick days
@click.command()
@click.option("--network-airport-codes", default="LGA", help="airport codes")
# @click.option("--failure", is_flag=True, help="Use failure prior")
@click.option("--svi-steps", default=1000, help="Number of SVI steps to run")
@click.option("--n-samples", default=800, help="Number of posterior samples to draw")
@click.option("--svi-lr", default=1e-3, help="Learning rate for SVI")
@click.option("--plot-every", default=100, help="Plot every N steps")
@click.option("--day", default=1, help="day")
def train_cmd(
    network_airport_codes, svi_steps, n_samples, svi_lr, plot_every, day
):
    # TODO: make this better

    nominal_days = [
        1,2,3,4,5,7,9,
        10,12,13,14,15,16,
        20,24,25,26,27,28,29
    ]

    nominal = day in nominal_days

    network_airport_codes = network_airport_codes.split(',')
    train(
        network_airport_codes,
        svi_steps,
        n_samples,
        svi_lr,
        plot_every,
        nominal,
        day_nums=[day], # TODO: this is not great way of setting the day
    )


if __name__ == "__main__":
    train_cmd()