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
    max_rows = 30
    max_pairs_per_row = n_pairs // max_rows + 1
    subplot_spec = []
    for i in range(max_rows):
        subplot_spec.append(
            [f"{i * max_pairs_per_row +j}" for j in range(max_pairs_per_row)]
        )

    fig = plt.figure(layout="constrained", figsize=(12, 4 * max_rows))
    axs = fig.subplot_mosaic(subplot_spec)

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
                f"{pair[1]}->{pair[0]}": np.zeros(len(tmp))
            }
        )


        if not wandb:
            # Wandb doesn't support this KDE plot.
            sns.kdeplot(
                x=f"{pair[0]}->{pair[1]}",
                y=f"{pair[1]}->{pair[0]}",
                hue="type",
                ax=axs[f"{i}"],
                data=plotting_df,
                color="blue",
            )
            axs[f"{i}"].plot([], [], "-", color="blue", label="Posterior")
        else:
            # axs[f"{i}"].scatter(
            #     plotting_df[f"{pair[0]}->{pair[1]}"],
            #     plotting_df[f"{pair[1]}->{pair[0]}"],
            #     marker=".",
            #     s=1,
            #     c="blue",
            #     label="Posterior",
            #     zorder=1,
            # )
            axs[f"{i}"].hist(
                plotting_df[f"{pair[0]}->{pair[1]}"],
                # marker=".",
                # s=1,
                color="blue",
                label="Posterior",
                density=True,
                zorder=1,
            )

        tmp = empirical_travel_times.loc[
            (empirical_travel_times.origin_airport == pair[0])
            & (empirical_travel_times.destination_airport == pair[1]),
            "travel_time",
        ].mean()
        # axs[f"{i}"].scatter(
        #     tmp,
        #     .5,
        #     # empirical_travel_times.loc[
        #     #     (empirical_travel_times.origin_airport == pair[1])
        #     #     & (empirical_travel_times.destination_airport == pair[0]),
        #     #     "travel_time",
        #     # ],
        #     marker="*",
        #     s=100,
        #     c="red",
        #     label="Empirical mean",
        #     zorder=10,
        # )
        axs[f"{i}"].axvline(tmp, color="red",zorder=9, label="empirical mean")

        axs[f"{i}"].set_xlabel(f"{pair[0]} -> {pair[1]}")
        axs[f"{i}"].set_ylabel(f"{pair[1]} -> {pair[0]}")
        axs[f"{i}"].set_xlim(0, 6)
        axs[f"{i}"].set_ylim(0, 5)

        if i == 0:
            axs[f"{i}"].legend()
        else:
            axs[f"{i}"].legend([], [], frameon=False)

    return fig


def plot_starting_aircraft(auto_guide, states, dt, n_samples):
    """Plot posterior samples of service times."""
    # Sample mean service time estimates from the posterior
    with pyro.plate("samples", n_samples, dim=-1):
        posterior_samples = auto_guide(states, dt)

    # Make subplots for each airport
    airport_codes = states[0].network_state.airports.keys()
    n_pairs = len(airport_codes)
    max_rows = 3
    max_plots_per_row = n_pairs // max_rows + 1
    subplot_spec = []
    for i in range(max_rows):
        subplot_spec.append(
            [f"{i * max_plots_per_row +j}" for j in range(max_plots_per_row)]
        )

    fig = plt.figure(layout="constrained", figsize=(12, 4 * max_rows))
    axs = fig.subplot_mosaic(subplot_spec)

    for i, code in enumerate(airport_codes):
        # Put all of the data into a DataFrame to plot it
        plotting_df = pd.DataFrame(
            {
                code: torch.exp(
                    posterior_samples[f"{code}_log_initial_available_aircraft"]
                )
                .detach()
                .cpu()
                .numpy(),
                "type": "Posterior",
            }
        )

        sns.histplot(
            x=code,
            hue="type",
            ax=axs[f"{i}"],
            data=plotting_df,
            color="blue",
            # kde=True,
        )
        axs[f"{i}"].set_xlim(-0.05, 30.0)

    return fig


def plot_service_times(auto_guide, states, dt, n_samples):
    """Plot posterior samples of service times."""
    # Sample mean service time estimates from the posterior
    with pyro.plate("samples", n_samples, dim=-1):
        posterior_samples = auto_guide(states, dt)

    # Make subplots for each airport
    airport_codes = states[0].network_state.airports.keys()
    n_pairs = len(airport_codes)
    max_rows = 1
    max_plots_per_row = n_pairs // max_rows #+ 1
    subplot_spec = []
    for i in range(max_rows):
        subplot_spec.append(
            [f"{i * max_plots_per_row +j}" for j in range(max_plots_per_row)]
        )

    fig = plt.figure(layout="constrained", figsize=(12, 4 * max_rows))
    axs = fig.subplot_mosaic(subplot_spec)

    for i, code in enumerate(airport_codes):
        # Put all of the data into a DataFrame to plot it
        plotting_df = pd.DataFrame(
            {
                code: posterior_samples[f"{code}_mean_service_time"]
                .detach()
                .cpu()
                .numpy(),
                "type": "Posterior",
            }
        )

        sns.histplot(
            x=code,
            hue="type",
            ax=axs[f"{i}"],
            data=plotting_df,
            color="blue",
            # kde=True,
        )
        # axs[f"{i}"].set_xlim(-0.05, 1.05)

    return fig


def plot_turnaround_times(auto_guide, states, dt, n_samples):
    """Plot posterior samples of service times."""
    # Sample mean service time estimates from the posterior
    with pyro.plate("samples", n_samples, dim=-1):
        posterior_samples = auto_guide(states, dt)

    # Make subplots for each airport
    airport_codes = states[0].network_state.airports.keys()
    n_pairs = len(airport_codes)
    max_rows = 1
    max_plots_per_row = n_pairs // max_rows #+ 1
    subplot_spec = []
    for i in range(max_rows):
        subplot_spec.append(
            [f"{i * max_plots_per_row +j}" for j in range(max_plots_per_row)]
        )

    fig = plt.figure(layout="constrained", figsize=(12, 4 * max_rows))
    axs = fig.subplot_mosaic(subplot_spec)

    for i, code in enumerate(airport_codes):
        # Put all of the data into a DataFrame to plot it
        plotting_df = pd.DataFrame(
            {
                code: posterior_samples[f"{code}_mean_turnaround_time"]
                .detach()
                .cpu()
                .numpy(),
                "type": "Posterior",
            }
        )

        sns.histplot(
            x=code,
            hue="type",
            ax=axs[f"{i}"],
            data=plotting_df,
            color="blue",
            # kde=True,
        )
        # axs[f"{i}"].set_xlim(-0.05, 1.05)

    return fig


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
    model, auto_guide, states, travel_times_dict, dt, 
    observations_df, n_samples, wandb=True
):
    
    with pyro.plate("samples", n_samples, dim=-1):
        posterior_samples = auto_guide(states, dt)

    predictive = pyro.infer.Predictive(
        model=model,
        posterior_samples=posterior_samples
    )

    samples = predictive(
        states, travel_times_dict, dt, 
        obs_none=True
    )

    # trace_pred = pyro.infer.TracePredictive(
    #     model, svi, num_samples=n_samples
    # ).run(states, travel_times_dict, dt)
    
    # samples = trace_pred()

    # print(samples)

    day_str_list = []
    flight_number_list = []
    origin_airport_list = []
    destination_airport_list = []
    sample_arrival_time_list = []
    sample_departure_time_list = []

    for key, sample in samples.items():
        split_key = key.split('_')
        if split_key[-1] != 'time':
            continue

        if split_key[-2] == 'arrival':
            arr_sample = sample.mean().item()
            dep_sample = None
        elif split_key[-2] == 'departure':
            arr_sample = None
            dep_sample = sample.mean().item()
        else:
            continue

        # print(split_key, sample)

        day_str_list.append(split_key[0])
        flight_number_list.append(split_key[1])
        origin_airport_list.append(split_key[2])
        destination_airport_list.append(split_key[3])
        sample_arrival_time_list.append(arr_sample)
        sample_departure_time_list.append(dep_sample)

    samples_df = pd.DataFrame(
        {   
            "date": pd.to_datetime(day_str_list),
            "flight_number": flight_number_list,
            "origin_airport": origin_airport_list,
            "destination_airport": destination_airport_list,
            "sample_arrival_time": sample_arrival_time_list,
            "sample_departure_time": sample_departure_time_list,
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

    dep_mask = merged_df["origin_airport"] == "LGA"
    arr_mask = merged_df["destination_airport"] == "LGA"

    arrivals_df = merged_df.loc[
        arr_mask,
        ["date", "flight_number",
         "sample_arrival_time", 
         "actual_arrival_time"]
    ]

    departures_df = merged_df.loc[
        dep_mask,
        ["date", "flight_number",
         "sample_departure_time", 
         "actual_departure_time"]
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

    # print(arrivals_df)
    # print(departures_df)

    # print(arrivals_df.nlargest(10, columns=['squared_dist']))
    # print(departures_df.nlargest(10, columns=['squared_dist']))

    print(arrivals_mse, departures_mse)
    print(arrivals_rmse, departures_rmse)

    print(arrivals_mse_adj, departures_mse_adj)
    print(arrivals_rmse_adj, departures_rmse_adj)
    # exit()

    return arrivals_rmse, departures_rmse, arrivals_rmse_adj, departures_rmse_adj



# TODO: deal with all of the above

def train(
    network_airport_codes, 
    # nominal_days, 
    # disrupted_days,
    svi_steps, 
    n_samples, 
    svi_lr, 
    plot_every, 
    nominal=True
):
    pyro.clear_param_store()  # avoid leaking parameters across runs
    pyro.enable_validation(True)
    pyro.set_rng_seed(1)

    # Avoid plotting error
    matplotlib.use("Agg")

    # Set the number of starting aircraft at each airport
    starting_aircraft = 50

    # Hyperparameters
    dt = 0.2
    # dt = 0.05

    # Use nominal or disrupted data
    if nominal:
        days_str = [
            f"2019-07-{day:02d}"
            for day in [
                1,4,5,
                # 1,4,5,9,10, #2,3
                # 14,15,24,25,28
            ]
        ]
        days = pd.to_datetime(days_str)
    else:
        days_str = [
            f"2019-07-{day:02d}"
            for day in [
                23,
                #6,8,11,17,18,
                #19,21,23,30,31
            ]
        ]
        days = pd.to_datetime(days_str)
    # days = pd.to_datetime(["12-01-2001", "12-02-2001"])
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
            "actual_arrival_time", "actual_departure_time"
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
    model = augmented_air_traffic_network_model
    model = pyro.poutine.scale(model, scale=1.0 / num_days)

    # Create an autoguide for the model
    auto_guide = pyro.infer.autoguide.AutoMultivariateNormal(model)
    # auto_guide = pyro.infer.autoguide.AutoDelta(model)

    # Set up SVI
    gamma = 0.1  # final learning rate will be gamma * initial_lr
    lrd = gamma ** (1 / svi_steps)
    optim = pyro.optim.ClippedAdam({"lr": svi_lr, "lrd": lrd})
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model, auto_guide, optim, elbo)

    run_name = f"[{','.join(network_airport_codes)}]_"
    run_name += f"{'nominal' if nominal else 'disrupted'}_"
    run_name += f"[{','.join(days.strftime('%Y-%m-%d').to_list())}]"
    
    wandb.init(
        project="bayes-air",
        name=run_name,
        group="nominal" if nominal else "disrupted",
        config={
            "type": "nominal" if nominal else "disrupted",
            "starting_aircraft": starting_aircraft,
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
        loss = svi.step(states, travel_times_dict, dt)
        losses.append(loss)

        pbar.set_description(f"ELBO loss: {loss:.2f}")

        if i % rmses_record_every == 0 or i == svi_steps - 1:
            arr_rmse, dep_rmse, arr_rmse_adj, dep_rmse_adj = \
            get_arrival_departures_rmses(
                model, auto_guide, states, travel_times_dict, dt, observations_df, 5
            )
            arr_rmses.append(arr_rmse)
            dep_rmses.append(dep_rmse)
            arr_rmses_adj.append(arr_rmse_adj)
            dep_rmses_adj.append(dep_rmse_adj)
            rmse_idxs.append(i)

        if i % plot_every == 0 or i == svi_steps - 1:
            fig = plot_travel_times(auto_guide, states, dt, n_samples, travel_times)
            wandb.log({"Travel times": wandb.Image(fig)}, commit=False)
            plt.close(fig)

            fig = plot_service_times(auto_guide, states, dt, n_samples)
            wandb.log({"Mean service times": wandb.Image(fig)}, commit=False)
            plt.close(fig)

            fig = plot_elbo_losses(losses)
            wandb.log({"ELBO loss": wandb.Image(fig)}, commit=False)
            plt.close(fig)

            # print(arr_rmses, dep_rmses, rmse_idxs)
            fig = plot_rmses(arr_rmses, dep_rmses, arr_rmses_adj, dep_rmses_adj, rmse_idxs)
            wandb.log({"Flight time RMSEs": wandb.Image(fig)}, commit=False)
            plt.close(fig)

            # get_arrival_departures_rmses(
            #     model, auto_guide, states, travel_times_dict, dt, observations_df, 5
            # )

            # fig = plot_turnaround_times(auto_guide, states, dt, n_samples)
            # wandb.log({"Mean turnaround times": wandb.Image(fig)}, commit=False)
            # plt.close(fig)

            # fig = plot_starting_aircraft(auto_guide, states, dt, n_samples)
            # wandb.log({"Starting aircraft": wandb.Image(fig)}, commit=False)
            # plt.close(fig)

            # Save the params and autoguide
            dir_path = os.path.dirname(__file__)
            save_path = os.path.join(dir_path, "..", "checkpoints", run_name, f"{i}")
            os.makedirs(save_path, exist_ok=True)
            pyro.get_param_store().save(os.path.join(save_path, "params.pth"))
            torch.save(auto_guide.state_dict(), os.path.join(save_path, "guide.pth"))

        wandb.log({"ELBO": loss})

    return loss


# TODO: add functionality to pick days
@click.command()
@click.option("--network-airport-codes", default="LGA", help="Use disrupted data")
@click.option("--disrupted", is_flag=True, help="Use disrupted data")
@click.option("--svi-steps", default=1000, help="Number of SVI steps to run")
@click.option("--n-samples", default=800, help="Number of posterior samples to draw")
@click.option("--svi-lr", default=1e-3, help="Learning rate for SVI")
@click.option("--plot-every", default=100, help="Plot every N steps")
def train_cmd(
    network_airport_codes, disrupted, svi_steps, n_samples, svi_lr, plot_every
):
    # TODO: make this better
    network_airport_codes = network_airport_codes.split(',')
    train(
        network_airport_codes,
        svi_steps,
        n_samples,
        svi_lr,
        plot_every,
        nominal=not disrupted,
    )


if __name__ == "__main__":
    train_cmd()


    # days = pd.to_datetime(["12-01-2001", "12-02-2001"])

    # data = ba_dataloader.load_remapped_data_bts(days)

    # network_airport_codes = ["LGA"]

    # states = []

    # for schedule_df in data:
    #     (
    #         network_flights, network_airports,
    #         incoming_flights, source_supernode,
    #     ) = \
    #         split_and_parse_full_schedule(
    #             schedule_df, 
    #             network_airport_codes,
    #         )
        
    #     network_state = NetworkState(
    #         airports={airport.code: airport for airport in network_airports},
    #         pending_flights=network_flights
    #     )

    #     state = AugmentedNetworkState(
    #         network_state=network_state,
    #         source_supernode=source_supernode,
    #         pending_incoming_flights=incoming_flights
    #     )

    #     states.append(state)

    # print(states)