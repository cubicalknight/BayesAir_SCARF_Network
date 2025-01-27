from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import seaborn as sns
import torch
import tqdm

import bayes_air.utils.dataloader as ba_dataloader
# from bayes_air.model import augmented_air_traffic_network_model
# from bayes_air.network import AugmentedNetworkState
# from bayes_air.schedule import parse_schedule
matplotlib.rcParams["figure.dpi"] = 300
# matplotlib.use('qtagg') 

class StopExecution(Exception):
    def _render_traceback_(self):
        return []
# raise StopExecution



def plot_days(year, month):

    start_day = f'{year}-{month}-1'
    days_in_month = pd.Period(start_day).days_in_month
    end_day = f'{year}-{month}-{days_in_month}'
    days_str = pd.date_range(start=start_day, end=end_day, freq='D').strftime('%Y-%m-%d').to_list()
    # days_str = [
    #     f"2019-07-{day:02d}"
    #     for day in range(1, 32)
    # ]
    days = pd.to_datetime(days_str)

    data = ba_dataloader.load_remapped_data_bts(days)

    # num_flights = sum([len(df) for df in data.values()])
    # print(f"Number of flights: {num_flights}")

    df = pd.concat(data.values()).reset_index(drop=True)

    print(df)

    max_plots_per_row = 2
    max_rows = int(np.ceil(len(data) / max_plots_per_row))
    subplot_spec = []
    for i in range(max_rows):
        subplot_spec.append(
            [f"{i * max_plots_per_row +j}" for j in range(max_plots_per_row)]
        )

    fig = plt.figure(layout="constrained", figsize=(12, 6 * max_rows))
    hosts = fig.subplot_mosaic(subplot_spec)

    marker_size = 5
    marker_size_cancel = 15
    first_legend_only = False

    for i, day_str in tqdm(enumerate(days_str)):
        host = hosts[f"{i}"]

        day_df = data[day_str]
        
        mask = day_df.cancelled | day_df.diverted
        day_df_successful = day_df.loc[~mask]
        day_df_cancelled = day_df.loc[day_df.cancelled]
        day_df_diverted = day_df.loc[day_df.diverted]

        success_percent = f'{len(day_df_successful) / len(day_df) :.2%}'
        cancel_percent = f'{len(day_df_cancelled) / len(day_df) :.2%}'
        divert_percent = f'{len(day_df_diverted) / len(day_df) :.2%}'

        host.scatter(
            day_df_successful.scheduled_departure_time, 
            day_df_successful.scheduled_arrival_time, 
            # day_df.scheduled_departure_time, 
            # day_df.scheduled_arrival_time, 
            s=marker_size, 
            color='deepskyblue',
            marker='.',
            label=f'scheduled = (S): {len(day_df_successful)} ({success_percent})',
        )
        host.scatter(
            day_df_successful.actual_departure_time, 
            day_df_successful.actual_arrival_time, 
            s=marker_size, 
            color='orange',
            marker='.',
            label=f'actual time = (A): {len(day_df_successful)} ({success_percent})'
        )

        host.scatter(
            day_df_cancelled.scheduled_departure_time, 
            day_df_cancelled.scheduled_arrival_time, 
            s=marker_size_cancel, 
            color='red',
            marker='x',
            label=f'cancelled (S): {len(day_df_cancelled)} ({cancel_percent})'
        )
        host.scatter(
            day_df_cancelled.actual_departure_time, 
            day_df_cancelled.actual_arrival_time, 
            s=marker_size_cancel, 
            color='darkred',
            marker='x',
            label=f'cancelled (A): {len(day_df_cancelled)} ({cancel_percent})'
        )

        host.scatter(
            day_df_diverted.scheduled_departure_time, 
            day_df_diverted.scheduled_arrival_time, 
            s=marker_size_cancel, 
            color='mediumseagreen',
            marker='v', 
            label=f'diverted (S): {len(day_df_diverted)} ({divert_percent})'
        )
        host.scatter(
            day_df_diverted.actual_departure_time, 
            day_df_diverted.actual_arrival_time, 
            s=marker_size_cancel, 
            color='seagreen',
            marker='v',
            label=f'diverted (A): {len(day_df_diverted)} ({divert_percent})'
        )

        # ax.plot([], [], color='white', label="test")

        host.set_xlim(4.9, 40.1)
        host.set_ylim(4.9, 40.1)
        host.set_xlabel('departure time (hours past midnight)')
        host.set_ylabel('arrival time (hours past midnight)')
        host.set_title(f'{day_str} (total flights: {len(day_df)})')

        ax_color = 'rebeccapurple'

        tmpx = host.twiny()
        tmpx.set_xlabel(
            'actual arrival/departure hour (for average delays)',
            color=ax_color
        )
        tmpx.tick_params(axis='x', colors=ax_color)
        tmpx.set_xlim(4.9, 40.1)

        tmpy = host.twinx()
        tmpy.set_ylabel(
            'hourly mean arrival/departure delay (hours)',
            color=ax_color
        )
        tmpy.tick_params(axis='y', colors=ax_color)
        tmpy.set_ylim(-.75, 4.25)

        tmpy.spines['top'].set_color(ax_color)
        tmpy.spines['right'].set_color(ax_color)

        actual_arrival_hour = (
            np.floor(day_df_successful.actual_arrival_time).astype(int)
        )
        hourly_arrival_delay = (
            day_df_successful
            .groupby(actual_arrival_hour)
            ["arrival_delay"]
            .mean()
        )
        tmpy.plot(
            hourly_arrival_delay.loc[:25],
            linestyle='dashed',
            color=ax_color,
            label="hourly arrival delay"
        )

        actual_departure_hour = (
            np.floor(day_df_successful.actual_departure_time).astype(int)
        )
        hourly_departure_delay = (
            day_df_successful
            .groupby(actual_departure_hour)
            ["departure_delay"]
            .mean()
        )
        tmpy.plot(
            hourly_departure_delay.loc[:25],
            linestyle='dotted',
            color=ax_color,
            label="hourly departure delay"
        )

        if not first_legend_only or i == 0:
            lgnd = host.legend(loc='lower right', fontsize=7)
            for i in range(len(lgnd.legend_handles)):
                lgnd.legend_handles[i]._sizes = [20]
            # red_patch = mpatches.Patch(color='red', label='actual')
            # blue_path = mpatches.Patch(color='blue', label='scheduled') 
            lines, labels = tmpy.get_legend_handles_labels()
            tmpy.legend(lines, labels, loc='upper left', fontsize=7)
        else:
            host.legend([], [], frameon=False)

    dir_path = Path(__file__).resolve().parent
    plt.savefig(dir_path / f'images/{year:04d}_{month:02d}_lga_flight_delays.png')
    # plt.show()
    # plt.close(fig)

    

if __name__ == '__main__':
    for year in tqdm(range(2018, 2019)):
        for month in tqdm(range(1, 12)):
            plot_days(year, month)