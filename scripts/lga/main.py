# %%
import lga_training_datagen 
# import weather_data_processing

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import dill


class ExtraDataProcessing:
    @staticmethod
    def process_series(
        s,
        *argv,
    ):
        # print(argv)
        for arg in argv:
            # print(arg, type(arg))
            if isinstance(arg, str):
                action = arg
                if action == 'no_inf':
                    s = s.loc[~s.isin([np.inf])]
                elif action == 'no_nan':
                    s = s.loc[~s.isin([np.nan])]
                elif action == 'inverse':
                    s = 1.0 / s
            elif isinstance(arg, (list, tuple, np.ndarray)):
                action = arg[0]
                if action == 'scale':
                    s = s * arg[1]
                elif action == 'clip':
                    s = s.clip(lower=arg[1], upper=arg[2])
                elif action == 'filter_index':
                    s = s.loc[(arg[1] <= s.index) & (s.index < arg[2])]
                elif action == 'apply_func':
                    s = s.apply(arg[1])
                elif action == 'apply_func_vec':
                    s = arg[1](s)
                else:
                    sg = s.groupby(pd.Grouper(freq=arg[1]))
                    if action == 'mean':
                        s = sg.mean()
                    elif action == 'min':
                        s = sg.min()
                    elif action == 'max':
                        s = sg.max()
                    elif action == 'size':
                        s = sg.size()
                    else:
                        raise ValueError(f'invalid step: {arg}')
        return s

    def process_extras(self, airport, start, end, freq='1D'):
        year = 2019
        month = 7
        start_date = start
        end_date = end

        data_dir = Path.cwd().resolve().parent / 'data'
        # weather_path = data_dir / 'noaa_lcdv2/lcd_lga_1987-2023_cleaned.parquet'
        weather_path = data_dir / f'noaa_lcdv2/cleaned/lcd_{airport.lower()}_2018-2019_cleaned.parquet'
        # TODO: handling in the bayesair remapped to like not have to do the "_decade" suffix
        # schedule_path = data_dir / 'bts_remapped/lga_reduced_2010-2019_clean_decade/parquet/lga_reduced_2010-2019_clean_decade.parquet'
        schedule_path = data_dir / f'bts_remapped/{airport}/2019/clean_daily/parquet/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_{year}_{month}_{airport}.parquet'

        wdf = pd.read_parquet(weather_path)
        sdf = pd.read_parquet(schedule_path)
        sdf = (
            sdf.set_index(
                pd.DatetimeIndex(sdf['date'])
            )
            .drop(['date'], axis=1)
        )

        wdf = wdf.loc[(wdf.index >= start_date) & (wdf.index < end_date)]
        sdf = sdf.loc[(sdf.index >= start_date) & (sdf.index < end_date)]

        processed_visibility = self.process_series(
            wdf.hourly_visibility,
            ('filter_index', start, end),
            ('clip', .001, 10), 
            # ('inverse'), 
            # ('mean', freq), 
            # ('inverse'),
            ('min', freq)
        )

        processed_ceiling = self.process_series(
            wdf.hsc_ceiling_height,
            ('filter_index', start, end),
            ('clip', .1, 100), 
            ('scale', 100), 
            ('inverse'), 
            ('mean', freq), 
            ('inverse'),
        )

        flight_counts = self.process_series(
            sdf.flight_number,
            ('size', freq)
        )

        tmp = sdf.scheduled_event_datetime.copy()
        tmp.index = pd.DatetimeIndex(sdf.scheduled_event_datetime)
        l = 1
        capacity_counts = self.process_series(
            tmp,
            ('filter_index', start, end),
            ('size', f'{l}H'),
            ('scale', 1/l),
            ('max', freq),
        )

        sdf_outgoing = sdf.loc[sdf.origin_airport == airport]
        sdf_incoming = sdf.loc[sdf.destination_airport == airport]

        d = self.process_series(
            sdf_outgoing.departure_delay_relu,
            ('filter_index', start, end),
            ('mean', freq),
        )

        a = self.process_series(
            sdf_incoming.arrival_delay_relu,
            ('filter_index', start, end),
            ('mean', freq),
        )

        processed_delay = self.process_series(
            sdf.event_delay_relu,
            ('filter_index', start, end),
            ('mean', freq),
        )
            
        for series, name in (
            (processed_visibility, f'{airport}_w_processed_visibility'),
            (processed_ceiling, f'{airport}_w_processed_ceiling'),
            (flight_counts, f'{airport}_x_flight_counts'),
            (capacity_counts, f'{airport}_x_capacity_counts'),
            (d, f'{airport}_y_departure_delays'),
            (a, f'{airport}_y_arrival_delays'),
            (processed_delay, f'{airport}_y_event_delays')
        ):
            series.to_csv(f'{name}.csv')
            series.to_frame(f'{name}').to_parquet(f'{name}.parquet')


def run_wx_data_processing():
    pass
    # execute weather data processing


def deal_with_model_logprobs(dir_path, airport, finer=False):
    model_logprobs_dir = (
        dir_path / f"{airport}_model_logprobs"
        if not finer else
        dir_path / f"{airport}_model_logprobs_finer_testing"
    )

    dir_path = dir_path / 'extras' / airport
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)

    combined_output_dict = {}

    dfs = []

    years = (2019,) #(2018, 2019)
    months =  range(7, 8)#range(1, 13)

    for year in tqdm(years):
        for month in tqdm(months):
            try:
                fname_base = f'{year:04d}_{month:02d}_output'
                with open(model_logprobs_dir / f'{fname_base}_dict.pkl', 'rb') as f:
                    model_logprobs_output_dict = dill.load(f)
            except FileNotFoundError:
                fname_base = 'output'
                with open(model_logprobs_dir / f'{fname_base}_dict.pkl', 'rb') as f:
                    model_logprobs_output_dict = dill.load(f)

            combined_output_dict |= model_logprobs_output_dict
            
            dfs.append(
                pd.read_parquet(model_logprobs_dir / f'{fname_base}.parquet')
            )

    df = pd.concat(dfs, axis=1)

    if len(years) == 1:
        fname_base = f'{years[0]}_output' if not finer else f'{years[0]}_finer_output'
    else:
        fname_base = f'{years[0]}-{years[-1]}_output' if not finer else f'{years[0]}-{years[-1]}_finer_output'

    df.to_csv(dir_path / f'{fname_base}.csv', index=False)
    print(f'Saved to {dir_path / f"{fname_base}.csv"}')
    df.to_parquet(dir_path / f'{fname_base}.parquet')
    print(f'Saved to {dir_path / f"{fname_base}.parquet"}')

    with open(dir_path / f'{fname_base}_dict.pkl', 'wb+') as handle:
        dill.dump(combined_output_dict, handle)
    print(f'Saved to {dir_path / f"{fname_base}_dict.pkl"}')


if __name__ == "__main__":
    # lga_training_datagen.train_cmd()
    dir_path = Path(__file__).parent
    airport = 'JFK'
    deal_with_model_logprobs(dir_path, airport, finer=True)

    # first need to run train code in lga_network, then go to lga_scratchpad, then lga_training
    # after this, run post processing
