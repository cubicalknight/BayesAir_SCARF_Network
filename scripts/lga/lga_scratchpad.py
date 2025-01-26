import dill
from pathlib import Path
import pandas as pd
import pyro
import torch

from tqdm import tqdm

import bayes_air.utils.dataloader as ba_dataloader

from scripts.lga.lga_network import (
    make_travel_times_dict_and_observation_df,
    make_states,
)


dir_path = Path(__file__).parent

def deal_with_model_logprobs(finer=False):

    model_logprobs_dir = (
        dir_path / "model_logprobs"
        if not finer else
        dir_path / "model_logprobs_finer"
    )

    combined_output_dict = {}

    dfs = []

    for year in tqdm((2018, 2019)):
        for month in tqdm(range(1, 13)):

            fname_base = f'{year:04d}_{month:02d}_output'
            with open(model_logprobs_dir / f'{fname_base}_dict.pkl', 'rb') as f:
                model_logprobs_output_dict = dill.load(f)
            combined_output_dict |= model_logprobs_output_dict
            
            dfs.append(
                pd.read_parquet(model_logprobs_dir / f'{fname_base}.parquet')
            )

    df = pd.concat(dfs, axis=1)

    fname_base = '2018-2019_output' if not finer else '2018-2019_finer_output'

    df.to_csv(dir_path / f'extras/{fname_base}.csv', index=False)
    df.to_parquet(dir_path / f'extras/{fname_base}.parquet')

    with open(dir_path / f'extras/{fname_base}_dict.pkl', 'wb+') as handle:
        dill.dump(combined_output_dict, handle)




def deal_with_checkpoints():

    checkpoints_dir = dir_path / "bayes-air-atrds-attempt-7/checkpoints/LGA/"
    days = pd.date_range(start='2018-01-01', end='2019-12-31', freq='D').strftime('%Y-%m-%d').to_list()

    s_guide_dist_dict = {}
    s_guide_dist_params_dict = {}

    pbar = tqdm(days)
    for name in pbar:

        with open(checkpoints_dir / f'{name}/empty_0.00_gaussian/final/output_dict.pkl', 'rb') as f:
            s_guide_output_dict = dill.load(f)
        
        s_guide = s_guide_output_dict["guide"]
        with pyro.plate("samples", 100000, dim=-1):
            posterior_samples = s_guide()
        posterior_samples = posterior_samples["LGA_0_mean_service_time"]

        mu = posterior_samples.mean().detach().item()
        sigma = torch.std(posterior_samples).detach().item()
        s_guide_dist = torch.distributions.Normal(mu, sigma)
        pbar.set_description(f'{name} {mu:.4f} {sigma:.4f}')

        s_guide_dist_dict[name] = s_guide_dist
        s_guide_dist_params_dict[name] = (mu, sigma)

    with open(dir_path / 'extras/2018-2019_s_guide_dist_dict.pkl', 'wb+') as handle:
        dill.dump(s_guide_dist_dict, handle)

    with open(dir_path / 'extras/2018-2019_s_guide_dist_params_dict.pkl', 'wb+') as handle:
        dill.dump(s_guide_dist_params_dict, handle)

    df = pd.DataFrame(
        [(k,) + v for k,v in s_guide_dist_params_dict.items()], 
        columns=['date', 'mu', 'sigma']
    )
    df.to_csv(dir_path / 'extras/2018-2019_s_guide_dist_params.csv', index=False)
    df.to_parquet(dir_path / 'extras/2019_s_guide_dist_params.parquet')



def deal_with_dataloader():
    day_strs = pd.date_range(start='2019-1-1', end='2019-12-31', freq='D').strftime('%Y-%m-%d').to_list()
    day_strs_list = [[day_str] for day_str in day_strs]
    network_airport_codes = ['LGA']

    output_dict = {}

    pbar = tqdm(day_strs_list)
    for day_strs in pbar:

        # gather data
        days = pd.to_datetime(day_strs)
        data = ba_dataloader.load_remapped_data_bts(days)
        name = ", ".join(day_strs)

        num_days = len(days)
        num_flights = sum([len(df) for df in data.values()])
        pbar.set_description(f"{name} -> days: {num_days}, flights: {num_flights}")

        # make things with the data
        travel_times_dict, observations_df = \
            make_travel_times_dict_and_observation_df(
                data, network_airport_codes
            ) 
        states = make_states(data, network_airport_codes)

        with open(dir_path / f'extras/cached_travel_times_dict/{name}.pkl', 'wb+') as handle:
            dill.dump(travel_times_dict, handle)
        
        with open(dir_path / f'extras/cached_states/{name}.pkl', 'wb+') as handle:
            dill.dump(states, handle)




if __name__ == '__main__':
    deal_with_checkpoints()
    # deal_with_model_logprobs()
    # deal_with_dataloader()