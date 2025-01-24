import dill
from pathlib import Path
import pandas as pd
import pyro
import torch

from tqdm import tqdm

dir_path = Path(__file__).parent

def deal_with_model_logprobs():

    model_logprobs_dir = dir_path / "model_logprobs"

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

    df.to_csv(dir_path / 'extras/2018-2019_output.csv', index=False)
    df.to_parquet(dir_path / 'extras/2018-2019_output.parquet')

    with open(dir_path / 'extras/2018-2019_output_dict.pkl', 'wb+') as handle:
        dill.dump(combined_output_dict, handle)




def deal_with_checkpoints():

    checkpoints_dir = dir_path / "bayes-air-atrds-attempt-7/checkpoints/LGA/"
    days = pd.date_range(start='2019-01-01', end='2019-12-31', freq='D').strftime('%Y-%m-%d').to_list()

    s_guide_dist_dict = {}

    pbar = tqdm(days)
    for name in pbar:

        with open(checkpoints_dir / f'{name}/empty_0.00_gaussian/final/output_dict.pkl', 'rb') as f:
            s_guide_output_dict = dill.load(f)
        
        s_guide = s_guide_output_dict["guide"]
        with pyro.plate("samples", 100000, dim=-1):
            posterior_samples = s_guide()
        posterior_samples = posterior_samples["LGA_0_mean_service_time"]

        mu = posterior_samples.mean().detach()
        sigma = torch.std(posterior_samples).detach()
        s_guide_dist = torch.distributions.Normal(mu, sigma)
        pbar.set_description(f'{name} {mu.item():.4f} {sigma.item():.4f}')

        s_guide_dist_dict[name] = s_guide_dist

    with open(dir_path / 'extras/2019_s_guide_dist_dict.pkl', 'wb+') as handle:
        dill.dump(s_guide_dist_dict, handle)


if __name__ == '__main__':
    # deal_with_checkpoints()
    deal_with_model_logprobs()