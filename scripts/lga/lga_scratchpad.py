import dill
from pathlib import Path
import pandas as pd


dir_path = Path(__file__).parent

def deal_with_model_logprobs():

    model_logprobs_dir = dir_path / "model_logprobs"

    combined_output_dict = {}

    dfs = []

    for year in (2018, 2019):
        for month in range(1, 13):

            fname_base = f'{year:04d}_{month:02d}_output'
            with open(model_logprobs_dir / f'{fname_base}_dict.pkl', 'rb') as f:
                model_logprobs_output_dict = dill.load(f)
            combined_output_dict |= model_logprobs_output_dict
            
            dfs.append(
                pd.read_parquet(model_logprobs_dir / f'{fname_base}.parquet')
            )

    df = pd.concat(dfs, axis=1)

    df.to_csv(model_logprobs_dir / '2018-2019_output.csv', index=False)
    df.to_parquet(model_logprobs_dir / '2018-2019_output.parquet')

def deal_with_checkpoints():

    days = pd.date_range(start='2019-01-01', end='2019-12-31', freq='D').strftime('%Y-%m-%d').to_list()

    with open(checkpoints_dir / f'{name}/empty_0.00_gaussian/final/output_dict.pkl', 'rb') as f:
        s_guide_output_dict = dill.load(f)
    
    s_guide = s_guide_output_dict["guide"]
    with pyro.plate("samples", 10000, dim=-1):
        posterior_samples = s_guide()
    posterior_samples = posterior_samples["LGA_0_mean_service_time"]

    mu = posterior_samples.mean().detach()
    sigma = torch.std(posterior_samples).detach()
    s_guide_dist = dist.Normal(mu, sigma)

    print(s_guide_dist.sample((10,)).squeeze())
    print(model_logprobs_output_dict[name])
