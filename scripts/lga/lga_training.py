# %%
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
import random
import itertools

# import bayes_air.utils.dataloader as ba_dataloader
import wandb
# from bayes_air.model import augmented_air_traffic_network_model_simplified
# from bayes_air.network import NetworkState, AugmentedNetworkState
# from bayes_air.schedule import split_and_parse_full_schedule

from tqdm import tqdm

# import pyro.distributions as dist
import torch.distributions as dist

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

# from scripts.lga.lga_network import (
#     make_travel_times_dict_and_observation_df,
#     make_states,
#     plot_time_indexed_network_var,
#     plot_hourly_delays,
#     get_hourly_delays,
# )

def plot_failure_nominal_prior(failure_prior, nominal_prior):
    fig = plt.figure()
    s = nominal_prior.sample((10000,)).detach().cpu()
    sns.histplot(s, color='b', alpha=.33, fill=True, label="nominal", kde=True, binwidth=.0001, edgecolor='k', linewidth=0)
    s = failure_prior.sample((10000,)).detach().cpu()
    sns.histplot(s, color='r', alpha=.33, fill=True, label="failure", kde=True, binwidth=.0001, edgecolor='k', linewidth=0)
    plt.savefig('ab_test.png')
    plt.legend()
    plt.close(fig)

a = .1

def transform_sample(sample):
    # return .02+.015*torch.tanh(a*sample)
    return .004 * sample + .02

# def transform_sample_deriv(sample):
#     return .004

def y_given_c_log_prob(model_logprobs, prior_log_prob_fn, device, finer=False):
    # index i corresonds to .0001*(100+i) ? if finer. otherwise x10 that
    if finer:
        samples = torch.arange(0.0100, 0.0400, 0.0001).to(device)
    else:
        samples = torch.arange(0.010, 0.040, 0.001).to(device)
    prior_logprobs = prior_log_prob_fn(samples)
    logprob = torch.logsumexp(model_logprobs + prior_logprobs, dim=-1)
    return logprob # i think technically this is  abit off but whatever just approx for now.

def setup_guide(posterior_guide, mst_split, device):
    # now define guide
    n_context = 5
    hidden_dim = 16
    bins = 4
    if posterior_guide == "nsf":
       guide = zuko.flows.NSF(
            features=mst_split,
            context=n_context,
            hidden_features=(hidden_dim, hidden_dim),
            bins=bins,
        ).to(device) #this is the only one that really works??
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
            # means=torch.tensor([.015, .015, .015]).reshape(3,1).to(device),
            # log_vars=torch.log(torch.tensor([.0001, .0001, .0001])).reshape(3,1).to(device),
        )
    else:
        raise ValueError
    
    return guide

class PriorMixture(object):
    def __init__(self, failure_prior, nominal_prior):
        self.failure_prior = failure_prior
        self.nominal_prior = nominal_prior
        return
    def log_prob(self, sample, label):
        return (
            (1-label) * self.nominal_prior.log_prob(sample)
            + (label) * self.failure_prior.log_prob(sample)
        )
    
class WeatherThreshold(torch.nn.Module):
    def __init__(self, a, device, init_visibility_threshold, init_ceiling_threshold):
        super().__init__()
        self.device = device
        self.visibility_threshold = torch.nn.Parameter(
            torch.tensor(init_visibility_threshold).to(device),
            requires_grad=True
        )
        self.ceiling_threshold = torch.nn.Parameter(
            torch.tensor(init_ceiling_threshold).to(device), # scale by 1k
            requires_grad=True
        )
        self.a = torch.tensor(a).to(device)

    def assign_label(self, visibility, ceiling):
        return 1 - torch.nn.functional.sigmoid(
            self.a * (visibility - self.visibility_threshold)
        ) * torch.nn.functional.sigmoid(
            self.a * (ceiling/1000.0 - self.ceiling_threshold)
        )
    
class ClusterThreshold(torch.nn.Module):
    def __init__(
            self, a, device, 
            # y_threshold,
            # x_threshold,
            init_visibility_threshold, 
            init_ceiling_threshold, 
            v=None, c=None
        ):
        super().__init__()
        self.device = device
        self.visibility_threshold = torch.nn.Parameter(
            torch.tensor(init_visibility_threshold).to(device),
            requires_grad=True
        ) if v is None else v
        self.ceiling_threshold = torch.nn.Parameter(
            torch.tensor(init_ceiling_threshold).to(device), # scale by 1k
            requires_grad=True
        ) if c is None else c
        # self.y_threshold = y_threshold
        # self.x_threshold = x_threshold
        self.a = torch.tensor(a).to(device)

    def assign_label(self, y_label, x_label, visibility, ceiling):
        # y_label = 1.0 if y > self.y_threshold else 0.0
        # x_label = 1.0 if x > self.x_threshold else 0.0
        w_label = (
            1 - torch.nn.functional.sigmoid(
                self.a * (visibility - self.visibility_threshold)
            ) * torch.nn.functional.sigmoid(
                self.a * (ceiling/1000.0 - self.ceiling_threshold)
            )
        ).unsqueeze(dim=-1)
        # TODO: should we split y,x into 4 separte as 1-hot ???
        label = torch.cat(
            (torch.zeros(4), w_label),
            axis=-1,
        )
        label[(y_label*2+x_label).long()] = 1.0
        return label



def train(
        project,
        network_airport_codes, # not used?
        svi_steps, 
        n_samples, 
        svi_lr, 
        gamma,
        dt,
        n_elbo_particles,
        finer,
        plot_every,
        rng_seed,
        posterior_guide,
        y_threshold,
        x_threshold,
        init_visibility_threshold,
        init_ceiling_threshold,
        day_strs_list,
        auto_split,
        auto_split_limit,
        auto_split_random,
        use_gpu=False,
    ):
    print('Starting training...')
    pyro.clear_param_store()  # avoid leaking parameters across runs
    pyro.enable_validation(True)
    pyro.set_rng_seed(int(rng_seed))
    torch.manual_seed(int(rng_seed))
    np.random.seed(int(rng_seed))

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
    extras_path = dir_path / 'extras' / network_airport_codes[0]

    # TODO generate these using the jupyter file
    processed_visibility = pd.read_csv(extras_path / f'{network_airport_codes[0]}_w_processed_visibility.csv')
    visibility_dict = dict(processed_visibility.values)
    processed_ceiling = pd.read_csv(extras_path / f'{network_airport_codes[0]}_w_processed_ceiling.csv')
    ceiling_dict = dict(processed_ceiling.values)

    processed_x = pd.read_csv(extras_path / f'{network_airport_codes[0]}_x_capacity_counts.csv') # TODO: option
    x_dict = dict(processed_x.values)
    processed_y = pd.read_csv(extras_path / f'{network_airport_codes[0]}_y_event_delays.csv') # TODO: option
    y_dict = dict(processed_y.values)

    # TODO regenerate wrt each apt
    model_logprobs_name = (
        '2019_finer_output_dict.pkl' 
        if finer else '2019_output_dict.pkl'
    )
    with open(extras_path / model_logprobs_name, 'rb') as f:
        model_logprobs_output_dict = dill.load(f)

    with open(extras_path / '2019_s_guide_dist_dict.pkl', 'rb') as f:
        s_guide_dist_dict = dill.load(f)

    # SETTING UP THE MODEL AND STUFF

    # Hyperparameters
    mst_split = 1 # not really used

    subsamples = {}

    yx_groups = {
        (i, j): []
        for i in range(2)
        for j in range(2)
    }
    
    # first go and do the easy stuff
    pbar = tqdm(day_strs_list)
    for day_strs in pbar:
        name = ", ".join(day_strs)
        
        y = y_dict[name]
        x = x_dict[name]
        y_label = 1.0 if y > y_threshold else 0.0
        x_label = 1.0 if x > x_threshold else 0.0
        yx_group = (int(y_label), int(x_label))
        yx_groups[yx_group].append(name)

        s_guide_dist = s_guide_dist_dict[name]

        subsamples[name] = {
            "y": y,
            "x": x,
            "yx_group": yx_group,
            "y_label": torch.tensor([y_label]).to(device),
            "x_label": torch.tensor([x_label]).to(device),
            # hardcode fix
            "visibility": visibility_dict[name + ' 00:00:00+00:00'],
            "ceiling": ceiling_dict[name + ' 00:00:00+00:00'],
            "s_guide_dist": s_guide_dist,
            "model_logprobs": model_logprobs_output_dict[name],
            "z_mu": s_guide_dist.loc.item(),
            "z_sigma": s_guide_dist.scale.item(),
        }

        pbar.set_description(f"{name} -> yx_group = {yx_group}")
    
    '''
    for k, v in subsamples.items():
        print(
            k, 
            int(v['y_label'].item()), 
            int(v['x_label'].item()), 
            f'{v["y"]:.3f}',
            v['x'],
        )
    print(yx_groups)
    '''

    if auto_split:
        for group, names in yx_groups.items():
            lim = min(auto_split_limit, len(names))
            if lim < auto_split_limit:
                print(f"warning: group {group} has only {lim} samples, less than limit of {auto_split_limit}")
            if auto_split_random:
                random.seed(rng_seed)
                yx_groups[group] = random.sample(yx_groups[group], lim)
            else:
                tmp = {k: v for k, v in subsamples.items() if v["yx_group"] == group}
                kvs = sorted(tmp.items(), key = lambda kv: kv[1]["z_mu"], reverse=(group[0]!=(0)))
                yx_groups[group] = [kv[0] for kv in kvs[:lim]]
                print(
                    group, 
                    f"{len(names):03d}",
                    f"{sum([kv[1]['z_mu'] for kv in kvs[:lim]])/lim:.6f}", 
                    f"{sum([kv[1]['z_mu'] for kv in kvs])/len(kvs):.6f}")

        # print(yx_groups)
        # return
        valid_names = list(itertools.chain.from_iterable(yx_groups.values()))
        
        subsamples = {
            k: v
            for k, v in subsamples.items()
            if k in valid_names
        }
        day_strs_list = [
            day_strs for day_strs in day_strs_list
            if ','.join(day_strs) in valid_names
        ]

    nominal_prior = dist.Normal(
        torch.tensor(.0120).to(device), 
        # torch.tensor(.0004).to(device)
        torch.tensor(.001).to(device)
    )

    failure_prior = dist.Normal(
        torch.tensor(.0190).to(device), 
        # torch.tensor(.0012).to(device)
        torch.tensor(.001).to(device)
    )

    plot_failure_nominal_prior(failure_prior, nominal_prior)
    prior_mixture = PriorMixture(failure_prior, nominal_prior)

    # setup guide
    guide = setup_guide(posterior_guide, mst_split, device)

    wt = WeatherThreshold(50.0, device, init_visibility_threshold, init_ceiling_threshold)
    ct = ClusterThreshold(50.0, device, init_visibility_threshold, init_ceiling_threshold)

    def objective_fn(subsample, n=1, posterior_scale=1, s_scale=1, prior_scale=1, yc_scale=1):

        visibility = subsample["visibility"]
        ceiling = subsample["ceiling"]
        s_guide_dist = subsample["s_guide_dist"]
        model_logprobs = subsample["model_logprobs"]

        x_label = subsample["x_label"]
        y_label = subsample["y_label"]

        prior_label = wt.assign_label(visibility, ceiling)
        posterior_label = ct.assign_label(y_label, x_label, visibility, ceiling)

        posterior_samples = guide(posterior_label).rsample((n,))
        posterior_logprobs = guide(posterior_label).log_prob(posterior_samples) \
            # + transform_sample_deriv(posterior_samples)

        s_logprobs = s_guide_dist.log_prob(transform_sample(posterior_samples))
        prior_logprobs = prior_mixture.log_prob(transform_sample(posterior_samples), prior_label)

        y_given_c_logprobs = y_given_c_log_prob(
            model_logprobs, 
            functools.partial(prior_mixture.log_prob, label=prior_label),
            device,
            finer=finer,
        )

        objective = (
            s_logprobs * s_scale 
            + prior_logprobs * prior_scale 
            - posterior_logprobs * posterior_scale
        ).mean() - y_given_c_logprobs * yc_scale

        # print(
        #     prior_label.detach().numpy(), 
        #     posterior_label.detach().numpy(), 
        #     s_logprobs.item(), prior_logprobs.item(), 
        #     posterior_logprobs.item(), y_given_c_logprobs.item()
        # )
        return -objective # negate to make it a loss
    
    def total_objective_fn(subsamples, **kwargs):
        loss = torch.tensor(0.0).to(device)
        for _, subsample in subsamples.items():
            loss += objective_fn(subsample, **kwargs)
        return loss / len(subsamples)
    
    guide_optimizer = torch.optim.Adam(
        guide.parameters(),
        lr=svi_lr,
        weight_decay=0.0, # fix this
    )
    guide_scheduler = torch.optim.lr_scheduler.StepLR(
        guide_optimizer, step_size=svi_steps, gamma=gamma
    )

    wt_optimizer = torch.optim.Adam(
        wt.parameters(),
        lr=svi_lr,
        weight_decay=0.0, # fix this
    )
    wt_scheduler = torch.optim.lr_scheduler.StepLR(
        wt_optimizer, step_size=svi_steps, gamma=gamma
    )

    ct_optimizer = torch.optim.Adam(
        ct.parameters(),
        lr=svi_lr,
        weight_decay=0.0, # fix this
    )
    ct_scheduler = torch.optim.lr_scheduler.StepLR(
        ct_optimizer, step_size=svi_steps, gamma=gamma
    )

    # run_name = f"[{','.join(network_airport_codes)}]_"
    # run_name += f"[{posterior_guide}]_"
    # run_name += f"[{day_strs_list[0][0]}_{day_strs_list[-1][0]}]" # TODO: fix this
    run_name = f'{posterior_guide}_{len(subsamples)}'
    # print(run_name)
    group_name = f"{len(subsamples)}"
    # print(group_name)
    # TODO: fix this for non-day-by-day???
    sub_dir = f"{project}/checkpoints/{run_name}/"

    wandb_init_config_dict = {
        # "starting_aircraft": starting_aircraft,
        "days": list(subsamples.keys()),
        # "num_flights": sum([v['num_flights']]),
        "num_days": len(subsamples),

        "dt": dt,
        "svi_lr": svi_lr,
        "svi_steps": svi_steps,
        "gamma": gamma,
        "n_samples": n_samples,
        "n_elbo_particles": n_elbo_particles,
        "finer": finer,

        "posterior_guide": posterior_guide,
        "y_threshold": y_threshold,
        "x_threshold": x_threshold,
        "init_visibility_threshold": init_visibility_threshold,
        "init_ceiling_threshold": init_ceiling_threshold,
    }
    
    wandb.init(
        project=project,
        name=run_name,
        group=group_name,
        config=wandb_init_config_dict,
    )

    losses = []

    pbar = tqdm(range(svi_steps))

    labels = []
    names = []
    for a in range(2):
        for b in range(2):
            for c in range(2):
                label = torch.zeros(5, dtype=torch.float)
                label[2*a+b] = 1.0
                label[4] = c
                labels.append(label)
                names.append(f'{a}{b}{c}')

    # kl_factor_start = 0.0
    # kl_factor_end = 1.0
    # kl_factor_schedule = torch.linspace(kl_factor_start, kl_factor_end, svi_steps).sqrt()

    for i in pbar:
        guide_optimizer.zero_grad()
        wt_optimizer.zero_grad()
        ct_optimizer.zero_grad()

        loss = total_objective_fn(
            subsamples,
            # s_scale=kl_factor_schedule[i],
            # posterior_scale=kl_factor_schedule[i],
            s_scale=2,
            prior_scale=.2,
        )
        loss.backward()

        if i < 500:
            guide_grad_norm = torch.nn.utils.clip_grad_norm_(
                guide.parameters(), 100.0 # TODO :grad clip param
            )
            guide_optimizer.step()
            guide_scheduler.step()

    # else:
        wt_grad_norm = torch.nn.utils.clip_grad_norm_(
            wt.parameters(), 10.0 # TODO :grad clip param
        )
        ct_grad_norm = torch.nn.utils.clip_grad_norm_(
            ct.parameters(), 10.0 # TODO :grad clip param
        )
        wt_optimizer.step()
        wt_scheduler.step()
        ct_optimizer.step()
        ct_scheduler.step()

        wt.ceiling_threshold.data.clamp_(0.0, 10.0)
        wt.visibility_threshold.data.clamp_(0.0, 10.0)
        ct.ceiling_threshold.data.clamp_(0.0, 10.0)
        ct.visibility_threshold.data.clamp_(0.0, 10.0)

        loss = loss.detach()
        losses.append(loss)

        desc = f"wt: v={wt.visibility_threshold.data.item():.4f}, c={wt.ceiling_threshold.data.item():.4f}; "
        desc += f"ct: v={ct.visibility_threshold.data.item():.4f}, c={ct.ceiling_threshold.data.item():.4f}; "
        desc += f"loss: {loss:.2f}" 
        pbar.set_description(desc)
                
        if i % plot_every == 0 or i == svi_steps - 1:
                
            fig, ax = plt.subplots()
            # for label, color in zip(labels, ['b', 'r']):
            bins = np.arange(.0050, .0351, .0001)
            for label, name in zip(labels, names):
                # print(label)
                samples = transform_sample(guide(label).sample((n_samples,)))
                ax.hist(
                    samples, 
                    # color='b', 
                    alpha=.5, 
                    fill=True, 
                    edgecolor='k', 
                    bins=bins,
                    linewidth=0,
                    label=f'{name}',
                )
            plt.xlim(0.005,.035)
            plt.legend()
            wandb.log({f"posteriors": wandb.Image(fig)}, commit=False)
            plt.close(fig)

            # Save the params and autoguide
            dir_path = os.path.dirname(__file__)
            save_path = os.path.join(dir_path, sub_dir)
            os.makedirs(save_path, exist_ok=True)
            # torch.save(guide.state_dict(), os.path.join(save_path, "guide.pth"))
            torch.save(
                {
                    "guide": guide.state_dict(),
                    "wt": wt.state_dict(),
                    "ct": ct.state_dict(),
                    "wt_vt": wt.visibility_threshold.data.item(),
                    "wt_ct": wt.ceiling_threshold.data.item(),
                    "ct_vt": ct.visibility_threshold.data.item(),
                    "ct_ct": ct.ceiling_threshold.data.item(),
                    "y_th": y_threshold,
                    "x_th": x_threshold,
                },
                # f"checkpoints/{run_name}/failure_checkpoint_{i}.pt",
                os.path.join(dir_path, sub_dir, f'checkpoint_{i}.pt')
            )

        # labels = [torch.tensor([a,b,c]).to(device) for a in (0.0,1.0) for b in (0.0,1.0) for c in (0.0,1.0)]
        # names = [(a,b,c) for a in (0,1) for b in (0,1) for c in (0,1)]
        samples = [transform_sample(guide(label).sample((n_samples,))).detach() for label in labels]

        log_dict = {
            "ELBO/loss (units = nats per dim)": loss.item(), 
            "wt/visibility": wt.visibility_threshold.data.item(),
            "wt/ceiling": wt.ceiling_threshold.data.item(),
            "ct/visibility": ct.visibility_threshold.data.item(),
            "ct/ceiling": ct.ceiling_threshold.data.item(),
        }
        for label, sample, name in zip(labels, samples, names):
            log_dict[f"q-mean/{name}"] = sample.mean().item()
            log_dict[f"q-std/{name}"] = torch.std(sample).item()

        wandb.log(log_dict)

    wandb.save(f"checkpoints/{run_name}/checkpoint_{svi_steps - 1}.pt")

    output_dict = {
        'guide': guide,
        'wt': wt,
        'ct': ct,
        'run_name': run_name,
        'group_name': group_name,
        'config': wandb_init_config_dict,
        "guide_state_dict": guide.state_dict(),
        "wt_state_dict": wt.state_dict(),
        "ct_state_dict": ct.state_dict(),
        "wt_vt": wt.visibility_threshold.data.item(),
        "wt_ct": wt.ceiling_threshold.data.item(),
        "ct_vt": ct.visibility_threshold.data.item(),
        "ct_ct": ct.ceiling_threshold.data.item(),
        "y_th": y_threshold,
        "x_th": x_threshold,
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


# TODO: add functionality to pick days
@click.command()
@click.option("--project", default="jfk-training-attempt-0")
@click.option("--network-airport-codes", default="JFK", help="airport codes")

@click.option("--svi-steps", default=1000, help="Number of SVI steps to run")
@click.option("--n-samples", default=10000, help="Number of posterior samples to draw")
@click.option("--svi-lr", default=5e-3, help="Learning rate for SVI")
@click.option("--plot-every", default=50, help="Plot every N steps")
@click.option("--rng-seed", default=1, type=int)
@click.option("--gamma", default=.2) # was .1
@click.option("--dt", default=.1)
@click.option("--n-elbo-particles", default=1)
@click.option("--finer/--no-finer", default=True)

@click.option("--posterior-guide", default="nsf", help="nsf/cnf/gmm") 
# TODO: weights for things in the objective

# thresholds: TODO
@click.option("--y-threshold", default=0.25, type=float) # hours (15 minutes)
@click.option("--x-threshold", default=72.0, type=float)
@click.option("--init-visibility-threshold", default=1.5, type=float) # hours
@click.option("--init-ceiling-threshold", default=1.0, type=float)

@click.option("--day-strs", default='2019-07-15')
# @click.option("--day-strs-path", default=None) # TODO: make this!!
@click.option("--year", default=None, type=int)
@click.option("--month", default=None, type=int)
@click.option("--start-day", default=None)
@click.option("--end-day", default=None)
@click.option("--all-days", is_flag=True)

@click.option("--auto-split", is_flag=True)
@click.option("--auto-split-limit", default=20, type=int) # was 10
@click.option("--auto-split-random", is_flag=True) 
def train_cmd(
        project, network_airport_codes, 
        svi_steps, n_samples, svi_lr, 
        plot_every, rng_seed,
        gamma, dt, n_elbo_particles, finer,
        posterior_guide, 
        y_threshold, x_threshold,
        init_visibility_threshold, init_ceiling_threshold,
        day_strs, year, month, start_day, end_day, all_days,
        auto_split, auto_split_limit, auto_split_random
    ):
    print(f"Running training with project: {project}, network_airport_codes: {network_airport_codes}, "
          f"svi_steps: {svi_steps}, n_samples: {n_samples}, svi_lr: {svi_lr}, "
            f"plot_every: {plot_every}, rng_seed: {rng_seed}, gamma: {gamma}, dt: {dt}, "
            f"n_elbo_particles: {n_elbo_particles}, finer: {finer}, "
            f"posterior_guide: {posterior_guide}, "
            f"y_threshold: {y_threshold}, x_threshold: {x_threshold}, "
            f"init_visibility_threshold: {init_visibility_threshold}, "
            f"init_ceiling_threshold: {init_ceiling_threshold}, "
            f"day_strs: {day_strs}, year: {year}, month: {month}, "
            f"start_day: {start_day}, end_day: {end_day}, all_days: {all_days}, "
            f"auto_split: {auto_split}, auto_split_limit: {auto_split_limit}, "
            f"auto_split_random: {auto_split_random}")
    network_airport_codes = network_airport_codes.split(',')
    if day_strs is not None:
        day_strs = day_strs.split(',')
    elif year is not None:
        if month is not None:
            start_day = f'{year}-{month}-1'
            days_in_month = pd.Period(start_day).days_in_month
            end_day = f'{year}-{month}-{days_in_month}'
            day_strs = pd.date_range(start=start_day, end=end_day, freq='D').strftime('%Y-%m-%d').to_list()
        else:
            start_day = f'{year}-1-1'
            end_day = f'{year}-12-31'
            day_strs = pd.date_range(start=start_day, end=end_day, freq='D').strftime('%Y-%m-%d').to_list()
    elif start_day is not None:
        if end_day is None:
            end_day = start_day
        day_strs = pd.date_range(start=start_day, end=end_day, freq='D').strftime('%Y-%m-%d').to_list()
    else: # basically all_days
        start_day = f'2018-1-1'
        end_day = f'2019-12-31'
        day_strs = pd.date_range(start=start_day, end=end_day, freq='D').strftime('%Y-%m-%d').to_list()
    
    if auto_split:
        pass
        if auto_split_limit:
            pass

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
        finer,
        plot_every,
        rng_seed,
        posterior_guide,
        y_threshold,
        x_threshold,
        init_visibility_threshold,
        init_ceiling_threshold,
        day_strs_list,
        auto_split,
        auto_split_limit,
        auto_split_random,
    )


if __name__ == "__main__":
    train_cmd()