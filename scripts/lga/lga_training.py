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
    ConditionalGaussianMixture
)
import zuko

from pbar_pool import PbarPool, Pbar

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


def asdf(model, sample, states):
    """
    p(y|z)
    """
    conditioning_dict = map_to_sample_sites_identity(sample)
    model_trace = pyro.poutine.trace(
        pyro.poutine.condition(model, data=conditioning_dict)
        # model
    ).get_trace(
        states=states,
        # mst_as_param=sample,
    )
    model_logprob = model_trace.log_prob_sum()
    return model_logprob


def train(
    project,
    network_airport_codes, 
    svi_steps, 
    n_samples, 
    svi_lr, 
    gamma,
    dt,
    n_elbo_particles,
    plot_every,
    rng_seed,
    multiprocess,
    pbars,

    rem_args,
    use_gpu=False
):
    pyro.clear_param_store()  # avoid leaking parameters across runs
    pyro.enable_validation(True)
    pyro.set_rng_seed(int(rng_seed))

    day_strs, prior_type, prior_scale, posterior_guide = rem_args

    if use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device("cpu")

    # Avoid plotting error . test
    matplotlib.use("Agg")

    # SETTING UP THE MODEL AND STUFF

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
    if not multiprocess:
        print(f"Number of days: {num_days}")
        print(f"Number of flights: {num_flights}")

    # make things with the data
    travel_times_dict, observations_df = \
        make_travel_times_dict_and_observation_df(
            data, network_airport_codes
        ) 
    states = make_states(data, network_airport_codes)

    # here, we make the nominal, failure, and uniform (not used) priors...
    # TODO: shifted gamma doesn't work. i have no idea why
    # mst_prior_nominal = _affine_beta_dist_from_mean_std(.0125, .001, .010, .020, device)
    mst_prior_nominal = _gamma_dist_from_mean_std(.0125, .0004, device)

    # mst_prior_failure = _affine_beta_dist_from_mean_std(.0200, .002, .010, .030, device)
    mst_prior_failure = _gamma_dist_from_mean_std(.0250, .0012, device)

    # mst_prior_default = _affine_beta_dist_from_alpha_beta(1.0, 1.0, .005, .030, device)
    mst_prior_default = _gamma_dist_from_mean_std(.0135, .0050, device)

    if not multiprocess:
        fig = plt.figure()
        s = mst_prior_nominal.sample((10000,)).detach().cpu()
        sns.histplot(s, color='b', alpha=.33, fill=True, label="nominal", kde=True, binwidth=.0001, edgecolor='k', linewidth=0)
        s = mst_prior_failure.sample((10000,)).detach().cpu()
        sns.histplot(s, color='r', alpha=.33, fill=True, label="failure", kde=True, binwidth=.0001, edgecolor='k', linewidth=0)
        plt.savefig('ab_test.png')
        s = mst_prior_default.sample((10000,)).detach().cpu()
        sns.histplot(s, color='purple', alpha=.33, fill=True, label="empty", kde=True, binwidth=.0001, edgecolor='k', linewidth=0)
        plt.legend()
        plt.title("prior distributions example")
        plt.savefig('abc_test.png')
        plt.close(fig)
    # return

    # by default, scale ELBO down by num flights
    model_scale = 1.0 / (num_flights)
    mst_prior_weight = mst_prior_scale / model_scale 
    # equals mst_prior_scale * num_flights / num_flights -> so should end up just as scale?

    # set up common model arguments
    model = functools.partial(
        augmented_air_traffic_network_model_simplified,

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

        # mst_prior=mst_prior_nominal,
        # mst_prior_weight=1e-12,
    )











    # re-scale ELBO
    model = pyro.poutine.scale(model, scale=model_scale)

    x = []
    y = []
    zs = []
    for val in np.arange(.005, .035, .001):
        n=1
        try:
            l = sum(
                [
                    asdf(model, torch.tensor(val).to(device), states)
                    for _ in range(n)
                ]
            ) / n
            z = mst_prior_nominal.log_prob(val)
            x.append(val)
            y.append(l.item())
            zs.append(z.item())
            print(f'{val:.3f}', f'{l.item():.4f}', f'{z.item():.4f}')
        except:
            pass
    fig = plt.figure()
    plt.plot(x,y)
    plt.plot(x,zs)
    plt.savefig('y_given_z_test.png')
    plt.close(fig)
    
    # print(f'{best_val:.3f}', best_l)

    return

    # Create an autoguide for the model
    init_loc_fn = pyro.infer.autoguide.initialization.init_to_value(
        values={
            f'LGA_{t_idx}_mean_service_time': torch.tensor(.015).to(device)
            for t_idx in range(3)
        },
        fallback=pyro.infer.autoguide.initialization.init_to_median
    )

    if posterior_guide == "nsf":
       _guide = zuko.flows.NSF(
            features=mst_split,
            context=1,
            hidden_features=(8, 8),
        ).to(device)
    elif posterior_guide == "cnf":
        _guide = zuko.flows.CNF(
            features=mst_split,
            context=1,
            hidden_features=(8, 8),
        ).to(device)
    elif posterior_guide == "gmm":
        _guide = ConditionalGaussianMixture(
            n_context=1, n_features=1,
        )
    else:
        raise ValueError
    
    # testing
    label = torch.tensor([0.0]).to(device).detach()
    guide = _guide(label)


    # Set up SVI
    gamma = gamma  # final learning rate will be gamma * initial_lr
    lrd = gamma ** (1 / svi_steps)

    optimizer = torch.optim.Adam(
        _guide.parameters(),
        lr=svi_lr,
        weight_decay=0.0, # fix this
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=svi_steps, gamma=gamma
    )

    run_name = f"[{','.join(network_airport_codes)}]_"
    run_name += f"[{prior_type},{prior_scale:.2f},{posterior_guide}]_"
    run_name += f"[{','.join(days.strftime('%Y-%m-%d').to_list())}]"
    # print(run_name)
    group_name = f"{prior_type}-{prior_scale:.2f}-{posterior_guide}"
    # print(group_name)
    # TODO: fix this for non-day-by-day???
    sub_dir = f"{project}/checkpoints/{'_'.join(network_airport_codes)}/{'_'.join(days.strftime('%Y-%m-%d').to_list())}/{prior_type}_{prior_scale:.2f}_{posterior_guide}/"

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
        project=project,
        name=run_name,
        group=group_name,
        config=wandb_init_config_dict,
    )

    losses = []
    losses_from_prior = []

    if not multiprocess:
        pbar = tqdm(range(svi_steps))
    else:
        if prior_type == "failure":
            color = (255, 100, 100)
        elif prior_type == "nominal":
            color = (100, 100, 255)
        else:
            color = (200, 200, 200)
        # pbar = Pbar(range(svi_steps), manager=pbars, name=f'{",".join(day_strs)} {group_name}\nprocess {pbars.id()}', color=color)
        pbar = Pbar(range(svi_steps), manager=pbars, name=f'{run_name}\n  task {pbars.id()}', color=color)


    for i in pbar:
        optimizer.zero_grad()
        loss = objective_fn(model, _guide(torch.tensor([0.0]).to(device)), states, device, 1)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            _guide.parameters(), 100.0 # TODO :grad clip param
        )
        optimizer.step()
        scheduler.step()
        # hidden.detach_()

        loss = loss.detach()

        losses.append(loss)

        posterior_samples = torch.exp(guide.sample((n_samples,)))

        z_samples = [posterior_samples[:,t_idx] for t_idx in range(mst_split)]
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
        if not multiprocess:
            pbar.set_description(desc)
                
        if i % plot_every == 0 or i == svi_steps - 1:

            # hourly_delays = get_hourly_delays(
            #     model, guide, states, observations_df, 1
            # )

            # fig = plot_hourly_delays(hourly_delays)
            # wandb.log({"Hourly delays": wandb.Image(fig)}, commit=False)
            # plt.close(fig)

            # plotting_dict = {
            #     "mean service times": plot_service_times,
            # }
        
            # for name, plot_func in plotting_dict.items():
            #     # for now require this common signature
            #     fig = plot_func(guide, states, n_samples)
            #     wandb.log({name: wandb.Image(fig)}, commit=False)
            #     plt.close(fig)

            # Save the params and autoguide
            dir_path = os.path.dirname(__file__)
            # save_path = os.path.join(dir_path, "checkpoints_final", run_name, f"{i}")
            save_path = os.path.join(dir_path, sub_dir, f"{i}")
            os.makedirs(save_path, exist_ok=True)
            pyro.get_param_store().save(os.path.join(save_path, "params.pth"))
            torch.save(_guide.state_dict(), os.path.join(save_path, "guide.pth"))

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
        # 'set_model': functools.partial(model, states),
        # 'set_guide': functools.partial(guide, states),
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

    # return loss


from multiprocessing import Process, Pool, cpu_count
# https://stackoverflow.com/questions/7207309/how-to-run-functions-in-parallel
# trying something
def run_cpu_tasks_in_parallel(tasks):
    running_tasks = [Process(target=task) for task in tasks]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()

import warnings

# TODO: add functionality to pick days
@click.command()
@click.option("--project", default="bayes-air-atrds-attempt-5")
@click.option("--network-airport-codes", default="LGA", help="airport codes")

@click.option("--svi-steps", default=500, help="Number of SVI steps to run")
@click.option("--n-samples", default=5000, help="Number of posterior samples to draw")
@click.option("--svi-lr", default=5e-3, help="Learning rate for SVI")
@click.option("--plot-every", default=50, help="Plot every N steps")
@click.option("--rng-seed", default=1, type=int)
@click.option("--gamma", default=.4) # was .1
@click.option("--dt", default=.1)
@click.option("--n-elbo-particles", default=1)

@click.option("--prior-type", default="empty", help="nominal/failure/empty")
@click.option("--posterior-guide", default="gaussian", help="gaussian/iafnormal/delta/laplace") 
# delta and laplace break plots rn 
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

@click.option("--multiprocess/--no-multiprocess", default=True)
@click.option("--processes", default=None)
@click.option("--wandb-silent", is_flag=True)


def train_cmd(
    project, network_airport_codes, 
    svi_steps, n_samples, svi_lr, 
    plot_every, rng_seed, gamma, dt, n_elbo_particles,
    prior_type, prior_scale, posterior_guide, 
    day_strs, year, month, start_day, end_day,
    learn_together, all_combos, multiprocess, processes, wandb_silent
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

    if wandb_silent or multiprocess:
        os.environ["WANDB_SILENT"] = "true"
    # else:
    #     os.environ["WANDB_SILENT"] = "false"

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
            for pguide in ("gmm", "nsf")
        ] + [
            ("empty", default_zero_scale, pguide)
            for pguide in ("gmm", "nsf")
        ]

    if multiprocess:
        pbars = PbarPool(width=100)
        def initializer():
            sys.stdout = open(os.devnull, 'w')
            return pbars.initializer()
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"
    else:
        pbars = None

    base_func = functools.partial(
        train,
        project,
        network_airport_codes,
        svi_steps,
        n_samples,
        svi_lr,
        gamma,
        dt,
        n_elbo_particles,
        plot_every,
        rng_seed,
        multiprocess,
        pbars,
    )

    rem_args = [
        (day_strs_list[i], *(ppp_params[j]))
        for i in range(len(day_strs_list))
        for j in range(len(ppp_params))
    ]
    # print(rem_args)
    # return

    if processes is None:
        processes = cpu_count()

    if multiprocess:
        with Pool(processes=processes, initializer=initializer()) as p:
            global_pbar = Pbar(p.imap_unordered(base_func, rem_args), manager=pbars, name='global', total=len(rem_args))
            for _ in global_pbar:
                pass

    else:
        for rem_arg in rem_args:
            base_func(rem_arg)


if __name__ == "__main__":
    train_cmd()