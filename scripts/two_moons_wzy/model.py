"""Define the model for the two moons toy problem."""
import pyro
import pyro.distributions as dist
import torch


def generate_two_moons_data(n, device, failure=False, sigma=0.1):
    """Generate two moons data.

    Args:
        n (int): Number of samples to generate.
        device (torch.device): Device to use.
        failure (bool): Whether to generate failure data.
    """
    theta = torch.pi * torch.rand(n).to(device)
    if failure:
        theta += torch.pi

    x = torch.stack(
        (
            torch.cos(theta) - 1 / 2,
            torch.sin(theta) - 1 / 4,
        ),
        axis=-1,
    )
    if failure:
        x += torch.tensor([1.0, 0.5]).to(device)

    return torch.normal(x, sigma)


def generate_two_moons_data_hierarchical(n, device, sigma=0.1, w_obs=None, z_obs=None, theta_obs=None):
    """Generate two moons data.

    Args:
        n (int): Number of samples to generate.
        device (torch.device): Device to use.
        failure (bool): Whether to generate failure data.
    """

    if w_obs is None:
        w = torch.rand(n).to(device)
    else:
        w = w_obs

    z = w > .5 if z_obs is None else z_obs

    if theta_obs is None:
        theta = torch.pi * torch.rand(n).to(device)
        theta[z] += torch.pi
    else:
        theta = theta_obs


    y = torch.stack(
        (
            torch.cos(theta) - 1 / 2,
            torch.sin(theta) - 1 / 4,
        ),
        axis=-1,
    )
        
    y[z] += (torch.tensor([1.0, 0.5]).to(device))

    y = torch.normal(y, sigma)

    return y.cpu(), w.cpu()



def two_moons_model(n, device, obs=None):
    """Define noisy observation for the two moons dataset.

    This function doesn't actually create the data, just models the observation.
    """
    with pyro.plate("data", n):
        x = pyro.sample(
            "x",
            dist.Normal(
                torch.zeros(2, device=device),
                5 * torch.ones(2, device=device),
            ).to_event(1),
        )

        noisy_obs = pyro.sample(
            "obs",
            dist.Normal(
                x,
                torch.tensor([0.1, 0.1]).to(device),
            ).to_event(1),
            obs=obs,
        )

    return noisy_obs


def two_moons_w_z_model(n, device, obs=None):

    with pyro.plate("data", n):
        w = pyro.sample(
            "w", 
            dist.Beta(
                torch.tensor(1.0, device=device),
                torch.tensor(1.0, device=device),
            )
        )

        z = pyro.sample(
            "z",
            dist.RelaxedBernoulliStraightThrough(
                temperature=torch.tensor(0.1, device=device),
                probs=w,
            ),
            obs=obs
        )

    return z


def two_moons_z_y_model(n, device, obs=None):

    with pyro.plate("data", n):
        z = pyro.sample(
            "z",
            dist.Beta(
                torch.tensor(1.0, device=device),
                torch.tensor(1.0, device=device),
            ),
        )

        failure = torch.tensor(1.0, device=device) * (z > .5)

        theta = pyro.sample(
            "theta",
            dist.Beta(
                torch.tensor(1.0, device=device),
                torch.tensor(1.0, device=device),
            ),
        )

        theta = (theta + failure) * torch.pi

        y_val = torch.stack(
            (
                torch.cos(theta) - 1 / 2,
                torch.sin(theta) - 1 / 4,
            ),
            axis=-1,
        )
        
        y_val += (torch.tensor([1.0, 0.5]).to(device)) * failure.unsqueeze(-1)

        y = pyro.sample(
            "y",
            dist.Normal(
                y_val,
                torch.tensor([0.1, 0.1]).to(device),
            ).to_event(1),
            obs=obs,
        )

    return y



def two_moons_w_z_y_model(n, device, w_obs=None, y_obs=None):


    with pyro.plate("data", n):
         
        w = pyro.deterministic(
            "w", 
            w_obs
        )

        z = pyro.sample(
            "z",
            dist.RelaxedBernoulliStraightThrough(
                temperature=torch.tensor(0.1, device=device),
                probs=w,
            ),
        )

        failure = torch.tensor(1.0, device=device) * z

        theta = pyro.sample(
            "theta",
            dist.Uniform(
                torch.tensor(0.0, device=device),
                torch.tensor(1.0, device=device),
            ),
        )

        theta = (theta + failure) * torch.pi

        y_val = torch.stack(
            (
                torch.cos(theta) - 1 / 2,
                torch.sin(theta) - 1 / 4,
            ),
            axis=-1,
        )
        
        y_val += (torch.tensor([1.0, 0.5]).to(device)) * failure.unsqueeze(-1)

        y = pyro.sample(
            "y",
            dist.Normal(
                y_val,
                torch.tensor([0.1, 0.1]).to(device),
            ).to_event(1),
            obs=y_obs,
        )

    return y



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y, w = generate_two_moons_data_hierarchical(1000, device)
     
    labels = (w > .5)
    samples = y

    plt.figure(figsize=(4, 4))
    plt.scatter(*samples.T, s=1, c=labels, cmap="bwr")
    # Turn off axis ticks
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.ylim([-1.1, 1.1])
    plt.xlim([-1.7, 1.7])
    # Equal aspect
    plt.gca().set_aspect("equal")
    plt.show()
