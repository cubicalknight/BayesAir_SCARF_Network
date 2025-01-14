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


def generate_two_moons_data_hierarchical(n, device, sigma=0.1):
    """Generate two moons data.

    Args:
        n (int): Number of samples to generate.
        device (torch.device): Device to use.
        failure (bool): Whether to generate failure data.
    """

    w = torch.rand(n).to(device)
    z = w > .5

    theta = torch.pi * torch.rand(n).to(device)

    # print(w)
    # print(w > .5)

    theta[z] += torch.pi

    y = torch.stack(
        (
            torch.cos(theta) - 1 / 2,
            torch.sin(theta) - 1 / 4,
        ),
        axis=-1,
    )
        
    y[z] += (torch.tensor([1.0, 0.5]).to(device))

    y = torch.normal(z, sigma)

    return y, w



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
            ).to_event(1),
        )

        z = pyro.sample(
            "z",
            dist.RelaxedBernoulliStraightThrough(
                temperature=torch.tensor(0.1, device=device),
                probs=w,
            ),
            obs=obs
        ).to_event(1)

    return z


def two_moons_z_y_model(n, device, obs=None):

    with pyro.plate("data", n):
        z = pyro.sample(
            "z",
            dist.Beta(
                torch.tensor(1.0, device=device),
                torch.tensor(1.0, device=device),
            ).to_event(1),
        )

        failure = torch.tensor(1.0 if z > .5 else 0.0, device=device)

        theta = pyro.sample(
            "theta",
            dist.Beta(
                torch.tensor(1.0, device=device),
                torch.tensor(1.0, device=device),
            ).to_event(1),
        )

        theta = (theta + failure) * torch.pi

        y_val = torch.stack(
            (
                torch.cos(theta) - 1 / 2,
                torch.sin(theta) - 1 / 4,
            ),
            axis=-1,
        )
        
        y_val += (torch.tensor([1.0, 0.5]).to(device)) * failure

        y = pyro.sample(
            "y",
            dist.Normal(
                y_val,
                torch.tensor([0.1, 0.1]).to(device),
            ).to_event(1),
            obs=obs,
        )

    return y



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y, w = [
        t.cpu() for t in
        generate_two_moons_data_hierarchical(1000, device)
    ]
     
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
