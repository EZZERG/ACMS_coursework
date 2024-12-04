import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# Set random seed for reproducibility
torch.manual_seed(42)


def bootstrap_particle_filter(
    model: StateSpaceModel, y_obs: torch.Tensor, num_particles: int
):
    T = y_obs.shape[0]
    state_dim = model.state_dim
    obs_dim = model.obs_dim

    # Initialize particles and weights
    particles = torch.zeros(T, num_particles, state_dim)
    weights = torch.zeros(T, num_particles)
    weights[0] = 1.0 / num_particles  # Uniform initial weights

    # Sample initial particles from the prior
    particles[0] = torch.randn(num_particles, state_dim)

    # Keep track of predicted observations
    predicted_observations = torch.zeros(T, num_particles, obs_dim)

    for t in range(1, T):
        # Resample particles based on previous weights
        indices = torch.multinomial(weights[t - 1], num_particles, replacement=True)
        resampled_particles = particles[t - 1, indices]

        # Propagate particles through the process model
        propagated_particles = []
        for i in range(num_particles):
            x = resampled_particles[i]
            new_x = model.get_next_state(x)
            propagated_particles.append(new_x)
        particles[t] = torch.stack(propagated_particles)

        # Compute predicted observations
        for i in range(num_particles):
            x = particles[t, i]
            pred_obs = model.C @ x
            predicted_observations[t, i] = pred_obs

        # Compute weights based on the likelihood of observations
        obs_diff = y_obs[t] - predicted_observations[t]
        # Assuming R is diagonal for simplicity
        inv_R_diag = 1.0 / torch.diag(model.R)
        exponent = -0.5 * torch.sum((obs_diff**2) * inv_R_diag, dim=1)
        normalization = torch.prod(torch.sqrt(2 * torch.pi * torch.diag(model.R)))
        obs_likelihoods = torch.exp(exponent) / normalization
        weights[t] = obs_likelihoods

        # Normalize weights
        weights_sum = weights[t].sum()
        if weights_sum > 0:
            weights[t] /= weights_sum
        else:
            weights[t] = 1.0 / num_particles  # Avoid division by zero

    # Estimate states by weighted average of particles
    x_est = torch.zeros(T, state_dim)
    x_std = torch.zeros(T, state_dim)
    for t in range(T):
        x_est[t] = torch.sum(particles[t] * weights[t].unsqueeze(1), dim=0)
        # Compute standard deviation over particles
        diff = particles[t] - x_est[t]
        x_std[t] = torch.sqrt(
            torch.sum(weights[t].unsqueeze(1) * diff**2, dim=0)
        )

    return x_est, x_std, particles, weights

# Parameters
state_dim = 4
obs_dim = 2
steps = 100
num_particles = 500

# Initialize the model
model = StateSpaceModel(
    state_dim=state_dim,
    obs_dim=obs_dim,
    damping=0.99,
    nonlinearity="quadratic",
    nonlinearity_scale=1.0,
)

# Simulate data
states, observations = model.simulate(steps=steps)

# Run the particle filter
x_est, x_std, particles, weights = bootstrap_particle_filter(
    model, observations, num_particles
)

# Plotting the results for the first two state dimensions
time = range(steps + 1)
fig, axs = plt.subplots(2, 1, figsize=(14, 10))

for dim in range(2):
    axs[dim].plot(time, states[:, dim].numpy(), label=f'True State {dim}', linestyle='--')
    axs[dim].plot(time, x_est[:, dim].numpy(), label=f'Estimated State {dim}')
    axs[dim].fill_between(
        time,
        (x_est[:, dim] - 2 * x_std[:, dim]).numpy(),
        (x_est[:, dim] + 2 * x_std[:, dim]).numpy(),
        color='red',
        alpha=0.2,
        label='Confidence Interval (±2 std)' if dim == 0 else ""
    )
    axs[dim].set_xlabel('Time')
    axs[dim].set_ylabel(f'State Dimension {dim}')
    axs[dim].legend()

plt.tight_layout()
plt.show()
