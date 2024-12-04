import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from acms.pyro_models.pyro_ssm import StateSpaceModel

class BootstrapParticleFilter:
    def __init__(self, model: StateSpaceModel, num_particles: int):
        self.model = model
        self.num_particles = num_particles
    
    def resample_particles(self, particles: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Resample particles based on their weights."""
        indices = torch.multinomial(weights, self.num_particles, replacement=True)
        return particles[indices]

    def propagate_particles(self, particles: torch.Tensor) -> torch.Tensor:
        """Propagate particles through the process model."""
        propagated_particles = []
        for i in range(self.num_particles):
            x = particles[i]
            new_x = self.model.get_next_state(x)
            propagated_particles.append(new_x)
        return torch.stack(propagated_particles)

    def compute_predicted_observations(self, particles: torch.Tensor) -> torch.Tensor:
        """Compute predicted observations for each particle."""
        predicted_obs = []
        for i in range(self.num_particles):
            x = particles[i]
            pred_obs = self.model.get_observation(x)
            predicted_obs.append(pred_obs)
        return torch.stack(predicted_obs)

    def compute_likelihood(self, y_obs: torch.Tensor, predicted_obs: torch.Tensor) -> torch.Tensor:
        """Compute weights based on the likelihood of observations."""
        obs_diff = y_obs - predicted_obs
        # Assuming R is diagonal for simplicity
        inv_R_diag = 1.0 / torch.diag(self.model.R)
        exponent = -0.5 * torch.sum((obs_diff**2) * inv_R_diag, dim=1)
        normalization = torch.prod(torch.sqrt(2 * torch.pi * torch.diag(self.model.R)))
        obs_likelihoods = torch.exp(exponent) / normalization
        return obs_likelihoods

    def estimate_state(self, particles: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate state by weighted average of particles."""
        x_est = torch.sum(particles * weights.unsqueeze(1), dim=0)
        # Compute standard deviation over particles
        diff = particles - x_est
        x_std = torch.sqrt(torch.sum(weights.unsqueeze(1) * diff**2, dim=0))
        return x_est, x_std
    
    def filter(self, y_obs: torch.Tensor):
        T = y_obs.shape[0]
        state_dim = self.model.state_dim
        obs_dim = self.model.obs_dim

        # Initialize particles and weights
        particles = torch.zeros(T, self.num_particles, state_dim)
        weights = torch.zeros(T, self.num_particles)
        weights[0] = 1.0 / self.num_particles  # Uniform initial weights

        # Sample initial particles from the prior
        particles[0] = self.model.sample_from_prior(self.num_particles)

        # Keep track of predicted observations
        predicted_observations = torch.zeros(T, self.num_particles, obs_dim)

        for t in range(1, T):
            # Resample particles based on previous weights
            resampled_particles = self.resample_particles(particles[t - 1], weights[t - 1])

            # Propagate particles through the process model
            particles[t] = self.propagate_particles(resampled_particles)

            # Compute predicted observations
            predicted_observations[t] = self.compute_predicted_observations(particles[t])

            # Compute weights based on the likelihood of observations
            weights[t] = self.compute_likelihood(y_obs[t], predicted_observations[t])

            # Normalize weights
            weights_sum = weights[t].sum()
            if weights_sum > 0:
                weights[t] /= weights_sum
            else:
                weights[t] = 1.0 / self.num_particles  # Avoid division by zero

        # Estimate states
        x_est = torch.zeros(T, state_dim)
        x_std = torch.zeros(T, state_dim)
        for t in range(T):
            x_est[t], x_std[t] = self.estimate_state(particles[t], weights[t])

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
bpf = BootstrapParticleFilter(model, num_particles)
x_est, x_std, particles, weights = bpf.filter(observations)

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
