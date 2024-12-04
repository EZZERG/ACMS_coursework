import torch
import numpy as np
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import pyro
from pyro.infer import SMCFilter

from acms.pyro_models.pyro_smc_model import PyroStateSpaceModel, PyroStateSpaceModel_Guide, generate_data
from acms.pyro_models.pyro_ssm import StateSpaceModel

class SMCExperiment:
    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        num_particles: int,
        dt: float = 0.01,
        nonlinearity: str = "quadratic",
        damping: float = 0.99,
        A_scale: float = 0.1,
        C_scale: float = 0.1,
        Q_terms_scale: float = 0.01,
        Q_scale: float = 0.01,
        R_scale: float = 0.01,
        nonlinearity_scale: float = 1.0,
    ):
        # Initialize state space model
        self.ssm = StateSpaceModel(
            state_dim=state_dim,
            obs_dim=obs_dim,
            dt=dt,
            nonlinearity=nonlinearity,
            damping=damping,
            A_scale=A_scale,
            C_scale=C_scale,
            Q_terms_scale=Q_terms_scale,
            Q_scale=Q_scale,
            R_scale=R_scale,
            nonlinearity_scale=nonlinearity_scale,
        )
        
        self.num_particles = num_particles
        # Create pyro model and guide
        self.model = PyroStateSpaceModel(self.ssm)
        self.guide = PyroStateSpaceModel_Guide(self.model)
        # Initialize SMC
        self.smc_filter = SMCFilter(self.model, self.guide, num_particles=num_particles, max_plate_nesting=0)

    def _compute_effective_sample_size(self, log_weights: torch.Tensor) -> float:
        """Compute the Effective Sample Size (ESS) from log weights."""
        weights = torch.exp(log_weights - torch.logsumexp(log_weights, dim=0))
        return 1.0 / torch.sum(weights ** 2)

    def _compute_posterior_stats(self, empirical: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and variance of the posterior distribution."""
        return empirical['z'].mean, empirical['z'].variance

    def _compute_rmse(self, true_states: torch.Tensor, estimated_states: torch.Tensor) -> float:
        """Compute Root Mean Square Error between true and estimated states."""
        return torch.sqrt(torch.mean((true_states - estimated_states) ** 2)).item()

    def run_experiment(self, T: int, save_plot: bool = True, plot_name: str = None) -> Dict[str, Any]:
        # Generate data from the state space model
        states, observations = generate_data(self.ssm, T)
        
        # Initialize estimates storage
        estimates = torch.zeros(T + 1, self.ssm.state_dim)
        variances = torch.zeros(T + 1, self.ssm.state_dim)
        log_weights = []
        
        # Initialize filter
        self.smc_filter.init(initial=torch.zeros(self.ssm.state_dim))
        empirical = self.smc_filter.get_empirical()
        estimates[0] = empirical['z'].mean
        variances[0] = empirical['z'].variance
        
        # Run filter
        for t in range(T):
            self.smc_filter.step(observations[t + 1])
            empirical = self.smc_filter.get_empirical()
            estimates[t + 1] = empirical['z'].mean
            variances[t + 1] = empirical['z'].variance
            # Store log weights for metrics
            log_weights.append(empirical['z'].log_weights)

        # Compute metrics
        final_log_weights = log_weights[-1]
        ess = self._compute_effective_sample_size(final_log_weights)
        rmse = self._compute_rmse(states, estimates)
        log_likelihood = torch.logsumexp(final_log_weights, dim=0).item()

        if save_plot:
            self._plot_trajectories(states, observations, estimates, variances, plot_name)

        # Return results dictionary
        results = {
            'true_states': states,
            'observations': observations,
            'filtered_states': estimates,
            'posterior_variance': variances,
            'metrics': {
                'effective_sample_size': ess.item(),
                'rmse': rmse,
                'log_likelihood': log_likelihood
            }
        }
        
        return results

    def _plot_trajectories(self, states: torch.Tensor, observations: torch.Tensor, 
                          estimates: torch.Tensor, variances: torch.Tensor,
                          plot_name: str = None):
        """Plot the true states, observations, and filtered states with confidence intervals."""
        time = range(states.shape[0])
        fig, axs = plt.subplots(2, 1, figsize=(14, 10))

        for dim in range(2):  # Plot first two dimensions
            axs[dim].plot(time, states[:, dim].numpy(), label=f'True State {dim}', linestyle='--')
            axs[dim].plot(time, estimates[:, dim].numpy(), label=f'Estimated State {dim}')
            std = torch.sqrt(variances[:, dim])
            axs[dim].fill_between(
                time,
                (estimates[:, dim] - 2 * std).numpy(),
                (estimates[:, dim] + 2 * std).numpy(),
                color='red',
                alpha=0.2,
                label='Confidence Interval (±2 std)' if dim == 0 else ""
            )
            axs[dim].scatter(time, observations[:, dim].numpy(), 
                           color='green', alpha=0.5, s=20, label='Observations')
            axs[dim].set_xlabel('Time')
            axs[dim].set_ylabel(f'State Dimension {dim}')
            axs[dim].legend()

        plt.tight_layout()
        if plot_name:
            plt.savefig(plot_name)
        else:
            plt.savefig('smc_filter_trajectories.png')
        plt.close()

if __name__ == "__main__":
    experiment = SMCExperiment(state_dim=2, obs_dim=2, num_particles=100)
    results = experiment.run_experiment(T=100, save_plot=True, plot_name='smc_filter_trajectories.png')
    print('results:', results)
