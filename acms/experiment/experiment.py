import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Literal
from acms.bootstrap_particle_filter.bootstrap_particle_filter import ParticleFilter
from acms.state_space_model.state_space_model import StateSpaceModel

@dataclass
class BPFMetrics:
    rmse: float
    mae: float
    state_rmse: np.ndarray
    temp_norm_const_var: float
    filter_mean_variance: np.ndarray
    effective_sample_size: float
    posterior_mean_bias: np.ndarray
    posterior_mean_std: np.ndarray
    posterior_mean_mse: np.ndarray

class Experiment:
    def __init__(
        self,
        n_particles: int,
        state_dim: int,
        obs_dim: int,
        n_steps: int = 100,
        create_plots: bool = False,
        save_dir: str = None,
        resampling_method: Literal["systematic", "multinomial"] = "multinomial",
        ess_resampling: bool = False,
    ):
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_steps = n_steps
        self.create_plots = create_plots
        self.save_dir = save_dir
        self.resampling_method = resampling_method
        self.ess_resampling = ess_resampling
        
        if self.create_plots and self.save_dir:
            self.experiment_dir = os.path.join(
                self.save_dir,
                f"exp_N{n_particles}_dx{state_dim}_dy{obs_dim}_n_steps{n_steps}"
            )
            os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.true_states = None
        self.state_estimates = None
        self.metrics = None

    def plot_tracking_results(self, dimensions=[0, 1]):
        """Plot particle filter tracking results for specified dimensions with confidence bands"""
        if not self.create_plots:
            return
            
        time = np.arange(len(self.true_states))
        
        # Time series plot
        fig, ax = plt.subplots(len(dimensions), 1, figsize=(10, 4 * len(dimensions)))
        if len(dimensions) == 1:
            ax = [ax]

        for i, dim in enumerate(dimensions):
            # Plot true and estimated states
            ax[i].plot(time, self.true_states[:, dim], "k-", label=f"True State {dim+1}")
            ax[i].plot(
                time, self.state_estimates[:, dim], "r--", label=f"Estimated State {dim+1}"
            )
            
            # Add confidence bands using standard deviation
            std_dev = self.metrics.posterior_mean_std[dim]
            ax[i].fill_between(
                time,
                self.state_estimates[:, dim] - 2*std_dev,
                self.state_estimates[:, dim] + 2*std_dev,
                color='r',
                alpha=0.2,
                label='95% Confidence'
            )
            
            ax[i].set_xlabel("Time")
            ax[i].set_ylabel(f"State {dim+1}")
            ax[i].legend()

        plt.tight_layout()
        if self.save_dir:
            plt.savefig(os.path.join(self.experiment_dir, "tracking.png"))
            plt.close()
        else:
            plt.show()

        # Phase plot if multiple dimensions
        if len(dimensions) >= 2:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(
                self.true_states[:, dimensions[0]],
                self.true_states[:, dimensions[1]],
                "k-",
                label="True Trajectory",
            )
            ax.plot(
                self.state_estimates[:, dimensions[0]],
                self.state_estimates[:, dimensions[1]],
                "r--",
                label="Estimated Trajectory",
            )
            ax.set_xlabel(f"State {dimensions[0]+1}")
            ax.set_ylabel(f"State {dimensions[1]+1}")
            ax.legend()

            if self.save_dir:
                plt.savefig(os.path.join(self.experiment_dir, "phase.png"))
                plt.close()
            else:
                plt.show()

    def run(self) -> BPFMetrics:
        model = StateSpaceModel(
            state_dim=self.state_dim,
            obs_dim=self.obs_dim,
            damping=0.99,
            nonlinearity="quadratic",
            nonlinearity_scale=1.0
        )
        
        self.true_states, observations = model.simulate(steps=self.n_steps)
        pf = ParticleFilter(
            model, 
            n_particles=self.n_particles,
            resampling_method=self.resampling_method,
            ess_resampling=self.ess_resampling
        )
        filter_results = pf.filter(observations)
        self.state_estimates = filter_results["state_estimates"]
        
        self.metrics = self._compute_bpf_metrics(pf, self.true_states, filter_results)

        if self.create_plots:
            self.plot_tracking_results()
            
        return self.metrics

    def _compute_bpf_metrics(
        self,
        pf: ParticleFilter,
        true_states: np.ndarray,
        filter_results: Dict[str, np.ndarray],
        n_variance_samples: int = 100
    ) -> BPFMetrics:
        state_estimates = filter_results["state_estimates"]
        particles = filter_results["particles"]
        weights = filter_results["weights"]
        normalizing_constants = filter_results["normalizing_constants"]
    
        # Original metrics
        rmse = np.sqrt(np.mean((true_states - state_estimates) ** 2))
        mae = np.mean(np.abs(true_states - state_estimates))
        state_rmse = np.sqrt(np.mean((true_states - state_estimates) ** 2, axis=0))
        
        norm_constant_var = np.var(normalizing_constants)
        
        filter_means = np.array([
            np.sum(particles * weights[:, np.newaxis], axis=0)
            for _ in range(n_variance_samples)
        ])
        filter_mean_var = np.var(filter_means, axis=0)
        
        # New posterior mean statistics
        posterior_mean = np.mean(state_estimates, axis=0)
        posterior_mean_bias = posterior_mean - np.mean(true_states, axis=0)
        posterior_mean_std = np.std(state_estimates, axis=0)
        posterior_mean_mse = np.mean((state_estimates - true_states) ** 2, axis=0)
        
        ess = 1.0 / np.sum(weights ** 2)
        
        return BPFMetrics(
            rmse=rmse,
            mae=mae,
            state_rmse=state_rmse,
            temp_norm_const_var=norm_constant_var,
            filter_mean_variance=filter_mean_var,
            effective_sample_size=ess,
            posterior_mean_bias=posterior_mean_bias,
            posterior_mean_std=posterior_mean_std,
            posterior_mean_mse=posterior_mean_mse
        )

if __name__ == "__main__":
    # Example of a single experiment with confidence bands
    print("\nRunning single experiment with confidence bands...")
    single_experiment = Experiment(
        n_particles=1000,
        state_dim=4,
        obs_dim=2,
        n_steps=100,
        create_plots=True,
        save_dir="experiment_results/single_experiment"
    )
    metrics = single_experiment.run()
    print(f"Single experiment RMSE: {metrics.rmse:.4f}")
    print(f"Single experiment ESS: {metrics.effective_sample_size:.4f}")
