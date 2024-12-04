import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal
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
    posterior_mean_bias: np.ndarray  # Add bias metric
    posterior_mean_std: np.ndarray   # Add standard deviation metric
    posterior_mean_mse: np.ndarray   # Add MSE metric

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
                f"exp_N{n_particles}_dx{state_dim}_dy{obs_dim}"
            )
            os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.true_states = None
        self.state_estimates = None
        self.metrics = None

    def plot_tracking_results(self, dimensions=[0, 1]):
        """Plot particle filter tracking results for specified dimensions"""
        if not self.create_plots:
            return
            
        time = np.arange(len(self.true_states))
        
        # Time series plot
        fig, ax = plt.subplots(len(dimensions), 1, figsize=(10, 4 * len(dimensions)))
        if len(dimensions) == 1:
            ax = [ax]

        for i, dim in enumerate(dimensions):
            ax[i].plot(time, self.true_states[:, dim], "k-", label=f"True State {dim+1}")
            ax[i].plot(
                time, self.state_estimates[:, dim], "r--", label=f"Estimated State {dim+1}"
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
        
        if self.create_plots:
            self.plot_tracking_results()
            
        self.metrics = self._compute_bpf_metrics(pf, self.true_states, filter_results)
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
class ExperimentScheduler:
    def __init__(
        self,
        n_particles_list: List[int],
        state_dims_list: List[int],
        obs_dims_list: List[int],
        n_trials: int = 10,
        n_steps: int = 100,
    ):
        self.n_particles_list = n_particles_list
        self.state_dims_list = state_dims_list
        self.obs_dims_list = obs_dims_list
        self.n_trials = n_trials
        self.n_steps = n_steps
        self.results_df = None
        self.trajectories = {}
        self.last_norm_constants = {}

    def run(self):
        results = []
        
        for d_x in self.state_dims_list:
            for d_y in self.obs_dims_list:
                for N in self.n_particles_list:
                    print(f'Running experiment for d_x={d_x}, d_y={d_y}, N={N}')
                    
                    last_constants = []
                    
                    for trial in range(self.n_trials):
                        experiment = Experiment(
                            n_particles=N,
                            state_dim=d_x,
                            obs_dim=d_y,
                            n_steps=self.n_steps
                        )
                        metrics = experiment.run()
                        
                        last_norm_const = experiment.metrics.temp_norm_const_var
                        last_constants.append(last_norm_const)
                        
                        # Extend the results dictionary with new metrics
                        results.append({
                            'state_dim': d_x,
                            'obs_dim': d_y,
                            'n_particles': N,
                            'trial': trial,
                            'rmse': metrics.rmse,
                            'mae': metrics.mae,
                            'temp_norm_const_var': metrics.temp_norm_const_var,
                            'filter_mean_var': np.mean(metrics.filter_mean_variance),
                            'ess': metrics.effective_sample_size,
                            'last_norm_const': last_norm_const,
                            # Add new posterior mean statistics
                            'posterior_mean_bias': np.mean(np.abs(metrics.posterior_mean_bias)),
                            'posterior_mean_std': np.mean(metrics.posterior_mean_std),
                            'posterior_mean_mse': np.mean(metrics.posterior_mean_mse)
                        })
                        
                        if trial == 0:
                            self.trajectories[(d_x, d_y, N)] = (
                                experiment.true_states,
                                experiment.state_estimates
                            )
                    
                    self.last_norm_constants[(d_x, d_y, N)] = last_constants
        
        self.results_df = pd.DataFrame(results)

    def plot_results(self, save_dir: str = None):
        if self.results_df is None:
            raise ValueError("No results available. Run experiment first.")
            
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # Original plots
        self._plot_rmse(axes[0, 0])
        self._plot_ess(axes[0, 1])
        self._plot_norm_const_var(axes[1, 0])
        self._plot_filter_mean_var(axes[1, 1])
        
        # New posterior mean plots
        self._plot_posterior_mean_bias(axes[2, 0])
        self._plot_posterior_mean_std(axes[2, 1])
        
        plt.tight_layout()
        if save_dir:
            # Save individual plots
            plt.figure()
            self._plot_rmse(plt.gca())
            plt.savefig(os.path.join(save_dir, "rmse.png"))
            plt.close()
            
            plt.figure()
            self._plot_ess(plt.gca())
            plt.savefig(os.path.join(save_dir, "ess.png"))
            plt.close()
            
            plt.figure()
            self._plot_norm_const_var(plt.gca())
            plt.savefig(os.path.join(save_dir, "norm_const_var.png"))
            plt.close()
            
            plt.figure()
            self._plot_filter_mean_var(plt.gca())
            plt.savefig(os.path.join(save_dir, "filter_mean_var.png"))
            plt.close()
            
            plt.figure()
            self._plot_posterior_mean_bias(plt.gca())
            plt.savefig(os.path.join(save_dir, "posterior_mean_bias.png"))
            plt.close()
            
            plt.figure()
            self._plot_posterior_mean_std(plt.gca())
            plt.savefig(os.path.join(save_dir, "posterior_mean_std.png"))
            plt.close()
            
            # Save combined plot
            fig.savefig(os.path.join(save_dir, "performance_metrics.png"))
            plt.close(fig)
        else:
            plt.show()

    def _plot_rmse(self, ax):
        for d_x in self.state_dims_list:
            data = self.results_df[self.results_df['state_dim'] == d_x]
            mean_rmse = data.groupby('n_particles')['rmse'].mean()
            ax.plot(mean_rmse.index, mean_rmse, 'o-', label=f'd_x={d_x}')
        ax.set_xlabel('Number of particles')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE vs Particles')
        ax.legend()

    def _plot_ess(self, ax):
        for d_x in self.state_dims_list:
            data = self.results_df[self.results_df['state_dim'] == d_x]
            mean_ess = data.groupby('n_particles')['ess'].mean()
            ax.plot(mean_ess.index, mean_ess, 'o-', label=f'd_x={d_x}')
        ax.set_xlabel('Number of particles')
        ax.set_ylabel('ESS')
        ax.set_title('Effective Sample Size vs Particles')
        ax.legend()

    def _plot_norm_const_var(self, ax):
        for d_x in self.state_dims_list:
            data = self.results_df[self.results_df['state_dim'] == d_x]
            mean_var = data.groupby('n_particles')['temp_norm_const_var'].mean()
            ax.plot(mean_var.index, mean_var, 'o-', label=f'd_x={d_x}')
        ax.set_xlabel('Number of particles')
        ax.set_ylabel('Normalizing Constant Variance')
        ax.set_title('Normalizing Constant Variance vs Particles')
        ax.legend()

    def _plot_filter_mean_var(self, ax):
        for d_x in self.state_dims_list:
            data = self.results_df[self.results_df['state_dim'] == d_x]
            mean_var = data.groupby('n_particles')['filter_mean_var'].mean()
            ax.plot(mean_var.index, mean_var, 'o-', label=f'd_x={d_x}')
        ax.set_xlabel('Number of particles')
        ax.set_ylabel('Filter Mean Variance')
        ax.set_title('Filter Mean Variance vs Particles')
        ax.legend()

    def _plot_posterior_mean_bias(self, ax):
        for d_x in self.state_dims_list:
            data = self.results_df[self.results_df['state_dim'] == d_x]
            mean_bias = data.groupby('n_particles')['posterior_mean_bias'].mean()
            ax.plot(mean_bias.index, mean_bias, 'o-', label=f'd_x={d_x}')
        ax.set_xlabel('Number of particles')
        ax.set_ylabel('Posterior Mean Bias')
        ax.set_title('Average Absolute Posterior Mean Bias vs Particles')
        ax.legend()

    def _plot_posterior_mean_std(self, ax):
        for d_x in self.state_dims_list:
            data = self.results_df[self.results_df['state_dim'] == d_x]
            mean_std = data.groupby('n_particles')['posterior_mean_std'].mean()
            ax.plot(mean_std.index, mean_std, 'o-', label=f'd_x={d_x}')
        ax.set_xlabel('Number of particles')
        ax.set_ylabel('Posterior Mean Std')
        ax.set_title('Average Posterior Mean Standard Deviation vs Particles')
        ax.legend()

    def print_summary(self):
        if self.results_df is None:
            raise ValueError("No results available. Run experiment first.")
            
        summary = self.results_df.groupby(['state_dim', 'obs_dim', 'n_particles']).agg({
            'rmse': ['mean', 'std'],
            'mae': ['mean', 'std'],
            'temp_norm_const_var': 'mean',
            'filter_mean_var': 'mean',
            'ess': 'mean',
            'last_norm_const': ['mean', 'std'],
            'posterior_mean_bias': ['mean', 'std'],
            'posterior_mean_std': ['mean', 'std'],
            'posterior_mean_mse': ['mean', 'std']
        }).round(4)
        
        print("\nExperiment Summary:")
        print(summary)
    
    def save_summary(self, save_dir: str):
        if self.results_df is None:
            raise ValueError("No results available. Run experiment first.")
            
        summary = self.results_df.groupby(['state_dim', 'obs_dim', 'n_particles']).agg({
            'rmse': ['mean', 'std'],
            'mae': ['mean', 'std'],
            'temp_norm_const_var': 'mean',
            'filter_mean_var': 'mean',
            'ess': 'mean',
            'last_norm_const': ['mean', 'std'],
            'posterior_mean_bias': ['mean', 'std'],
            'posterior_mean_std': ['mean', 'std'],
            'posterior_mean_mse': ['mean', 'std']
        }).round(4)
        
        summary.to_csv(f"{save_dir}/summary.csv")

if __name__ == "__main__":
    # Example usage
    scheduler = ExperimentScheduler(
        n_particles_list=[100, 500, 1000],
        state_dims_list=[2, 4, 6],
        obs_dims_list=[2, 4],
        n_trials=2
    )
    
    # Create results directory
    results_dir = "experiment_results"
    os.makedirs(results_dir, exist_ok=True)
    
    scheduler.run()
    scheduler.print_summary()
    scheduler.save_summary(save_dir=results_dir)
    scheduler.plot_results(save_dir=results_dir)
