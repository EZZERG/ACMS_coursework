import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from acms.experiment.experiment import Experiment

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
    # Fix random seed for reproducibility
    np.random.seed(42)

    # Example usage of ExperimentScheduler
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
