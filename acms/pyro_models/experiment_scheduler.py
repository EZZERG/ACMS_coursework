import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict, Any
from itertools import product
from acms.pyro_models.smc_experiment import SMCExperiment

class ExperimentScheduler:
    def __init__(
        self,
        state_dims: List[int],
        observation_dims: List[int],
        n_particles_list: List[int],
        n_steps: int,
        n_runs: int = 3,
        save_dir: str = "results/smc_experiments",
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
        """
        Initialize the ExperimentScheduler.
        
        Args:
            state_dims: List of state dimensions to test
            observation_dims: List of observation dimensions to test
            n_particles_list: List of particle counts to test
            n_steps: Number of time steps for each experiment
            n_runs: Number of runs per configuration
            save_dir: Directory to save results and plots
            Additional parameters are passed to SMCExperiment
        """
        self.state_dims = state_dims
        self.observation_dims = observation_dims
        self.n_particles_list = n_particles_list
        self.n_steps = n_steps
        self.n_runs = n_runs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Store SMCExperiment parameters
        self.experiment_params = {
            'dt': dt,
            'nonlinearity': nonlinearity,
            'damping': damping,
            'A_scale': A_scale,
            'C_scale': C_scale,
            'Q_terms_scale': Q_terms_scale,
            'Q_scale': Q_scale,
            'R_scale': R_scale,
            'nonlinearity_scale': nonlinearity_scale,
        }
        
        # Initialize results DataFrame
        self.results_df = pd.DataFrame()

    def _get_experiment_name(self, state_dim: int, obs_dim: int, n_particles: int) -> str:
        """Generate a unique name for the experiment configuration."""
        return f"exp_s{state_dim}_o{obs_dim}_p{n_particles}"

    def _get_plot_path(self, exp_name: str, run: int) -> Path:
        """Generate the path for saving plots."""
        plot_dir = self.save_dir / exp_name
        plot_dir.mkdir(exist_ok=True)
        return plot_dir / f"smc_filter_trajectories_d_x_{exp_name.split('_s')[1].split('_')[0]}_d_y_{exp_name.split('_o')[1].split('_')[0]}_N_{exp_name.split('_p')[1]}_T_{self.n_steps}_run_{run}.png"

    def run_experiments(self) -> pd.DataFrame:
        """Run all experiments with the specified configurations."""
        
        # Generate all combinations of parameters
        configs = list(product(self.state_dims, self.observation_dims, self.n_particles_list))
        
        for state_dim, obs_dim, n_particles in configs:
            exp_name = self._get_experiment_name(state_dim, obs_dim, n_particles)
            print(f"\nRunning experiments for configuration: {exp_name}")
            
            # Create experiment instance
            experiment = SMCExperiment(
                state_dim=state_dim,
                obs_dim=obs_dim,
                num_particles=n_particles,
                **self.experiment_params
            )
            
            # Run multiple times
            for run in range(self.n_runs):
                print(f"Run {run + 1}/{self.n_runs}")
                
                # Only save plot for first run
                save_plot = (run == 0)
                plot_path = self._get_plot_path(exp_name, run) if save_plot else None
                
                # Run experiment
                results = experiment.run_experiment(
                    T=self.n_steps,
                    save_plot=save_plot,
                    plot_name=str(plot_path) if plot_path else None
                )
                
                # Extract metrics and add to DataFrame
                metrics = results['metrics']
                run_data = {
                    'state_dim': state_dim,
                    'observation_dim': obs_dim,
                    'n_particles': n_particles,
                    'run': run,
                    'effective_sample_size': metrics['effective_sample_size'],
                    'rmse': metrics['rmse'],
                    'log_likelihood': metrics['log_likelihood']
                }
                
                self.results_df = pd.concat([self.results_df, pd.DataFrame([run_data])], 
                                         ignore_index=True)
        
        # Save results to CSV
        results_path = self.save_dir / "experiment_results.csv"
        self.results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}")
        
        return self.results_df

    def get_results(self) -> pd.DataFrame:
        """Return the results DataFrame."""
        return self.results_df


if __name__ == "__main__":
    # Example usage of ExperimentScheduler
    
    # Initialize scheduler with different configurations
    scheduler = ExperimentScheduler(
        state_dims=[2, 4],              # Test with 2 and 4 state dimensions
        observation_dims=[2],           # Keep observation dimension fixed at 2
        n_particles_list=[100, 200],    # Test with 100 and 200 particles
        n_steps=100,                    # Run each experiment for 100 steps
        n_runs=3,                       # Run each configuration 3 times
        save_dir="results/smc_experiments"  # Save results here
    )

    # Run all experiments
    results_df = scheduler.run_experiments()

    # Print summary statistics grouped by configuration
    print("\nSummary Statistics:")
    summary = results_df.groupby(['state_dim', 'observation_dim', 'n_particles']).agg({
        'effective_sample_size': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'log_likelihood': ['mean', 'std']
    }).round(4)
    print(summary)

