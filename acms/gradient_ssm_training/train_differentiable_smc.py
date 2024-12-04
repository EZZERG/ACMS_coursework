from acms.gradient_ssm_training.differebtiable_smc import DifferentiableSMC, DifferentiableSMCTrainer
from acms.pyro_models.pyro_ssm import StateSpaceModel
from acms.pyro_models.pyro_smc_model import generate_data
import pandas as pd
import torch
import os

def run_experiment(state_dim, obs_dim, n_particles, num_timesteps=100):
    """Run a single experiment with given configuration."""
    # Generate data from true model
    true_model = StateSpaceModel(state_dim=state_dim, obs_dim=obs_dim)
    _, observations = generate_data(true_model, num_timesteps=num_timesteps)
    
    # Create differentiable SMC model
    diff_smc = DifferentiableSMC(
        state_dim=state_dim,
        obs_dim=obs_dim,
        num_particles=n_particles
    )
    
    # Create trainer and train
    trainer = DifferentiableSMCTrainer(
        model=diff_smc,
        learning_rate=0.001,
        max_epochs=1000
    )
    
    # Train using observations
    history = trainer.train(observations)
    
    # Get final marginal loglikelihood (negative of final loss)
    final_loglik = -history['loss'][-1]
    
    # Plot training progress
    plot_filename = f'training_history_s{state_dim}_o{obs_dim}_p{n_particles}.png'
    trainer.plot_training_history(history, plot_filename)
    
    return final_loglik

def main():
    # Define configurations to test
    configs = [
        {'state_dim': 2, 'obs_dim': 2, 'n_particles': 100},
        {'state_dim': 2, 'obs_dim': 2, 'n_particles': 200},
        {'state_dim': 4, 'obs_dim': 2, 'n_particles': 100},
        {'state_dim': 4, 'obs_dim': 2, 'n_particles': 200},
    ]
    
    # Create results directory if it doesn't exist
    results_dir = 'results/smc_experiments'
    os.makedirs(results_dir, exist_ok=True)
    
    # Store results
    results = []
    
    # Run experiments for each configuration
    for config in configs:
        print(f"\nRunning experiment with config: {config}")
        
        # Create experiment directory
        exp_name = f"exp_s{config['state_dim']}_o{config['obs_dim']}_p{config['n_particles']}"
        exp_dir = os.path.join(results_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Change to experiment directory
        os.chdir(exp_dir)
        
        # Run experiment
        final_loglik = run_experiment(
            state_dim=config['state_dim'],
            obs_dim=config['obs_dim'],
            n_particles=config['n_particles']
        )
        
        # Store results
        result = {
            'state_dim': config['state_dim'],
            'obs_dim': config['obs_dim'],
            'n_particles': config['n_particles'],
            'marginal_loglik': final_loglik
        }
        results.append(result)
        
        # Change back to original directory
        os.chdir('../../..')
    
    # Create DataFrame and save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(results_dir, 'differentiable_experiment_results.csv'), index=False)
    print("\nFinal Results:")
    print(df)

if __name__ == "__main__":
    main()
