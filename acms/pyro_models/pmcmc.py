import torch
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt

from acms.pyro_models.smc_experiment import SMCExperiment

@dataclass
class PMCMCParameters:
    """Parameters to estimate in the state space model."""
    A_scale: float
    C_scale: float
    Q_scale: float
    R_scale: float
    nonlinearity_scale: float
    
    def to_vector(self) -> torch.Tensor:
        """Convert parameters to vector form."""
        return torch.tensor([
            self.A_scale,
            self.C_scale,
            self.Q_scale,
            self.R_scale,
            self.nonlinearity_scale
        ])
    
    @classmethod
    def from_vector(cls, vector: torch.Tensor) -> 'PMCMCParameters':
        """Create parameters from vector form."""
        return cls(
            A_scale=vector[0].item(),
            C_scale=vector[1].item(),
            Q_scale=vector[2].item(),
            R_scale=vector[3].item(),
            nonlinearity_scale=vector[4].item()
        )

class ParticleMCMC:
    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        num_particles: int,
        n_mcmc_steps: int = 1000,
        proposal_scale: float = 0.1,
        unscaled_matrices: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Initialize Particle MCMC for parameter estimation.
        
        Args:
            state_dim: Dimension of the state space
            obs_dim: Dimension of the observations
            num_particles: Number of particles for SMC
            n_mcmc_steps: Number of MCMC iterations
            proposal_scale: Scale of the random walk proposal
            unscaled_matrices: Dictionary of unscaled matrices from the true model
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.num_particles = num_particles
        self.n_mcmc_steps = n_mcmc_steps
        self.proposal_scale = proposal_scale
        self.unscaled_matrices = unscaled_matrices
        
    def _create_experiment(self, params: PMCMCParameters) -> SMCExperiment:
        """Create SMC experiment with given parameters."""
        return SMCExperiment(
            state_dim=self.state_dim,
            obs_dim=self.obs_dim,
            num_particles=self.num_particles,
            A_scale=params.A_scale,
            C_scale=params.C_scale,
            Q_scale=params.Q_scale,
            R_scale=params.R_scale,
            nonlinearity_scale=params.nonlinearity_scale,
            unscaled_matrices=self.unscaled_matrices
        )
    
    def _log_prior(self, params: PMCMCParameters) -> float:
        """Compute log prior probability of parameters."""
        # Using log-normal priors for scale parameters
        param_vector = params.to_vector()
        log_prior = torch.distributions.LogNormal(-2.0, 1.0).log_prob(param_vector).sum()
        return log_prior.item()
    
    def _propose_parameters(self, current_params: PMCMCParameters) -> PMCMCParameters:
        """Generate proposal parameters using random walk."""
        current_vector = current_params.to_vector()
        # Random walk proposal in log space
        log_proposal = torch.log(current_vector) + torch.randn_like(current_vector) * self.proposal_scale
        proposal_vector = torch.exp(log_proposal)
        return PMCMCParameters.from_vector(proposal_vector)
    
    def run_pmcmc(
        self,
        observations: torch.Tensor,
        initial_params: Optional[PMCMCParameters] = None
    ) -> Tuple[List[PMCMCParameters], List[float]]:
        """
        Run Particle MCMC algorithm.
        
        Args:
            observations: Observed data (T+1, obs_dim)
            initial_params: Initial parameter values (optional)
            
        Returns:
            parameters: List of parameter samples
            log_likelihoods: List of log-likelihood values
        """
        T = observations.shape[0] - 1
        
        # Initialize parameters if not provided
        if initial_params is None:
            initial_params = PMCMCParameters(
                A_scale=0.1,
                C_scale=0.1,
                Q_scale=0.01,
                R_scale=0.01,
                nonlinearity_scale=1.0
            )
        
        # Initialize storage
        parameters = [initial_params]
        log_likelihoods = []
        
        # Run initial SMC
        current_experiment = self._create_experiment(initial_params)
        current_results = current_experiment.run_experiment(T, save_plot=False)
        current_log_likelihood = current_results['metrics']['log_likelihood']
        current_log_prior = self._log_prior(initial_params)
        
        # MCMC loop
        for step in tqdm(range(self.n_mcmc_steps), desc="Running PMCMC"):
            # Propose new parameters
            proposal_params = self._propose_parameters(parameters[-1])
            proposal_log_prior = self._log_prior(proposal_params)
            
            # Run SMC with proposed parameters
            proposal_experiment = self._create_experiment(proposal_params)
            proposal_results = proposal_experiment.run_experiment(T, save_plot=False)
            proposal_log_likelihood = proposal_results['metrics']['log_likelihood']
            
            # Compute acceptance ratio
            log_acceptance_ratio = (
                proposal_log_likelihood + proposal_log_prior -
                current_log_likelihood - current_log_prior
            )
            
            # Accept/reject
            if torch.log(torch.rand(1)) < log_acceptance_ratio:
                parameters.append(proposal_params)
                log_likelihoods.append(proposal_log_likelihood)
                current_log_likelihood = proposal_log_likelihood
                current_log_prior = proposal_log_prior
            else:
                parameters.append(parameters[-1])
                log_likelihoods.append(current_log_likelihood)
        
        return parameters, log_likelihoods

    def analyze_results(
        self,
        parameters: List[PMCMCParameters],
        log_likelihoods: List[float],
        burnin: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze PMCMC results.
        
        Args:
            parameters: List of parameter samples
            log_likelihoods: List of log-likelihood values
            burnin: Number of initial samples to discard
            
        Returns:
            Dictionary containing analysis results
        """
        # Convert parameters to array form
        param_array = torch.stack([p.to_vector() for p in parameters[burnin:]])
        
        # Compute statistics
        mean_params = PMCMCParameters.from_vector(param_array.mean(dim=0))
        std_params = param_array.std(dim=0)
        
        # Compute acceptance rate
        n_unique = len(set(tuple(p.to_vector().numpy()) for p in parameters))
        acceptance_rate = n_unique / len(parameters)
        
        return {
            'mean_parameters': mean_params,
            'std_parameters': std_params,
            'acceptance_rate': acceptance_rate,
            'final_log_likelihood': log_likelihoods[-1]
        }

    def plot_traces(
        self,
        parameters: List[PMCMCParameters],
        log_likelihoods: List[float],
        true_params: Optional[PMCMCParameters] = None,
        save_path: Optional[str] = None
    ):
        """Plot MCMC traces and histograms."""
        param_names = ['A_scale', 'C_scale', 'Q_scale', 'R_scale', 'nonlinearity_scale']
        param_array = torch.stack([p.to_vector() for p in parameters])
        
        fig, axs = plt.subplots(len(param_names) + 1, 2, figsize=(15, 4 * (len(param_names) + 1)))
        
        # Plot parameter traces and histograms
        for i, name in enumerate(param_names):
            # Trace plot
            axs[i, 0].plot(param_array[:, i])
            axs[i, 0].set_title(f'{name} Trace')
            if true_params is not None:
                true_value = true_params.to_vector()[i]
                axs[i, 0].axhline(y=true_value, color='r', linestyle='--')
            
            # Histogram
            axs[i, 1].hist(param_array[:, i], bins=50, density=True)
            axs[i, 1].set_title(f'{name} Histogram')
            if true_params is not None:
                axs[i, 1].axvline(x=true_value, color='r', linestyle='--')
        
        # Plot log-likelihood
        axs[-1, 0].plot(log_likelihoods)
        axs[-1, 0].set_title('Log-likelihood Trace')
        axs[-1, 1].hist(log_likelihoods, bins=50, density=True)
        axs[-1, 1].set_title('Log-likelihood Histogram')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
