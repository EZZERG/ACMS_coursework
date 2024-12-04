import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt

class DifferentiableSMC(nn.Module):
    """Differentiable Sequential Monte Carlo implementation."""
    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        num_particles: int,
        dt: float = 0.01,
        nonlinearity: str = "quadratic",
        damping: float = 0.99,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.num_particles = num_particles
        self.dt = dt
        self.nonlinearity = nonlinearity
        self.damping = damping

        # Learnable parameters
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)
        self.C = nn.Parameter(torch.randn(obs_dim, state_dim) * 0.1)
        self.Q_terms = nn.Parameter(torch.randn(state_dim, state_dim, state_dim) * 0.01)
        self._Q_chol = nn.Parameter(torch.eye(state_dim) * 0.1)
        self._R_chol = nn.Parameter(torch.eye(obs_dim) * 0.1)

    @property
    def Q(self) -> torch.Tensor:
        """Get positive definite Q matrix from its Cholesky factor."""
        return self._Q_chol @ self._Q_chol.t()

    @property
    def R(self) -> torch.Tensor:
        """Get positive definite R matrix from its Cholesky factor."""
        return self._R_chol @ self._R_chol.t()

    def nonlinear_dynamics(self, x: torch.Tensor) -> torch.Tensor:
        """Compute nonlinear dynamics with batched operations."""
        if self.nonlinearity == "quadratic":
            batch_size = x.shape[0]
            x_expanded = x.unsqueeze(-1)  # [batch, state_dim, 1]
            outer_products = x_expanded @ x_expanded.transpose(-2, -1)  # [batch, state_dim, state_dim]
            quad_terms = torch.einsum('ijk,bjk->bi', self.Q_terms, outer_products)
            return quad_terms
        elif self.nonlinearity == "sine":
            return torch.sin(x)
        elif self.nonlinearity == "tanh":
            return torch.tanh(x)
        else:
            return torch.zeros_like(x)

    def transition_prob(self, prev_state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """Compute transition probability p(x_t|x_{t-1})."""
        mean = prev_state + self.dt * (
            prev_state @ self.A.t() + self.nonlinear_dynamics(prev_state)
        ) * self.damping
        
        dist = torch.distributions.MultivariateNormal(mean, self.Q)
        return dist.log_prob(next_state)

    def observation_prob(self, state: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Compute observation probability p(y_t|x_t)."""
        mean = state @ self.C.t()
        dist = torch.distributions.MultivariateNormal(mean, self.R)
        return dist.log_prob(obs)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Run differentiable particle filter and compute marginal likelihood.
        
        Args:
            observations: [T+1, obs_dim] tensor of observations
            
        Returns:
            log_marginal_likelihood: scalar tensor
        """
        T = observations.shape[0] - 1
        device = observations.device
        
        # Initialize particles and weights
        particles = torch.randn(self.num_particles, self.state_dim, device=device)
        log_weights = torch.zeros(self.num_particles, device=device)
        
        # Initial observation likelihood
        log_weights = self.observation_prob(particles, observations[0])
        
        # Storage for marginal likelihood
        log_marginal_likelihood = torch.zeros(1, device=device)
        
        # SMC loop
        for t in range(T):
            # Resample (using differentiable relaxation)
            log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
            weights = torch.softmax(log_weights, dim=0)
            
            # Compute ancestor indices using Gumbel-Softmax trick for differentiability
            temperature = 0.5
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(weights) + 1e-10) + 1e-10)
            ancestor_probs = torch.softmax((log_weights + gumbel_noise) / temperature, dim=0)
            
            # Resample particles
            resampled_particles = torch.sum(
                ancestor_probs.unsqueeze(-1) * particles.unsqueeze(0),
                dim=1
            )
            
            # Propagate particles
            noise = torch.randn_like(particles) @ self._Q_chol.t()
            mean = resampled_particles + self.dt * (
                resampled_particles @ self.A.t() + 
                self.nonlinear_dynamics(resampled_particles)
            ) * self.damping
            particles = mean + noise
            
            # Update weights
            log_weights = self.observation_prob(particles, observations[t + 1])
            
            # Accumulate marginal likelihood
            log_marginal_likelihood = log_marginal_likelihood + torch.logsumexp(log_weights, dim=0)
        
        return log_marginal_likelihood

class DifferentiableSMCTrainer:
    """Trainer for differentiable SMC parameter estimation."""
    def __init__(
        self,
        model: DifferentiableSMC,
        learning_rate: float = 0.001,
        max_epochs: int = 1000,
    ):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.max_epochs = max_epochs
        
    def train(self, observations: torch.Tensor) -> Dict[str, List[float]]:
        """Train the model using gradient descent on the negative log marginal likelihood."""
        history = {
            'loss': [],
            'grad_norm': []
        }
        
        best_loss = float('inf')
        patience = 100
        patience_counter = 0
        
        pbar = tqdm(range(self.max_epochs))
        for epoch in pbar:
            self.optimizer.zero_grad()
            
            # Forward pass: compute negative log marginal likelihood
            log_marginal_likelihood = self.model(observations)
            loss = -log_marginal_likelihood
            
            # Backward pass
            loss.backward()
            
            # Compute gradient norm and clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Record metrics
            current_loss = loss.item()
            history['loss'].append(current_loss)
            history['grad_norm'].append(grad_norm.item())
            
            # Update progress bar
            pbar.set_description(f"Loss: {current_loss:.4f}")
            
            # Early stopping check
            if current_loss < best_loss - 1e-4:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
        return history

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        """Plot training metrics."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot loss
        ax1.plot(history['loss'])
        ax1.set_ylabel('Negative Log Marginal Likelihood')
        ax1.set_xlabel('Epoch')
        ax1.set_title('Training Loss')
        
        # Plot gradient norm
        ax2.plot(history['grad_norm'])
        ax2.set_ylabel('Gradient Norm')
        ax2.set_xlabel('Epoch')
        ax2.set_title('Gradient Norm During Training')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
