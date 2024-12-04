import torch
from typing import Optional, Tuple, Dict

class StateSpaceModel:
    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        dt: float = 0.01,
        nonlinearity: str = "quadratic",
        damping: float = 0.99,
        A_scale: float = 0.1,
        C_scale: float = 0.1,
        Q_terms_scale: float = 0.01,
        Q_scale: float = 0.01,
        R_scale: float = 0.01,
        nonlinearity_scale: float = 1.0,
        unscaled_matrices: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.dt = dt
        self.nonlinearity = nonlinearity
        self.damping = damping  # For system stability
        self.nonlinearity_scale = nonlinearity_scale

        # Store scales for later unscaling
        self.A_scale = A_scale
        self.C_scale = C_scale
        self.Q_terms_scale = Q_terms_scale
        self.Q_scale = Q_scale
        self.R_scale = R_scale

        # Initialize matrices either from provided unscaled matrices or sample them
        if unscaled_matrices is not None:
            self.A = unscaled_matrices['A'] * A_scale
            self.C = unscaled_matrices['C'] * C_scale
            self.Q_terms = unscaled_matrices['Q_terms'] * Q_terms_scale
            self.Q = unscaled_matrices['Q'] * Q_scale
            self.R = unscaled_matrices['R'] * R_scale
        else:
            self.A = torch.randn(state_dim, state_dim) * A_scale
            self.C = torch.randn(obs_dim, state_dim) * C_scale
            self.Q_terms = torch.randn(state_dim, state_dim, state_dim) * Q_terms_scale
            self.Q = torch.eye(state_dim) * Q_scale
            self.R = torch.eye(obs_dim) * R_scale

    def get_unscaled_matrices(self) -> Dict[str, torch.Tensor]:
        """Export the unscaled matrices for initialization of a new model."""
        return {
            'A': self.A / self.A_scale,
            'C': self.C / self.C_scale,
            'Q_terms': self.Q_terms / self.Q_terms_scale,
            'Q': self.Q / self.Q_scale,
            'R': self.R / self.R_scale
        }

    def sample_from_prior(self, num_samples: int = 1) -> torch.Tensor:
        """Sample initial states from the prior distribution."""
        return torch.randn(num_samples, self.state_dim)

    def nonlinear_dynamics(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute nonlinear dynamics. Handles both batched and unbatched inputs.
        x can be either (state_dim,) or (batch_size, state_dim)
        """
        if self.nonlinearity == "quadratic":
            # Add batch dimension if not present
            if x.dim() == 1:
                x = x.unsqueeze(0)
            batch_size = x.shape[0]
            
            # Initialize output tensor
            quad_terms = torch.zeros(batch_size, self.state_dim)
            
            # Compute quadratic terms for each batch element
            for b in range(batch_size):
                for i in range(self.state_dim):
                    quad_terms[b, i] = torch.sum(self.Q_terms[i] * torch.outer(x[b], x[b]))
            
            # Remove batch dimension if input was unbatched
            if batch_size == 1:
                quad_terms = quad_terms.squeeze(0)
                
            return quad_terms
        elif self.nonlinearity == "sine":
            return torch.sin(x)
        elif self.nonlinearity == "tanh":
            return torch.tanh(x)
        else:
            return torch.zeros_like(x)

    def get_next_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute next state. Handles both batched and unbatched inputs.
        x can be either (state_dim,) or (batch_size, state_dim)
        """
        # Add batch dimension if not present
        unbatched = x.dim() == 1
        if unbatched:
            x = x.unsqueeze(0)
        
        # Compute dynamics
        new_x = x + self.dt * (
            x @ self.A + self.nonlinearity_scale * self.nonlinear_dynamics(x)
        )
        new_x = new_x * self.damping
        
        # Add process noise
        batch_size = x.shape[0]
        noise = torch.distributions.MultivariateNormal(
            torch.zeros(self.state_dim), 
            self.Q
        ).sample((batch_size,))
        new_x = new_x + noise
        
        # Remove batch dimension if input was unbatched
        if unbatched:
            new_x = new_x.squeeze(0)
            
        return new_x

    def get_observation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute observation. Handles both batched and unbatched inputs.
        x can be either (state_dim,) or (batch_size, state_dim)
        """
        # Add batch dimension if not present
        unbatched = x.dim() == 1
        if unbatched:
            x = x.unsqueeze(0)
            
        # Compute observation
        obs = x @ self.C.t()
        
        # Add observation noise
        batch_size = x.shape[0]
        noise = torch.distributions.MultivariateNormal(
            torch.zeros(self.obs_dim), 
            self.R
        ).sample((batch_size,))
        obs = obs + noise
        
        # Remove batch dimension if input was unbatched
        if unbatched:
            obs = obs.squeeze(0)
            
        return obs

    def step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one step of the model. Handles both batched and unbatched inputs.
        x can be either (state_dim,) or (batch_size, state_dim)
        """
        new_x = self.get_next_state(x)
        obs = self.get_observation(new_x)
        return new_x, obs

    def simulate(
        self, steps: int, x0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x0 is None:
            x0 = torch.randn(self.state_dim)  # Initialize x0 with random values

        states = torch.zeros(steps + 1, self.state_dim)
        observations = torch.zeros(steps + 1, self.obs_dim)

        states[0] = x0
        observations[0] = self.get_observation(x0)

        for t in range(steps):
            states[t + 1], observations[t + 1] = self.step(states[t])

        return states, observations
