from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from acms.state_space_model.general_state_space_model import GeneralStateSpaceModel


class StateSpaceModel(GeneralStateSpaceModel):
    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        dt: float = 0.01,
        nonlinearity: Literal["quadratic", "sine", "tanh", ""] = "quadratic",
        damping: float = 0.99,
        A_scale: float = 0.1,
        C_scale: float = 0.1,
        Q_terms_scale: float = 0.01,
        Q_scale: float = 0.01,
        R_scale: float = 0.01,
        nonlinearity_scale: float = 1.0,
    ) -> None:
        super().__init__(state_dim, obs_dim, dt)
        self.nonlinearity = nonlinearity
        self.damping = damping  # add damping for stability of the system when simulating with high number of steps

        self.A = np.random.randn(state_dim, state_dim) * A_scale
        self.C = np.random.randn(obs_dim, state_dim) * C_scale
        self.Q_terms = np.random.randn(state_dim, state_dim, state_dim) * Q_terms_scale
        self.nonlinearity_scale = nonlinearity_scale

        self.Q = np.eye(state_dim) * Q_scale
        self.R = np.eye(obs_dim) * R_scale

    def nonlinear_dynamics(self, x: NDArray) -> NDArray:
        if self.nonlinearity == "quadratic":
            quad_terms = np.zeros(self.state_dim)
            for i in range(self.state_dim):
                quad_terms[i] = np.sum(self.Q_terms[i] * np.outer(x, x))
            return quad_terms
        elif self.nonlinearity == "sine":
            return np.sin(x)
        elif self.nonlinearity == "tanh":
            return np.tanh(x)
        else:
            return np.zeros(self.state_dim)

    def get_next_state(self, x: NDArray) -> NDArray:
        new_x = x + self.dt * (
            self.A @ x + self.nonlinearity_scale * self.nonlinear_dynamics(x)
        )
        new_x *= self.damping
        new_x += np.random.multivariate_normal(np.zeros(self.state_dim), self.Q)
        return new_x

    def get_observation(self, x: NDArray) -> NDArray:
        obs = self.C @ x
        obs += np.random.multivariate_normal(np.zeros(self.obs_dim), self.R)
        return obs

    def step(self, x: NDArray) -> Tuple[NDArray, NDArray]:
        new_x = self.get_next_state(x)
        obs = self.get_observation(new_x)
        return new_x, obs
    
    def get_likelihood(self, pred_obs: NDArray, observation: NDArray) -> float:
        diff = observation - pred_obs
        inv_R_diag = 1.0 / np.diag(self.R)  # Inverse of the diagonal elements
        exponent = -0.5 * np.sum((diff ** 2) * inv_R_diag)
        normalization = (2 * np.pi) ** (len(observation) / 2) * np.sqrt(np.prod(np.diag(self.R)))
        likelihood = np.exp(exponent) / normalization
        return likelihood

    def simulate(
        self, steps: int, x0: Optional[NDArray] = None
    ) -> Tuple[NDArray, NDArray]:
        if x0 is None:
            x0 = np.random.randn(self.state_dim)  # Initialize x0 with random values

        states = np.zeros((steps + 1, self.state_dim))
        observations = np.zeros((steps + 1, self.obs_dim))

        states[0] = x0
        observations[0] = self.C @ x0

        for t in range(steps):
            states[t + 1], observations[t + 1] = self.step(states[t])

        return states, observations


# Example usage:
if __name__ == "__main__":
    # Testing the implementation
    # Fix the random seed for reproducibility
    np.random.seed(42)

    # Run simulation
    model = StateSpaceModel(
        state_dim=4,
        obs_dim=2,
        damping=0.99,
        nonlinearity="quadratic",
        nonlinearity_scale=1,
    )
    states, observations = model.simulate(steps=10000)
    model.plot_simulation(
        states,
        observations,
        state_dims=[0, 1],
        obs_dims=[0],
        traj_state_dims=[0, 1],
        traj_obs_dims=[0, 1],
    )

    # Save the model
    model.save_model("state_space_model.pkl")

    # Load the model
    loaded_model = StateSpaceModel.load_model("state_space_model.pkl")
    loaded_states, loaded_observations = loaded_model.simulate(steps=100)
    loaded_model.plot_simulation(
        loaded_states,
        loaded_observations,
        state_dims=[0, 1],
        obs_dims=[0],
        traj_state_dims=[0, 1],
        traj_obs_dims=[0, 1],
    )
