from typing import Literal, Dict, Tuple

import numpy as np
from tqdm import tqdm

from acms.state_space_model.general_state_space_model import GeneralStateSpaceModel


class ParticleFilter:
    def __init__(
        self,
        ssm: GeneralStateSpaceModel,
        n_particles: int = 100,
        resampling_method: Literal["systematic", "multinomial"] = "multinomial",
        ess_resampling: bool = False,
    ):
        self.ssm = ssm
        self.n_particles = n_particles
        self.resampling_method = resampling_method
        self.ess_resampling = ess_resampling

    def initialize_particles(self) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize particles and weights."""
        particles = np.random.randn(self.n_particles, self.ssm.state_dim)
        weights = np.ones(self.n_particles) / self.n_particles
        return particles, weights

    def predict(self, particles: np.ndarray) -> np.ndarray:
        """Propagate particles through state dynamics."""
        new_particles = np.array([self.ssm.get_next_state(p) for p in particles])
        return new_particles

    def update(self, particles: np.ndarray, weights: np.ndarray, observation: np.ndarray) -> Tuple[np.ndarray, float]:
        """Update weights using observation likelihood."""
        new_weights = np.array([
            w * self.ssm.get_likelihood(pred_obs=self.ssm.get_observation(p), observation=observation)
            for p, w in zip(particles, weights)
        ])
        normalizing_constant = np.sum(new_weights)
        new_weights /= normalizing_constant
        return new_weights, normalizing_constant

    def resample(self, particles: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resample particles based on their weights."""
        if self.resampling_method == "systematic":
            new_particles, new_weights = self._resample_systematic(particles, weights)
        elif self.resampling_method == "multinomial":
            new_particles, new_weights = self._resample_multinomial(particles, weights)
        else:
            raise ValueError("Invalid resampling method")
        return new_particles, new_weights

    def _resample_systematic(self, particles: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Systematic resampling."""
        positions = (np.random.random() + np.arange(self.n_particles)) / self.n_particles
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0

        i, j = 0, 0
        new_particles = np.zeros_like(particles)

        while i < self.n_particles:
            if positions[i] < cumsum[j]:
                new_particles[i] = particles[j]
                i += 1
            else:
                j += 1

        new_weights = np.ones(self.n_particles) / self.n_particles
        return new_particles, new_weights

    def _resample_multinomial(self, particles: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resample particles according to their weights."""
        indices = np.random.choice(self.n_particles, size=self.n_particles, p=weights)
        new_particles = particles[indices]
        new_weights = np.ones(self.n_particles) / self.n_particles
        return new_particles, new_weights

    def get_state_estimate(self, particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Return weighted mean of particles."""
        return np.sum(particles * weights[:, np.newaxis], axis=0)

    def get_state_estimate_covariance(self, particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Return weighted covariance of particles."""
        return np.cov(particles, aweights=weights)

    def filter(self, observations: np.ndarray) -> Dict[str, np.ndarray]:
        """Run filter over full observation sequence."""
        T = len(observations)
        state_estimates = np.zeros((T, self.ssm.state_dim))
        normalizing_constants = np.zeros(T)

        particles, weights = self.initialize_particles()

        print("Running Particle Filter...")
        for t in tqdm(range(T)):
            particles = self.predict(particles)
            weights, normalizing_constants[t] = self.update(particles, weights, observations[t])
            if self.ess_resampling and 1.0 / np.sum(weights**2) < self.n_particles / 2:
                particles, weights = self.resample(particles, weights)
            state_estimates[t] = self.get_state_estimate(particles, weights)

        return {
            "state_estimates": state_estimates,
            "particles": particles,
            "weights": weights,
            "normalizing_constants": normalizing_constants
        }
