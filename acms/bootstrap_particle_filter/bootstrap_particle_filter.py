from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from acms.state_space_model.general_state_space_model import GeneralStateSpaceModel

class ParticleFilter:
    def __init__(self, ssm: GeneralStateSpaceModel, n_particles: int=100, resampling_method: Literal["systematic", "multinomial"]="multinomial", ess_resampling: bool=False):
        self.ssm = ssm
        self.n_particles = n_particles
        # TODO: particles are initialized as coming from a standard normal distribution
        # This assumes that all ssm models use a standard normal distribution for the initial state
        # Make it more general by using the initial state distribution of the ssm model
        self.particles = np.random.randn(n_particles, ssm.state_dim)
        self.weights = np.ones(n_particles) / n_particles
        self.resampling_method = resampling_method
        # if True, resample when effective sample size is too small, otherwise resample at every time step
        self.ess_resampling = ess_resampling
    
    def predict(self):
        """Propagate particles through state dynamics"""
        for i in range(self.n_particles):
            self.particles[i] = self.ssm.get_next_state(self.particles[i])
    
    def update(self, observation):
        """Update weights using observation likelihood"""
        for i in range(self.n_particles):
            pred_obs = self.ssm.get_observation(self.particles[i])
            self.weights[i] *= np.exp(-0.5 * np.sum((observation - pred_obs)**2) / 
                                    np.diag(self.ssm.R).mean())
        
        self.weights /= np.sum(self.weights)
        
        # resample when effective sample size if becoming too small
        if self.ess_resampling:
            if 1.0 / np.sum(self.weights**2) < self.n_particles/2:
                self.resample()
        else:
            self.resample()
    
    def resample(self):
        if self.resampling_method == "systematic":
            self._resample_systematic()
        elif self.resampling_method == "multinomial":
            self._resample_multinomial()
        else:
            raise ValueError("Invalid resampling method")
    
    def _resample_systematic(self):
        """Systematic resampling"""
        positions = (np.random.random() + np.arange(self.n_particles)) / self.n_particles
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0
        
        i, j = 0, 0
        new_particles = np.zeros_like(self.particles)
        
        while i < self.n_particles:
            if positions[i] < cumsum[j]:
                new_particles[i] = self.particles[j]
                i += 1
            else:
                j += 1
                
        self.particles = new_particles
        self.weights.fill(1.0 / self.n_particles)
    
    def _resample_multinomial(self):
        """Resample particles according to their weights"""
        indices = np.random.choice(
            self.n_particles,
            size=self.n_particles,
            p=self.weights
        )
        self.particles = self.particles[indices]
        # note: I missed in the lecture that I need to reset the weights to uniform after resampling
        # but it seems to gratly help with keeping the std from collapsing to 0
        # which is indicative of some level of particle degeneracy
        self.weights.fill(1.0 / self.n_particles)

    def get_state_estimate(self):
        """Return weighted mean of particles"""
        return np.sum(self.particles * self.weights[:, np.newaxis], axis=0)

    def filter(self, observations):
        """Run filter over full observation sequence"""
        T = len(observations)
        state_estimates = np.zeros((T, self.ssm.state_dim))
        
        for t in range(T):
            self.predict()
            self.update(observations[t])
            state_estimates[t] = self.get_state_estimate()
            
        return state_estimates

    def plot_results(self, true_states, state_estimates, dimensions=[0, 1], save_dir=None):
        """Plot particle filter tracking results for specified dimensions"""
        time = np.arange(len(true_states)) * self.ssm.dt
        
        # Time series plot
        fig, ax = plt.subplots(len(dimensions), 1, figsize=(10, 4*len(dimensions)))
        if len(dimensions) == 1:
            ax = [ax]
            
        for i, dim in enumerate(dimensions):
            ax[i].plot(time, true_states[:, dim], 'k-', label=f'True State {dim+1}')
            ax[i].plot(time, state_estimates[:, dim], 'r--', label=f'Estimated State {dim+1}')
            ax[i].fill_between(time, 
                             state_estimates[:, dim] - 2*np.std(self.particles[:, dim]),
                             state_estimates[:, dim] + 2*np.std(self.particles[:, dim]),
                             color='r', alpha=0.2)
            ax[i].set_xlabel('Time')
            ax[i].set_ylabel(f'State {dim+1}')
            ax[i].legend()
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(f'{save_dir}/pf_tracking.png')
            plt.close()
        else:
            plt.show()

        # Phase plot if multiple dimensions
        if len(dimensions) >= 2:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(true_states[:, dimensions[0]], true_states[:, dimensions[1]], 
                   'k-', label='True Trajectory')
            ax.plot(state_estimates[:, dimensions[0]], state_estimates[:, dimensions[1]], 
                   'r--', label='Estimated Trajectory')
            ax.set_xlabel(f'State {dimensions[0]+1}')
            ax.set_ylabel(f'State {dimensions[1]+1}')
            ax.legend()
            
            if save_dir:
                plt.savefig(f'{save_dir}/pf_phase.png')
                plt.close()
            else:
                plt.show()

if __name__ == "__main__":
    from acms.state_space_model.state_space_model import StateSpaceModel

    # Fix random seed for reproducibility
    np.random.seed(42)

    # Initialize model
    model = StateSpaceModel(
        state_dim=4, 
        obs_dim=2,
        damping=0.99,
        nonlinearity='quadratic',
        nonlinearity_scale=1.0,
        A_scale=0.1,
        C_scale=0.1,
        Q_scale=0.01,
        R_scale=0.01
    )

    # Generate true trajectory
    n_steps = 100
    true_states, observations = model.simulate(steps=n_steps)

    # Initialize and run particle filter
    pf = ParticleFilter(model, n_particles=1000)
    state_estimates = pf.filter(observations)

    # Plot results
    pf.plot_results(true_states, state_estimates, dimensions=[0, 1])

    # Print RMSE for each dimension
    rmse = np.sqrt(np.mean((true_states - state_estimates)**2, axis=0))
    print(f"RMSE per dimension: {rmse}")

    # Plot model trajectories
    model.plot_simulation(true_states, observations)