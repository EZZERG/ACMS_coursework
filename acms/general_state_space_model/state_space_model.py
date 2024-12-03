import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

class StateSpaceModel:
    def __init__(self, state_dim, obs_dim, dt=0.01, nonlinearity='quadratic', damping=0.99, 
                 A_scale=0.1, C_scale=0.1, Q_terms_scale=0.01, Q_scale=0.01, R_scale=0.01, nonlinearity_scale=1.0):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.dt = dt
        self.nonlinearity = nonlinearity
        self.damping = damping # add damping for stability of the system when simulating with high number of steps
        
        self.A = np.random.randn(state_dim, state_dim) * A_scale
        self.C = np.random.randn(obs_dim, state_dim) * C_scale
        self.Q_terms = np.random.randn(state_dim, state_dim, state_dim) * Q_terms_scale
        self.nonlinearity_scale = nonlinearity_scale
        
        self.Q = np.eye(state_dim) * Q_scale
        self.R = np.eye(obs_dim) * R_scale
    
    def nonlinear_dynamics(self, x):
        if self.nonlinearity == 'quadratic':
            quad_terms = np.zeros(self.state_dim)
            for i in range(self.state_dim):
                quad_terms[i] = np.sum(self.Q_terms[i] * np.outer(x, x))
            return quad_terms
        elif self.nonlinearity == 'sine':
            return np.sin(x)
        elif self.nonlinearity == 'tanh':
            return np.tanh(x)
        else:
            return np.zeros(self.state_dim)
    
    def get_next_state(self, x):
        new_x = x + self.dt * (self.A @ x + self.nonlinearity_scale * self.nonlinear_dynamics(x))
        new_x *= self.damping
        new_x += np.random.multivariate_normal(np.zeros(self.state_dim), self.Q)
        return new_x
    
    def get_observation(self, x):
        obs = self.C @ x
        obs += np.random.multivariate_normal(np.zeros(self.obs_dim), self.R)
        return obs
    
    def step(self, x):
        new_x = self.get_next_state(x)
        obs = self.get_observation(new_x)
        return new_x, obs
        
    def simulate(self, steps, x0=None):
        if x0 is None:
            x0 = np.random.randn(self.state_dim)  # Initialize x0 with random values
        
        states = np.zeros((steps + 1, self.state_dim))
        observations = np.zeros((steps + 1, self.obs_dim))
        
        states[0] = x0
        observations[0] = self.C @ x0
        
        for t in range(steps):
            states[t+1], observations[t+1] = self.step(states[t])
            
        return states, observations

    def plot_simulation(self, states, observations, state_dims=None, obs_dims=None, traj_state_dims=None, traj_obs_dims=None, save_dir=None):
        time = np.arange(states.shape[0]) * self.dt
        
        if state_dims is None:
            state_dims = range(self.state_dim)
        if obs_dims is None:
            obs_dims = range(self.obs_dim)
        if traj_state_dims is None:
            traj_state_dims = [0, 1]
        if traj_obs_dims is None:
            traj_obs_dims = [0, 1]
        
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        for i in state_dims:
            axs[0].plot(time, states[:, i], label=f'State {i+1}')
        axs[0].set_title('States over Time')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('State values')
        axs[0].legend()
        
        for i in obs_dims:
            axs[1].plot(time, observations[:, i], label=f'Observation {i+1}')
        axs[1].set_title('Observations over Time')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Observation values')
        axs[1].legend()
        
        plt.tight_layout()
        
        if save_dir:
            fig.savefig(os.path.join(save_dir, 'states_observations.png'))
            plt.close(fig)
        else:
            plt.show()
        
        if len(traj_state_dims) >= 2:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(states[:, traj_state_dims[0]], states[:, traj_state_dims[1]], label='State Trajectory', zorder=1)
            ax.scatter(states[0, traj_state_dims[0]], states[0, traj_state_dims[1]], color='green', marker='o', s=100, label='Start', zorder=2)
            ax.scatter(states[-1, traj_state_dims[0]], states[-1, traj_state_dims[1]], color='red', marker='x', s=100, label='End', zorder=2)
            ax.set_title('State Trajectory')
            ax.set_xlabel(f'State {traj_state_dims[0]+1}')
            ax.set_ylabel(f'State {traj_state_dims[1]+1}')
            ax.legend()
            if save_dir:
                fig.savefig(os.path.join(save_dir, 'state_trajectory.png'))
                plt.close(fig)
            else:
                plt.show()
        else:
            print("Warning: Not enough state dimensions to plot trajectory.")
        
        if len(traj_obs_dims) >= 2:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(observations[:, traj_obs_dims[0]], observations[:, traj_obs_dims[1]], label='Observation Trajectory', zorder=1)
            ax.scatter(observations[0, traj_obs_dims[0]], observations[0, traj_obs_dims[1]], color='green', marker='o', s=100, label='Start', zorder=2)
            ax.scatter(observations[-1, traj_obs_dims[0]], observations[-1, traj_obs_dims[1]], color='red', marker='x', s=100, label='End', zorder=2)
            ax.set_title('Observation Trajectory')
            ax.set_xlabel(f'Observation {traj_obs_dims[0]+1}')
            ax.set_ylabel(f'Observation {traj_obs_dims[1]+1}')
            ax.legend()
            if save_dir:
                fig.savefig(os.path.join(save_dir, 'observation_trajectory.png'))
                plt.close(fig)
            else:
                plt.show()
        else:
            print("Warning: Not enough observation dimensions to plot trajectory.")
    
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

# Example usage:
if __name__ == "__main__":
    # Testing the implementation
    model = StateSpaceModel(state_dim=4, obs_dim=2, damping=0.99, nonlinearity='quadratic', nonlinearity_scale=1)
    states, observations = model.simulate(steps=10000)
    model.plot_simulation(states, observations, state_dims=[0, 1], obs_dims=[0], traj_state_dims=[0, 1], traj_obs_dims=[0, 1])
    
    # Save the model
    model.save_model('state_space_model.pkl')
    
    # Load the model
    loaded_model = StateSpaceModel.load_model('state_space_model.pkl')
    loaded_states, loaded_observations = loaded_model.simulate(steps=100)
    loaded_model.plot_simulation(loaded_states, loaded_observations, state_dims=[0, 1], obs_dims=[0], traj_state_dims=[0, 1], traj_obs_dims=[0, 1])