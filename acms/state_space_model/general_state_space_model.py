import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from abc import ABC, abstractmethod

class GeneralStateSpaceModel(ABC):
    def __init__(self, state_dim, obs_dim, dt=0.01):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.dt = dt

    @abstractmethod
    def get_next_state(self, x):
        pass

    @abstractmethod
    def get_observation(self, x):
        pass

    @abstractmethod
    def step(self, x):
        pass
    
    @abstractmethod
    def simulate(self, steps, x0=None):
        pass

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