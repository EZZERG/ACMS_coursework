import argparse
import logging
import matplotlib.pyplot as plt

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SMCFilter

from acms.pyro_models.pyro_ssm import StateSpaceModel

logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)

class PyroStateSpaceModel:
    def __init__(self, ssm_model):
        self.model = ssm_model
        self.t = 0

    def init(self, state, initial):
        self.t = 0
        state["z"] = pyro.sample("z_init", dist.Delta(initial, event_dim=1))

    def step(self, state, y=None):
        self.t += 1
        
        # Get mean for next state using model dynamics
        mean = state["z"] + self.model.dt * (
            state["z"] @ self.model.A.t() + 
            self.model.nonlinearity_scale * self.model.nonlinear_dynamics(state["z"])
        ) * self.model.damping

        # Sample next state
        state["z"] = pyro.sample(
            f"z_{self.t}",
            dist.MultivariateNormal(mean, self.model.Q).to_event(0)
        )

        # Sample observation
        y = pyro.sample(
            f"y_{self.t}",
            dist.MultivariateNormal(state["z"] @ self.model.C.t(), self.model.R).to_event(0),
            obs=y
        )

        return state["z"], y

class PyroStateSpaceModel_Guide:
    def __init__(self, model):
        self.model = model
        self.t = 0

    def init(self, state, initial):
        self.t = 0
        pyro.sample("z_init", dist.Delta(initial, event_dim=1))

    def step(self, state, y=None):
        self.t += 1
        
        # Simple proposal using the same dynamics as the model
        mean = state["z"] + self.model.model.dt * (
            state["z"] @ self.model.model.A.t() + 
            self.model.model.nonlinearity_scale * self.model.model.nonlinear_dynamics(state["z"])
        ) * self.model.model.damping

        # Use a slightly larger covariance for the proposal
        proposal_cov = 1.2 * self.model.model.Q

        pyro.sample(
            f"z_{self.t}",
            dist.MultivariateNormal(mean, proposal_cov).to_event(0)
        )

def generate_data(model, args):
    states = torch.zeros(args.num_timesteps + 1, model.state_dim)
    observations = torch.zeros(args.num_timesteps + 1, model.obs_dim)
    
    # Initialize
    states[0] = torch.randn(model.state_dim)
    observations[0] = model.get_observation(states[0])
    
    # Generate sequence
    for t in range(args.num_timesteps):
        states[t + 1], observations[t + 1] = model.step(states[t])
    
    return states, observations

def main(args):
    pyro.set_rng_seed(args.seed)

    # Initialize state space model
    ssm = StateSpaceModel(
        state_dim=args.state_dim,
        obs_dim=args.obs_dim,
        dt=args.dt,
        nonlinearity=args.nonlinearity,
        damping=args.damping,
        nonlinearity_scale=args.nonlinearity_scale
    )

    # Create pyro model and guide
    model = PyroStateSpaceModel(ssm)
    guide = PyroStateSpaceModel_Guide(model)

    # Initialize SMC
    smc = SMCFilter(model, guide, num_particles=args.num_particles, max_plate_nesting=0)

    logging.info("Generating data")
    states, observations = generate_data(ssm, args)

    logging.info("Filtering")
    
    # Initialize estimates storage
    estimates = torch.zeros(args.num_timesteps + 1, args.state_dim)
    variances = torch.zeros(args.num_timesteps + 1, args.state_dim)
    
    # Initialize filter
    smc.init(initial=torch.zeros(args.state_dim))
    z = smc.get_empirical()["z"]
    estimates[0] = z.mean
    variances[0] = z.variance
    
    # Run filter
    for t in range(args.num_timesteps):
        smc.step(observations[t + 1])
        z = smc.get_empirical()["z"]
        estimates[t + 1] = z.mean
        variances[t + 1] = z.variance

    logging.info("At final time step:")
    logging.info(f"truth: {states[-1]}")
    logging.info(f"mean: {estimates[-1]}")
    logging.info(f"std: {torch.sqrt(variances[-1])}")

    # Plotting
    time = range(args.num_timesteps + 1)
    fig, axs = plt.subplots(2, 1, figsize=(14, 10))

    for dim in range(2):  # Plot first two dimensions
        axs[dim].plot(time, states[:, dim].numpy(), label=f'True State {dim}', linestyle='--')
        axs[dim].plot(time, estimates[:, dim].numpy(), label=f'Estimated State {dim}')
        std = torch.sqrt(variances[:, dim])
        axs[dim].fill_between(
            time,
            (estimates[:, dim] - 2 * std).numpy(),
            (estimates[:, dim] + 2 * std).numpy(),
            color='red',
            alpha=0.2,
            label='Confidence Interval (±2 std)' if dim == 0 else ""
        )
        axs[dim].set_xlabel('Time')
        axs[dim].set_ylabel(f'State Dimension {dim}')
        axs[dim].legend()

    plt.tight_layout()
    plt.savefig('smc_filter_trajectories.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="State Space Model w/ SMC Filtering")
    parser.add_argument("-n", "--num-timesteps", default=100, type=int)
    parser.add_argument("-p", "--num-particles", default=100, type=int)
    parser.add_argument("--state-dim", default=4, type=int)
    parser.add_argument("--obs-dim", default=2, type=int)
    parser.add_argument("--dt", default=0.01, type=float)
    parser.add_argument("--nonlinearity", default="quadratic", type=str)
    parser.add_argument("--damping", default=0.99, type=float)
    parser.add_argument("--nonlinearity-scale", default=1.0, type=float)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    main(args)
