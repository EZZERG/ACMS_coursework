import torch
import pyro
import pyro.distributions as dist

from acms.pyro_models.pyro_ssm import StateSpaceModel

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

def generate_data(model, num_timesteps):
    states = torch.zeros(num_timesteps + 1, model.state_dim)
    observations = torch.zeros(num_timesteps + 1, model.obs_dim)
    
    # Initialize
    states[0] = torch.randn(model.state_dim)
    observations[0] = model.get_observation(states[0])
    
    # Generate sequence
    for t in range(num_timesteps):
        states[t + 1], observations[t + 1] = model.step(states[t])
    
    return states, observations
