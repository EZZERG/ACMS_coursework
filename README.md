# ACMS (Advanced Computational Methods for State-space models)

This repository contains implementations of various state-space models and particle filtering methods, with a focus on Sequential Monte Carlo (SMC) and Particle MCMC techniques.

## Project Structure

```
acms/
├── bootstrap_particle_filter/     # Bootstrap particle filter implementation
├── experiment/                    # Experiment framework and schedulers
├── gradient_ssm_training/         # Differentiable SMC training
├── pyro_models/                   # Pyro-based implementations
└── state_space_model/            # State space model base implementations
```

## Key Components

- **State Space Models**: Base implementations for general state space models
- **Bootstrap Particle Filter**: Implementation of the bootstrap particle filter algorithm
- **Pyro Models**: Probabilistic programming implementations using Pyro
- **Experiment Framework**: Modular system for running and managing experiments
- **Gradient-based Training**: Differentiable SMC implementation for gradient-based training

## Installation

This project uses Poetry for dependency management. To install:

1. Make sure you have Poetry installed:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository and install dependencies:
```bash
git clone [repository-url]
cd ACMS_coursework
poetry install
```

## Usage

Running experiment with simulated data

### Running Experiments

```python
from acms.experiment import experiment_scheduler
from acms.experiment import ssm_experiment

# Configure and run experiments
experiment = ssm_experiment.SSMExperiment(...)
scheduler = experiment_scheduler.ExperimentScheduler(...)
```

### Using Particle Filters

```python
from acms.bootstrap_particle_filter import bootstrap_particle_filter
from acms.state_space_model import state_space_model

# Initialize and run particle filter
model = state_space_model.StateSpaceModel(...)
filter = bootstrap_particle_filter.BootstrapParticleFilter(...)
```

### Pyro Models

```python
from acms.pyro_models import pyro_smc_model
from acms.pyro_models import pyro_ssm

# Use Pyro-based implementations
smc_model = pyro_smc_model.PyroSMCModel(...)
```

## Results

Experiment results are stored in the `results/` directory, organized by experiment type:
- `results/smc_experiments/`: Sequential Monte Carlo experiment results
- `results/pmcmc_experiments/`: Particle MCMC experiment results

