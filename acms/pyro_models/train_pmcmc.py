from acms.pyro_models.pmcmc import PMCMCParameters, ParticleMCMC
from acms.pyro_models.pyro_smc_model import generate_data
from acms.pyro_models.smc_experiment import SMCExperiment

# Generate data with true parameters
true_params = PMCMCParameters(
    A_scale=0.1,
    C_scale=0.1,
    Q_scale=0.01,
    R_scale=0.01,
    nonlinearity_scale=1.0
)

experiment = SMCExperiment(
    state_dim=2,
    obs_dim=2,
    num_particles=100,
    **vars(true_params)
)

# Generate synthetic data
T = 100
states, observations = generate_data(experiment.ssm, T)

# Initialize PMCMC
pmcmc = ParticleMCMC(
    state_dim=2,
    obs_dim=2,
    num_particles=100,
    n_mcmc_steps=1000,
    proposal_scale=0.1
)

# Run PMCMC
parameters, log_likelihoods = pmcmc.run_pmcmc(observations)

# Analyze results
results = pmcmc.analyze_results(parameters, log_likelihoods)
print("Estimated parameters:", results['mean_parameters'])
print("Acceptance rate:", results['acceptance_rate'])

# Plot results
pmcmc.plot_traces(parameters, log_likelihoods, true_params, 'pmcmc_traces.png')
