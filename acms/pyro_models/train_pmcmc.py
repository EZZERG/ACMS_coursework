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

# Create experiment for generating true data
experiment = SMCExperiment(
    state_dim=2,
    obs_dim=2,
    num_particles=100,
    **vars(true_params)
)

# Get unscaled matrices from the true model
unscaled_matrices = experiment.ssm.get_unscaled_matrices()

# Generate synthetic data
T = 100
states, observations = generate_data(experiment.ssm, T)

# Initialize PMCMC with unscaled matrices
pmcmc = ParticleMCMC(
    state_dim=2,
    obs_dim=2,
    num_particles=100,
    n_mcmc_steps=200,
    proposal_scale=0.1,
    unscaled_matrices=unscaled_matrices  # Pass unscaled matrices here
)

# Run PMCMC
parameters, log_likelihoods = pmcmc.run_pmcmc(observations)

# Analyze results
results = pmcmc.analyze_results(parameters, log_likelihoods)
print("True parameters:", true_params)
print("Estimated parameters:", results['mean_parameters'])
print("Acceptance rate:", results['acceptance_rate'])
print("Final log likelihood:", results['final_log_likelihood'])

# Plot results
pmcmc.plot_traces(parameters, log_likelihoods, true_params, 'pmcmc_traces.png')
