import pandas as pd
import numpy as np
import numpy.random as random

# Initialize paramters:
num_sims = 100000
N_per_sim = [20]
sigma = 0.10
mu = 0.00

# Seed for RNG:
np.random.seed(1414)

# Initialize output structure for sample stdevs:
nans_stdevs = np.empty( ( num_sims , len(N_per_sim) ) )
nans_stdevs[:] = np.nan
sample_stdevs = pd.DataFrame(nans_stdevs)

# Same structure for population stdevs:
nans_stdevs_pop = np.empty( ( num_sims , len(N_per_sim) ) )
nans_stdevs_pop[:] = np.nan
population_stdevs = pd.DataFrame(nans_stdevs_pop)

# GARCH(1,1) paramters:
#
# The greeks {alpha, beta, omega} are from 
# "GARCH 101": https://pubs.aeaweb.org/doi/pdfplus/10.1257/jep.15.4.157
long_run_volatility = sigma
garch_alpha = 0.05 # coefficient of lagged residuals
garch_beta = 0.90 # coefficient of lagged variance
garch_omega = 0.05 # coefficient of long term / equilibrium variance 

# Compute derived parameter:
long_run_variance = sigma ** 2

# Simulate N days/observations (do this "num_sims" times), then observe the 
# sampling distribution of N-day volatility:
for nn in range(0,len(N_per_sim)):
    print(str("N = ")+str(N_per_sim[nn])+str(':'))
    for ss in range(0,num_sims):
        
        # Initialize structure to hold each day's OBSERVATIONS
        # (this is what we'll actually take the stdev of...)
        nans_obs = np.empty( [N_per_sim[nn] , 1] )
        nans_obs[:] = np.nan
        observations = nans_obs
        
        # For parsimony assume variance "yesterday" at t=0 is the equilibrium / long run 
        # variance:
        sigma_sqrd_previous = long_run_variance
        sigma_previous = sigma_sqrd_previous ** (1/2)
        
        # Using "yesterday's" sigma, generate an "observation" - then we can compute 
        # a residual.
        epsilon_previous = random.normal(loc=mu, scale=sigma_previous, size=None)
        
        # Now walk through the remaining N-1 days:
        for tt in range(0,N_per_sim[nn]):
            # Compute today's variance based on GARCH parameters:
            # (1) coefficient of lagged residual, 
            # (2) coefficient of lagged variance, 
            # (3) coefficient of long term / equilibrium variance 
            var_t = (garch_omega * long_run_variance) + \
                    (garch_alpha * (epsilon_previous ** 2)) + \
                    (garch_beta * (sigma_previous ** 2))
                
            # Generate "today's" observation and residual:
            sigma_t = var_t ** (1/2)
            observations[tt] = random.normal(loc=mu, scale=sigma_t, size=None)
            epsilon_previous = observations[tt] - mu
            sigma_previous = sigma_t
            
        # Put sample stdev (ddof=1) into output structure:
        sample_stdevs.iloc[ss,nn] = observations.std(ddof=1)
        # Put population stdev (ddof=0) into output structure:
        population_stdevs.iloc[ss,nn] = observations.std(ddof=0)   

# Print mean, median, etc of "sample_stdevs" & "population_stdevs":
for nn in range(0,len(N_per_sim)):
    print(str("N = ")+str(N_per_sim[nn])+str(':'))
    print(str("Mean: ")+str(sample_stdevs.iloc[:,nn].mean()))
    print(str("Median: ")+str(sample_stdevs.iloc[:,nn].median()))
    print(str("Bias of Mean: ")+str(sample_stdevs.iloc[:,nn].mean() / sigma))
    print(str("Bias of Median: ")+str(sample_stdevs.iloc[:,nn].median() / sigma))
    print(str("Mean (Population): ")+str(population_stdevs.iloc[:,nn].mean()))
    print(str("Median (Population): ")+str(population_stdevs.iloc[:,nn].median()))
    print(str("Bias of Mean (Population): ")+str(population_stdevs.iloc[:,nn].mean() / sigma))
    print(str("Bias of Median (Population): ")+str(population_stdevs.iloc[:,nn].median() / sigma))

