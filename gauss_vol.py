import pandas as pd
import numpy as np
import numpy.random as random

# Initialize paramters:
num_sims = 250000
N_per_sim = [5, 10, 20, 50, 100]
sigma = 0.1

# Initialize output structure:
nans_stdevs = np.empty( ( num_sims , len(N_per_sim) ) )
nans_stdevs[:] = np.nan
sample_stdevs = pd.DataFrame(nans_stdevs)

# Same structure for population stdevs:
nans_stdevs_pop = np.empty( ( num_sims , len(N_per_sim) ) )
nans_stdevs_pop[:] = np.nan
population_stdevs = pd.DataFrame(nans_stdevs_pop)

# Run simulations for each sample size in "N_per_sim":
for nn in range(0,len(N_per_sim)):
    for ss in range(0,num_sims):
        
        rand_ss = random.normal(loc=0.0, scale=sigma, size=N_per_sim[nn])
    
        # Put sample stdev (ddof=1) into output structure:
        sample_stdevs.iloc[ss,nn] = rand_ss.std(ddof=1)
        # Put population stdev (ddof=0) into output structure:
        population_stdevs.iloc[ss,nn] = rand_ss.std(ddof=0)   

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

