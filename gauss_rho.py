# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
"""
"""

import pandas as pd # only used for plots
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

# Initialize:
num_sims = 50000
N_per_sim = [5, 10, 20, 50, 100]
correls = [0.1, 0.3, 0.5, 0.7, 0.9]

nans_correls = np.empty( ( num_sims , len(N_per_sim) , len(correls)) )
nans_correls[:] = np.nan
sample_correls = nans_correls

#####################################################
# Now do correlations:
#####################################################


sigma = 0.10

for cc in range(0,len(correls)):
    
    print(str('rho = ')+str(correls[cc]))

    vols = [sigma, sigma]
    correl_mtrx = np.empty( ( 2 , 2 ) )
    correl_mtrx[0,0] = 1
    correl_mtrx[1,1] = 1
    correl_mtrx[1,0] = correls[cc]
    correl_mtrx[0,1] = correls[cc]
    
    cov_mtrx = np.empty( ( 2 , 2 ) )
    cov_mtrx[0,0] = sigma * sigma
    cov_mtrx[1,1] = sigma * sigma
    cov_mtrx[1,0] = correls[cc] * sigma * sigma
    cov_mtrx[0,1] = correls[cc] * sigma * sigma
    
    for nn in range(0,len(N_per_sim)):
        for ss in range(0,num_sims):
            
            rand_ss = random.multivariate_normal(mean=[0,0], cov=cov_mtrx, size=N_per_sim[nn], \
                                                 check_valid='warn', tol=1e-8)
        
            # Put sample correlation into output structure:
            sample_correls[ss,nn,cc] = np.corrcoef(rand_ss[:,0],rand_ss[:,1])[1,0]
        
# Print mean, median, etc of "sample_correls", and put 
# the biases into a dataframe for later plotting:
nans = np.empty( ( len(N_per_sim) , len(correls) ) )
nans = np.nan
plot_df = pd.DataFrame(nans, index=N_per_sim, columns=correls)
print("CORRELATIONS:")
for cc in range(0,len(correls)):
    print(str('-----------------------'))
    print(str('rho = ')+str(correls[cc]))
    print(str('-----------------------'))
    for nn in range(0,len(N_per_sim)):
        print(str("N = ")+str(N_per_sim[nn])+str(':'))
        print(str("Mean: ")+str(sample_correls[:,nn,cc].mean()))
        bias_of_mean = sample_correls[:,nn,cc].mean() / correls[cc]
        print(str("Bias of Mean: ")+str(bias_of_mean))
        plot_df.iloc[nn,cc] = bias_of_mean

# Create plot:
fig = plt.figure(figsize=(8,5))
x = N_per_sim
rho_30 = plot_df.iloc[:,1]
rho_50 = plot_df.iloc[:,2]
rho_70 = plot_df.iloc[:,3]
rho_90 = plot_df.iloc[:,4]
plt.plot(x, rho_30, color='red', linestyle='dashdot')
plt.plot(x, rho_50, color='yellow')
plt.plot(x, rho_70, color='green')
plt.plot(x, rho_90, color='blue')
fig.set_xlabel('Lag (Months)')
plt.show()