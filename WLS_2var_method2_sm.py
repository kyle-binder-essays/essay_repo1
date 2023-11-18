import pandas as pd
import numpy as np
import os
import time

# At end of file, print time elapsed in seconds:
tic = time.time()

import statsmodels.api as sm

# Global variables:
cwd = os.getcwd()
file_loc_path = cwd + str('\\etf_returns_raw.csv')
etf_rtns = pd.read_csv(file_loc_path, index_col=0)
x = sm.add_constant(etf_rtns.iloc[:, [0, 1]])  # SPY & TLT & constant
y = etf_rtns.iloc[:, 2]  # "Frankenstein"
t = etf_rtns.shape[0]
w = 30  # window length -- for this example 30 will do...

# Assign weights:
base_weight = 0.80
zero_matrix = np.zeros(shape=(w, 1))
weight_vector = [base_weight ** (w - ii) for ii in range(0, w)]

# Initialize structure for rolling regression coefficients:
rolling_coefs = pd.DataFrame(index=etf_rtns.index, columns=['Alpha', 'Beta_SPY', 'Beta_TLT'])

for ii in range(w-1, t):

    x_input = x.iloc[ii-(w-1):ii+1, :]
    y_input = y.iloc[ii-(w-1):ii+1]

    wls_model = sm.WLS(y_input, x_input, weights=weight_vector)
    model_coefs = wls_model.fit()
    rolling_coefs.iloc[ii, :] = model_coefs.params

toc = time.time()
print(str(toc-tic) + ' seconds elapsed.')
