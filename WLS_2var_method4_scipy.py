import pandas as pd
import os
import time

# At end of file, print time elapsed in seconds:
tic = time.time()

import numpy as np
from scipy.optimize import minimize

# Global variables:
cwd = os.getcwd()
file_loc_path = cwd + str('\\etf_returns_raw.csv')
etf_rtns = pd.read_csv(file_loc_path, index_col=0)
x = etf_rtns.iloc[:, [0,1]]  # SPY + TLT
y = etf_rtns.iloc[:, 2]  # "Frankenstein"
t = etf_rtns.shape[0]
w = 30  # window length -- for this example 30 will do...

# Initialize structure for rolling regression coefficients:
rolling_coefs = pd.DataFrame(index=etf_rtns.index, columns=['Alpha', 'Beta_SPY', 'Beta_TLT'])

base_weight = 0.80
zero_matrix = np.zeros(shape=(w, 1))
weight_vector_sqrts = [(base_weight ** (w - ii)) ** (1/2) for ii in range(0, w)]

# Initial guess for alpha & beta:
b0 = [0, 0.5, 0.5]

for ii in range(w-1, t):

    x_input = x.iloc[ii-(w-1):ii+1, :]
    y_input = y.iloc[ii-(w-1):ii+1]

    # Subfunction to get sum of squared residuals...which is what
    # we are minimizing:
    def residuals_wtd_sum_of_sqs(b):
        wtd_residuals = (y_input - (b[0] + (b[1] * x_input.iloc[:, 0]) + (b[2] * x_input.iloc[:, 1]))) * weight_vector_sqrts
        return sum(wtd_residuals ** 2)

    beta_coefs = minimize(residuals_wtd_sum_of_sqs, b0, tol=1e-12)

    # I don't love that "x" is a field of the beta_coefs structure...but for
    # some frameworks, alpha is same as any other input variable...constants
    # have feelings too...
    rolling_coefs.iloc[ii, 0] = beta_coefs.x[0]
    rolling_coefs.iloc[ii, 1:] = beta_coefs.x[1:]

toc = time.time()
print(str(toc-tic) + ' seconds elapsed.')
