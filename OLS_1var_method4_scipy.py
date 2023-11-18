import pandas as pd
import os
import time

# At end of file, print time elapsed in seconds:
tic = time.time()

from scipy.optimize import minimize

# Global variables:
cwd = os.getcwd()
file_loc_path = cwd + str('\\etf_returns_raw.csv')
etf_rtns = pd.read_csv(file_loc_path, index_col=0)
x = etf_rtns.iloc[:, 0]  # SPY
y = etf_rtns.iloc[:, 2]  # "Frankenstein"
t = etf_rtns.shape[0]
w = 30  # window length -- for this example 30 will do...

# Initialize structure for rolling regression coefficients:
rolling_coefs = pd.DataFrame(index=etf_rtns.index, columns=['Alpha', 'Beta'])

# Initial guess for alpha & beta:
b0 = [0, 1]

for ii in range(w-1, t):

    x_input = x.iloc[ii-(w-1):ii+1]
    y_input = y.iloc[ii-(w-1):ii+1]

    # Subfunction to get sum of squared residuals...which is what
    # we are minimizing:
    def residuals_sum_of_sqs(b):
        residuals = y_input - (b[0] + b[1] * x_input)
        return sum(residuals ** 2)

    beta_coefs = minimize(residuals_sum_of_sqs, b0)

    # I don't love that "x" is a field of the beta_coefs structure...but for
    # some frameworks, alpha is same as any other input variable...constants
    # have feelings too...
    rolling_coefs.iloc[ii, 0] = beta_coefs.x[0]
    rolling_coefs.iloc[ii, 1] = beta_coefs.x[1]

toc = time.time()
print(str(toc-tic) + ' seconds elapsed.')
