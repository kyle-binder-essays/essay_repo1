import pandas as pd
import os
import time

# At end of file, print time elapsed in seconds:
tic = time.time()

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

# Global variables:
cwd = os.getcwd()
file_loc_path = cwd + str('\\etf_returns_raw.csv')
etf_rtns = pd.read_csv(file_loc_path, index_col=0)
x = sm.add_constant(etf_rtns.iloc[:, [0,1]])  # SPY & TLT & constant
y = etf_rtns.iloc[:, 2]  # "Frankenstein"
t = etf_rtns.shape[0]
w = 30  # window length -- for this example 30 will do...

# Initialize structure for rolling regression coefficients:
rolling_coefs = pd.DataFrame(index=etf_rtns.index, columns=['Alpha', 'Beta_SPY', 'Beta_TLT'])

rolls = RollingOLS(y, x, window=w)
rollresids = rolls.fit()
rolling_coefs = rollresids.params.copy()

toc = time.time()
print(str(toc-tic) + ' seconds elapsed.')
