import pandas as pd
import os
import time
from sklearn.linear_model import LinearRegression

# At end of file, print time elapsed in seconds:
tic = time.time()

# Global variables:
cwd = os.getcwd()
file_loc_path = cwd + str('\\etf_returns_raw.csv')
etf_rtns = pd.read_csv(file_loc_path, index_col=0)
x = pd.DataFrame(etf_rtns.iloc[:, 0])  # SPY
y = etf_rtns.iloc[:, 2]  # "Frankenstein"
t = etf_rtns.shape[0]
w = 30  # window length -- for this example 30 will do...

rolling_coefs = pd.DataFrame(index=etf_rtns.index, columns=['Alpha', 'Beta'])

for ii in range(w-1, t):
    reg = LinearRegression()
    reg.fit(x[ii-(w-1):ii+1], y[ii-(w-1):ii+1])
    rolling_coefs.iloc[ii, 0] = reg.intercept_
    rolling_coefs.iloc[ii, 1] = reg.coef_[0]

toc = time.time()
print(str(toc - tic) + ' seconds elapsed.')
