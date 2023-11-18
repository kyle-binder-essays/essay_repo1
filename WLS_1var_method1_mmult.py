import pandas as pd
import os
import time
import numpy as np

# At end of file, print time elapsed in seconds:
tic = time.time()

# Global variables:
cwd = os.getcwd()
file_loc_path = cwd + str('\\etf_returns_raw.csv')
etf_rtns = pd.read_csv(file_loc_path, index_col=0)
spy = pd.DataFrame(etf_rtns.iloc[:, 0])  # SPY
y = etf_rtns.iloc[:, 2]  # "Frankenstein"
t = etf_rtns.shape[0]
win = 30  # window length -- for this example 30 will do...

rolling_coefs = pd.DataFrame(index=etf_rtns.index, columns=['Alpha', 'Beta'])

ones = pd.DataFrame(index=etf_rtns.index, columns=['Ones'], data=1)
dframes = [ones, spy]
x = pd.concat(dframes, axis=1)

base_weight = 0.80

for ii in range(win-1, t):
    # wts: square diagonal matrix of weights; all off-diagonal entries are zero:
    wts = pd.DataFrame(index=etf_rtns.index[ii - (win-1):ii + 1], columns=etf_rtns.index[ii - (win-1):ii + 1], data=0)
    for jj in range(0, win):
        wts.iloc[jj, jj] = (base_weight ** (win - jj)) ** (1/2)
    rolling_coefs.iloc[ii, :] = (
        (np.linalg.inv(x[ii - (win - 1):ii + 1].T.dot(wts.T).dot(wts).dot(x[ii - (win - 1):ii + 1]))).dot(
            x[ii - (win - 1):ii + 1].T)).dot(wts.T).dot(wts).dot(y[ii - (win - 1):ii + 1])

toc = time.time()
print(str(toc-tic) + ' seconds elapsed.')