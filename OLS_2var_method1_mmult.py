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
etfs = pd.DataFrame(etf_rtns.iloc[:, [0, 1]])  # SPY + TLT
y = etf_rtns.iloc[:, 2]  # "Frankenstein"
t = etf_rtns.shape[0]
w = 30  # window length -- for this example 30 will do...

rolling_coefs = pd.DataFrame(index=etf_rtns.index, columns=['Alpha', 'Beta_SPY', 'Beta_TLT'])

ones = pd.DataFrame(index=etf_rtns.index, columns=['Ones'], data=1)
dframes = [ones, etfs]
x = pd.concat(dframes, axis=1)

rolling_coefs.loc[w-1:, :] = [
    ((np.linalg.inv(x[ii - (w-1):ii + 1].T.dot(x[ii - (w-1):ii + 1]))).dot(x[ii - (w-1):ii + 1].T)).dot(y[ii - (w-1):ii + 1])[:] for
    ii in range(w-1, t)]

toc = time.time()
print(str(toc-tic) + ' seconds elapsed.')