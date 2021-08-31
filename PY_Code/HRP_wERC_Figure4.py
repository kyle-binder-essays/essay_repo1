# -*- coding: utf-8 -*-
"""
@author: Kyle
"""
import matplotlib.pyplot as mpl
import scipy.cluster.hierarchy as sch, random, numpy as np, pandas as pd
import HRP as hrpm
import ERC_2p7 as ercm
import os

# ------------------------------------------------------------------------------

# Load data:
cwd = os.getcwd()
csv_dir_01 = cwd + '\\Simulated_Gaussians_Seed12345.csv'

x_all = pd.read_csv(csv_dir_01, index_col=0)

# x_A = x_all.iloc[:,2]
# x_C = x_all.iloc[:,3]

x_A = pd.DataFrame(index=x_all.index, data=x_all.iloc[:,2].values, columns=['x_A'])
x_B = pd.DataFrame(index=x_all.index, data=x_all.iloc[:,2].values, columns=['x_B'])
x_C = pd.DataFrame(index=x_all.index, data=x_all.iloc[:,3].values, columns=['x_C'])

# Normalize all columns to have variance of 1:
x_Az = x_A / x_A.std()
x_Bz = x_B / x_B.std()
x_Cz = x_C / x_C.std()

x_concat = [x_Az,x_Bz,x_Cz]
x = pd.concat(x_concat, axis=1)
# x = x_all

# Get covariance to be input to HRC + ERC:
covariances, corr = x.cov(), x.corr()

# Get ERC Weights:
assets_risk_budget = [float(1) / float(covariances.shape[1])] * covariances.shape[1]
init_weights = [float(1) / float(covariances.shape[1])] * covariances.shape[1]
N = covariances.shape[1]
cov_npy = np.zeros([N,N])
cov_npy[:] = covariances.values
wts_erc = ercm.get_risk_parity_weights(cov_npy, assets_risk_budget, init_weights)

# Get HRP Weights:
dist = hrpm.correlDist(corr)
link = sch.linkage(dist, 'single')
sortIx = hrpm.getQuasiDiag(link)
sortIx = corr.index[sortIx].tolist()  # recover labels
df0 = corr.loc[sortIx, sortIx]  # reorder
# plotCorrMatrix('HRP3_corr1.png',df0,labels=df0.columns)
# 4) Capital allocation
hrp_s = hrpm.getRecBipart(covariances, sortIx)
hrp_df = pd.DataFrame(index=hrp_s.index, data=hrp_s.values, columns=['wts'])
