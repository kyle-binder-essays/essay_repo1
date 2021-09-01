# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 21:01:16 2021

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
csv_dir = cwd + '\\Simulated_Gaussians_Seed12345.csv'
x = pd.read_csv(csv_dir, index_col=0)

# Get covariance to be input to HRC + ERC:
covariances, corr = x.cov(), x.corr()

# Get ERC Weights:
assets_risk_budget = [float(1) / float(covariances.shape[1])] * covariances.shape[1]
init_weights = [float(1) / float(covariances.shape[1])] * covariances.shape[1]
cov_npy = np.zeros([10,10])
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
