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

# Iterate over several values of epsilon and collect all outputs into single dataframe:
iterator_array = pd.DataFrame(range(0,20,5))
# Initialize:
figure6 = pd.DataFrame(columns=list(range(0,101,10)), data=None, \
                               index=['epsilon', 'Min_Eig', 'Correl_AB', 'Correl_BD', 'Correl_CE', 'Correl_AE', \
                                      'ERC_A', 'ERC_B', 'ERC_C', 'ERC_D', 'ERC_E', \
                                      'HRP_A', 'HRP_B', 'HRP_C', 'HRP_D', 'HRP_E', \
                                      ])
for iterator in figure6.columns:
    print(iterator)
    # Ugh, don't need to specify float in Python 3+:
    epsilon = float(iterator) / float(100)
    print(epsilon)

    # Define columns {A,B,C,D,E}:
    x_A = pd.DataFrame(index=x_all.index, data=x_all.iloc[:, 2].values, columns=['x_A'])
    x_B = pd.DataFrame(index=x_all.index, data=( \
                (1 - epsilon) * x_all.iloc[:, 2].values + \
                (epsilon * x_all.iloc[:, 4].values) \
        ), columns=['x_B'])
    x_D = pd.DataFrame(index=x_all.index, data=( \
                (1 - epsilon) * x_all.iloc[:, 2].values + \
                (epsilon * x_all.iloc[:, 0].values) \
        ), columns=['x_D'])
    x_C = pd.DataFrame(index=x_all.index, data=x_all.iloc[:, 3].values, columns=['x_C'])
    x_E = pd.DataFrame(index=x_all.index, data=( \
                (1 - epsilon) * x_all.iloc[:, 3].values + \
                (epsilon * x_all.iloc[:, 1].values) \
        ), columns=['x_E'])

    # Normalize all columns to have variance of 1:
    x_Az = x_A / x_A.std()
    x_Bz = x_B / x_B.std()
    x_Cz = x_C / x_C.std()
    x_Dz = x_D / x_D.std()
    x_Ez = x_E / x_E.std()

    x_concat = [x_Az, x_Bz, x_Cz, x_Dz, x_Ez]
    x = pd.concat(x_concat, axis=1)

    # Get covariance to be input to HRC + ERC:
    covariances, corr = x.cov(), x.corr()

    # Get ERC Weights:
    assets_risk_budget = [float(1) / float(covariances.shape[1])] * covariances.shape[1]
    init_weights = [float(1) / float(covariances.shape[1])] * covariances.shape[1]
    N = covariances.shape[1]
    cov_npy = np.zeros([N, N])
    cov_npy[:] = covariances.values
    wts_erc = ercm.get_risk_parity_weights(cov_npy, assets_risk_budget, init_weights)

    # Get HRP Weights:
    dist = hrpm.correlDist(corr)
    link = sch.linkage(dist, 'single')
    sortIx = hrpm.getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()  # recover labels
    df0 = corr.loc[sortIx, sortIx]  # reorder

    # Capital allocation of HRP weights:
    hrp_s = hrpm.getRecBipart(covariances, sortIx)
    hrp_df = pd.DataFrame(index=hrp_s.index, data=hrp_s.values, columns=['wts'])

    # Collect all outputs into single structure:
    figure6.loc['epsilon', iterator] = epsilon
    figure6.loc['Min_Eig', iterator] = min(np.linalg.eigvals(corr))
    figure6.loc['Correl_AB', iterator] = corr.iloc[0,1]
    figure6.loc['Correl_BD', iterator] = corr.iloc[1,3]
    figure6.loc['Correl_CE', iterator] = corr.iloc[2,4]
    figure6.loc['Correl_AE', iterator] = corr.iloc[0,4]
    figure6.loc['ERC_A', iterator] = wts_erc[0]
    figure6.loc['ERC_B', iterator] = wts_erc[1]
    figure6.loc['ERC_C', iterator] = wts_erc[2]
    figure6.loc['ERC_D', iterator] = wts_erc[3]
    figure6.loc['ERC_E', iterator] = wts_erc[4]
    figure6.loc['HRP_A', iterator] = hrp_df.loc['x_A', 'wts']
    figure6.loc['HRP_B', iterator] = hrp_df.loc['x_B', 'wts']
    figure6.loc['HRP_C', iterator] = hrp_df.loc['x_C', 'wts']
    figure6.loc['HRP_D', iterator] = hrp_df.loc['x_D', 'wts']
    figure6.loc['HRP_E', iterator] = hrp_df.loc['x_E', 'wts']


