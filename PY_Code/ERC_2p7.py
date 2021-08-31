# -*- coding: utf-8 -*-
"""
@author: Kyle
"""

import pandas as pd
import numpy as np
import datetime
from scipy.optimize import minimize
TOLERANCE = 1e-10


def allocation_risk(weights, covariances):

    # We calculate the risk of the weights distribution
    portfolio_risk = np.sqrt((weights * covariances * weights.T))[0, 0]

    # It returns the risk of the weights distribution
    return portfolio_risk


def assets_risk_contribution_to_allocation_risk(weights, covariances):

    # We calculate the risk of the weights distribution
    portfolio_risk = allocation_risk(weights, covariances)

    # We calculate the contribution of each asset to the risk of the weights
    # distribution
    assets_risk_contribution = np.multiply(weights.T, covariances * weights.T) \
        / portfolio_risk

    # It returns the contribution of each asset to the risk of the weights
    # distribution
    return assets_risk_contribution


def risk_budget_objective_error(weights, args):

    # The covariance matrix occupies the first position in the variable
    covariances = args[0]

    # The desired contribution of each asset to the portfolio risk occupies the
    # second position
    assets_risk_budget = args[1]

    # We convert the weights to a matrix
    weights = np.matrix(weights)

    # We calculate the risk of the weights distribution
    portfolio_risk = allocation_risk(weights, covariances)

    # We calculate the contribution of each asset to the risk of the weights
    # distribution
    assets_risk_contribution = \
        assets_risk_contribution_to_allocation_risk(weights, covariances)

    # We calculate the desired contribution of each asset to the risk of the
    # weights distribution
    assets_risk_target = \
        np.asmatrix(np.multiply(portfolio_risk, assets_risk_budget))

    # Error between the desired contribution and the calculated contribution of
    # each asset
    error = \
        sum(np.square(assets_risk_contribution - assets_risk_target.T))[0, 0]

    # It returns the calculated error
    return error


def get_risk_parity_weights(covariances, assets_risk_budget, initial_weights):

    # Restrictions to consider in the optimisation: only long positions whose
    # sum equals 100%
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                   {'type': 'ineq', 'fun': lambda x: x})

    # Optimisation process in scipy
    optimize_result = minimize(fun=risk_budget_objective_error,
                               x0=initial_weights,
                               args=[covariances, assets_risk_budget],
                               method='SLSQP',
                               constraints=constraints,
                               tol=TOLERANCE,
                               options={'disp': False})

    # Recover the weights from the optimised object
    weights = optimize_result.x

    # It returns the optimised weights
    return weights


def get_weights_v3a():

    # We calculate the covariance matrix
    # Example 1: validate MATLAB example:
#    covariances = np.zeros([4,4])
#    covariances[0,0]= 0.0324
#    covariances[0,1]= 0.04131
#    covariances[0,2]= -0.0108
#    covariances[0,3]= 0
#    
#    covariances[1,0]= 0.04131
#    covariances[1,1]= 0.0729
#    covariances[1,2]= -0.00972
#    covariances[1,3]= 0
#    
#    covariances[2,0]= -0.0108
#    covariances[2,1]= -0.00972
#    covariances[2,2]= 0.0144
#    covariances[2,3]= 0
#    
#    covariances[3,0]=0
#    covariances[3,1]=0
#    covariances[3,2]=0
#    covariances[3,3]=0.0256
    
    # Example 2a: same as Example 1, but diagonal entries are equal (want to 
    # see that last entry gets exactly 25% weight.)
    # Note: it doesn't; it's 26.4%
#    covariances = np.zeros([4,4])
#    covariances[0,0]= 0.03
#    covariances[0,1]= 0.04131
#    covariances[0,2]= -0.0108
#    covariances[0,3]= 0
#    
#    covariances[1,0]= 0.04131
#    covariances[1,1]= 0.03
#    covariances[1,2]= -0.00972
#    covariances[1,3]= 0
#    
#    covariances[2,0]= -0.0108
#    covariances[2,1]= -0.00972
#    covariances[2,2]= 0.03
#    covariances[2,3]= 0
#    
#    covariances[3,0]=0
#    covariances[3,1]=0
#    covariances[3,2]=0
#    covariances[3,3]=0.03
    
    # Example 2b: same as Example 2a, but let's increase diagonal entries a lot:
    # Yup, all weights converging toward 25%:
    covariances = np.zeros([4,4])
    covariances[0,0]= 0.49
    covariances[0,1]= 0.04131
    covariances[0,2]= -0.0108
    covariances[0,3]= 0
    
    covariances[1,0]= 0.04131
    covariances[1,1]= 0.49
    covariances[1,2]= -0.00972
    covariances[1,3]= 0
    
    covariances[2,0]= -0.0108
    covariances[2,1]= -0.00972
    covariances[2,2]= 0.49
    covariances[2,3]= 0
    
    covariances[3,0]=0
    covariances[3,1]=0
    covariances[3,2]=0
    covariances[3,3]=0.49

    # The desired contribution of each asset to the portfolio risk: we want all
    # asset to contribute equally
    assets_risk_budget = [1 / covariances.shape[1]] * covariances.shape[1]

    # Initial weights: equally weighted
    init_weights = [1 / covariances.shape[1]] * covariances.shape[1]

    # Optimisation process of weights
    weights = \
        get_risk_parity_weights(covariances, assets_risk_budget, init_weights)

    # Convert the weights to a pandas Series
    weights = pd.Series(weights, name='weight')

    # It returns the optimised weights
    return weights, covariances
