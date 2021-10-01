# -*- coding: utf-8 -*-
"""
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
#import datetime as dt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)

def get_lower_triangle_returns_matrix(rtns, \
                                      percent_N_per_leg, \
                                      max_lags, \
                                      total_N ):
    
    # Initialize output:
    nans = np.empty( ( rtns.shape[0] , max_lags ) )
    nans[:] = np.nan
    triangle_mtrx = pd.DataFrame(nans)
    triangle_mtrx = triangle_mtrx.set_index(rtns.index)
    triangle_mtrx.columns = list(range(0+1,max_lags+1))
    
    # Populate ranks for each row:
    ranks = initialize_output(rtns)
    for rr in range(0,ranks.shape[0]):
        # Lowest = 1; largest = n ... when "ascending=True"
        temp_rank = rtns.iloc[rr,:].rank(method='average',ascending=False) 
        # Put transpose in output variable:
        ranks.iloc[rr,:] = temp_rank.transpose()
        
    # Based on ranks, populate weights:
    historical_all_weights      = initialize_output(rtns)
    historical_long_weights     = initialize_output(rtns)
    historical_short_weights    = initialize_output(rtns)
    
    # Populate Long weights:
    for tt in range( 0 , rtns.shape[0] ):
        # Lowest = 1 (1=most desirable: lowest vol, lowest rho, highest momo); 
        # largest = n
        num_non_nan_securities_tt = ranks.iloc[tt,:].count()
        num_securities_long_leg_tt = math.ceil(percent_N_per_leg * num_non_nan_securities_tt)
        long_thresh_tt = num_securities_long_leg_tt + 0.99 # threshold rank for long leg inclusion
        
        # Initialze entire row to zeroes:
        historical_long_weights.iloc[tt,:] = np.zeros([1,rtns.shape[1]]) 
        
        # Check if there are ties (happens more than you'd think, 
        # especially for (1,0) or (x,x-1) formations)):
        ranks_tt = ranks.iloc[tt,:]
        num_secs_tt = ranks_tt[ranks_tt < long_thresh_tt].count()
        wt = 1 / num_secs_tt
        
        # Assign the non-zero weights:
        for nn in range( 0 , rtns.shape[1] ):
            if(ranks_tt[nn] < long_thresh_tt):
                historical_long_weights.iloc[tt,nn] = wt
                
    # Populate SHORT weights:
#    short_thresh = total_N - num_securities_short_leg + 0.01 # threshold rank for short leg inclusion
    for tt in range( 0 , rtns.shape[0] ):
        # Lowest = 1 (1=most desirable: lowest vol, lowest rho, highest momo); 
        # largest = n
        #
        # threshold rank for short leg inclusion:
        # "count()" counts non-nan elements:
        num_non_nan_securities_tt = ranks.iloc[tt,:].count()
        num_securities_short_leg_tt = math.ceil(percent_N_per_leg * num_non_nan_securities_tt)
        short_thresh_tt = num_non_nan_securities_tt - num_securities_short_leg_tt + 0.01
        
        # Initialze entire row to zeroes:
        historical_short_weights.iloc[tt,:] = np.zeros([1,rtns.shape[1]]) 
        
        if (num_securities_short_leg_tt > 0) :
            # Check if there are ties (happens more than you'd think):
            ranks_tt = ranks.iloc[tt,:]
            num_secs_tt = ranks_tt[ranks_tt > short_thresh_tt].count()
            wt = 1 / num_secs_tt
            
            # Assign the non-zero weights:
            for nn in range( 0 , rtns.shape[1] ):
                if(ranks_tt[nn] > short_thresh_tt ):
                    historical_short_weights.iloc[tt,nn] = -wt
                
    # All weights:
    historical_all_weights = historical_long_weights + historical_short_weights
    
    
    # Get MOMO returns for each of {(1,0), (2,1), (3,2), ..., (max_lags,max_lags-1) } :
    for tt in range( 0 , rtns.shape[0] ):
        
        # Print for visibility to console:
        if (tt % 100 == 0):
            print(tt)
        
        for lags in range( 0 , max_lags):
            
            if (tt >= (lags+1) ):
                wts = historical_all_weights.iloc[tt-(lags+1),:].values
                # Need to account for NaNs - if we had no NaNs, we could
                # simply do: pos_rtns = rtns.iloc[tt,:].values
                pos_rtns = rtns.iloc[tt,:] # initialize
                for cc in range(0,rtns.shape[1]):
                    if (pd.isna(rtns.iloc[tt,cc])):
                        pos_rtns.iloc[cc] = 0 # overwrite NAN with ZERO
                    else:
                        pos_rtns.iloc[cc] = rtns.iloc[tt,cc] # do nothing
                
                triangle_mtrx.iloc[tt,lags] = sum(wts * pos_rtns)
                
    # Check that weights sum to zero:
    weight_test_1 = 1 # initialize to TRUE
    
    # Check that max weight is not > 1/num_securities_long_leg (within machine precision)
    # &
    # Check that min weight is not < -1/num_securities_short_leg:
    weight_test_2 = 1 # initialize to TRUE
    
    
    return  triangle_mtrx, historical_all_weights, ranks, \
            weight_test_1, weight_test_2 


def initialize_output(rtns):
    
    # Function to initialize output & keep row headings (dates) and column headings (tickers):
    nans = np.empty( ( rtns.shape[0] , rtns.shape[1] ) )
    nans[:] = np.nan
    output_df = pd.DataFrame(nans)
    output_df = output_df.set_index(rtns.index)
    output_df.columns = rtns.columns
    
    return output_df


def get_arith_means_chart(triangle_mtrx):
    
    # Determine "max_lag_tested" from structure of matrix itself:
    max_lags_tested = triangle_mtrx.shape[1]
    
    # Initialize outputs:
    nans = np.empty( ( 1 , max_lags_tested ) )
    nans[:] = np.nan
    row_of_means_same_history = pd.DataFrame(nans)
    row_of_means_same_history.columns = triangle_mtrx.columns
    
    # Initialize again because of how python handles pointers:
    nans2 = np.empty( ( 1 , max_lags_tested ) )
    nans2[:] = np.nan
    row_of_means_all_rows = pd.DataFrame(nans2)
    row_of_means_all_rows.columns = triangle_mtrx.columns
    
    # Aritmetic means of all columns for common subperiod of no NANs:
    temp_row_of_means_same_history = \
        pd.DataFrame.mean(triangle_mtrx.iloc[(max_lags_tested-1):,:], axis=0)
    # Convert from series to DFrame:    
    row_of_means_same_history.iloc[0,:] = temp_row_of_means_same_history
    
    # Aritmetic means of all columns without regard to NANs:
    temp_row_of_means_all_rows = \
        pd.DataFrame.mean(triangle_mtrx, axis=0)    
    # Convert from series to DFrame: 
    row_of_means_all_rows.iloc[0,:] = temp_row_of_means_all_rows
    
    # Unit test: check that these two entries match:
    # (1) row_of_means_same_history.iloc[0,max_lag_tested-1]
    # (2) row_of_means_all_rows.iloc[0,max_lag_tested-1]
    test_result = 1 # initialize to TRUE
    
    return row_of_means_same_history, row_of_means_all_rows, test_result
    
    
def get_volatility_chart(triangle_mtrx):
    
    # Determine "max_lag_tested" from structure of matrix itself:
    max_lags_tested = triangle_mtrx.shape[1]
    
    # Initialize outputs:
    nans = np.empty( ( 1 , max_lags_tested ) )
    nans[:] = np.nan
    row_of_vols_same_history = pd.DataFrame(nans)
    row_of_vols_same_history.columns = triangle_mtrx.columns
    
    # Initialize again because of how python handles pointers:
    nans2 = np.empty( ( 1 , max_lags_tested ) )
    nans2[:] = np.nan
    row_of_vols_all_rows = pd.DataFrame(nans2)
    row_of_vols_all_rows.columns = triangle_mtrx.columns
    
    # Sample Std Devs of all columns for common subperiod of no NANs:
    temp_row_same_history = \
        pd.DataFrame.std(triangle_mtrx.iloc[(max_lags_tested-1):,:], axis=0)
    # Convert from series to DFrame:    
    row_of_vols_same_history.iloc[0,:] = temp_row_same_history
    
    # Sample Std Devs of all columns without regard to NANs:
    temp_row_all_rows = \
        pd.DataFrame.std(triangle_mtrx, axis=0)    
    # Convert from series to DFrame: 
    row_of_vols_all_rows.iloc[0,:] = temp_row_all_rows
    
    # Unit test: check that these two entries match:
    # (1) row_of_means_same_history.iloc[0,max_lag_tested-1]
    # (2) row_of_means_all_rows.iloc[0,max_lag_tested-1]
    test_result = 1 # initialize to TRUE
    
    return row_of_vols_same_history, row_of_vols_all_rows, test_result
    
    
def get_tstats_chart(triangle_mtrx, \
                     row_of_means_same_history, row_of_means_all_rows, \
                     row_of_vols_same_history, row_of_vols_all_rows):
    
    # Need 3 inputs to get a t-stat: {mean, vol, N}.
    
    # Determine "max_lag_tested" from structure of matrix itself:
    max_lags_tested = triangle_mtrx.shape[1]
    
    # Initialize outputs:
    nans = np.empty( ( 1 , max_lags_tested ) )
    nans[:] = np.nan
    row_of_tstats_same_history = pd.DataFrame(nans)
    row_of_tstats_same_history.columns = triangle_mtrx.columns
    
    # Initialize again because of how python handles pointers:
    nans2 = np.empty( ( 1 , max_lags_tested ) )
    nans2[:] = np.nan
    row_of_tstats_all_rows = pd.DataFrame(nans2)
    row_of_tstats_all_rows.columns = triangle_mtrx.columns
    
    # Populate outputs:
    N_same_history = triangle_mtrx.shape[0] - triangle_mtrx.shape[1]
    print(N_same_history)
    for ii in range(0, max_lags_tested):
        
        # Populate row_of_tstats_same_history[ii]:
        tstat_same_ii = (row_of_means_same_history.iloc[0,ii] - 0) / \
            (row_of_vols_same_history.iloc[0,ii] / (N_same_history ** (0.5)) )
        row_of_tstats_same_history.iloc[0,ii] = tstat_same_ii
        
        # Populate row_of_tstats_all_rows[ii]:
        N_ii = triangle_mtrx.shape[0] - ii - 1
        tstat_all_ii = (row_of_means_all_rows.iloc[0,ii] - 0) / \
            (row_of_vols_all_rows.iloc[0,ii] / (N_ii ** (0.5)) )
        row_of_tstats_all_rows.iloc[0,ii] = tstat_all_ii
    
    
    return row_of_tstats_same_history, row_of_tstats_all_rows
    
    
def plot_three_panel_chart(row_of_means, row_of_vols, row_of_tstats, title_string):

    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)

    fig.suptitle(title_string)
    ax1.title.set_text('Panel A: Average (Monthly) Returns')
    ax2.title.set_text('Panel B: Volatility')
    ax3.title.set_text('Panel C: T-Statistics')

    ax1.set_xlabel('Lag (Months)')
    ax2.set_xlabel('Lag (Months)')
    ax3.set_xlabel('Lag (Months)')
    
    ax1.set_ylabel('Monthly Returns (0.001 = 0.1%)')
    ax2.set_ylabel('Volatility (0.01 = 1.0%)')
    ax3.set_ylabel('T-Statistics (Red Line: +/-1.96)')
    
    ax1.set_ylim(-0.005, 0.007)
    ax2.set_ylim(0, 0.05)
    ax3.set_ylim(-4, 4)

    ax3.axhline(y=1.96, color='r', linestyle='-')
    ax3.axhline(y=-1.96, color='r', linestyle='-')
    
    # For Panel A, make the major tick label in multiples of 0.10%, and
    # make the format '%1.3f':
    ax1.yaxis.set_major_locator(MultipleLocator(0.001))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%1.3f'))
    ax1.bar(row_of_means.columns, \
           row_of_means.iloc[0, :], \
           width=0.8, bottom=None, align='center', data=None, \
           tick_label=row_of_means.columns)
    
    # For Panel B, make the major tick label in multiples of 0.5%, and
    # make the format '%1.1f':
    ax2.yaxis.set_major_locator(MultipleLocator(0.005))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%1.3f'))
    ax2.bar(row_of_vols.columns, \
               row_of_vols.iloc[0, :], \
               width=0.8, bottom=None, align='center', data=None, \
               tick_label=row_of_vols.columns)
    
    # For Panel C, make the major tick label in multiples of 2.0, and
    # make the format '%1.1f':
    ax3.yaxis.set_major_locator(MultipleLocator(2))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
    ax3.bar(row_of_tstats.columns, \
               row_of_tstats.iloc[0, :], \
               width=0.8, bottom=None, align='center', data=None, \
               tick_label=row_of_tstats.columns)

    return

    