# -*- coding: utf-8 -*-
"""
"""

from datetime import datetime as dtdt
import pandas as pd
import numpy as np
import math # for ceil function

##################################
#
# UMD: "Up Minus Down"
#
# Desired outputs from this PY file:
# (1) Weights: historical + current
# (2) Backtest returns
# That's it.
#
# Eventually:  VIX 9-Day & 6-Month (Etc)
#              CBOE IMPLIED CORRELATIONS
#
##################################


def initialize_output(rtns):
    
    # Function to initialize output & keep row headings (dates) and column headings (tickers):
    nans = np.empty( ( rtns.shape[0] , rtns.shape[1] ) )
    nans[:] = np.nan
    output_df = pd.DataFrame(nans)
    output_df = output_df.set_index(rtns.index)
    output_df.columns = rtns.columns
    
    return output_df
    

def get_historical_long_short_weights(rtns, momo_ranks, \
                                  percent_per_leg, \
                                  num_securities_total):
    
    # Initialize output & keep row headings (dates) and column headings (tickers):
    historical_all_weights      = initialize_output(rtns)
    historical_long_weights     = initialize_output(rtns)
    historical_short_weights    = initialize_output(rtns)
    
    # Number of securities to put in "long" portfolio:
#    num_secs = num_securities_long_leg
    
    # Populate combo ranks:
    combo_ranks = momo_ranks
        
    # Now, rank the combo_ranks.  Why?
    # Say we want the top 3 securities, per combo ranks.  It's conceivable that 
    # combo ranks for n=4 could be ( 1.4, 1.9, 3.8, 3.9 ) [sum to 10].  So we can't
    # simply say "get the securities with rank <= 3"...The following "ranking of ranks"
    # solves this:
    
    # Initialize output: keep row headings (dates) and column headings (tickers):
    combo_ranks_ranked = initialize_output(combo_ranks)
    
    # Populate combo_ranks_ranked:
    for rr in range(0,combo_ranks.shape[0]):
        # Lowest = 1; largest = n ... when "ascending=True"
        temp_rank = combo_ranks.iloc[rr,:].rank(method='average',ascending=True) 
        # Put transpose in output variable:
        combo_ranks_ranked.iloc[rr,:] = temp_rank.transpose()
    
    # Populate Long weights:
    for tt in range( 0 , rtns.shape[0] ):
        # Lowest = 1 (1=most desirable: lowest vol, lowest rho, highest momo); 
        # largest = n
        #
        # threshold rank for long leg inclusion:
        # "count()" counts non-nan elements:
        num_non_nan_securities_tt = momo_ranks.iloc[tt,:].count()
        num_securities_long_leg_tt = math.ceil(percent_per_leg * num_non_nan_securities_tt)
        long_thresh_tt = num_securities_long_leg_tt + 0.99 # threshold rank for long leg inclusion
        
        # Initialze entire row to zeroes:
        historical_long_weights.iloc[tt,:] = np.zeros([1,rtns.shape[1]]) 
        
        # Check if there are ties (happens more than you'd think):
        ranks_tt = combo_ranks_ranked.iloc[tt,:]
        num_secs_tt = ranks_tt[ranks_tt < long_thresh_tt].count()
        wt = 1 / num_secs_tt
        
        # Assign the non-zero weights:
        for nn in range( 0 , rtns.shape[1] ):
            if(ranks_tt[nn] < long_thresh_tt):
                historical_long_weights.iloc[tt,nn] = wt
                
    # Populate SHORT weights:
    for tt in range( 0 , rtns.shape[0] ):
        # Lowest = 1 (1=most desirable: lowest vol, lowest rho, highest momo); 
        # largest = n
        #
        # threshold rank for short leg inclusion:
        # "count()" counts non-nan elements:
        num_non_nan_securities_tt = momo_ranks.iloc[tt,:].count()
        num_securities_short_leg_tt = math.ceil(percent_per_leg * num_non_nan_securities_tt)
        short_thresh_tt = num_non_nan_securities_tt - num_securities_short_leg_tt + 0.01
        
        # Initialze entire row to zeroes:
        historical_short_weights.iloc[tt,:] = np.zeros([1,rtns.shape[1]]) 
        
        if (num_securities_short_leg_tt > 0) :
            # Check if there are ties (happens more than you'd think, 
            # especially for (1,0) or (x,x-1) formations)):
            ranks_tt = combo_ranks_ranked.iloc[tt,:]
            num_secs_tt = ranks_tt[ranks_tt > short_thresh_tt].count()
            wt = 1 / num_secs_tt
            
            # Assign the non-zero weights:
            for nn in range( 0 , rtns.shape[1] ):
                if(ranks_tt[nn] > short_thresh_tt ):
                    historical_short_weights.iloc[tt,nn] = -wt
                
    # All weights:
    historical_all_weights = historical_long_weights + historical_short_weights
    
    # Check that weights sum to zero:
    # (or, sum to 1 if num_securities_short_leg==0)
    weight_test_1 = 1 # initialize to TRUE
    
    # Check that max weight is not > 1/num_securities_long_leg (within machine precision)
    # &
    # Check that min weight is not < -1/num_securities_short_leg:
    weight_test_2 = 1 # initialize to TRUE
            
    return  historical_all_weights, combo_ranks, combo_ranks_ranked, \
            weight_test_1, weight_test_2


def get_historical_momo_ranks(historical_signals):
    
    # Initialize output: keep row headings (dates) and column headings (tickers):
    df = initialize_output(historical_signals)
    
    # Populate output:
    for rr in range(0,df.shape[0]):
        # Lowest = 1; largest = n ... when "ascending=True"
        temp_rank = historical_signals.iloc[rr,:].rank(method='average',ascending=False) 
        # Put transpose in output variable:
        df.iloc[rr,:] = temp_rank.transpose()
    momo_ranks = df
    
    return momo_ranks


def get_historical_cumulative_returns(rtns, momo_window_1, momo_window_2):
    
    # INPUTS:
    # rtns: hopefully self explanatory
    # momo_window_1:  for classic "(12,1) UMD", this is the "12".
    # momo_window_2:  for classic "(12,1) UMD", this is the "1".
    
    # Initialize output:
    cum_rtns = initialize_output(rtns)
    
    rtns_plus = rtns + 1
    for tt in range( momo_window_1 , rtns.shape[0] ):
        for nn in range( 0 , rtns.shape[1] ):
            cum_rtns.iloc[tt,nn] = \
                rtns_plus.iloc[(tt - momo_window_1 + 1):
                         (tt - momo_window_2 + 1),nn].prod() - 1
                    
    # Default behavior of pandas 1.0+ is to treat NaN as zero - so e.g. 12 NaNs
    # would have a cmulative return of 0...we want this to remain NaN so that the 
    # cumulative return of 0 is not included in the downstream sortings.
    for tt in range( momo_window_1 , rtns.shape[0] ):
        for nn in range( 0 , rtns.shape[1] ):
            # Assume we will not populate cum_rtns.iloc[tt,nn] with nan until
            # proven otherwise:
            do_assign_nan_tt_nn = False
            for ss in range(tt, tt-momo_window_1, -1):
                if pd.isna(rtns.iloc[ss,nn]):
                    do_assign_nan_tt_nn = True
            if (do_assign_nan_tt_nn == True):
                cum_rtns.iloc[tt,nn] = np.NaN
    
    return cum_rtns
 
    
def main(rtns_df, momentum_window_1, momentum_window_2, \
         percent_N_per_leg, total_N):
    
    # Print to console for user visibility:
    print('Starting...')
    print(str(dtdt.now().strftime("%d-%b-%Y (%H:%M:%S)")))
    
    # Print to console for user visibility:
    print('Getting tickers...')
    print(str(dtdt.now().strftime("%d-%b-%Y (%H:%M:%S)")))
    rtns_all = rtns_df
    
    # Print to console for user visibility:
    print('Momentum...')
    print(str(dtdt.now().strftime("%d-%b-%Y (%H:%M:%S)")))    
    
    # Sector signals: MOMO
    historical_momo_signals = \
        get_historical_cumulative_returns(rtns_all, momentum_window_1, momentum_window_2)
    historical_momo_ranks = \
        get_historical_momo_ranks(historical_momo_signals)

    
    # Print to console for user visibility:
    print('Getting weights...')
    print(str(dtdt.now().strftime("%d-%b-%Y (%H:%M:%S)")))
    
    # Get weights of main portfolio, before applying any hedges:
    historical_weights, combo_ranks, combo_ranks_ranked, \
        weight_test_1, weight_test_2 = \
        get_historical_long_short_weights(rtns_all, historical_momo_ranks,\
                                          percent_N_per_leg,\
                                          total_N)
      
    return rtns_all, historical_weights, \
                historical_momo_signals, \
                historical_momo_ranks, \
                weight_test_1, weight_test_2
    