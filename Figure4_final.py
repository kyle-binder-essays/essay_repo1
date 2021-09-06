# -*- coding: utf-8 -*-
"""
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import calendar
import datetime as dt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)


def get_input_parameters():

    ##############################################
    # LOAD AS DATAFRAME FROM CSV:
    ##############################################

    # Load data:
    cwd = os.getcwd()
    #    csv_daily = cwd + str('\Data\Fama_French\\12_Industry_Portfolios_Daily.CSV'
    csv_monthly = cwd + str('\Data\Fama_French\\12_Industry_Portfolios.CSV')
    #    df_daily = pd.read_csv(csv_daily,skiprows=9,nrows=24966-9,index_col=0)
    df_monthly = pd.read_csv(csv_monthly, skiprows=11, nrows=1148 - 11, index_col=0)

    # Date conversion from "YYYYMM" to an actual date:
    dates_fama_m = df_monthly.index
    dates_good_m = convert_fama_monthly_to_dates(dates_fama_m)
    df_monthly = df_monthly.set_index(dates_good_m.index)

    df_to_use = df_monthly

    # Convert from Fama French format (1% return = 1.00) to more common format (1% return = 0.01)
    df_to_use = df_to_use / 100

    num_securities_long_leg = 3
    num_securities_short_leg = 3

    total_N = df_to_use.shape[1]

    data_frequency = 'M'

    return df_to_use, num_securities_long_leg, num_securities_short_leg, total_N, \
           data_frequency, dates_good_m


def last_business_day_in_month(year: int, month: int) -> int:
    return max(calendar.monthcalendar(year, month)[-1:][0][:5])


def convert_fama_monthly_to_dates(fama_dates):
    nans = np.empty((len(fama_dates)))
    nans[:] = np.nan
    good_dates = pd.DataFrame(nans)

    for tt in range(0, len(fama_dates)):
        yyyy = int(str(fama_dates[tt])[:4])
        mm = int(str(fama_dates[tt])[4:])
        dd = last_business_day_in_month(yyyy, mm)
        good_dates.iloc[tt, 0] = dt.datetime(yyyy, mm, dd)

    good_dates = good_dates.set_index(0)
    return good_dates


def get_lower_triangle_returns_matrix(rtns, \
                                      num_securities_long_leg, \
                                      num_securities_short_leg, \
                                      max_lags, \
                                      total_N ):

    ##############################################
    # See repo directory "/Data/Excel_Validation/" for
    # how+why this data structure is created
    ##############################################

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
    long_thresh = num_securities_long_leg + 0.99 # threshold rank for long leg inclusion
    for tt in range( 0 , rtns.shape[0] ):
        # Lowest = 1 (1=most desirable: lowest vol, lowest rho, highest momo); 
        # largest = n
        
        # Initialze entire row to zeroes:
        historical_long_weights.iloc[tt,:] = np.zeros([1,rtns.shape[1]]) 
        
        # Check if there are ties (happens more than you'd think, 
        # especially for (1,0) or (x,x-1) formations)):
        ranks_tt = ranks.iloc[tt,:]
        num_secs_tt = ranks_tt[ranks_tt < long_thresh].count()
        wt = 1 / num_secs_tt
        
        # Assign the non-zero weights:
        for nn in range( 0 , rtns.shape[1] ):
            if(ranks_tt[nn] < long_thresh):
                historical_long_weights.iloc[tt,nn] = wt
                
    # Populate SHORT weights:
    short_thresh = total_N - num_securities_short_leg + 0.01 # threshold rank for short leg inclusion
    for tt in range( 0 , rtns.shape[0] ):
        # Lowest = 1 (1=most desirable: lowest vol, lowest rho, highest momo); 
        # largest = n
        
        # Initialze entire row to zeroes:
        historical_short_weights.iloc[tt,:] = np.zeros([1,rtns.shape[1]]) 
        
        if (num_securities_short_leg > 0) :
            # Check if there are ties (happens more than you'd think):
            ranks_tt = ranks.iloc[tt,:]
            num_secs_tt = ranks_tt[ranks_tt > short_thresh].count()
            wt = 1 / num_secs_tt
            
            # Assign the non-zero weights:
            for nn in range( 0 , rtns.shape[1] ):
                if(ranks_tt[nn] > short_thresh ):
                    historical_short_weights.iloc[tt,nn] = -wt
                
    # All weights:
    historical_all_weights = historical_long_weights + historical_short_weights

    # Get MOMO returns for each of {(1,0), (2,1), (3,2), ..., (max_lags,max_lags-1) } :
    for tt in range( 0 , rtns.shape[0] ):
        
        # Print for visibility to console:
        if (tt % 100 == 0):
            print(tt)
        
        for lags in range(0 , max_lags):
            
            if (tt >= (lags+1) ):
                wts = historical_all_weights.iloc[tt-(lags+1),:].values
                pos_rtns = rtns.iloc[tt,:].values
                
                triangle_mtrx.iloc[tt,lags] = sum(wts * pos_rtns)
                
    # TO DO: Check that weights sum to zero:
    weight_test_1 = 1 # initialize to TRUE
    
    # TO DO: Check that max weight is not > 1/num_securities_long_leg (within machine precision)
    # & check that min weight is not < -1/num_securities_short_leg:
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
    

def plot_means_vols_chart(row_of_means, row_of_vols):

    fig = plt.figure(figsize=(8,12))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    fig.suptitle('Figure 4')
    ax1.title.set_text('Panel A: Momentum (Lag, Lag-1) Average (Monthly) Returns, 1926-2021')
    ax2.title.set_text('Panel B: Momentum (Lag, Lag-1) Standard Deviations, 1926-2021')

    ax1.set_xlabel('Lag (Months)')
    ax2.set_xlabel('Lag (Months)')
    
    ax1.set_ylabel('Monthly Returns (0.001 = 0.1%)')
    ax2.set_ylabel('Volatility (0.01 = 1.0%)')
    
    ax1.set_ylim(-0.003, 0.007)
    ax2.set_ylim(0, 0.04)
    
    # For Panel A, make the major tick label in multiples of 0.10%, and
    # make the format '%1.3f':
    ax1.yaxis.set_major_locator(MultipleLocator(0.001))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%1.3f'))
    
    # For Panel B, make the major tick label in multiples of 0.5%, and
    # make the format '%1.1f':
    ax2.yaxis.set_major_locator(MultipleLocator(0.005))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%1.3f'))
    

    # fig.suptitle('Vertically stacked subplots')
    ax1.bar(row_of_means.columns, \
               row_of_means.iloc[0, :], \
               width=0.8, bottom=None, align='center', data=None, \
               tick_label=row_of_means.columns)
    ax2.bar(row_of_vols.columns, \
               row_of_vols.iloc[0, :], \
               width=0.8, bottom=None, align='center', data=None, \
               tick_label=row_of_vols.columns)

    return

# Get input parameters:
rtns_df, num_securities_long_leg, num_securities_short_leg, total_N, \
    data_frequency, dates_good_m = \
    get_input_parameters()

# Get "fell off cliff" charts:
max_lags_to_test = 18
triangle_rtns_mtrx, weights_for_triangle, ranks_for_triangle, \
triangle_weight_test_1, triangle_weight_test_2 = \
    get_lower_triangle_returns_matrix(rtns_df, \
                                        num_securities_long_leg, \
                                        num_securities_short_leg, \
                                        max_lags_to_test, \
                                        total_N)

row_of_means_same_history, row_of_means_all_rows, test_result_arith = \
    get_arith_means_chart(triangle_rtns_mtrx)

row_of_vols_same_history, row_of_vols_all_rows, test_result_vols = \
    get_volatility_chart(triangle_rtns_mtrx)

## plot_arith_means_chart(row_of_means_all_rows)
plot_means_vols_chart(row_of_means_same_history, row_of_vols_same_history)



