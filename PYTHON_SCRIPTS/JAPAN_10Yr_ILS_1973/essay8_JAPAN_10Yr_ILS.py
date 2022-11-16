# -*- coding: utf-8 -*-
"""
"""

import pandas as pd
import numpy as np
import bond_pricing
import os
from datetime import datetime


def define_input_parameters():
    
    ##############################################
    # LOAD AS DATAFRAME FROM CSV:
    ##############################################
        
    cwd = os.getcwd()

    print('loading monthly real yields')
    csv_monthly = cwd + str('\monthly_yields_1973_JAPAN_ILS_10Y.CSV')
    df_monthly = pd.read_csv(csv_monthly,skiprows=0,nrows=2685-1,index_col=0)
    print('FINISHED loading monthly real yields')

    # Maturity (years) of the time series of yields to be loaded:
    shorter_mat_yrs = 5
    longer_mat_yrs  = 10
    
    return df_monthly, shorter_mat_yrs, longer_mat_yrs
       
   
def initialize_output(df, column_heading):
    
    # Function to initialize output & keep row headings (dates) with
    # new input column heading "column_heading":
    nans = np.empty( ( df.shape[0] , 1 ) )
    nans[:] = np.nan
    output_df = pd.DataFrame(nans)
    output_df = output_df.set_index(df.index)
    output_df.columns = column_heading
    
    return output_df


def monthly_yields_to_returns(df_yields, shorter_mat_yrs, longer_mat_yrs):
    
    ############################################
    # Requirements/inputs:
    ############################################
    #
    # df_yields: two-column dataframe: the 2nd column should be
    #    the desired maturity to which we convert to yields.
    #    the first column should be a smaller maturity than column 2, and
    #    the closer the maturities are, the more accurate the (linear)
    #    interpolation that will be performed.
    #
    # Maturity (years) of the time series of yields to be loaded:
    # ex: longer_mat_yrs = 10
    # ex: shorter_mat_yrs = 5
    #
    ############################################
    # Outputs:
    ############################################
    #
    # df_returns: represents the return on a hypothetical constant maturity 
    # index of treasury bonds, rebalanced monthly; that is, at the end of each 
    # month, the previous 10-year bond is now a 9.9167-year bond, which we sell 
    # (at a yield determined by interpolation of available data), and proceed to 
    # purchase a new bond whose yield matches that of that day’s published 10-year rate.
    #
    ############################################
    
    # First, get series of (linearly) interpolated yields;
    # This series will be a constant maturity series with maturity 
    # equal to "longer_mat_yrs" years minus 1 month:
    interpolated_yields = linear_interp_monthly(df_yields, \
                                        shorter_mat_yrs, \
                                        longer_mat_yrs)
    
    # Initialize some variables to be used next:
    price_carry_return_m = initialize_output(df_yields, ['Price_Rtn'])
    coupon_return_m = initialize_output(df_yields, ['Cpn_Rtn'])
    price_sold_at_m = initialize_output(df_yields, ['Sold_Price'])
    total_return_m = initialize_output(df_yields, ['Total_Rtn'])
    
    for tt in range(1, interpolated_yields.shape[0]):
        # Decompose into price return + coupon return
        annualized_cpn = df_yields.iloc[tt-1,1]
        coupon_return_m.iloc[tt,0] = annualized_cpn / 1200
        prev_long_yield_today_tt = interpolated_yields.iloc[tt,0]
        # Need to have package installed
        price_sold_at_m.iloc[tt,0] = bond_pricing.simple_bonds.bond_price(settle=None, \
            cpn=annualized_cpn/100, 
            mat=longer_mat_yrs - (1/12), 
            yld=prev_long_yield_today_tt/100, \
            freq=2, 
            redeem=100, daycount=None)
        price_carry_return_m.iloc[tt,0] = (price_sold_at_m.iloc[tt,0]/100) - 1
        total_return_m.iloc[tt,0] = price_carry_return_m.iloc[tt,0] + coupon_return_m.iloc[tt,0]
    
    df_returns = total_return_m
    
    return df_returns, interpolated_yields, \
           price_carry_return_m, coupon_return_m, total_return_m, \
           price_sold_at_m


def daily_yields_to_returns(df_yields, shorter_mat_yrs, longer_mat_yrs):
    
    ############################################
    # Requirements/inputs:
    ############################################
    #
    # df_yields: two-column dataframe: the 2nd column should be
    #    the desired maturity to which we convert to yields.
    #    the first column should be a smaller maturity than column 2, and
    #    the closer the maturities are, the more accurate the (linear)
    #    interpolation that will be performed.
    #
    # Maturity (years) of the time series of yields to be loaded:
    # ex: longer_mat_yrs = 10
    # ex: shorter_mat_yrs = 5
    #
    ############################################
    # Outputs:
    ############################################
    #
    # df_returns: represents the return on a hypothetical constant maturity 
    # index of treasury bonds, rebalanced daily; that is, at the end of each 
    # day, the previous 10-year bond is now a ~9.99726-year bond (that's if weekday, 
    # is less if weekend), which we sell (at a yield determined by interpolation 
    # of available data), and proceed to purchase a new bond whose yield matches that 
    # of that day’s published 10-year rate.
    #
    ############################################
    
    # First, get series of (linearly) interpolated yields;
    # This series will be a constant maturity series with maturity 
    # equal to "longer_mat_yrs" years minus 1 month:
    interpolated_yields_d, left_fraction_d_tt, right_fraction_d_tt, \
        days_elapsed_d = linear_interp_daily(df_yields, \
                                        shorter_mat_yrs, \
                                        longer_mat_yrs)
    
    # Initialize some variables to be used next:
    price_carry_return_d = initialize_output(df_yields, ['Price_Rtn'])
    coupon_return_d = initialize_output(df_yields, ['Cpn_Rtn'])
    price_sold_at_d = initialize_output(df_yields, ['Sold_Price'])
    total_return_d = initialize_output(df_yields, ['Total_Rtn'])
    
    for tt in range(1, interpolated_yields_d.shape[0]):
        # Decompose into price return + coupon return
        annualized_cpn = df_yields.iloc[tt-1,1]
        coupon_return_d.iloc[tt,0] = days_elapsed_d.iloc[tt,0] * annualized_cpn / 36500
        prev_long_yield_today_tt = interpolated_yields_d.iloc[tt,0]
        # Need to have package installed
        price_sold_at_d.iloc[tt,0] = bond_pricing.simple_bonds.bond_price(settle=None, \
            cpn=annualized_cpn/100, 
            mat=longer_mat_yrs - (days_elapsed_d.iloc[tt,0]/365), 
            yld=prev_long_yield_today_tt/100, \
            freq=2, 
            redeem=100, daycount=None)
        price_carry_return_d.iloc[tt,0] = (price_sold_at_d.iloc[tt,0]/100) - 1
        total_return_d.iloc[tt,0] = price_carry_return_d.iloc[tt,0] + coupon_return_d.iloc[tt,0]
    
    df_returns = total_return_d
    
    return df_returns, interpolated_yields_d, \
            price_carry_return_d, coupon_return_d, total_return_d, \
            price_sold_at_d, \
            left_fraction_d_tt, right_fraction_d_tt, days_elapsed_d


def linear_interp_monthly(df_yields, shorter_mat_yrs, longer_mat_yrs):

    ################
    # INPUTS:
    # shorter_mat_yrs, longer_mat_yrs: same as what's in define_input_parameters()
    ################
    #
    # Output series will be a constant maturity series with maturity 
    # equal to "longer_mat_yrs" years minus 1 month:
    ################
    

    # Initialize output:
    column_heading = [str(longer_mat_yrs-1)+str('Y_11M')]
    interp_yields = initialize_output(df_yields, column_heading)
    
    # Ratios for interpolation:
    year_length = longer_mat_yrs - shorter_mat_yrs
    if (year_length < 0):
        raise(ValueError('Long maturity is smaller than short maturity...'))
    right_fraction = (year_length - (1/12)) / year_length
    left_fraction = 1 - right_fraction
    
    
    for tt in range(0, interp_yields.shape[0]):
        interp_yields.iloc[tt,0] = (left_fraction * df_yields.iloc[tt,0]) + \
            (right_fraction * df_yields.iloc[tt,1])
    
    return interp_yields


def linear_interp_daily(df_yields, shorter_mat_yrs, longer_mat_yrs):

    ################
    # INPUTS:
    # shorter_mat_yrs, longer_mat_yrs: same as what's in define_input_parameters()
    ################
    #
    # Output series will be a constant maturity series with maturity 
    # equal to "longer_mat_yrs" years minus 1 business day:
    ################
    
    # Initialize output:
    column_heading = [str(longer_mat_yrs-1)+str('Y_11M')]
    interp_yields = initialize_output(df_yields, column_heading)
    
    # Ratios for interpolation:
    year_length = longer_mat_yrs - shorter_mat_yrs
    if (year_length < 0):
        raise(ValueError('Long maturity is smaller than short maturity...'))
        
    left_fraction_d_tt = initialize_output(df_yields, ['DailyLeftFraction'])
    right_fraction_d_tt = initialize_output(df_yields, ['DailyRightFraction'])
    days_elapsed_d = initialize_output(df_yields, ['DaysElapsed'])
    
    for tt in range(1, interp_yields.shape[0]):
        
        # Probably overkill here (man I do not miss all the 360/365 bullshit), but 
        # use "timedelta" objects to determine actual number of days that have elapsed:
        datetime_object1 = datetime.strptime(df_yields.index[tt], '%m/%d/%Y')
        datetime_object2 = datetime.strptime(df_yields.index[tt-1], '%m/%d/%Y')
        delta_tt = datetime_object1 - datetime_object2
        days_elapsed = delta_tt.days
        days_elapsed_d.iloc[tt,0] = days_elapsed
        
        right_fraction_d_tt.iloc[tt,0] = (year_length - (days_elapsed/365)) / year_length
        left_fraction_d_tt.iloc[tt,0] = 1 - right_fraction_d_tt.iloc[tt,0]
    
    for tt in range(0, interp_yields.shape[0]):
        interp_yields.iloc[tt,0] = (left_fraction_d_tt.iloc[tt,0] * df_yields.iloc[tt,0]) + \
            (right_fraction_d_tt.iloc[tt,0] * df_yields.iloc[tt,1])
    
    return interp_yields, left_fraction_d_tt, right_fraction_d_tt, days_elapsed_d


def main():
    
    # Get input parameters:
    print("getting_data")
    yields_monthly, shorter_mat_yrs, longer_mat_yrs = define_input_parameters()
    
    # Call main function for monthly data:
    print("calling main")
    returns_monthly, interp_yields_monthly, \
    price_carry_return_m, coupon_return_m, total_return_m, \
    price_sold_at_m = monthly_yields_to_returns(\
                                                yields_monthly, \
                                                shorter_mat_yrs, \
                                                longer_mat_yrs)
    
    # # Call main function for daily data:
    # returns_daily, interp_yields_daily, \
    # price_carry_return_d, coupon_return_d, total_return_d, \
    # price_sold_at_d, \
    # left_fraction_d_tt, right_fraction_d_tt, days_elapsed_d = \
    #     daily_yields_to_returns(yields_daily, shorter_mat_yrs, longer_mat_yrs)

        
    return  yields_monthly, \
            interp_yields_monthly, \
            returns_monthly, \
            price_carry_return_m, coupon_return_m, total_return_m, \
            price_sold_at_m
            
# Desired output variable used in Essay #8 is "price_carry_return_m":
yields_monthly, interp_yields_monthly, \
returns_monthly, \
price_carry_return_m, coupon_return_m, total_return_m, \
price_sold_at_m = main()
print('done done')

