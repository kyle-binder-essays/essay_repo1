# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 20:52:43 2021
"""

##################################################
#
# For given time series of returns, provide
# typical modern portfolio stats like beta, Sharpe, 
# geometric/arithmetic reutnrs, INFORMATION RATIO, TRACKING ERROR
#
# And also a python-level graphic like a dataframe calendar
# of monthlly returns, yearly returns.
#
##################################################

from datetime import datetime as dtdt
import pandas as pd
import numpy as np
import calendar # for getting "business days" ('BM','BY') for monthly+yearly returns


def actual_weights_from_ideal_weights( ideal_weights , rebal_dates ):
    
    actual_weights = 3
    return actual_weights


def initialize_output(rtns):
    
    # Function to initialize output & keep row headings (dates) and column headings (tickers):
    nans = np.empty( ( rtns.shape[0] , rtns.shape[1] ) )
    nans[:] = np.nan
    output_df = pd.DataFrame(nans)
    output_df = output_df.set_index(rtns.index)
    output_df.columns = rtns.columns
    
    return output_df


def get_current_weights(historical_weights):
    
    # Initialize output & keep row headings (dates) and column headings (tickers):
    nans = np.empty( ( 1 , historical_weights.shape[1] ) )
    nans[:] = np.nan
    current_weights = pd.DataFrame(nans)
    current_weights = historical_weights.set_index(historical_weights.index)
    current_weights.columns = historical_weights.columns
    
    current_weights = historical_weights.iloc[-1,:]
    return current_weights


def get_calendar_returns_from_daily_returns(rtns):
    
    # Must have index (row headings) be daily frequency dates.
    monthlies = convert_daily_rtns_to_monthly_rtns(rtns)
    yearlies = convert_daily_rtns_to_yearly_rtns(rtns)
    calendar_df = calendar_dataframe( monthlies , yearlies )
    
    return calendar_df


def total_return_from_returns(returns):
    """Retuns the return between the first and last value of the DataFrame.
    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
    Returns
    -------
    total_return : float or pandas.Series
        Depending on the input passed returns a float or a pandas.Series.
    """
    return (returns + 1).prod() - 1


def convert_daily_rtns_to_monthly_rtns(daily_returns):
    
    # Variable "daily_returns" can take single or multiple columns.
    
    r = daily_returns.copy()
    monthlies = r.groupby((r.index.year, r.index.month))\
             .apply(total_return_from_returns)
    ############################################################
    # Date manipulations:
    ############################################################
    m_start_date = dtdt(monthlies.index[0][0] , \
                            monthlies.index[0][1] , 1 )
    last_day_m = calendar.monthrange(monthlies.index[-1][0],\
                                     monthlies.index[-1][1])[1]
    m_end_date = dtdt(monthlies.index[-1][0] , \
                          monthlies.index[-1][1] , last_day_m )
    business_m =pd.date_range(m_start_date, m_end_date, freq='BM')
    
    # Need to convert "monthlies" from series to Dataframe for the 
    # following "monthlies['BusDate'] = business_m" to work:
    monthlies = pd.DataFrame(monthlies)
    
    monthlies['BusDate'] = business_m  
    monthlies.loc[:,'BusDate'] = business_m  
    
    monthlies['BusDate'] = [time.date() for time in monthlies['BusDate']]
    monthlies = monthlies.set_index('BusDate')
    monthly_returns = monthlies
    return monthly_returns
    

def convert_daily_rtns_to_yearly_rtns(daily_returns):
    
    # Variable "daily_returns" can take single or multiple columns.
    r = daily_returns.copy()
    yearlies = r.groupby((r.index.year))\
             .apply(total_return_from_returns)
    ############################################################
    # Date manipulations:
    ############################################################
    y_start_date = dtdt(yearlies.index[0],1,1)
    y_end_date = dtdt(yearlies.index[-1],12,31)
    business_y =pd.date_range(y_start_date, y_end_date, freq='BY')
    
    yearlies = pd.DataFrame(yearlies)
    
    yearlies['BusDate'] = business_y 
    yearlies['BusDate'] = [time.date() for time in yearlies['BusDate']]
    yearlies = yearlies.set_index('BusDate')
    yearly_returns = yearlies
    return yearly_returns


def convert_monthly_rtns_to_yearly_rtns(monthly_returns):
    
    # Variable "monthly_returns" can take single or multiple columns.
    r = monthly_returns.copy()
    
    # Here we go: before applying (as we did in daily-to-monthly) the 
    # following "index.year", we have to convert "r.index" to a DATETIME 
    # object.  Right now the dtype of "r.index" is "object".
    r = r.set_index(pd.to_datetime(r.index))
    
    yearlies = r.groupby((r.index.year))\
             .apply(total_return_from_returns)
    ############################################################
    # Date manipulations:
    ############################################################
    y_start_date = dtdt(yearlies.index[0],1,1)
    y_end_date = dtdt(yearlies.index[-1],12,31)
    business_y =pd.date_range(y_start_date, y_end_date, freq='BY')
    
    yearlies = pd.DataFrame(yearlies)
    
    yearlies['BusDate'] = business_y 
    yearlies['BusDate'] = [time.date() for time in yearlies['BusDate']]
    yearlies = yearlies.set_index('BusDate')
    yearly_returns = yearlies
    return yearly_returns


def calendar_dataframe( monthly_returns , yearly_returns ):
    
    years = pd.DatetimeIndex(yearly_returns.index).year
    months_tt = pd.DatetimeIndex(monthly_returns.index).month
    years_tt = pd.DatetimeIndex(monthly_returns.index).year
    
    # Initialize output: 
    # 13 columns: 12 months plus YEAR
    # Rows/Index = number of years (including partial years):
    nans = np.empty( ( yearly_returns.shape[0] , 13 ) )
    nans[:] = np.nan
    calendar_df = pd.DataFrame(nans)
    calendar_df = calendar_df.set_index(years)
    calendar_df.columns = ['ANNUAL', 'Jan','Feb','Mar', 'Apr','May','Jun',\
                         'Jul','Aug','Sep', 'Oct','Nov','Dec']
    
    # Dictionary to map month numbers to names:
    month_dict={1:'Jan', 2:'Feb',3:'Mar', 4:'Apr',5:'May',6:'Jun',\
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
#    month_dict_offset={0:'Jan', 1:'Feb',2:'Mar', 3:'Apr',4:'May',5:'Jun',\
#           6:'Jul',7:'Aug',8:'Sep',9:'Oct',10:'Nov',11:'Dec'}

    # Populate Yearly Returns:
    # lol, don't forget ".values"
    calendar_df.loc[:,'ANNUAL'] = yearly_returns.values
    
    # Populate Monthly Returns:
    for tt in range(0,monthly_returns.shape[0]):
        y = years_tt[tt]
        m = months_tt[tt]
        
        output_column = month_dict[m]
        
        # Populate output:
        calendar_df.loc[y,output_column] = monthly_returns.iloc[tt,0]
    
    return calendar_df


def get_backtest_rtns_simple_rebal( ideal_weights , rtns ):
    
    # Won't have a return for first period:
    position_rtns = ideal_weights.iloc[0:-1,:].values * rtns.iloc[1:,:].values
    
    # Assign dates and column headers:
    output_position_rtns = pd.DataFrame(position_rtns)
    output_position_rtns.columns = rtns.columns
    output_position_rtns = output_position_rtns.set_index(rtns.index[1:])    
    
    # Collapse all position returns into single period return:
    backtest_rtns = np.sum(position_rtns,axis=1)
    output_backtest_rtns = pd.DataFrame(backtest_rtns)
    output_backtest_rtns = output_backtest_rtns.set_index(rtns.index[1:]) 
    
    return output_position_rtns, output_backtest_rtns
    