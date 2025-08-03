import pandas as pd
import numpy as np
from data.load_data import load_compustat,  load_earning_calls, load_returns
from scipy.stats.mstats import winsorize

def keep_matching_data(df1, df2, id):
    ''' keep only data shared by two datasets, matching date, quarter and id (gvkey or cusip)'''
    df1 = df1.set_index([id,'year', 'quarter'])
    df2 = df2.set_index([id,'year', 'quarter'])
    df1 = df1.loc[list(set(df1.index) & set(df2.index))]
    df2 = df2.loc[list(set(df1.index) & set(df2.index))]
    return df1.reset_index(), df2.reset_index()



def fill_missing(data):
    '''Fill missing data, ensuring that data is grouped by stock and sorted by date.'''
    data.reset_index()
    filled_stocks = []
    
    for stock_id, stock in data.groupby('gvkey'):
        stock_sorted = stock.sort_values(by=['year', 'quarter']).copy()
        filled_stock = stock_sorted.ffill()
        filled_stocks.append(filled_stock)
    
    return pd.concat(filled_stocks)


def clean_compustat_data(small_compstat, threshold = 0.9):
    ''' Drop columns with more that 100*threshold% NaN, constant columns and fill with forard fill'''
    small_compstat = small_compstat.set_index(['gvkey', 'cusip', 'year', 'quarter']).dropna(axis = 1, thresh = threshold*small_compstat.shape[0])
    small_compstat = small_compstat.loc[:, small_compstat.nunique(dropna=True) > 1]
    small_compstat = fill_missing(small_compstat).dropna().reset_index()

    return small_compstat


def get_rid_outliers(df, MIN_PERCENTILE=0.01, MAX_PERCENTILE=0.99):
    '''Get rid of outliers using winsorization'''
    df_winsorized = df.copy()
    for col in df.columns:
        df_winsorized[col] = winsorize(df[col], limits=[MIN_PERCENTILE, 1 - MAX_PERCENTILE])
    return df_winsorized

def shift(returns):
    '''Shift return of one quarter, stock by stock'''
    returns['ret'] = returns.sort_values(['cusip', 'year', 'quarter']) \
                             .groupby('cusip')['ret'] \
                             .shift(1)
    return returns




def get_clean_data():
    ''' get compustat, returns and earning calls cleaned data with matching indices'''
    small_compstat = load_compustat()
    
    #fyr represents the month that define the fiscal year, we might have couple entries for same date if firm has changed its fiscal year at some point
    small_compstat = small_compstat.sort_values(by = 'fyr', ascending=False).drop_duplicates(subset=['cusip', 'year', 'quarter']).drop(columns=['fyearq', 'fqtr', 'fyr'])
    #keep only one for cusip first 8 
    small_compstat['cusip'] = small_compstat['cusip'].str[:8]
    small_compstat = small_compstat.drop_duplicates(subset=['cusip', 'year', 'quarter'])
    
    #We first keep data shared by the three datasets before cleaning
    small_ret = load_returns()
    small_ret = shift(small_ret).dropna()
    small_compstat, small_ret = keep_matching_data(small_compstat, small_ret, 'cusip')

    conversation = load_earning_calls()
    small_compstat, conversation = keep_matching_data(small_compstat, conversation, 'gvkey')

    small_compstat = clean_compustat_data(small_compstat)
    

    #keeping only data not thrown away when cleaning
    small_compstat, small_ret = keep_matching_data(small_compstat, small_ret, 'cusip')
    small_compstat, conversation = keep_matching_data(small_compstat, conversation, 'gvkey')

    small_compstat = get_rid_outliers(small_compstat.set_index(['cusip', 'gvkey', 'year', 'quarter'])).reset_index()
    small_ret = get_rid_outliers(small_ret.set_index(['cusip', 'year', 'quarter'])).reset_index()

    return small_compstat, small_ret, conversation

