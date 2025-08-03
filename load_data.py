import pandas as pd
import numpy as np

#First date of earning calls
STARTDATE = '2010-01-04'


def load_returns(path='targets/monthly_crsp.csv'):
    ret = pd.read_csv(path)
    #Drop data with no ID and no Return
    ret = ret.dropna(subset=['CUSIP', 'MthRet'])
    ret['date'] = pd.to_datetime(ret['MthCalDt'])
    ret = ret.loc[ret['date'] >= STARTDATE].drop(columns=['MthCalDt'])
    ret = ret[['CUSIP', 'MthRet', 'sprtrn', 'date']].rename(
        columns={'CUSIP': 'cusip', 'MthRet': 'ret', 'sprtrn': 'mrkt_ret'}
    )
    #splitting date into year and quarter
    ret['year'] = ret['date'].dt.year
    ret['quarter'] = np.ceil(ret['date'].dt.month / 3).astype(int)

    # Compute quarterly cumulative returns
    ret = ret.drop(columns='date').groupby(['cusip', 'year', 'quarter']).apply(
        lambda x: (x + 1).prod() - 1
    ).reset_index()
    return ret



def load_compustat(path='predictors/CompFirmCharac.csv'):
    df = pd.read_csv(path)
    df['datadate'] = pd.to_datetime(df['datadate'])
    df = df[df['datadate'] > STARTDATE]
    #splitting date into year and quarter
    df['year'] = df['datadate'].dt.year
    df['quarter'] = np.ceil(df['datadate'].dt.month / 3).astype(int)
    df= df.drop(columns=['datadate'])
    #dropping non numerical columns
    df = df.drop(columns = [col for col in df.select_dtypes(include = ['object', 'string']).columns if col != 'cusip'])
    return df




def load_earning_calls(path = 'predictors/earnings_calls.parquet'):
    earning_calls = pd.read_parquet(path)

    small_calls = earning_calls.loc[earning_calls['mostimportantdateutc']> STARTDATE]
    #Concatenate full conversation into one input
    dict_data = dict({'gvkey':[], 'date':[], 'text':[]})
    for id , df in small_calls.groupby(['gvkey', 'mostimportantdateutc']):
        dict_data['gvkey'].append(id[0])
        dict_data['date'].append(id[1])
        df = df.sort_values('componentorder')
        df['text'] = df['transcriptcomponenttypename'] + ' : ' + df['componenttext']
        all_text = df["text"].str.cat(sep=' \n')
        dict_data['text'].append(all_text)

    conversation = pd.DataFrame(dict_data)

    #splitting date into year and quarter
    conversation['year'] = conversation['date'].dt.year
    conversation['quarter'] = np.ceil(conversation['date'].dt.month/3)
    conversation = conversation.drop(columns='date').groupby(['gvkey', 'year','quarter']).sum().reset_index()
    conversation['gvkey'] = conversation['gvkey'].astype('int')

    return conversation


