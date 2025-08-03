import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from data.process_data import get_clean_data



def split_train_test(df, split = 0.8):
    ''' split data for train and test based on quarter'''
    df = df.sort_values(by=['year', 'quarter']).reset_index(drop=True)

    # Get unique time periods
    time_periods = df[['year', 'quarter']].drop_duplicates()
    n_total = len(time_periods)
    n_train = int(n_total * split)

    # Define the cutoff date for train/test
    train_periods = time_periods.iloc[:n_train]
    #test_periods = time_periods.iloc[n_train:]

    # Merge to get train and test sets
    df['is_train'] = df[['year', 'quarter']].apply(tuple, axis=1).isin(train_periods.apply(tuple, axis=1))

    train_df = df[df['is_train']].drop(columns='is_train')
    test_df = df[~df['is_train']].drop(columns='is_train')

    return train_df, test_df



def transform_PCA(X_train, X_test, variance_threshold=0.95):
    ''' Apply PCA to features in order to reduce dimensionality'''
    pca = PCA(variance_threshold)

    # Set index
    index_cols = ['cusip', 'gvkey', 'year', 'quarter']
    X_train_idx = X_train[index_cols]
    X_test_idx = X_test[index_cols]

    # Standardize test and train data, both with train moments
    X_test_scaled = (X_test.drop(columns=index_cols) - X_train.drop(columns=index_cols).mean()) / X_train.drop(columns=index_cols).std()
    X_train_scaled = (X_train.drop(columns=index_cols) - X_train.drop(columns=index_cols).mean()) / X_train.drop(columns=index_cols).std()
    
    

    corr_matrix = X_train_scaled.corr()
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True)
    # plt.title("Correlation matrix of cleaned firm characteristics (before PCA)")
    # plt.tight_layout()
    # plt.savefig("plots/corr_matrix.png", dpi=300)
    # plt.close()


    # Fit PCA on training data and transform
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # explained_var_ratio = np.cumsum(pca.explained_variance_ratio_)

    # plt.figure(figsize=(8, 6))
    # plt.plot(np.arange(1, len(explained_var_ratio) + 1), explained_var_ratio, marker='o')
    # plt.xlabel("Number of principal components")
    # plt.ylabel("Cumulative explained variance")
    # plt.title("Cumulative explained variance ratio of PCA components")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("plots/pca_explained_var.png", dpi=300)
    # plt.close()

    # Return as DataFrame with index columns
    X_train_pca_df = pd.DataFrame(X_train_pca, index=X_train.index)
    X_test_pca_df = pd.DataFrame(X_test_pca, index=X_test.index)

    X_train_pca_df = pd.concat([X_train_idx.reset_index(drop=True), X_train_pca_df.reset_index(drop=True)], axis=1)
    X_test_pca_df = pd.concat([X_test_idx.reset_index(drop=True), X_test_pca_df.reset_index(drop=True)], axis=1)

    return X_train_pca_df, X_test_pca_df



def get_train_test_data():
    ''' get preprocessed train and test data'''
    comstat, ret, conv = get_clean_data()
    comstat = comstat.set_index(['year', 'quarter'])
    non_constant_cols = comstat.nunique(dropna=True)
    comstat = comstat.loc[:, non_constant_cols[non_constant_cols > 1].index].reset_index()

    X_train, X_test = split_train_test(comstat)
    X_train, X_test = transform_PCA(X_train, X_test)

    y_train = X_train.merge(ret, on =['cusip', 'year','quarter'], how = 'inner').set_index(['cusip','gvkey', 'year', 'quarter'])[['ret']]
    X_train = X_train.merge(conv, on =['gvkey', 'year','quarter'], how = 'inner').set_index(['cusip','gvkey', 'year', 'quarter'])
    y_test = X_test.merge(ret, on =['cusip', 'year','quarter'], how = 'inner').set_index(['cusip','gvkey', 'year', 'quarter'])[['ret']]
    X_test = X_test.merge(conv, on =['gvkey', 'year','quarter'], how = 'inner').set_index(['cusip','gvkey', 'year', 'quarter'])

    return X_train, X_test, y_train, y_test
