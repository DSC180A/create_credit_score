from zipfile import ZipFile
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

file_path = 'data/raw/forStudents.pkl.zip'

def run(path):
    zip_file = ZipFile(path)
    df = pd.read_pickle(zip_file.open('forStudents.pkl'))

    df_clean = df[df['all0000'].notna()]
    df_clean = df_clean[df_clean['bad'] != -1.0]
    df_clean = df_clean[df_clean['is_app_approved'] == 1]
    df_clean.drop(columns=['bad_v2','evaluation_dt','vintage', "vantage3_score", "bad_balance", "current_balance", "net_spend"], inplace=True)
    described = df_clean["annual_income"].describe()
    quartiles = list(described.iloc[4:7])

    def assign_quartile(row):
        income = row["annual_income"]
        if income < quartiles[0]:
            return 0
        elif ((quartiles[0] < income) & (quartiles[1] >= income)):
            return 1
        elif ((quartiles[1] < income) & (quartiles[2] >= income)):
            return 2
        else:
            return 3

    df_clean['income_quartile'] = df_clean.apply(assign_quartile, axis=1)

    dfs = []
    for i in range(4):
        quartile = df_clean.loc[df_clean["income_quartile"] == i]
        dfs.append(quartile.fillna(quartile.mean()))
        
    df_clean = pd.concat(dfs)

    temp = df_clean.isna().sum().to_frame()
    temp[temp[0] != 0]

    df_clean.drop(columns=['all9230', 'all9240', 'all9249', 'all9280'], inplace=True)
    df_clean = df_clean.fillna(df_clean.mean())

    def process_categorical_data(dataset):
        """ One hot encodes all of the categorial columns of the dataset.  Removes the original columns """
        # select categorical data
        categorical_data = dataset.select_dtypes('object')
        
        enc = OneHotEncoder(handle_unknown='ignore')
        
        ohe_data = enc.fit_transform(categorical_data).toarray()
        column_names = enc.get_feature_names(categorical_data.columns)
        
        add = dataset.select_dtypes(exclude='object')
        added = pd.DataFrame(ohe_data,columns=column_names).astype(int)
        
        add.reset_index(drop = True, inplace= True)
        added.reset_index(drop = True, inplace= True)
        X = pd.concat([add, added], axis=1)
        
        return X

    df_clean = process_categorical_data(df_clean)

    compression_opts = dict(method='zip',archive_name='clean.csv')
    df_clean.to_csv('data/out/clean.zip', index=False,compression=compression_opts)  

run(file_path)
    