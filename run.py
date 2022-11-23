#!/usr/bin/env python

import pickle as pkl
import pandas as pd
from sklearn.preprocessing import StandardScaler



def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''
    predictions_fp = 'predictions.csv'
    if 'test' in targets:
        data = pd.read_csv('test/testdata.csv')

        print(data.shape)

        X_test = data.drop(columns='bad')
        y_test = data.bad.astype(int)

        print(X_test.shape)

        scaler = StandardScaler()
        scaler.fit(X_test)
        scaled_data = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
        X_test = scaled_data

        pklmodel = pkl.load(open('src/data/model.pkl', 'rb'))

        predictions = pd.Series(pklmodel.predict(X_test))


        predictions.to_csv(predictions_fp, index_label=False)

        return


if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)