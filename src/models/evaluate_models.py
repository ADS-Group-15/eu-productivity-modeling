import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.data import data_loader

project_dir = Path(__file__).resolve().parents[2]


def evaluate():
    print('Load dataset...')
    x_train, y_train, x_test, y_test = data_loader.load()

    print('Load saved models...')
    with open(os.path.join(project_dir, 'models', 'lr.pkl'), 'rb') as file:
        lr = pickle.load(file)
    with open(os.path.join(project_dir, 'models', 'gbm.pkl'), 'rb') as file:
        gbm = pickle.load(file)

    print('Evaluate models on test...')
    rows = []
    rows.append({
        'model': 'LinearRegression',
        'error': np.sqrt(mean_squared_error(y_test, lr.predict(x_test)))
    })
    rows.append({
        'model': 'Tuned LightGBM',
        'error': np.sqrt(mean_squared_error(y_test, gbm.predict(x_test)))
    })
    df = pd.DataFrame(rows)
    print(df)

    print('Save results...')
    df.to_csv(os.path.join(project_dir, 'reports', 'evaluation.csv'), index=False)
    print('Done')


if __name__ == '__main__':
    evaluate()
