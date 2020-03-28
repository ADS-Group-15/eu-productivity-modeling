import os
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.features.features import columns_to_fit


def train():
    print('Train LinearRegression...')
    project_dir = Path(__file__).resolve().parents[2]
    data_interim_dir = os.path.join(project_dir, 'data', 'interim')

    train_df = pd.read_csv(os.path.join(data_interim_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_interim_dir, 'test.csv'))

    train_temp = train_df[columns_to_fit + ['per_hour_worked']].dropna()
    test_temp = test_df[columns_to_fit + ['per_hour_worked']].dropna()
    x_train, y_train = train_temp[columns_to_fit], train_temp[['per_hour_worked']]
    x_test, y_test = test_temp[columns_to_fit], test_temp[['per_hour_worked']]

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    error = mean_squared_error(y_test, lr.predict(x_test))
    print(f'Error on test: {error}')


if __name__ == '__main__':
    train()
