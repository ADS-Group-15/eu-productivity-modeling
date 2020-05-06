import os
from pathlib import Path

import pandas as pd

from src.features.features import columns_to_fit

project_dir = Path(__file__).resolve().parents[2]


def load():
    data_interim_dir = os.path.join(project_dir, 'data', 'interim')

    train_df = pd.read_csv(os.path.join(data_interim_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_interim_dir, 'test.csv'))

    train_temp = train_df[columns_to_fit + ['compensation']].dropna()
    test_temp = test_df[columns_to_fit + ['compensation']].dropna()
    x_train, y_train = train_temp[columns_to_fit], train_temp[['compensation']]
    x_test, y_test = test_temp[columns_to_fit], test_temp[['compensation']]

    return x_train, y_train, x_test, y_test
