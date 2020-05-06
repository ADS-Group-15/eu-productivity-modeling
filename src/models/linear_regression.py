import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.data import data_loader

project_dir = Path(__file__).resolve().parents[2]


def train():
    print('Load dataset...')
    x_train, y_train, x_test, y_test = data_loader.load()

    print('Train LinearRegression...')
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    error = np.sqrt(mean_squared_error(y_test, lr.predict(x_test)))
    print(f'Error on test: {error}')

    print(f'Save trained model...')
    with open(os.path.join(project_dir, 'models', 'lr.pkl'), 'wb') as file:
        pickle.dump(lr, file)
    print('Done')


if __name__ == '__main__':
    train()
