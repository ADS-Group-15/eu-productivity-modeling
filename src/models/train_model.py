import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def main():
    project_dir = Path(__file__).resolve().parents[2]
    data_interim_dir = os.path.join(project_dir, 'data', 'interim')

    df = pd.read_csv(os.path.join(data_interim_dir, 'dataset.csv'))

    scaler = StandardScaler()
    df['per_hour_worked'] = scaler.fit_transform(df[['Compensation of employees per hour worked (Euro)']])
    df['per_employee'] = scaler.fit_transform(df[['Compensation per employee (Euro)']])
    df['education'] = scaler.fit_transform(df[['education']])
    df['population'] = scaler.fit_transform(df[['population']])
    df['rd_expenditure'] = scaler.fit_transform(df[['rd_expenditure']])

    data = df.dropna()
    features = ['education', 'population', 'rd_expenditure']
    x = data[features]
    y = data[['per_hour_worked']]
    scores_on_train = []
    scores_on_test = []
    kfold = KFold(n_splits=2, shuffle=True, random_state=42)
    for train_index, test_index in kfold.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = LinearRegression()
        model.fit(x_train, y_train)
        scores_on_train.append(mean_squared_error(y_train, model.predict(x_train)))
        scores_on_test.append(mean_squared_error(y_test, model.predict(x_test)))

    print(f'Average score on train: {np.mean(scores_on_train)}')
    print(f'Average score on test: {np.mean(scores_on_test)}')


if __name__ == '__main__':
    main()
