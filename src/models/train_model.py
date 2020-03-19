import pandas as pd
import os
import pickle
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    project_dir = Path(__file__).resolve().parents[2]
    data_interim_dir = os.path.join(project_dir, 'data', 'interim')

    df = pd.read_csv(os.path.join(data_interim_dir, 'dataset.csv'))

    scaler = StandardScaler()
    df['normalized_per_hour_worked'] = scaler.fit_transform(
        df[['Compensation of employees per hour worked (Euro)']])
    df['normalized_per_employee'] = scaler.fit_transform(
        df[['Compensation per employee (Euro)']])
    df['normalized_education'] = scaler.fit_transform(df[['education']])

    data = df.dropna()
    x = data[['normalized_education']]
    y = data[['normalized_per_hour_worked']]
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    model = LinearRegression()
    model.fit(x_train, y_train)
    print(mean_squared_error(y_train, model.predict(x_train)))
    print(mean_squared_error(y_test, model.predict(x_test)))
    print(model.coef_)

    with open(os.path.join(data_interim_dir, 'linear_regression.pkl'), 'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    main()
