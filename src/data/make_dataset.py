import os
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data import nama_lp_ulc, trng_lfs_02, tps00001
from src.features.features import columns_to_fit

pd.options.mode.chained_assignment = None
project_dir = Path(__file__).resolve().parents[2]
data_interim_dir = os.path.join(project_dir, 'data', 'interim')


def process_dfs():
    nama_lp_ulc.process()
    trng_lfs_02.process()
    tps00001.process()


def merge_dfs():
    compensation_df = pd.read_csv(os.path.join(data_interim_dir, 'compensation.csv'))
    education_df = pd.read_csv(os.path.join(data_interim_dir, 'education.csv'))
    population_df = pd.read_csv(os.path.join(data_interim_dir, 'population.csv'))
    rd_expenditure_df = pd.read_csv(os.path.join(data_interim_dir, 'rd_expenditure.csv'))

    df = compensation_df.merge(education_df, on=['year', 'GEO'])
    df = df.merge(population_df, on=['year', 'GEO'])
    df = df.merge(rd_expenditure_df, on=['year', 'GEO'])

    df.rename(columns={
        'Compensation of employees per hour worked (Euro)': 'per_hour_worked',
        'Compensation per employee (Euro)': 'per_employee'
    }, inplace=True)

    return df


def split_dataset(df, train_size=0.7):
    years = df['year'].unique()
    years = sorted(years)
    pivot = int(len(years) * train_size)
    train_index = years[:pivot]
    test_index = years[pivot:]

    train_df = df[df['year'].isin(train_index)]
    test_df = df[df['year'].isin(test_index)]

    return train_df, test_df


def add_features(df, features):
    shift_range = [1, 2]
    for feature in features:
        for shift in shift_range:
            df[f'{feature}_shift_{shift}'] = df.sort_values('year').groupby(['GEO'])[feature].shift(shift)
            df[f'{feature}_diff_{shift}'] = df[feature] - df[f'{feature}_shift_{shift}']

    agg_params = {}
    for feature in features:
        for shift in shift_range:
            agg_params[f'{feature}_shift_{shift}_mean'] = (f'{feature}_shift_{shift}', 'mean')

    agg = df.groupby('year').agg(**agg_params).reset_index()
    df = df.merge(agg, on='year')

    for feature in features:
        for shift in shift_range:
            df[f'{feature}_shift_{shift}_mean_diff'] = df[f'{feature}_shift_{shift}'] - df[
                f'{feature}_shift_{shift}_mean']
    return df


def scale_features(df, columns):
    print(df.columns)
    for column in columns:
        print(column)
        scaler = MinMaxScaler()
        df[column] = scaler.fit_transform(df[[column]])


def main():
    process_dfs()
    df = merge_dfs()

    # We need to drop data unless we have a good solution to impute missing data
    df = df[df['year'] < 2018]
    df.to_csv(os.path.join(data_interim_dir, 'dataset.csv'), index=False)

    df = add_features(df, ['education', 'population', 'rd_expenditure'])
    scale_features(df, columns_to_fit)

    train_df, test_df = split_dataset(df)
    train_df.to_csv(os.path.join(data_interim_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(data_interim_dir, 'test.csv'), index=False)


if __name__ == '__main__':
    main()
