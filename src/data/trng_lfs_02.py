import os
import re
from pathlib import Path

import pandas as pd
from iso3166 import countries

country_code_dict = {
    'EL': 'GR',
    'UK': 'GB'
}


def country_code_to_name(code):
    if code in country_code_dict:
        code = country_code_dict[code]
    try:
        return countries.get(code).name
    except KeyError:
        return None


country_name_dict = {
    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
}


def normalize_country_name(name):
    if name in country_name_dict:
        return country_name_dict[name]
    else:
        return name


flag_dict = {
    ':': 'not available',
    'p': 'provisional',
    'b': 'break in time series',
    'd': 'definition differs',
    'c': 'conditional',
    'e': 'estimated',
}


def extract_flags(p_str):
    flag = None
    for char in flag_dict:
        if char in p_str:
            flag = flag_dict[char]
            break
    return flag


def preprocess(df):
    # Rename
    columns_to_rename = {c: c.rstrip() for c in df.columns if c.endswith(' ')}
    raw_base_column = [c for c in df.columns if c.endswith('\\time')][0]  # e.g. 'unit,isced11,sex,age,geo\\time'
    base_column = raw_base_column.replace('\\time', '')
    columns_to_rename[raw_base_column] = base_column
    df.rename(columns=columns_to_rename, inplace=True)

    # Unpivot
    df = df.melt(id_vars=base_column, var_name='year', value_name='value')

    # Split
    columns_to_split = base_column.split(',')
    df[columns_to_split] = df[base_column].str.split(',', expand=True)
    df.drop(columns=[base_column], inplace=True)

    # Normalize country name
    df.rename(columns={'geo': 'GEO'}, inplace=True)
    df['GEO'] = df['GEO'].apply(country_code_to_name)
    df['GEO'] = df['GEO'].apply(normalize_country_name)

    # Extract flags
    df['flags'] = df['value'].apply(lambda p_str: extract_flags(p_str))
    df['value'] = df['value'].apply(lambda p_str: re.sub('[^\d\.]', '', p_str))
    df['value'] = df['value'].replace('', None, regex=True)

    # Convert data types
    df['year'] = df['year'].astype(int)
    df['value'] = df['value'].astype(float)

    return df


def process():
    project_dir = Path(__file__).resolve().parents[2]
    data_raw_dir = os.path.join(project_dir, 'data', 'raw')
    data_interim_dir = os.path.join(project_dir, 'data', 'interim')

    file_path = os.path.join(data_raw_dir, 'trng_lfs_02.tsv')
    df = pd.read_csv(file_path, delimiter='\t')
    df = preprocess(df)
    df = df[df['age'] == 'Y18-24']
    df.drop(columns=['age', 'unit', 'isced11', 'flags'], inplace=True)
    df = df.groupby(['GEO', 'year'])['value'].mean().reset_index()
    df.rename(columns={
        'value': 'education',
        'sex': 'education_sex',
    }, inplace=True)

    df.to_csv(os.path.join(data_interim_dir, 'education.csv'), index=False)


if __name__ == '__main__':
    process()
