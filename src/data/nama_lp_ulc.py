import locale
import os
from pathlib import Path

import pandas as pd


def atof(value):
    if value == ':':
        return None
    return locale.atof(value.replace(',', ''))


def nantonone(flag):
    if flag == 'nan':
        return None
    return flag


def truncate(geo, limit=20):
    return (geo[:limit] + '..') if len(geo) > limit else geo


country_name_dict = {
    'Germany (until 1990 former territory of the FRG)': 'Germany'
}


def normalize_country_name(name):
    if name in country_name_dict:
        return country_name_dict[name]
    else:
        return name


def process():
    project_dir = Path(__file__).resolve().parents[2]
    data_raw_dir = os.path.join(project_dir, 'data', 'raw')
    data_interim_dir = os.path.join(project_dir, 'data', 'interim')

    dfs = []
    for i in [1, 2]:
        filename = f'nama_10_lp_ulc_{i}_Data.csv'
        file_path = os.path.join(data_raw_dir, filename)
        df = pd.read_csv(file_path)

        geos_to_exclude = [
            'European Union - 27 countries (from 2020)',
            'European Union - 28 countries (2013-2020)',
            'European Union - 15 countries (1995-2004)',
            'Euro area (EA11-1999, EA12-2001, EA13-2007, EA15-2008, EA16-2009, EA17-2011, EA18-2014, EA19-2015)',
            'Euro area - 19 countries  (from 2015)',
            'Euro area - 12 countries (2001-2006)']
        df = df[~df['GEO'].isin(geos_to_exclude)]

        df['TIME'] = df['TIME'].astype(int)
        df['GEO'] = df['GEO'].apply(normalize_country_name)
        df['Value'] = df['Value'].apply(atof)
        df['Flag and Footnotes'] = df['Flag and Footnotes'].apply(nantonone)
        dfs.append(df)

    compensation_per_hour_df = dfs[0]
    compensation_per_hour_df.rename(columns={'Value': 'Compensation of employees per hour worked (Euro)'}, inplace=True)

    compensation_per_employee_df = dfs[1]
    compensation_per_employee_df.rename(columns={'Value': 'Compensation per employee (Euro)'}, inplace=True)

    compensation_per_hour_df = compensation_per_hour_df[
        ['TIME', 'GEO', 'Compensation of employees per hour worked (Euro)']]
    compensation_per_employee_df = compensation_per_employee_df[
        ['TIME', 'GEO', 'Compensation per employee (Euro)']]

    df = compensation_per_hour_df.merge(compensation_per_employee_df, on=['TIME', 'GEO'])
    df.rename(columns={'TIME': 'year'}, inplace=True)

    df.to_csv(os.path.join(data_interim_dir, 'compensation.csv'), index=False)


if __name__ == '__main__':
    process()
