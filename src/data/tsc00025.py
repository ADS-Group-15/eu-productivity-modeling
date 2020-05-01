import os
from pathlib import Path

import pandas as pd

from src.data import preprocessor


def process():
    project_dir = Path(__file__).resolve().parents[2]
    data_raw_dir = os.path.join(project_dir, 'data', 'raw')
    data_interim_dir = os.path.join(project_dir, 'data', 'interim')

    file_path = os.path.join(data_raw_dir, 'tsc00025.tsv.gz')
    df = pd.read_csv(file_path, delimiter='\t')
    df = preprocessor.process(df)
    df = df[['year', 'GEO', 'value', 'sex']]
    df = df.groupby(['GEO', 'year']).mean().reset_index()
    df.rename(columns={
        'value': 'hrst'
    }, inplace=True)

    df.to_csv(os.path.join(data_interim_dir, 'hrst.csv'), index=False)


if __name__ == '__main__':
    process()
