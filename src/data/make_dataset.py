import os
from pathlib import Path

import pandas as pd

from src.data import nama_lp_ulc
from src.data import trng_lfs_02


def main():
    project_dir = Path(__file__).resolve().parents[2]
    data_interim_dir = os.path.join(project_dir, 'data', 'interim')

    nama_lp_ulc.process()
    trng_lfs_02.process()

    compensation_df = pd.read_csv(os.path.join(data_interim_dir, 'compensation.csv'))
    education_df = pd.read_csv(os.path.join(data_interim_dir, 'education.csv'))

    df = compensation_df.merge(education_df, on=['year', 'GEO'])
    df.to_csv(os.path.join(data_interim_dir, 'dataset.csv'), index=False)


if __name__ == '__main__':
    main()
