from src.data import make_dataset

import pandas as pd


def test_split_dataset():
    df = pd.DataFrame([
        {'year': 2010, 'value': 1},
        {'year': 2011, 'value': 2},
        {'year': 2012, 'value': 3}
    ])
    train_df, test_df = make_dataset.split_dataset(df, train_size=0.7)
    assert len(train_df) == 2
    assert len(test_df) == 1


def test_add_features():
    df = pd.DataFrame([
        {'year': 2010, 'value': 1, 'GEO': 'a'},
        {'year': 2011, 'value': 2, 'GEO': 'a'},
        {'year': 2012, 'value': 3, 'GEO': 'a'}
    ])
    make_dataset.add_features(df, ['value'])
    assert df.iloc[0]['value_mean'] == 2
    assert df.iloc[0]['value_sum'] == 6
    assert df.iloc[2]['value_shift_1'] == 2.0
    assert df.iloc[2]['value_shift_2'] == 1.0


def test_scale_features():
    df = pd.DataFrame([
        {'year': 2010, 'value': 1},
        {'year': 2011, 'value': 2},
        {'year': 2012, 'value': 3}
    ])
    make_dataset.scale_features(df, ['value'])
    assert df.iloc[0]['value'] == 0.0
    assert df.iloc[1]['value'] == 0.5
    assert df.iloc[2]['value'] == 1.0
