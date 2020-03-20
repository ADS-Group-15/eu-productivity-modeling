from src.data import preprocessor

import pandas as pd
import pytest


def test_preprocess_with_empty_df():
    df = pd.DataFrame()
    with pytest.raises(IndexError):
        preprocessor.process(df)


def test_preprocess_with_standard_df():
    df = pd.DataFrame({
        'unit,geo\\time': ['Euro,AR', 'Euro,BR', 'Euro,CZ'],
        '2010 ': ['1,000.1', '2', '3c'],
        '2011 ': ['4', '5', '6'],
    })
    df = preprocessor.process(df)
    assert set(df.columns) == {'GEO', 'flags', 'unit', 'value', 'year'}
    assert df[df['GEO'] == 'Argentina'].iloc[0]['value'] == 1000.1
    assert df[df['GEO'] == 'Czechia'].iloc[0]['flags'] == 'conditional'
