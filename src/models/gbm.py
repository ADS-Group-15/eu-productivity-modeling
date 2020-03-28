import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error

from src.features.features import columns_to_fit


def train():
    print('Train LightGBM...')
    project_dir = Path(__file__).resolve().parents[2]
    data_interim_dir = os.path.join(project_dir, 'data', 'interim')

    train_df = pd.read_csv(os.path.join(data_interim_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_interim_dir, 'test.csv'))

    train_temp = train_df[columns_to_fit + ['per_hour_worked']].dropna()
    test_temp = test_df[columns_to_fit + ['per_hour_worked']].dropna()
    x_train, y_train = train_temp[columns_to_fit], train_temp[['per_hour_worked']]
    x_test, y_test = test_temp[columns_to_fit], test_temp[['per_hour_worked']]

    import lightgbm as lgb
    import optuna

    def objective(trial):
        param = {
            "objective": "regression",
            "metric": "l2",
            "verbosity": 0,
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }

        dtrain = lgb.Dataset(x_train, label=y_train)
        gbm = lgb.train(param, dtrain)
        error = mean_squared_error(y_test, gbm.predict(x_test))
        return error

    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print(" {}: {}".format(key, value))

    dtrain = lgb.Dataset(x_train, label=y_train)
    params = trial.params
    params['objective'] = 'regression'
    tuned_gbm = lgb.train(params, dtrain)
    error = mean_squared_error(y_test, tuned_gbm.predict(x_test))
    print(f'Error on test: {error}')


if __name__ == '__main__':
    train()
