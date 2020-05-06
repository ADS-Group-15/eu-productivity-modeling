import os
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error

from src.data import data_loader

project_dir = Path(__file__).resolve().parents[2]


def train():
    print('Load dataset...')
    x_train, y_train, x_test, y_test = data_loader.load()

    print('Search LightGBM parameters...')

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

    print('Train LightGBM...')
    dtrain = lgb.Dataset(x_train, label=y_train)
    params = trial.params
    params['objective'] = 'regression'
    tuned_gbm = lgb.train(params, dtrain)
    error = np.sqrt(mean_squared_error(y_test, tuned_gbm.predict(x_test)))
    print(f'Error on test: {error}')

    print(f'Save trained model...')
    with open(os.path.join(project_dir, 'models', 'gbm.pkl'), 'wb') as file:
        pickle.dump(tuned_gbm, file)
    print('Done')


if __name__ == '__main__':
    train()
