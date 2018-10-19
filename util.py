import numpy as np

import enlighten
import json
import os

from sklearn.metrics import roc_auc_score as AUC, log_loss, accuracy_score as accuracy
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE

progress_manager = enlighten.get_manager()


def makedirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


def get_config_name(config):
    return ','.join(sorted('%s=%s' % (k, v) for k, v in config.items()))


def format_json(dict_):
    return json.dumps(dict_, indent=4, sort_keys=True)


def load_json(path):
    with open(path, 'rb') as f:
        return json.load(f)


def save_json(dict_, path):
    with open(path, 'w+') as f:
        json.dump(dict_, f, indent=4, sort_keys=True)


def train_and_eval_sklearn_classifier(clf, data):

    x_train = data['x_train']
    y_train = data['y_train']

    x_test = data['x_test']
    y_test = data['y_test']

    clf.fit(x_train, y_train)

    try:
        p = clf.predict_proba(x_train)[:, 1]   # sklearn convention
    except IndexError:
        p = clf.predict_proba(x_train)

    ll = log_loss(y_train, p)
    auc = AUC(y_train, p)
    acc = accuracy(y_train, np.round(p))

    try:
        p = clf.predict_proba(x_test)[:, 1]    # sklearn convention
    except IndexError:
        p = clf.predict_proba(x_test)

    ll = log_loss(y_test, p)
    auc = AUC(y_test, p)
    acc = accuracy(y_test, np.round(p))

    # return { 'loss': 1 - auc, 'log_loss': ll, 'auc': auc }
    return {'loss': ll, 'log_loss': ll, 'auc': auc}


def train_and_eval_sklearn_regressor(reg, data):

    x_train = data['x_train']
    y_train = data['y_train']

    x_test = data['x_test']
    y_test = data['y_test']

    reg.fit(x_train, y_train)
    p = reg.predict(x_train)

    mse = MSE(y_train, p)
    rmse = math.sqrt(mse)
    mae = MAE(y_train, p)

    print '\n# training | RMSE: {:.4f}, MAE: {:.4f}'.format(rmse, mae)

    #

    p = reg.predict(x_test)

    mse = MSE(y_test, p)
    rmse = math.sqrt(mse)
    mae = MAE(y_test, p)

    print '# testing  | RMSE: {:.4f}, MAE: {:.4f}'.format(rmse, mae)

    return {'loss': rmse, 'rmse': rmse, 'mae': mae}
