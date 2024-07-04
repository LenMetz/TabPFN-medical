from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
import numpy as np
import sklearn
from data_prep_utils import *

def scores(y_test, y_pred):
    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), roc_auc_score(y_test, y_pred)

def cross_validate_sample(model, X, y, metrics, cv=3, sampling=None):
    X_folds = np.array_split(X, cv)
    y_folds = np.array_split(y, cv)
    results = np.zeros((len(metrics)))
    for run in range(cv):
        model_clean = sklearn.base.clone(model)
        X_folds = X_folds[-run:] + X_folds[:-run]
        y_folds = y_folds[-run:] + y_folds[:-run]
        X_train, y_train = np.concatenate(tuple(X_folds[:-1])), np.concatenate(tuple(y_folds[:-1]))
        X_test, y_test = X_folds[-1], y_folds[-1]
        if sampling: X_train, y_train = sampling(X_train, y_train)
        X_train, y_train = unison_shuffled_copies(X_train, y_train)
        if model_clean.__class__.__name__=="TabPFNClassifier": X_train, y_train = reduce_n_samples(X_train, y_train)
        model_clean.fit(X_train, y_train)
        preds = model_clean.predict(X_test)
        for i, m in enumerate(metrics):
            results[i] += sklearn.metrics.get_scorer(m)._score_func(y_test, preds)
    results = results/cv
    return results
        