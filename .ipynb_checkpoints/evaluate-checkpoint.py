from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
import numpy as np
import sklearn
from data_prep_utils import *
import time

def scores(y_test, y_pred):
    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), roc_auc_score(y_test, y_pred)

# return cv folds preserving original class distribution
def stratified_split(data, labels, cv=3):
    size = labels.shape[0]
    fold_size = size//cv
    counts = np.unique(labels, return_counts=True)
    c0_size = np.floor(fold_size*counts[1][0]/size).astype(int)
    c1_size = np.floor(fold_size*counts[1][1]/size).astype(int)#fold_size-c0_size
    
    c0_data = data[labels==0]
    c1_data = data[labels==1]
    np.random.shuffle(c0_data)
    np.random.shuffle(c1_data)
    
    data_folds, labels_folds = [], []
    for f in range(cv):
        data_single_fold = np.concatenate((c0_data[c0_size*f:c0_size*(f+1),:],c1_data[c1_size*f:c1_size*(f+1),:]))
        labels_single_fold = np.concatenate((np.zeros((c0_size)), np.ones((c1_size)))).astype(int)
        data_single_fold, labels_single_fold = unison_shuffled_copies(data_single_fold, labels_single_fold)
        data_folds.append(data_single_fold)
        labels_folds.append(labels_single_fold)
        
    return data_folds, labels_folds

def cross_validate_sample(model, X, y, metrics, strat_split=True, cv=3, sampling=None):
    if strat_split:
        X_folds, y_folds = stratified_split(X, y, cv)
    else:
        X_folds, y_folds = np.array_split(X, cv), np.array_split(y, cv)
    results = np.zeros((len(metrics)+1))
    for run in range(cv):
        model_clean = sklearn.base.clone(model)
        X_folds = X_folds[-run:] + X_folds[:-run]
        y_folds = y_folds[-run:] + y_folds[:-run]
        X_train, y_train = np.concatenate(tuple(X_folds[:-1])), np.concatenate(tuple(y_folds[:-1]))
        X_test, y_test = X_folds[-1], y_folds[-1]
        if sampling: X_train, y_train = sampling(X_train, y_train)
        X_train, y_train = unison_shuffled_copies(X_train, y_train)
        start_time = time.time()
        if model_clean.__class__.__name__=="TabPFNClassifier": 
            model_clean.fit(X_train, y_train, overwrite_warning=True)
        else:
            model_clean.fit(X_train, y_train)
        preds = model_clean.predict(X_test)
        results[len(metrics)] += time.time() - start_time
        for i, m in enumerate(metrics):
            if m == "roc_auc_ovr":
                results[i] += sklearn.metrics.roc_auc_score(y_test, preds, multi_class="ovr")
            else:
                results[i] += sklearn.metrics.get_scorer(m)._score_func(y_test, preds)
    results = results/cv
    return results
        