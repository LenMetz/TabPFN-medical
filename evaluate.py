from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
import numpy as np
import sklearn
from data_prep_utils import *
import time
import torch

def scores(y_test, y_pred):
    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), roc_auc_score(y_test, y_pred)

# return cv folds preserving original class distribution
def stratified_split(data, labels, cv=3, max_samples=None, seed=42):
    if max_samples is None:
        size = labels.shape[0]
    else:
        size = min(labels.shape[0],np.floor(max_samples*(cv/(cv-1))))
    fold_size = size//cv
    counts = np.unique(labels, return_counts=True)
    c0_size = np.floor(fold_size*counts[1][0]/labels.shape[0]).astype(int)
    c1_size = np.floor(fold_size*counts[1][1]/labels.shape[0]).astype(int)#fold_size-c0_size
    c0_data = data[labels==0]
    c1_data = data[labels==1]
    np.random.default_rng(seed=seed).shuffle(c0_data)
    np.random.default_rng(seed=seed).shuffle(c1_data)
    
    data_folds, labels_folds = [], []
    for f in range(cv):
        data_single_fold = np.concatenate((c0_data[c0_size*f:c0_size*(f+1),:],c1_data[c1_size*f:c1_size*(f+1),:]))
        labels_single_fold = np.concatenate((np.zeros((c0_size)), np.ones((c1_size)))).astype(int)
        data_single_fold, labels_single_fold = unison_shuffled_copies(data_single_fold, labels_single_fold, seed)
        data_folds.append(data_single_fold)
        labels_folds.append(labels_single_fold)
        
    return data_folds, labels_folds

def cross_validate_sample(model, X, y, metrics, strat_split=True, cv=3, sampling=None, 
                          reducer=NonZeroSelect(), max_samples=None, seed=42, overwrite=True,
                         n_best_delete=0, recomp=False):
    if strat_split:
        X_folds, y_folds = stratified_split(X, y, cv, max_samples,seed=seed)
    else:
        X_folds, y_folds = np.array_split(X, cv), np.array_split(y, cv)
    results = [[] for _ in range(len(metrics)+1)]
    for run in range(cv):
        model_clean = sklearn.base.clone(model)
        X_folds = X_folds[-run:] + X_folds[:-run]
        y_folds = y_folds[-run:] + y_folds[:-run]
        X_train, y_train = np.concatenate(tuple(X_folds[:-1])), np.concatenate(tuple(y_folds[:-1]))
        X_test, y_test = X_folds[-1], y_folds[-1]
        if sampling: X_train, y_train = sampling(X_train, y_train)
        X_train, y_train = unison_shuffled_copies(X_train, y_train,seed=seed)
        X_test, y_test = unison_shuffled_copies(X_test, y_test,seed=seed)
        if reducer is not None:
            X_train, X_test = remove_same_features_traintest(X_train, X_test)
            if reducer.__class__.__name__=="AnovaSelect":
                reducer.fit(X_train, y_train)
            else:
                reducer.fit(np.concatenate((X_train,X_test),axis=0), np.concatenate((y_train,y_test),axis=0))
            if n_best_delete>0:
                to_delete = reducer.feature_indices[:n_best_delete]
                X_train, X_test = np.delete(X_train, to_delete,1), np.delete(X_test, to_delete,1)
                reducer.fit(X_train, y_train)
            #print(X_train.shape)
            X_train = reducer.transform(X_train)
            X_test = reducer.transform(X_test)
            if recomp:
                X_train, X_test = data_to_comp(X_train), data_to_comp(X_test)
        start_time = time.time()
        if model_clean.__class__.__name__=="TabPFNClassifier" or  model_clean.__class__.__name__=="MedPFNClassifier":
            if overwrite:
                model_clean.fit(X_train, y_train, overwrite_warning=True)
            else:
                X_train, y_train = reduce_n_samples(X_train, y_train, 1024)
                model_clean.fit(X_train, y_train, overwrite_warning=True)
        else:
            model_clean.fit(X_train, y_train)
        with torch.no_grad():
            #preds = model_clean.predict(X_test)
            probs = model_clean.predict_proba(X_test)
            #if model_clean.__class__.__name__=="MedPFNClassifier":
            if len(probs.shape)>1:
                preds = np.argmax(probs, axis=1)
                probs = (probs[:,1]-probs[:,0]+1)*0.5
            else:
                preds = (probs>0.5).astype(float)
        results[-1].append(time.time() - start_time)
        for i, m in enumerate(metrics):
            if m =="roc_auc":
                results[i].append(sklearn.metrics.get_scorer(m)._score_func(y_test, probs))
            else:
                results[i].append(sklearn.metrics.get_scorer(m)._score_func(y_test, preds))
        #print(results)
        del model_clean
    results_mean = np.mean(np.array(results), axis=1)
    results_std = np.std(np.array(results), axis=1)
    return results_mean, results_std
        