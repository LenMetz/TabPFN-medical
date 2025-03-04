import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split, GridSearchCV
from tabularbench.core.dataset_split import make_stratified_dataset_split
from pathlib import Path
from tabularbench.config.config_run import ConfigRun
from tabularbench.core.get_trainer import get_trainer
from tabularbench.core.trainer_finetune import TrainerFinetune
from tabularbench.models.foundation.foundation_transformer import FoundationTransformer
import xgboost as xgb
from xgboost import XGBClassifier
import torch
import catboost as cb
import optuna
from math import e
#from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from data_prep_utils import *
from sklearn.linear_model import LogisticRegression


optuna.logging.set_verbosity(optuna.logging.WARNING)


class TabForestPFNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, model_path="", cfg_path="", max_epochs=10):
        super().__init__()
        self.model_path = model_path
        self.cfg_path = cfg_path
        self.max_epochs = max_epochs
        self.cfg = ConfigRun.load(Path(cfg_path))
        self.cfg.hyperparams["max_epochs"]=max_epochs
        self.cfg.hyperparams["max_samples_query"] = 512
        self.cfg.hyperparams["max_samples_support"] = 4096
        self.cfg.device = "cpu"
        model = FoundationTransformer(
            n_features=self.cfg.hyperparams['n_features'],
            n_classes=self.cfg.hyperparams['n_classes'],
            dim=self.cfg.hyperparams['dim'],
            n_layers=self.cfg.hyperparams['n_layers'],
            n_heads=self.cfg.hyperparams['n_heads'],
            attn_dropout=self.cfg.hyperparams['attn_dropout'],
            y_as_float_embedding=self.cfg.hyperparams['y_as_float_embedding'],
            use_pretrained_weights=self.cfg.hyperparams['use_pretrained_weights'],
            path_to_weights=Path(model_path)
        )
        self.trainer = TrainerFinetune(self.cfg, model, n_classes=2)

    
    def fit(self, X, y):

        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        X_train, X_valid, y_train, y_valid = make_stratified_dataset_split(X, y)
        self.trainer.train(X_train, y_train, X_valid, y_valid)

        return self
    
    def predict(self, X):

        logits =  self.trainer.predict(self.X_, self.y_, X)
        return logits.argmax(axis=1)
    
    def predict_proba(self, X):
        logits = self.trainer.predict(self.X_, self.y_, X)
        return np.exp(logits) / np.exp(logits).sum(axis=1)[:, None]

class LogisticRegressionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression()

    def fit(self, X, y):
        X = normalize(X)
        self.model.fit(X,y)

    def predict(self, X):
        return self.model.predict(normalize(X))
    
    def predict_proba(self, X):
        return self.model.predict_proba(normalize(X))
        

class MajorityClass(BaseEstimator, ClassifierMixin):
    def __init__(self, maj_class=0):
        super().__init__()
        self.maj_class = maj_class
    
    def fit(self, X, y):
        counts = np.unique(y, return_counts=True)
        self.maj_class = counts[0][np.argmax(counts[1])]
    def predict(self, X):
        return np.full(X.shape[0], self.maj_class)
    
    def predict_proba(self, X):
        return np.full(X.shape[0], self.maj_class)

'''class CatBoostOptim(BaseEstimator, ClassifierMixin):
    
    def __init__(self, X=None, y=None, n_optim=10):
        super().__init__()
        self.X = X
        self.y = y
        self.n_optim=n_optim
        self.model = cb.CatBoostClassifier()

    def fit(self, X, y):
        self.X = X
        self.y = y
        def objective(trial):
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25)
            #train = xgb.DMatrix(train_x, label=train_y)
            #dvalid = xgb.DMatrix(valid_x, label=valid_y)
        
            param = {
                "objective": trial.suggest_categorical("objective", ["CrossEntropy"]),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
                "depth": trial.suggest_int("depth", 1, 12),
                "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                "bootstrap_type": trial.suggest_categorical(
                    "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
                ),
                "used_ram_limit": "3gb",
            }
        
            if param["bootstrap_type"] == "Bayesian":
                param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 1)
            elif param["bootstrap_type"] == "Bernoulli":
                param["subsample"] = trial.suggest_float("subsample", 0.1, 1)
        
            model = cb.CatBoostClassifier(**param, num_trees=100)
            model.fit(X_train, y_train, verbose=False, eval_set=[(X_test, y_test)], early_stopping_rounds=10)
            preds = model.predict(X_test)
            roc_auc = sklearn.metrics.roc_auc_score(y_test, preds)
            return roc_auc
            
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_optim, timeout=600)
        self.model = cb.CatBoostClassifier(**study.best_params)
        self.model.fit(self.X,self.y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)'''

class XGBoostOptim(BaseEstimator, ClassifierMixin):
    
    def __init__(self, X=None, y=None, n_optim=10):
        super().__init__()
        self.X = X
        self.y = y
        self.n_optim=n_optim
        self.model = XGBClassifier()

    def fit(self, X, y):
        self.X = X
        self.y = y
        def objective(trial):
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1, stratify=self.y)
            #train = xgb.DMatrix(train_x, label=train_y)
            #dvalid = xgb.DMatrix(valid_x, label=valid_y)
        
            param = {
                "verbosity": 0,
                "objective": "binary:logistic",
                # use exact for small dataset.
                "tree_method": "auto",
                # defines booster, gblinear for linear functions.
                "booster": "gbtree",
                # L2 regularization weight.
                "lambda": trial.suggest_float("lambda", 1e-8, 1e2, log=True),
                # L1 regularization weight.
                "alpha": trial.suggest_float("alpha", 1e-8, 1e5, log=True),
                # sampling ratio for training data.
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                # sampling according to each tree.
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.2, 1.0),
            }
        
            if param["booster"] in ["gbtree", "dart"]:
                # maximum depth of the tree, signifies complexity of the tree.
                param["max_depth"] = trial.suggest_int("max_depth", 1, 10, step=1)
                # minimum child weight, larger the term more conservative the tree.
                param["min_child_weight"] = trial.suggest_float("min_child_weight", 1e-8, 1e5, log=True)
                param["eta"] = trial.suggest_float("eta", 1e-5, 1.0, log=True)
                # defines how selective algorithm is.
                param["gamma"] = trial.suggest_float("gamma", 1e-8, 1e2, log=True)
                param["grow_policy"] = "depthwise"
        
            if param["booster"] == "dart":
                param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
                param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
                param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
                param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        
            model = XGBClassifier(n_estimators=5, **param)
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_test)
            roc_auc = sklearn.metrics.roc_auc_score(torch.nn.functional.one_hot(torch.tensor(y_test).to(torch.int64)).numpy(), preds, multi_class="ovr")
            return roc_auc
            
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_optim, timeout=600)
        self.model = XGBClassifier(n_estimators=5, **study.best_params, objective='binary:logistic')
        self.model.fit(self.X,self.y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

class XGBoostGrid(BaseEstimator, ClassifierMixin):
    
    def __init__(self, X=None, y=None, n_optim=10):
        super().__init__()
        self.X = X
        self.y = y
        self.n_optim=n_optim
        self.model = XGBClassifier()

    def fit(self, X, y):
        self.X = X
        self.y = y
        
        param_grid = {
            'learning_rate': [0.01, 0.1, 1.0],
            'max_depth': [5, 7, 9],
            'subsample': [0.5, 0.7],
            'n_estimators': [5,25,50],
            'gamma': [0.1,0.5,0.9]
            
        }
        grid_search = GridSearchCV(estimator = self.model, param_grid=param_grid, cv=3, scoring="roc_auc")
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        self.model =  XGBClassifier(**best_params,)
        self.model.fit(X,y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)



class CatBoostGrid(BaseEstimator, ClassifierMixin):
    
    def __init__(self, X=None, y=None, n_optim=10):
        super().__init__()
        self.X = X
        self.y = y
        self.n_optim=n_optim
        self.model = cb.CatBoostClassifier(silent=True)

    def fit(self, X, y):
        self.X = X
        self.y = y
        
        param_grid = {
            'depth':[6,9],
            'iterations':[100, 200],
            'learning_rate':[0.01,0.1,0.3], 
            'l2_leaf_reg':[1,10]
        }
        grid_search = GridSearchCV(estimator = self.model, param_grid=param_grid, cv=3, scoring="roc_auc")
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        self.model =  cb.CatBoostClassifier(**best_params, silent=True)
        self.model.fit(X,y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

class AutoGluon(BaseEstimator, ClassifierMixin):

    def __init__(self):
        super().__init__()
        self.model = TabularPredictor(label="label", eval_metric='roc_auc', verbosity=0)
    def fit(self, X, y):
        data = pd.DataFrame(X)
        data.insert(0,"label",y)
        self.model.fit(data)

    def predict(self,X):
        data = pd.DataFrame(X)
        return self.model.predict(data)
        
    def predict_proba(self,X):
        data = pd.DataFrame(X)
        return self.model.predict_proba(data)
