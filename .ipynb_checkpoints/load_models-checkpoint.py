import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from tabularbench.core.dataset_split import make_stratified_dataset_split
from pathlib import Path
from tabularbench.config.config_run import ConfigRun
from tabularbench.core.get_trainer import get_trainer
from tabularbench.core.trainer_finetune import TrainerFinetune
from tabularbench.models.foundation.foundation_transformer import FoundationTransformer
import xgboost as xgb
from xgboost import XGBClassifier
import optuna

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
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25)
            #train = xgb.DMatrix(train_x, label=train_y)
            #dvalid = xgb.DMatrix(valid_x, label=valid_y)
        
            param = {
                "verbosity": 0,
                "objective": "binary:logistic",
                # use exact for small dataset.
                "tree_method": "exact",
                # defines booster, gblinear for linear functions.
                "booster": trial.suggest_categorical("booster", ["gbtree"]),
                # L2 regularization weight.
                "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
                # L1 regularization weight.
                "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
                # sampling ratio for training data.
                "subsample": trial.suggest_float("subsample", 0.4, 0.8),
                # sampling according to each tree.
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            }
        
            if param["booster"] in ["gbtree", "dart"]:
                # maximum depth of the tree, signifies complexity of the tree.
                param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
                # minimum child weight, larger the term more conservative the tree.
                param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
                param["eta"] = trial.suggest_float("eta", 1e-3, 1e-1, log=True)
                # defines how selective algorithm is.
                param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
                param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        
            if param["booster"] == "dart":
                param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
                param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
                param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
                param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        
            model = XGBClassifier(**param)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            roc_auc = sklearn.metrics.roc_auc_score(y_test, preds)
            return roc_auc
            
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_optim, timeout=600)
        self.model = XGBClassifier(**study.best_params, objective='binary:logistic')
        self.model.fit(self.X,self.y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)