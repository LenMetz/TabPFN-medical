import random
import math

import torch
from torch import nn
import numpy as np
from sklearn.tree import DecisionTreeRegressor


from tabpfn.utils import default_device
from .utils import get_batch_to_dataloader

class GaussianNoise(nn.Module):
    def __init__(self, std, device):
        super().__init__()
        self.std = std
        self.device=device

    def forward(self, x):
        return x + torch.normal(torch.zeros_like(x), self.std)


def causes_sampler_f(num_causes):
    means = np.random.normal(0, 1, (num_causes))
    std = np.abs(np.random.normal(0, 1, (num_causes)) * means)
    return means, std

def get_batch(batch_size, seq_len, num_features, hyperparameters, device=default_device, num_outputs=1, sampling='normal'
              , epoch=None, **kwargs):
    
    def get_sample():
        n_classes = get_n_classes(hyperparameters["max_classes"])
        categorical_perc = get_categorical_perc(hyperparameters["categorical_x"])
        depth = get_depth(hyperparameters["min_depth"], hyperparameters["max_depth"])
        n_features = get_n_features(hyperparameters["min_features"], hyperparameters["max_features"])
        n_categorical_features = get_n_categorical_features(categorical_perc, n_features)
        n_categorical_classes = get_n_categorical_classes(n_categorical_features)
        x = np.random.normal(size=(hyperparameters["base_size"], n_features))
        y = np.random.normal(0, 1, size=(hyperparameters["base_size"],))

        clf = DecisionTreeRegressor(
            max_depth=depth,
            max_features='sqrt',
        )
        clf.fit(x, y)

        x2 = np.random.normal(size=(hyperparameters["n_samples"], n_features))
        x2 = transform_some_features_to_categorical(x2, n_categorical_features, n_categorical_classes)

        z = clf.predict(x2)
        #z = quantile_transform(z)
        z = put_in_buckets(z, n_classes)
        return x2, z
    

    sample = [get_sample() for _ in range(0, batch_size)]

    x, y = zip(*sample)
    x = torch.tensor(x)
    y = torch.tensor(y)
    
    return x, y, y
    
def get_n_classes(max_classes: int) -> int:
    return np.random.randint(2, max_classes, size=1).item()

def get_categorical_perc(categorical_x: bool) -> float:
    if categorical_x:
        return np.random.uniform(0, 1, size=(1,)).item()
    else:
        return 0
    
def get_depth(min_depth: int, max_depth: int) -> int:
    if min_depth == max_depth:
        return min_depth
    else:
        return np.random.randint(min_depth, max_depth, size=1).item()
    
def get_n_features(min_features: int, max_features: int) -> int:
    if min_features == max_features:
        return min_features
    else:
        return np.random.randint(min_features, max_features, size=1).item()
    
def get_n_categorical_features(categorical_perc: float, n_features: int) -> int:
    return int(categorical_perc * (n_features + 1))

def get_n_categorical_classes(n_categorical_features: int) -> np.ndarray:
    return np.random.geometric(p=0.5, size=(n_categorical_features,)) + 1

def put_in_buckets(z: np.ndarray, n_classes: int) -> np.ndarray:
    buckets = np.random.uniform(0, 1, size=(n_classes-1,))
    buckets.sort()
    buckets = np.hstack([buckets, 1])
    b = np.argmax(z <= buckets[:, None], axis=0)

    return b

def transform_some_features_to_categorical(
        x: np.ndarray, 
        n_categorical_features: int, 
        n_categorical_classes: int
    ) -> np.ndarray:

    if n_categorical_features == 0:
        return x
    
    x_index_categorical = np.random.choice(np.arange(x.shape[1]), size=(n_categorical_features,), replace=False)
    x_categorical = x[:, x_index_categorical]

    #quantile_transformer = QuantileTransformer(output_distribution='uniform')
    #x_categorical = quantile_transformer.fit_transform(x_categorical)

    for i in range(n_categorical_features):
        x_categorical[:, i] = put_in_buckets(x_categorical[:, i], n_categorical_classes[i])

    x[:, x_index_categorical] = x_categorical

    return x

DataLoader = get_batch_to_dataloader(get_batch)