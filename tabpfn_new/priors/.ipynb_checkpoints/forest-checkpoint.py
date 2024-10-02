import random
import math

import torch
from torch import nn
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

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

    # zero-inflated negative binomial distribution
    def zinb(size=(1000,100)):
        pi = 0.25
        p = np.random.uniform(0.95,0.99, size=size[1])
        p = np.repeat(np.expand_dims(p,axis=0),size[0],axis=0)
        X = np.random.negative_binomial(100,p)
        X = np.random.binomial(1,1-pi,size)*X
        return X
    
    # makes dataset compositional
    def to_comp(X):
        return np.expand_dims(1/np.sum(X,axis=1),axis=1)*X
    
    # multinomial dirichlet distribution
    def multinomial_dirichlet(size=(1000,100)):
        M = hyperparameters["mnd_M"] if "mnd_M" in hyperparameters else 1000
        a1 = hyperparameters["mnd_a1"] if "mnd_a1" in hyperparameters else 1
        a2 = hyperparameters["mnd_a2"] if "mnd_a2" in hyperparameters else 5
        a1 = np.random.uniform(0.5, 5, size[1])
        a2 = np.random.uniform(0.5, 10, size[1])
        alphas = np.random.beta(a1,a2)#,size[1])
        thetas = [np.random.dirichlet(alphas) for i in range(size[0])]
        #print(thetas, np.sum(thetas))
        X = np.asarray([np.random.multinomial(M, theta)/M for theta in thetas])
        #X = X + np.random.normal(0,1e-2,X.shape)
        return X
    def get_sample_function(name):
        if name=="normal":
            return np.random.normal
        if name=="zinb":
            return zinb
        if name=="mnd":
            return multinomial_dirichlet
        else:
            raise Exception("Sample function "+ name + " not found!")

    y_std = hyperparameters.get("y_std", 1)
    n_classes = get_n_classes(hyperparameters["num_classes"])
    #categorical_perc = get_categorical_perc(hyperparameters["categorical_x"])
    depth = get_depth(hyperparameters["min_depth"], hyperparameters["max_depth"])
    #print("forest tree depth: ", depth)
    n_features = get_n_features(hyperparameters["min_features"], hyperparameters["max_features"])
    print(n_features, depth)
    #n_categorical_features = get_n_categorical_features(categorical_perc, n_features)
    #n_categorical_classes = get_n_categorical_classes(n_categorical_features)
    if "data_sample_func" in hyperparameters:
        data_sample_func = get_sample_function(hyperparameters["data_sample_func"])
    else: 
        data_sample_func = np.random.normal
        
    def get_sample():
        
        x = data_sample_func(size=(hyperparameters["base_size"]*2, n_features))
        x1, x2 = x[:hyperparameters["base_size"]], x[hyperparameters["base_size"]:]
        if hyperparameters["comp"]:
            x1 = to_comp(x1)
        y = np.random.normal(0, y_std, size=(hyperparameters["base_size"],))
        if hyperparameters.get("alt_ys", False):
            y = np.concatenate((np.abs(np.random.normal(0, y_std, size=(int(hyperparameters["base_size"]/2),)))+0.15,
                -np.abs(np.random.normal(0, y_std, size=(int(hyperparameters["base_size"]/2),)))-0.15))
            np.random.shuffle(y)
        #y = np.concatenate((np.ones(int(hyperparameters["base_size"]/2)), np.zeros(int(hyperparameters["base_size"]/2))))
        #np.random.shuffle(y)
        #c = np.random.choice(np.arange(hyperparameters["base_size"]), size=int(hyperparameters["base_size"]/2), replace=False)
        clf = DecisionTreeRegressor(
            max_depth=depth,
            max_features='sqrt',#'sqrt',
        )
        clf.fit(x1, y)
        if hyperparameters["comp"]:
            x2 = to_comp(x2)
        #x2 = transform_some_features_to_categorical(x2, n_categorical_features, n_categorical_classes)

        z = clf.predict(x2)
        #z = quantile_transform(z)
        #z = put_in_buckets(z, n_classes)
        return np.expand_dims(x2,1), np.expand_dims(z,1)
    

    sample = [get_sample() for _ in range(0, batch_size)]
    x, y = zip(*sample)
    x, y = np.concatenate(x,1), np.concatenate(y,1)
    x = torch.tensor(x).float().to(device)
    y = torch.tensor(y).float().to(device)
    
    return x, y, y
    
def get_n_classes(max_classes: int) -> int:
    if max_classes>2:
        return np.random.randint(2, max_classes, size=1).item()
    else: return 2
        
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