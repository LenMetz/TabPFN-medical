import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# method to fix class imbalance by adding random samples from less frequent class
def oversample(X, y):
    min_class = np.argmin([X[np.where(y==0)].shape[0],X[np.where(y==1)].shape[0]])
    X_min = X[np.where(y==min_class)]
    n_new_samples = X.shape[0]-X_min.shape[0]
    indices = np.random.randint(0,X_min.shape[0], n_new_samples)
    new_samples = X_min[indices]
    X_new = np.concatenate((X, new_samples), axis=0)
    y_new = np.concatenate((y, np.full(min_class,n_new_samples)))
    return X_new, y_new

# method to fix class imbalance by removing random samples from more frequent class
def undersample(X, y):
    min_class = np.argmin([X[np.where(y==0)].shape[0],X[np.where(y==1)].shape[0]])
    max_class = np.argmax([X[np.where(y==0)].shape[0],X[np.where(y==1)].shape[0]])
    X_min = X[np.where(y==min_class)]
    X_maj = X[np.where(y!=min_class)]
    n_sample_min = X_min.shape[0]
    X_new = np.concatenate((X_min, X_maj[:n_sample_min]))
    y_new = np.concatenate((np.full(n_sample_min, min_class),np.full(n_sample_min, max_class)))
    return X_new, y_new

# fetches microbiome data, reduces to healthy/CRC binary classification and converts to numpy
def get_microbiome(path):
    df = pd.read_csv(path)
    df_binary = df.loc[(df["disease"] == "healthy") | (df["disease"]=="CRC")]
    df_data = df_binary.iloc[:,2:-5]
    data = df_data.to_numpy()
    labels = df_binary["disease"].to_numpy()
    labels[labels=="healthy"] = 0
    labels[labels=="CRC"] = 1
    return data, labels.astype(int)

# select features with most non-zero entries
def top_non_zero(data, reduced_length=100):
    counts = np.count_nonzero(data, axis=0)
    indices = np.argsort(counts)[::-1]
    return data[:,indices[:reduced_length]]

# applies same shuffle to two array with the same shape in first dimension
def unison_shuffled_copies(a, b):
    p = np.random.permutation(a.shape[0])
    return a[p], b[p]

# TabPFN friendly train test split
def tabpfn_split(X,y,split=0.8, seed=42):
    n_samples = X.shape[0]
    train_size = min(1024, int(n_samples*0.8))
    test_size = min(1024, int(n_samples*0.2))
    return train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=seed)

# reduce dataset to fit into tabpfn max length in samples = 1024, reduction is intended for non-trivial sample selection
def reduce_n_samples(X, y, max_length=1024, reduction=None):
    if not reduction:
        choice = np.arange(min(X.shape[0],1024))
    return X[choice], y[choice]

#reduce dataset to fit max number of features (=100 for tabpfn), selection is intended for non-trivial feature selection
def reduce_n_features(X, max_features=100, selection=None):
    if not selection:
        choice = np.arange(min(X.shape[1],max_features))
    return X[:,choice]

# coverts datafram to numpy array, changes categorical variables to unique indexes (int)
def df_to_numpy(df):
    if len(df.shape)>1:
        data = df.apply(lambda x: pd.factorize(x)[0]).to_numpy()
    else:
        data = pd.factorize(df)[0]
    return data

#
def normalize(data):
    scaler = preprocessing.StandardScaler().fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled, scaler