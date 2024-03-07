import numpy as np
from scipy.special import erf
from scipy.io import loadmat
import os
from fxpmath import Fxp


data_filename = 'colon.mat'
TEMPLATE = Fxp(None, True, 32, 16)

def load_dataset(directory=None):
    directory = (os.path.dirname(os.path.realpath(__file__)) + '/../datasets' if directory is None else directory) + '/colon/'
    dataset = load_data(directory)
    return dataset

def load_dataset_pr(directory=None):
    directory = (os.path.dirname(os.path.realpath(__file__)) + '/../datasets' if directory is None else directory) + '/colon/'
    dataset = load_data_pr(directory)
    return dataset

def load_data(directory):
    info = {
        'raw': {}
    }

    mat = loadmat(directory + data_filename)

    with open(directory + data_filename) as f:
        info['raw']['data'] = np.asarray(mat['data']).T
        info['raw']['label'] = np.asarray(mat['labels']).astype(int)
        info['raw']['label'][info['raw']['label'] == 2] = 0

    return info

def load_data_pr(directory):
    info = {
    'raw': {}
    }

    mat = loadmat(directory + data_filename)

    with open(directory + data_filename) as f:
        info['raw']['data'] = Fxp(np.asarray(mat['data']).T).like(TEMPLATE)
        info['raw']['label'] = Fxp(np.asarray(mat['labels']).astype(int)).like(TEMPLATE)
        info['raw']['label'][info['raw']['label'] == 2] = Fxp(0).like(TEMPLATE)
        
    return info

class Normalize:

    def __init__(self):
        self.stats = None

    def fit(self, X):
        X_mean = np.mean(X, axis=0)
        X_std = np.sqrt(np.square(X - X_mean).sum(axis=0) / max(1, len(X) - 1))
        self.stats = (X_mean, X_std)

    def transform(self, X):
        transformed_X = erf((X - self.stats[0]) / (np.maximum(1e-6, self.stats[1]) * np.sqrt(2.)))
        return transformed_X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
