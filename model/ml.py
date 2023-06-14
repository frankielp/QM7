import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.preprocess import process_data

import torch
import torch.nn as nn
Y_SCALE_FACTOR=2000.0
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.Sigmoid())
        
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(nn.Sigmoid())
        
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def train_val_split(data,val_fold):
    val_idx = data['P'][val_fold].flatten()
    train_idx = np.array(list(set(data['P'].flatten()) - set(val_idx)))

    X = np.concatenate((data['cm'], data['eigs'], data['centralities']), axis=1)
    y = data['T'][0]/Y_SCALE_FACTOR
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    return X_train,y_train,X_val,y_val

    
def linear_regression(X_train,y_train,X_val,y_val):
    model=LinearRegression()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_val)
    loss=mean_absolute_error(y_val,y_pred)
    return loss
    
def kernel_ridge(X_train,y_train,X_val,y_val):
    model = KernelRidge(alpha = 1e-8, kernel = "laplacian", gamma=1/4000)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_val)
    loss=mean_absolute_error(y_val,y_pred)
    return loss
def svr (X_train, y_train, X_val, y_val):
    model = SVR(C=10, epsilon=0.2, kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    loss = mean_absolute_error(y_val, y_pred)
    return loss

def gaussian_process_regression(X_train, y_train, X_val, y_val):
    kernel = RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_val)
    loss = mean_absolute_error(y_val, y_pred)
    return loss

def check_svr(X_train, y_train, X_val, y_val):
    # Define the parameter grid
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf'],
        'C': [0.1, 1, 10],
        'epsilon': [0.1, 0.2, 0.3]
    }
    # Create an SVR instance
    model = SVR()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(np.concatenate((X_train, X_val), axis=0), np.append(y_train,y_val))

    # Get the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Parameters:", best_params)
    print("Best Score:", best_score)

if __name__=='__main__':
    train_ml()

