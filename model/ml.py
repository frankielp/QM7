import numpy as np  # linear algebra
import torch.nn as nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

Y_SCALE_FACTOR = 2000.0


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        # self.layers.append(nn.Dropout(0.2))

        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.layers.append(nn.ReLU())
            # self.layers.append(nn.Dropout(0.2))

        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def linear_regression(X_train, y_train, X_val, y_val):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    loss = mean_absolute_error(y_val * Y_SCALE_FACTOR, y_pred * Y_SCALE_FACTOR)
    return loss


def kernel_ridge(X_train, y_train, X_val, y_val):
    model = KernelRidge(alpha=1e-8, kernel="laplacian", gamma=1 / 4000)
    # model = KernelRidge(alpha=1.67e-7, kernel="rbf", gamma=1 / (2 * (77 ** 2)))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    loss = mean_absolute_error(y_val * Y_SCALE_FACTOR, y_pred * Y_SCALE_FACTOR)
    return loss


def svr(X_train, y_train, X_val, y_val):
    model = SVR(C=10, epsilon=0.2, kernel="linear")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    loss = mean_absolute_error(y_val * Y_SCALE_FACTOR, y_pred * Y_SCALE_FACTOR)
    return loss


def gaussian_process_regression(X_train, y_train, X_val, y_val):
    kernel = RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_val)
    loss = mean_absolute_error(y_val * Y_SCALE_FACTOR, y_pred * Y_SCALE_FACTOR)
    return loss


def check_svr(X_train, y_train, X_val, y_val):
    # Define the parameter grid
    param_grid = {
        "kernel": ["linear", "poly", "rbf"],
        "C": [0.1, 1, 10],
        "epsilon": [0.1, 0.2, 0.3],
    }
    # Create an SVR instance
    model = SVR()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(np.concatenate((X_train, X_val), axis=0), np.append(y_train, y_val))

    # Get the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Parameters:", best_params)
    print("Best Score:", best_score)


def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
