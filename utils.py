import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

Y_SCALE_FACTOR = 2000.0


def process_data_gnn(datadir):
    data = scipy.io.loadmat(datadir)
    dataset = data["X"]
    folds = data["P"]
    label = data["T"]
    A_hat_list = np.zeros((len(dataset), 23, 23))
    D_list = np.zeros((len(dataset), 23, 23))
    print("Processing data")
    for index, item in tqdm(enumerate(dataset)):
        item = item.tolist()
        A = np.zeros((23, 23))
        for row in range(len(item)):
            for col in range(len(item[row])):
                if item[row][col] > 0.0:
                    A[row][col] = 1
        # Get degree matrix
        A_hat = A + np.eye(23)
        degree = []
        for row in A_hat:
            degree.append(row.sum())
            D = np.diag(degree)
        # Normalize
        D = np.linalg.inv(np.linalg.cholesky(D))
        A_hat_list[index] = A_hat
        D_list[index] = D
    return dataset, A_hat_list, D_list, folds, label


def process_data_sorted(datadir):
    warnings.simplefilter("ignore")

    rand_state = 42
    np.random.seed(rand_state)

    qm7 = scipy.io.loadmat(datadir)
    # Cartesian Coordinate matrix
    R = qm7["R"]

    # Reshape energy to (7165,) and normalize
    y = np.transpose(qm7["T"]).reshape((7165,))
    std = (y - y.mean()).std()
    y_scaled = (y - y.mean()) / std

    ## Step 1: Take upper half
    # (idx) to access the elements of the matrix in the upper triangle including the diagonal (k=0).
    # (idx_dist) to access the elements of the distance matrix in the upper triangle excluding the diagonal (k=1).
    num_atoms = 23
    idx = np.triu_indices(num_atoms, k=0)
    idx_dist = np.triu_indices(num_atoms, k=1)

    # Initialize placeholder
    coulomb_matrix = np.zeros(
        (qm7["X"].shape[0], num_atoms * (num_atoms + 1) // 2), dtype=float
    )
    eigs = np.zeros((qm7["X"].shape[0], num_atoms), dtype=float)
    centralities = np.zeros((qm7["X"].shape[0], num_atoms), dtype=float)
    dist_matrix = np.zeros(
        (qm7["X"].shape[0], ((num_atoms * num_atoms) - num_atoms) // 2), dtype=float
    )

    for i in tqdm(range(len(qm7["X"]))):
        # Extract the Coulomb vector from the Coulomb matrix (cm) and sort the elements in descending order.
        cm = qm7["X"][i]
        coulomb_vector = cm[idx]
        sorted_cv = np.argsort(-coulomb_vector)
        coulomb_matrix[i] = coulomb_vector[sorted_cv]

        ## Step 2: Compute the distance matrix (dist) from the coordinate matrix (R)
        # Extract the distance vector from the distance matrix, then sort the elements in descending order.
        dist = squareform(pdist(R[i]))
        dist_vector = dist[idx_dist]
        sorted_dv = np.argsort(-dist_vector)
        dist_matrix[i] = dist_vector[sorted_dv]

        # Compute the eigenvalues (w) and eigenvectors (v) of the distance matrix.
        eig_value, eig_vector = np.linalg.eig(dist)
        eigs[i] = eig_value[np.argsort(-eig_value)]

        ## Step 3: Compute the eigenvector centrality of the graph created from the distance matrix
        # Assign the centrality values to the corresponding row in the centralities matrix.
        centralities[i] = np.array(
            list(nx.eigenvector_centrality(nx.Graph(dist)).values())
        )

    qm7["cm"] = coulomb_matrix
    qm7["eigs"] = eigs
    qm7["centralities"] = centralities
    qm7["dist_matrix"] = dist_matrix
    qm7["y_scaled"] = y_scaled
    return qm7


def plot_loss(mae_scores_train, mae_scores_val, save_dir):
    # Generate the iteration numbers based on the number of MAE scores
    iterations = np.arange(1, len(mae_scores_train) + 1)

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the training MAE curve
    axes[0].plot(iterations, mae_scores_train)
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Training MAE")
    axes[0].set_title("Training Loss Curve")
    axes[0].grid(True)

    iterations = np.arange(1, len(mae_scores_val) + 1)

    # Plot the validation MAE curve
    axes[1].plot(iterations, mae_scores_val)
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Validation MAE")
    axes[1].set_title("Validation Loss Curve")
    axes[1].grid(True)

    # Adjust the layout to prevent overlapping of subplots
    fig.tight_layout()

    # Save the plot to the specified directory
    plt.savefig(save_dir)
    plt.close()


def process_data_random(datadir):
    data = scipy.io.loadmat(datadir)
    X = data["X"]

    step = 1.0
    noise = 1.0

    # Sort X according to norm and get upper half
    triuind = np.triu_indices(23, k=0)

    sorted_triuind_X = []
    for x in X:
        # Sort according to norm
        idx = np.argsort(
            -(x**2).sum(axis=0) ** 0.5 + np.random.normal(0, noise, x[0].shape)
        )
        x = x[np.ix_(idx, idx)].copy()
        x = x[triuind[0], triuind[1]].flatten()
        sorted_triuind_X.append(x)
    sorted_triuind_X = np.array(sorted_triuind_X)

    # Get max value of norm
    max_val = 0
    for _ in range(10):
        max_val = np.maximum(max_val, sorted_triuind_X.max(axis=0))

    # Augment the matrix
    X_aug = []
    for i in range(sorted_triuind_X.shape[1]):
        for k in np.arange(0, max_val[i] + step, step):
            X_aug += [np.tanh((sorted_triuind_X[:, i] - k) / step)]
    X_aug = np.array(X_aug).T

    # normalize
    mean = X_aug.mean(axis=0)
    std = (X_aug - mean).std()

    X_aug_norm = (X_aug - mean) / std

    qm7 = data
    qm7["X"] = X_aug_norm

    return qm7


def train_val_split(data, val_fold, option):
    val_idx = data["P"][val_fold].flatten()
    train_idx = np.array(list(set(data["P"].flatten()) - set(val_idx)))

    if option == "random":
        X = data["X"]
    elif option == "sorted":
        X = np.concatenate((data["cm"], data["eigs"], data["centralities"]), axis=1)
    y = data["T"][0] / Y_SCALE_FACTOR
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    dataset = process_data_random("data/qm7.mat")
