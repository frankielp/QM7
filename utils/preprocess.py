import pandas as pd
import scipy.io
import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time
from tqdm import tqdm
import os

import warnings


def process_data(datadir):
    warnings.simplefilter('ignore')

    rand_state = 42
    np.random.seed(rand_state)

    qm7 = scipy.io.loadmat(datadir)
    # Cartesian Coordinate matrix
    R=qm7['R']


    # Reshape energy to (7165,) and normalize
    y = np.transpose(qm7['T']).reshape((7165,))
    std=(y-y.mean()).std()
    y_scaled = (y -y.mean())/ std

    ## Step 1: Take upper half
    # (idx) to access the elements of the matrix in the upper triangle including the diagonal (k=0). 
    # (idx_dist) to access the elements of the distance matrix in the upper triangle excluding the diagonal (k=1).
    num_atoms = 23
    idx = np.triu_indices(num_atoms,k=0) 
    idx_dist = np.triu_indices(num_atoms,k=1)

    # Initialize placeholder
    coulomb_matrix = np.zeros((qm7['X'].shape[0], num_atoms*(num_atoms+1)//2), dtype=float)
    eigs = np.zeros((qm7['X'].shape[0], num_atoms), dtype=float)
    centralities = np.zeros((qm7['X'].shape[0], num_atoms), dtype=float)
    dist_matrix = np.zeros((qm7['X'].shape[0], ((num_atoms*num_atoms)-num_atoms)//2), dtype=float)

    for i in tqdm(range(len(qm7['X']))):
        # Extract the Coulomb vector from the Coulomb matrix (cm) and sort the elements in descending order.
        cm=qm7['X'][i]
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
        centralities[i] = np.array(list(nx.eigenvector_centrality(nx.Graph(dist)).values()))

    qm7['cm']=coulomb_matrix
    qm7['eigs']=eigs
    qm7['centralities']=centralities
    qm7['dist_matrix']=dist_matrix
    qm7['y_scaled']=y_scaled
    return qm7