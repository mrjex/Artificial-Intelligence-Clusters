from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import rand_score
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from itertools import combinations
import numpy as np
import time

import models.principal_component_analysis as PCA
import utils


OUTPUT_PATH = "output/data/normalized_cut.json"



# Normalized Cut
def normalized_cut_clustering(X, n_clusters):
    # Compute pairwise distances and create a valid affinity matrix
    affinity_matrix = -pairwise_distances(X, metric='euclidean')
    np.fill_diagonal(affinity_matrix, 0)  # Ensure diagonal is zero
    affinity_matrix = np.maximum(affinity_matrix, 0)  # Ensure non-negative values only

    # Compute Laplacian matrix
    laplacian_matrix, _ = laplacian(affinity_matrix, normed=True, return_diag=True)
    
    # Check and clean invalid values
    laplacian_matrix = np.nan_to_num(laplacian_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Eigen decomposition
    _, eigenvectors = eigsh(laplacian_matrix, k=n_clusters, which='SM')
    
    # Apply Gaussian Mixture Model for clustering
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    labels = gmm.fit_predict(eigenvectors)
    return labels



# Normalized Cut - Entrypoint from Main
def run_normalized_cut(X, y, num_mnist_instances):
    output_obj = utils.loadSchema("Cluster Method Schema (K-value)")
    utils.fill_schema_combinations(output_obj, "Clusters (K)", num_mnist_instances)
    output_obj["Method"] = "Normalized Cut"

    total_start_time = time.time()  # Start timer

    for n_pca in utils.n_pca_list:
        pca_start_time = time.time()  # Start timer

        current_pca_obj = utils.loadSchema("PCA Schema (K-value)")
        current_pca_obj["PCA Level"] = n_pca

        X_pca = PCA.preprocess_data(X, n_pca)

        for k in utils.k_list:
            k_start_time = time.time()  # Start timer

            ncut_labels = normalized_cut_clustering(X_pca, k)
            ncut_rand = utils.compute_rand_index(y, ncut_labels)

            current_k_obj = utils.loadSchema("Object Schema (K-value)")

            time_elapsed_k = utils.getDuration(k_start_time)
            utils.assign_object_schema(current_k_obj, "Clusters (K)", k, ncut_rand, time_elapsed_k)

            current_pca_obj["K-Combinations"].append(current_k_obj)

        current_pca_obj["TimeElapsedSum"] = utils.getDuration(pca_start_time)
        output_obj["Run Time Data"]["Results"].append(current_pca_obj)

    output_obj["Run Time Data"]["Total Execution Run Time"] = utils.getDuration(total_start_time)
    utils.writeOutputJSON(OUTPUT_PATH, output_obj)