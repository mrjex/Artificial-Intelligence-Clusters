from sklearn.mixture import GaussianMixture
import time
import utils
import models.principal_component_analysis as PCA

OUTPUT_PATH = "output/data/gmm.json"

def gmm_clustering(X, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag', random_state=42)
    labels = gmm.fit_predict(X)
    return labels


# Gaussian Mixture Model - Entrypoint from Main
def runGMM(X, y, num_mnist_instances):
    output_obj = utils.loadSchema("Cluster Method Schema (K-value)")
    utils.fill_schema_combinations(output_obj, "Clusters (K)", num_mnist_instances)
    output_obj["Method"] = "GMM"

    total_start_time = time.time()  # Start timer

    for n_pca in utils.n_pca_list:
        pca_start_time = time.time()  # Start timer

        current_pca_obj = utils.loadSchema("PCA Schema (K-value)")
        current_pca_obj["PCA Level"] = n_pca

        X_pca = PCA.preprocess_data(X, n_pca)

        for k in utils.k_list:
            k_start_time = time.time()  # Start timer

            gmm_labels = gmm_clustering(X_pca, k)
            gmm_rand = utils.compute_rand_index(y, gmm_labels)

            current_k_obj = utils.loadSchema("Object Schema (K-value)")

            time_elapsed_k = utils.getDuration(k_start_time)
            utils.assign_object_schema(current_k_obj, "Clusters (K)", k, gmm_rand, time_elapsed_k)

            current_pca_obj["K-Combinations"].append(current_k_obj)

        current_pca_obj["TimeElapsedSum"] = utils.getDuration(pca_start_time)
        output_obj["Run Time Data"]["Results"].append(current_pca_obj)

    output_obj["Run Time Data"]["Total Execution Run Time"] = utils.getDuration(total_start_time)
    utils.writeOutputJSON(OUTPUT_PATH, output_obj)