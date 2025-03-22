from sklearn.cluster import MeanShift, estimate_bandwidth

import utils
import time
import models.principal_component_analysis as PCA


OUTPUT_PATH = "output/data/mean_shift.json"


# Mean Shift
def mean_shift_clustering(X, bandwidth):
    if not bandwidth:
        bandwidth = estimate_bandwidth(X, quantile=0.2)
    mean_shift = MeanShift(bandwidth=bandwidth)
    labels = mean_shift.fit_predict(X)
    return labels


# Gaussian Mixture Model - Entrypoint from Main
def runMeanShift(X, y, num_mnist_instances):
    output_obj = utils.loadSchema("Cluster Method Schema (Kernel-width-value)")
    utils.fill_schema_combinations(output_obj, "Kernel Width", num_mnist_instances)
    output_obj["Method"] = "Mean Shift"

    total_start_time = time.time()  # Start timer

    for n_pca in utils.n_pca_list:
        pca_start_time = time.time()  # Start timer

        current_pca_obj = utils.loadSchema("PCA Schema (Kermel-width-value)")
        current_pca_obj["PCA Level"] = n_pca

        X_pca = PCA.preprocess_data(X, n_pca)

        for bw in utils.bandwidths:
            bw_start_time = time.time()  # Start timer

            ms_labels = mean_shift_clustering(X_pca, bw)
            ms_rand = utils.compute_rand_index(y, ms_labels)

            current_bw_obj = utils.loadSchema("Object Schema (Kermel-width-value)")

            time_elapsed_bandwidth = utils.getDuration(bw_start_time)
            utils.assign_object_schema(current_bw_obj, "Kernel Width", bw, ms_rand, time_elapsed_bandwidth)

            current_pca_obj["Kernel-width Combinations"].append(current_bw_obj)

        current_pca_obj["TimeElapsedSum"] = utils.getDuration(pca_start_time)
        output_obj["Run Time Data"]["Results"].append(current_pca_obj)

    output_obj["Run Time Data"]["Total Execution Run Time"] = utils.getDuration(total_start_time)
    utils.writeOutputJSON(OUTPUT_PATH, output_obj)