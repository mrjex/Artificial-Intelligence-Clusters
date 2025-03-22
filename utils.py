# Rename 'utils.py' to 'config.py'?

from itertools import combinations
import numpy as np
import json
import yaml
import time


DATA_SCHEMAS_PATH = "output/data/schemas.yml"


##  SETTINGS SECTION  ##

# Methods Selection
runAllMethods = False
allMethods = ["GMM", "Mean Shift", "Normalized Cut"] # All clustering methods. PCA is always applied
selectedMethods = ["GMM"]


# Combinations/Adjustment Selection
runAllCombinations = True
n_pca_list = [10, 50, 100, 200]
k_list = range(5, 16)  # Clusters for GMM and Normalized Cut --> 5-15
bandwidths = [0.5, 1.0, 1.5, 2.0]  # Bandwidths for Mean Shift



# Rand Index calculation - Returns a number within the range [0, 1] that quantifies the
# quality of the clustering method's results, with respect to provided ground truth labels
def compute_rand_index(y_true, y_pred):
    n = len(y_true)
    same_true = np.array([y_true[i] == y_true[j] for i, j in combinations(range(n), 2)])
    same_pred = np.array([y_pred[i] == y_pred[j] for i, j in combinations(range(n), 2)])
    a = np.sum(np.logical_and(same_true, same_pred))
    b = np.sum(np.logical_and(~same_true, ~same_pred))
    rand_index = 2 * (a + b) / (n * (n - 1))
    return rand_index



def writeOutputJSON(filePath, outputObj):
    with open(filePath, 'w') as f:
        json.dump(outputObj, f, indent=4)



# Returns the duration of a period, where the start is marked by 'start_time' parameter
def getDuration(start_time):
    current_time = time.time()
    return current_time - start_time



def loadSchema(customSchemaType):
    try:
        with open(f"{DATA_SCHEMAS_PATH}", "r") as file:
            schema_skeleton = yaml.safe_load(file)

        # writeOutputJSON("output_schema.json", schema_skeleton[f"{customSchemaType}"])
        return schema_skeleton[f"{customSchemaType}"]
    except:
        return f"'{customSchemaType}' isn't an existing customized schema type, or the relative path to the YML schema file is incorrect"




def fill_schema_combinations(schema_skeleton_structure, schema_type, num_mnist_instances):

    keyName = "K Values" if schema_type == "Clusters (K)" else "Kernel-width Values"

    schema_skeleton_structure["Combinatorial Configurations"]["PCA Levels"] = n_pca_list

    combination_list = list(k_list) if keyName == "K Values" else bandwidths
    schema_skeleton_structure["Combinatorial Configurations"][f"{keyName}"] = combination_list

    schema_skeleton_structure["Run Time Data"]["Num MNIST Instances"] = num_mnist_instances


# 'clustering_variable' is either "K" or "Kernel-width"
def assign_object_schema(schema_obj, schema_type, clustering_variable, rand_index_performance, time_elapsed):

    if schema_type != "Clusters (K)":
        schema_type = "Kernel Width"

    # schema_type = "Clusters (K)" if schema_type == "K" else "Kernel Width"

    schema_obj[f"{schema_type}"] = clustering_variable
    schema_obj["Rand Index Performance"] = rand_index_performance
    schema_obj["Time Elapsed"] = time_elapsed