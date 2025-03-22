from sklearn.datasets import fetch_openml

import models.gaussian_mixture_model as gmmModel
import models.normalized_cut as normalizedCutModel
import models.mean_shift as meanShiftModel

import utils
import visualization


def load_mnist_subset(subsetSize):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target
    X_subset = X[:subsetSize]
    y_subset = y[:subsetSize].astype(int)
    return X_subset, y_subset



def run_models(X, y, NUM_MNIST_INSTANCES):
    if utils.runAllMethods:
        pass
    else:
        # gmmModel.runGMM(X, y, NUM_MNIST_INSTANCES)
        # normalizedCutModel.run_normalized_cut(X, y, NUM_MNIST_INSTANCES)
        meanShiftModel.runMeanShift(X, y, NUM_MNIST_INSTANCES)


##  MAIN LOGIC  ##


NUM_MNIST_INSTANCES = 10000
TASK = "RUN MODELS" # NOTE: Change this value if you wish to toggle between running the models and visualize data


# Load a subset of 50000 instances from the MNIST database
X, y = load_mnist_subset(NUM_MNIST_INSTANCES)


# Developer specifies whether to run the models or visualize the generated JSON data
if TASK == "RUN MODELS":
    run_models(X, y, NUM_MNIST_INSTANCES)
else:
    visualization.visualize_model_results()