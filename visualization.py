import pandas as pd
import matplotlib.pyplot as plt


DATA_PATHS = {
    "GMM": "output/data/gmm.json",
    "Mean Shift": "output/data/mean_shift.json",
    "Normalized Cut": "output/data/normalized_cut.json"
}



def read_gmm_data():
    return [
        {"Method": "GMM", "PCA Level": pca, "Clusters (K)": k["Clusters (K)"], "Rand Index": k["Rand Index Performance"]}
        for pca_data in [
            {"PCA Level": result["PCA Level"], "K-Combinations": result["K-Combinations"]}
            for result in [
                {
                    "PCA Level": r["PCA Level"],
                    "K-Combinations": r["K-Combinations"],
                }
                for r in eval(open(f'{DATA_PATHS["GMM"]}').read())['Run Time Data']['Results']
            ]
        ]
        for pca in [pca_data["PCA Level"]]
        for k in pca_data["K-Combinations"]
    ]



def read_mean_shift_data():
    return [
        {"Method": "Mean Shift", "PCA Level": pca, "Kernel Width": k["Kernel Width"], "Rand Index": k["Rand Index Performance"]}
        for pca_data in [
            {"PCA Level": result["PCA Level"], "Kernel-width Combinations": result["Kernel-width Combinations"]}
            for result in [
                {
                    "PCA Level": r["PCA Level"],
                    "Kernel-width Combinations": r["Kernel-width Combinations"],
                }
                for r in eval(open(f'{DATA_PATHS["Mean Shift"]}').read())['Run Time Data']['Results']
            ]
        ]
        for pca in [pca_data["PCA Level"]]
        for k in pca_data["Kernel-width Combinations"]
    ]


def read_normalized_cut_data():
    return [
        {"Method": "Normalized Cut", "PCA Level": pca, "Clusters (K)": k["Clusters (K)"], "Rand Index": k["Rand Index Performance"]}
        for pca_data in [
            {"PCA Level": result["PCA Level"], "K-Combinations": result["K-Combinations"]}
            for result in [
                {
                    "PCA Level": r["PCA Level"],
                    "K-Combinations": r["K-Combinations"],
                }
                for r in eval(open(f'{DATA_PATHS["Normalized Cut"]}').read())['Run Time Data']['Results']
            ]
        ]
        for pca in [pca_data["PCA Level"]]
        for k in pca_data["K-Combinations"]
    ]




# Returns a Tuple of each clustering method's associated data obtained in the latest runtime execution
def read_json_data():
    return read_gmm_data(), read_mean_shift_data(), read_normalized_cut_data()


# Combines all collected data into one cohesive Pandas Dataframe that contains all data at one place
def mergeDataframe():
    gmm_data, ms_data, nc_data = read_json_data()
    merged_data_frame = pd.DataFrame(gmm_data + ms_data + nc_data)
    return merged_data_frame




def visualize_gmm(dataframe):
    plt.figure(figsize=(10, 6))
    gmm_data = dataframe[dataframe["Method"] == "GMM"]
    for pca_level in gmm_data["PCA Level"].unique():
        subset = gmm_data[gmm_data["PCA Level"] == pca_level]
        plt.plot(subset["Clusters (K)"], subset["Rand Index"], label=f'PCA {pca_level}')
    plt.title("Rand Index vs Clusters (K) for GMM")
    plt.xlabel("Clusters (K)")
    plt.ylabel("Rand Index")
    plt.legend()
    plt.grid()
    plt.show()



def visualize_mean_shift(dataframe):
    plt.figure(figsize=(10, 6))
    mean_shift_data = dataframe[dataframe["Method"] == "Mean Shift"]
    for pca_level in mean_shift_data["PCA Level"].unique():
        subset = mean_shift_data[mean_shift_data["PCA Level"] == pca_level]
        plt.plot(subset["Kernel Width"], subset["Rand Index"], label=f'PCA {pca_level}')
    plt.title("Rand Index vs Kernel Width for Mean Shift")
    plt.xlabel("Kernel Width")
    plt.ylabel("Rand Index")
    plt.legend()
    plt.grid()
    plt.show()



def visualize_normalized_cut(dataframe):
    plt.figure(figsize=(10, 6))
    nc_data = dataframe[dataframe["Method"] == "Normalized Cut"]
    for pca_level in nc_data["PCA Level"].unique():
        subset = nc_data[nc_data["PCA Level"] == pca_level]
        plt.plot(subset["Clusters (K)"], subset["Rand Index"], label=f'PCA {pca_level}')
    plt.title("Rand Index vs Clusters (K) for Normalized Cut")
    plt.xlabel("Clusters (K)")
    plt.ylabel("Rand Index")
    plt.legend()
    plt.grid()
    plt.show()



def visualize_line_charts(all_data):
    visualize_gmm(all_data)
    visualize_mean_shift(all_data)
    visualize_normalized_cut(all_data)


def visualize_bar_chart(all_data):
    fixed_pca_level = 50 # Filter data for a specific PCA level (e.g., PCA Level = 50)
    filtered_data = all_data[all_data["PCA Level"] == fixed_pca_level]

    # Separate performance metrics (Rand Index and Time Elapsed) by method
    methods = filtered_data["Method"].unique()
    performance_data = {
        "Method": [],
        "Rand Index": [],
        "Time Elapsed": []
    }

    for method in methods:
        method_data = filtered_data[filtered_data["Method"] == method]
        performance_data["Method"].append(method)
        performance_data["Rand Index"].append(method_data["Rand Index"].mean())
        performance_data["Time Elapsed"].append(method_data.get("Time Elapsed", pd.Series([0])).mean())

    # Convert to DataFrame for visualization
    performance_df = pd.DataFrame(performance_data)

    # Plotting Rand Index
    plt.figure(figsize=(12, 6))
    plt.bar(performance_df["Method"], performance_df["Rand Index"], alpha=0.7, color='b', label="Rand Index")
    plt.title(f"Comparative Rand Index Performance at PCA Level {fixed_pca_level}")
    plt.xlabel("Method")
    plt.ylabel("Rand Index")
    plt.grid(axis="y")
    plt.legend()
    plt.show()



def visualize_model_results():
    all_data = mergeDataframe()
    visualize_line_charts(all_data)
    visualize_bar_chart(all_data)