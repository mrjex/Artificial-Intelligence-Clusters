# Artificial Intelligence Clusters

![PCA](https://img.shields.io/badge/Implemented-Principal%20Component%20Analysis-blue)
![GMM](https://img.shields.io/badge/Implemented-Gaussian%20Mixture%20Model-red)
![MS](https://img.shields.io/badge/Implemented-Mean%20Shift-green)
![NC](https://img.shields.io/badge/Implemented-Normalized%20Cut-purple)
![RI](https://img.shields.io/badge/Evaluation-Rand%20Index-yellow)

> Unsupervised Learning on MNIST Handwritten Digits

## Table of Contents

- [Artificial Intelligence Clusters](#artificial-intelligence-clusters)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Highlights](#project-highlights)
  - [Getting Started](#getting-started)
  - [Performance Analysis](#performance-analysis)

## Overview

This project explores the powerful domain of unsupervised machine learning through clustering analysis on the MNIST handwritten digit dataset, and it demonstrates how different clustering algorithms can discover intrinsic patterns and group similar handwritten digits without any labeled training data.

## Project Highlights

- **Unsupervised Pattern Recognition**: Discovering natural groupings in visual data without labeled examples
- **Dimensionality Reduction**: Applying *PCA* to reduce computational complexity while preserving information
- **Multiple Clustering Approaches**: Implementation and comparison of three distinct clustering techniques
- **Performance Evaluation**: Quantitative assessment using *Rand Index* scoring
- **Visual Analysis**: Comprehensive visualization of clustering performance across parameters


## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure execution mode in `main.py`:
   ```python
   # To generate new clustering results
   run_models(True)  # Set to False to skip execution
   
   # To visualize existing results
   visualize_results(True)  # Set to False to skip visualization
   ```
3. Run the main script:
   ```bash
   python main.py
   ```

## Performance Analysis

**Gaussian Mixture Model Performance:**
![gmm-visualization](output/visualization/gmm_rand_index.PNG)
*GMM performance with varying PCA dimensions and cluster counts*

**Mean Shift Performance:**
![mean-shift-performance](output/visualization/mean_shift_rand_index.PNG)
*Mean Shift performance with different bandwidth parameters and PCA dimensions*

**Normalized Cut Performance:**
![normalized-cut-performance](output/visualization/normalized_cut_rand_index.PNG)
*Normalized Cut performance across cluster counts and PCA dimensions*

**Comparative Analysis:**
![average-performances](output/visualization/rand_index_method_avg_comparison.PNG)
*Average Rand Index scores across all clustering methods*
