import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.stats import energy_distance


def compute_distance(features, labels, method="euclidean"):
    """
    Compute distance of each gRNA group from Non-Targeting control.

    Parameters
    ----------
    features : np.ndarray
        Latent features (cells × latent_dims).
    labels : array-like
        gRNA labels for each cell.
    method : str
        'euclidean' or 'energy' to select distance type.

    Returns
    -------
    pd.DataFrame
        DataFrame with distance values per gRNA.
    """
    df_encoded = pd.DataFrame(features)
    df_encoded["gRNA"] = labels

    # Group by gRNA
    gRNA_groups = {g: group.drop(columns="gRNA").values for g, group in df_encoded.groupby("gRNA")}

    # Find control group(s)
    non_targeting = [g for g in gRNA_groups if "Non-Targeting" in g]
    if not non_targeting:
        raise ValueError("No Non-Targeting group found in labels.")
    control = gRNA_groups[non_targeting[0]]

    results = {}
    for g, group in gRNA_groups.items():
        if method == "euclidean":
            # Compute Euclidean distance between centroids
            dist = np.linalg.norm(group.mean(axis=0) - control.mean(axis=0))
        elif method == "energy":
            # Compute Energy distance between distributions
            dist = max(energy_distance(group.mean(axis=1), control.mean(axis=1)), 0)
        else:
            raise ValueError("Invalid method. Choose 'euclidean' or 'energy'.")
        results[g] = dist

    return pd.DataFrame.from_dict(results, orient="index", columns=["distance"])


def compute_threshold_and_plot_hist(distance, plot=False, method="euclidean", log1p=False):
    # Find best threshold minimizing within-group variance
    distances = np.sort(np.array(distance).flatten())
    if log1p:
        distances = np.log1p(distances)
    best_threshold, best_total_var = None, np.inf

    for i in range(1, len(distances) - 1):
        thr = (distances[i] + distances[i+1]) / 2
        left, right = distances[distances <= thr], distances[distances > thr]
        total_var = (len(left) * np.var(left) + len(right) * np.var(right)) / len(distances)
        if total_var < best_total_var:
            best_total_var, best_threshold = total_var, thr

    # Optional: plot histogram with threshold
    if plot:
        sns.histplot(distances, kde=False, bins=20, kde_kws={'clip': (0, None)})
        plt.axvline(x=best_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {best_threshold:.2f}')
        plt.legend()
        plt.title('Distance Distribution')
        plt.xlabel(method.capitalize() + ' Distance')

        plt.show()

    return best_threshold

