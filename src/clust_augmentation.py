import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from collections import defaultdict
from tqdm import tqdm

def prep_data(arr):
    if arr.ndim <= 1:
        arr_use = arr.copy().reshape(-1,1)
    else:
        arr_use = arr.copy()
    
    return arr_use

def construct_interval(x_train, y_train, n_clusters=[10]):
    """Construct interval from 

    Args:
        x_train (_type_): Train data features
        y_train (_type_): Train data target
        n_clusters (list, optional): List of n_clusters. Defaults to [10].

    Returns:
        np.ndarray: Array with shape (nsamp, ndim, 2)
    """
    x_use = prep_data(x_train)
    y_use = prep_data(y_train)
    
    x_train_int = []
    y_train_int = []
    
    for n_clus in tqdm(n_clusters):
        kmeans = KMeans(n_clusters=n_clus, random_state=42).fit(x_use)
        labels = kmeans.predict(x_use)

        for cluster_id in range(n_clus):
            x_cluster = x_use[labels == cluster_id]
            y_cluster = y_use[labels == cluster_id]
            if len(x_cluster) <= 1:
                continue
            x_min = x_cluster.min(axis=0).flatten()
            x_max = x_cluster.max(axis=0).flatten()
            y_min = y_cluster.min(axis=0).flatten()
            y_max = y_cluster.max(axis=0).flatten()
            
            x_train_int.append(np.stack([x_min,x_max], axis=-1))
            y_train_int.append(np.stack([y_min,y_max], axis=-1))
    
    x_train_int = np.array(x_train_int)
    y_train_int = np.array(y_train_int)

    return x_train_int, y_train_int