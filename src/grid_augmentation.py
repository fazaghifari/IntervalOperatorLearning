import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from tqdm import tqdm

def prep_data(arr):
    if arr.ndim <= 1:
        arr_use = arr.copy().reshape(-1,1)
    else:
        arr_use = arr.copy()
    
    return arr_use


def grid_grouping(data, grid_sizes, shift=None):
    """
    Group n-dimensional data points into grid cells with optional shift.

    Parameters:
    - data: ndarray of shape (n_samples, n_dims)
    - grid_sizes: list or array of length n_dims, grid size per dimension
    - shift: list or array of length n_dims, shift per dimension (default 0)

    Returns:
    - groups: dict mapping grid index tuples to arrays of point indices
    """
    data = np.asarray(data)
    grid_sizes = np.asarray(grid_sizes)
    if shift is None:
        shift = np.zeros_like(grid_sizes)
    else:
        shift = np.asarray(shift)

    # Shift and compute grid coordinates
    shifted_data = data - shift
    grid_indices = np.floor(shifted_data / grid_sizes).astype(int)

    # Group points
    groups = defaultdict(list)
    for idx, grid_idx in enumerate(map(tuple, grid_indices)):
        groups[grid_idx].append(idx)
    for key in groups:
        groups[key] = np.array(groups[key])
    
    return groups


def construct_interval_grid(x_train, y_train, grid_res, n_shift=0, nmax_per_res=25):
    x_use = prep_data(x_train)
    y_use = prep_data(y_train)
    
    x_train_int = []
    y_train_int = []

    for res in tqdm(grid_res):
        if type(res) == list:
            linspaces = [np.linspace(lo, hi, n_shift, endpoint=False).tolist() for lo, hi in zip([0]*len(res), res)]
            shifts = list(map(list, zip(*linspaces))) # transposed the linspaces
        else:
            raise ValueError("grid_res must be list of lists, e.g. for 1-D [[0.5],[0.3],[0.2]]")

        d_count = 0

        sublist_x = []
        sublist_y = []
        for offset in shifts:
            groups = grid_grouping(x_use, res, offset)

            for i, (cell, indices) in enumerate(groups.items()):
                x_points = x_use[indices]
                y_points = y_use[indices]
                # Skip single groups
                if len(x_points) <= 1:
                    continue

                x_min = x_points.min(axis=0).flatten()
                x_max = x_points.max(axis=0).flatten()
                y_min = y_points.min(axis=0).flatten()
                y_max = y_points.max(axis=0).flatten()
                
                temp_x = np.stack([x_min,x_max], axis=-1)
                temp_y = np.stack([y_min,y_max], axis=-1)

                # Avoid duplicate
                if not any(np.array_equal(temp_x, a) for a in sublist_x):
                    sublist_x.append(temp_x)
                    sublist_y.append(temp_y)
                    d_count += 1
        
        # Sample max number per resolution
        if len(sublist_x) > nmax_per_res:
            indices = np.random.choice(len(sublist_x), size=nmax_per_res, replace=False)
            sublist_x = [sublist_x[i] for i in indices]
            sublist_y = [sublist_y[i] for i in indices]

        print(f"Grid resolution {res} has {len(sublist_x)} data")
        x_train_int.append(sublist_x)
        y_train_int.append(sublist_y)
    
    res_x = [item for row in x_train_int for item in row]
    res_y = [item for row in y_train_int for item in row]

    res_x = np.array(res_x)
    res_y = np.array(res_y)

    return res_x, res_y