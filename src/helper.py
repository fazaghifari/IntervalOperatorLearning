import numpy as np

def interval_midpoint(x):
    """Compute interval midpoint

    Args:
        x (np.array): input array

    Raises:
        ValueError: if the input dimension not 1 or 2

    Returns:
        np.array: interval midpoint
    """
    if x.ndim == 1:
        mid = 0.5*(x[0]+x[1])
    elif x.ndim == 2:
        mid = 0.5*(x[:,0]+x[:,1])
        mid = np.reshape(mid, (-1,1))
    else:
        raise ValueError("Does not support ndim>2")
    
    return mid

def interval_radius(x):
    """Compute interval radius

    Args:
        x (np.array): input array

    Raises:
        ValueError: if the input dimension not 1 or 2

    Returns:
        np.array: interval radius
    """
    if x.ndim == 1:
        rad = 0.5*(x[1]-x[0])
    elif x.ndim == 2:
        rad = 0.5*(x[:,1]-x[:,0])
        rad = np.reshape(rad, (-1,1))
    else:
        raise ValueError("Does not support ndim>2")
    
    return rad

def interval_width(x):
    """Compute interval width

    Args:
        x (np.array): input array

    Returns:
        np.array: interval width
    """
    return 2 * interval_radius(x)

def beta_interval(x):
    """Construct beta interval

    Args:
        x (np.array): input array


    Returns:
        np.array: beta interval from data
    """
    radius = interval_radius(x)
    midpoint = interval_midpoint(x)
    ratio = radius/midpoint

    beta_int = np.zeros(x.shape)
    if beta_int.ndim == 1:
        beta_int[0] = -ratio
        beta_int[1] = ratio
    else:
        beta_int[:,0] = -ratio
        beta_int[:,1] = ratio
    
    return beta_int

def beta_level(beta_int):
    """compute beta level from data

    Args:
        beta_int (np.array): input array

    Returns:
        np.array: beta level
    """
    return interval_width(beta_int)

def interval_from_beta(x_mid, beta_level):
    """construct interval from beta

    Args:
        x_mid (np.array): interval midpoint
        beta_level (float): beta level

    Returns:
        np.array: interval 
    """
    assert x_mid.ndim <= 2, "cannot handle dimension >2"

    if x_mid.ndim == 0:
        beta = [-0.5*beta_level, 0.5*beta_level]
        x = [x_mid * (1 + beta[0]), x_mid * (1 + beta[1]) ]
        x = np.array(x)
    else:
        if x_mid.ndim == 1:
            x_mid = x_mid.reshape(-1,1)
        beta = np.ones(shape=(x_mid.shape[0],2))
        beta[:,0] = beta[:,0] * 0.5 * beta_level * -1
        beta[:,1] = beta[:,1] * 0.5 * beta_level
        x = np.zeros(shape=(x_mid.shape[0],2))
        x[:,0] = x_mid.flatten() * (1 + beta[:,0])
        x[:,1] = x_mid.flatten() * (1 + beta[:,1])
        
    return x


def interval_from_radius(x_mid, radius):
    """construct interval from midpoint and radius

    Args:
        x_mid (np.array): interval midpoint
        radius (np.array): non-negative interval radius
    
    Returns:
        np.array: interval
    """
    assert x_mid.ndim <= 2, "cannot handle dimension >2"
    assert x_mid.ndim == radius.ndim, "dimension of midpoint and radius should be the same"
    assert np.any(radius >= 0), "negative value detected in radius array, negative value is not allowed."

    if x_mid.ndim == 0:
        x = [x_mid - radius, x_mid + radius]
        x = np.array(x)
    else:
        if x_mid.ndim == 2:
            x_mid = x_mid.flatten()
            radius = radius.flatten()
        x = np.zeros(shape=(x_mid.shape[0],2))
        x[:,0] = x_mid - radius
        x[:,1] = x_mid + radius
    
    return x