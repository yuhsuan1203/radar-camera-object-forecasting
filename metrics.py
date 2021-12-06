import numpy as np


def calc_fde(outputs, targets, n, return_mean=True):
    '''
    Calculates the final displacement error (L2 distance) between outputs and
    targets (final output and final target)
    Args:
        outputs: np array. 1D array formated [x,x,x,x... y,y,y,y...]
        targets: np array. 1D array formated [x,x,x,x... y,y,y,y...]
        n: Number of predictions
    Returns:
        Final displacement error at n timesteps between outputs and targets
    '''

    # Reshape to [[x,y],[x,y],...)
    outputs = outputs.reshape(-1, n * 4, order='A')
    outputs = outputs.reshape(-1, n, 4)
    outputs = outputs[:, :, 0:2]

    # Reshape to [[x,y],[x,y],...)
    targets = targets.reshape(-1, n * 4, order='A')
    targets = targets.reshape(-1, n, 4)
    targets = targets[:, :, 0:2]

    # Get the final prediction
    outputs = outputs[:, -1, :]
    targets = targets[:, -1, :]

    # L2 Distance
    diff = (outputs - targets) * (outputs - targets)

    if return_mean:
        return np.mean(np.sqrt(np.sum(diff, axis=1)))
    else:
        return np.sqrt(np.sum(diff, axis=1))


def calc_ade(outputs, targets, return_mean=True):
    '''
    Calculates the average displacement error (L2 distance) between outputs and
    targets
    Args:
        outputs: np array. 1D array formated [x,x,x,x... y,y,y,y...]
        targets: np array. 1D array formated [x,x,x,x... y,y,y,y...]
    Returns:
        Final displacement error at n timesteps between outputs and targets
    '''
    # Reshape to [[x,y],[x,y],...)
    outputs = outputs.reshape(-1, 96, order='A')
    outputs = outputs.reshape(-1, 24, 4)
    # Just the centroids
    outputs = outputs[:, :, 0:2]
    # Reshape to [[x,y],[x,y],...)
    targets = targets.reshape(-1, 96, order='A')
    targets = targets.reshape(-1, 24, 4)
    targets = targets[:, :, 0:2]
    # Get the final prediction
    # outputs = outputs[:,:,-1]
    # targets = targets[:,:,-1]
    # L2 Distance

    out_mid_xs = outputs[:, :, 0]
    out_mid_ys = outputs[:, :, 1]
    tar_mid_xs = targets[:, :, 0]
    tar_mid_ys = targets[:, :, 1]

    diff = ((out_mid_xs - tar_mid_xs) * (out_mid_xs - tar_mid_xs)) + \
        ((out_mid_ys - tar_mid_ys) * (out_mid_ys - tar_mid_ys))

    if return_mean:
        return np.mean(np.sqrt(np.mean(diff, axis=1)))
    else:
        return np.sqrt(np.mean(diff, axis=1))
