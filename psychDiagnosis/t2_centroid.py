import numpy as np

from .EKM_algorithm import kmalg
from .calculate_mf_from_alpha_cuts import mu_sf


def t2_centroid(au, al, n):
    """
    Centroid interval of Type-2 MF computed using KM algorithm.
    Au and Al are arrays of α-cuts of upper and lower MFs.
    """
    # Compose the x and w arrays
    xinc = (au[0, 1] - au[0, 0]) / n
    if au[0, 0] != au[0, 1]:
        x = np.zeros(n + 1)
        x[0] = au[0, 0]
    else:
        return np.array([au[0, 0], au[0, 1]])

    for i in range(n + 1):
        x[i] = x[0] + i * xinc

    # xx is an array of intervals of zero width
    xx = np.vstack((x, x)).T

    # Generate lower and upper MF values for each x
    w = np.zeros((n + 1, 2))
    for i in range(n + 1):
        w[i, 0] = mu_sf(x[i], al)
        w[i, 1] = mu_sf(x[i], au)

    # We don't want the wc and wp intervals to be zero width, else we'd have a conventional centroid
    for i in range(n):
        if (w[i, 1] - w[i, 0]) == 0:
            w[i, 1] += 0.00001

    # Compute centroid interval using KM
    center = kmalg(xx, w)
    out = np.array([center[0], center[1]])
    return out



def defuzz(x):
    """
    Defuzzify the Type-2 fuzzy set x from its left and right centroid
    """
    return (x[0] + x[1]) / 2
