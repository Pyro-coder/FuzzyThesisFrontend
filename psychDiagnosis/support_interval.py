import numpy as np


def support(f, N, xmin, xmax):
    """
    Calculate support interval of f defined on [xmin, xmax] using N increments on the x-axis.

    Parameters:
    f (function): The function for which to calculate the support interval.
    N (int): The number of increments on the x-axis.
    xmin (float): The minimum x value.
    xmax (float): The maximum x value.

    Returns:
    np.ndarray: A 1x2 array containing the left and right endpoints of the support interval.
    """
    # Calculate x increment length
    xincr = (xmax - xmin) / N
    x = xmin
    fx = f(x)

    # Find left endpoint of support interval
    while fx == 0 and x <= xmax:
        x += xincr
        fx = f(x)

    # Back up to last value of x where f(x) = 0, if necessary
    if x != xmin:
        xleft = x - xincr
    else:
        xleft = xmin

    # Find right endpoint of support interval
    x = xmax
    fx = f(x)

    while fx == 0 and x >= xmin:
        x -= xincr
        fx = f(x)

    # Advance to last value of x where f(x)=0, if necessary
    if x != xmax:
        xright = x + xincr
    else:
        xright = xmax

    out = np.array([xleft, xright])
    return out
