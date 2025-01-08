import numpy as np
from scipy.optimize import bisect


# THIS FUNCTION HAS CAUSED ERRORS IN THE PAST, PROCEED WITH CAUTION, REMEMBER WHERE IT IS USED IF YOU NEED TO FIX
# ERRORS. (I THINK I FIXED MOST OF IT) IF THERE ARE ERRORS, THEY ARE PROBABLY IN THIS FUNCTION.

def alpha_cut(mu, xmin, xmax, m, n):
    """
    Compute alpha-cut intervals of fuzzy membership function mu.
    mu is a fuzzy membership function.
    [xmin, xmax] is the support interval of mu.
    m+1 is the number of alpha-cut intervals; n is the number of steps in x to discretize the MF.
    """
    xincr = (xmax - xmin) / n
    maxy = 0
    maxxleft = xmin
    maxxright = xmin

    # Find the x-interval of the mu function maximum, to bound the alpha-cut endpoints
    for i in range(n + 1):
        x = xmin + i * xincr
        mu_x = mu(x)
        if mu_x > maxy:
            maxxleft = x
            maxxright = x
            maxy = mu_x
        elif mu_x == maxy:
            maxxright = x
        elif mu_x < maxy:
            maxxright = x - xincr
            break

    alphacut = np.zeros((m + 1, 3))
    alphacut[0, 0] = xmin
    alphacut[0, 1] = xmax
    alphacut[0, 2] = 0

    # Find the non-zero alpha cut intervals
    for i in range(1, m + 1):
        alpha = i / m
        if alpha < maxy:
            if (xmin == maxxleft) or (abs(mu(xmin) - alpha) < 0.0001) or (mu(xmin) >= alpha):
                alphacut[i, 0] = xmin
            else:
                if abs(mu(maxxleft) - alpha) < 0.0001:
                    alphacut[i, 0] = maxxleft
                else:
                    alphacut[i, 0] = bisect(lambda x: mu(x) - alpha, xmin, maxxleft)

            if (xmax == maxxright) or (abs(mu(xmax) - alpha) < 0.0001) or (mu(xmax) >= alpha):
                alphacut[i, 1] = xmax
            else:
                if abs(mu(maxxright) - alpha) < 0.0001:
                    alphacut[i, 1] = maxxright
                else:
                    alphacut[i, 1] = bisect(lambda x: mu(x) - alpha, maxxright, xmax)

        if alpha == maxy:
            alphacut[i, 0] = maxxleft
            alphacut[i, 1] = maxxright

        if alpha > maxy:
            break

        alphacut[i, 2] = alpha


    # Remove extra rows of zeros before returning
    alphacut = alphacut[~np.all(alphacut == 0, axis=1)]
    return alphacut
