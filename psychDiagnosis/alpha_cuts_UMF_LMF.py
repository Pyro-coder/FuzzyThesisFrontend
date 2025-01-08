import numpy as np

from .alpha_cut_algorithm import alpha_cut
from .support_interval import support


def alpha_t2(f, Ninc, xmin, xmax, Nα, Nincr):
    """
    Calculate α-cuts of a UMF and LMF of a Type-2 MF using Ninc increments to find the support interval.
    Returns nested lists of np.array objects containing the α-cuts of the UMF and LMF.
    """

    # Define the UMF and LMF functions
    def fu(x):
        return f(x)[0]

    def fl(x):
        return f(x)[1]

    # Calculate support intervals of UMF and LMF
    usup = support(fu, Ninc, xmin, xmax)
    lsup = support(fl, Ninc, xmin, xmax)

    # Calculate α-cuts of the UMF and LMF
    au = alpha_cut(fu, usup[0], usup[1], Nα, Nincr)
    al = alpha_cut(fl, lsup[0], lsup[1], Nα, Nincr)

    # Output a nested list
    out = [au, al]

    return out