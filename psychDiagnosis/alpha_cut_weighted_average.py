from EKM_algorithm import kmalg
from max_alpha_for_lmf import alpha_max_lmf

import numpy as np

def alpha_to_alpha_t2wa(aux, alx, auw, alw):
    """
    Calculate alpha-cuts of Type-2 fuzzy weighted average given UMF/LMF alpha-cuts of inputs.
    
    Parameters:
    aux, alx, auw, alw - Nested lists or arrays where each element contains an array of alpha-cuts.
    
    Returns:
    A nested 2-vector, each element containing an array of corresponding alpha-cut intervals of the FWA UMF/LMF.
    """
    n = len(aux)  # Number of terms in the fuzzy WA
    m = len(aux[0])  # Number of alpha-cuts of UMF
    
    # Initialize the output structures for UMF and LMF
    out = [None, None]

    # Process each j-th alpha-cut for UMF
    for j in range(m):
        xuab = [aux[i][j] for i in range(n)]
        wucd = [auw[i][j] for i in range(n)]

        # Compute IWA using the KM algorithm for each UMF alpha-cut
        zu = kmalg(xuab, wucd)
        if j == 0:
            out[0] = zu
        else:
            out[0] = np.vstack((out[0], zu))

    # Determine the minimum number of alpha-cuts using the provided helper function
    hmin, _ = alpha_max_lmf(alx, alw)
    
    # Process each j-th alpha-cut for LMF up to hmin
    for j in range(hmin + 1):
        xlab = [alx[i][j] for i in range(n)]
        wlcd = [alw[i][j] for i in range(n)]

        # Compute IWA using the KM algorithm for each LMF alpha-cut
        zl = kmalg(xlab, wlcd)
        if j == 0:
            out[1] = zl
        else:
            out[1] = np.vstack((out[1], zl))

    return out