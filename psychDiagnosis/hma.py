import noncenteral_tstatic_cdf
import openpyxl
import numpy as np
import nct_cdf
import dataclean
import statistics

def hma_fou_class0(x, x0, x1):
    # Step 1 of HMA: determine the FOU class for naturally bounded (class 0) sets
    # x is an n x 2 array of intervals that survived the dataclean process
    # [x0, x1] is the bound interval
    ml = np.mean(x[:, 0])
    sigma_l = statistics.stdev(x[:, 0])
    mr = np.mean(x[:, 1])
    sigma_r = statistics.stdev(x[:, 1])
    k = nct_cdf.ktol(0.05, 0.05, len(x), 10 ** -5)
    al = ml - k * sigma_l
    bu = mr + k * sigma_r
    if al <= x0:
        out = "Left shoulder FOU"
    else:
        if bu >= x1:
            out = "Right shoulder FOU"
        else:
            out = "Interior FOU"
    return out

def hma_fou_class1(x, x0):
    # Step 1 of HMA: determine the FOU class for sets bounded only on the left (class 1)
    # In these cases, there are only Left shoulder or Interior FOUs
    # x is an n x 2 array of intervals that survived the dataclean processing
    # x0 is the left bound (typically 0) for the intervals
    ml = np.mean(x[:, 0])
    sigma_l = statistics.stdev(x[:, 0])
    k = nct_cdf.ktol(0.05, 0.05, len(x), 10 ** -5)
    al = ml - k * sigma_l
    if al <= x0:
        out = "Left shoulder FOU"
    else:
        out = "Interior FOU"
    return out


# debugging tests
# print(hma_fou_class0(dataclean.yVB, 0, 10))
# print(hma_fou_class0(dataclean.yB, 0, 10))
# print(hma_fou_class0(dataclean.ySB, 0, 10))
# print(hma_fou_class0(dataclean.yF, 0, 10))
# print(hma_fou_class0(dataclean.ySG, 0, 10))
# print(hma_fou_class0(dataclean.yG, 0, 10))
# print(hma_fou_class0(dataclean.yVG, 0, 10))

def hma_overlap(x, c, x0, x1):
    # Compute overlap interval of x rows for FOU class c
    # Note: the data part eliminates non-overlapping intervals, so we are assured of overlap
    if c == "Left shoulder FOU":
        out = [x0, min(x[:, 1])]
    elif c == "Interior FOU":
        out = [max(x[:, 0]), min(x[:, 1])]
    elif c == "Right shoulder FOU":
        out = [max(x[:, 0]), x1]
    return out

# tests for debugging hma_overlap
# print(hma_overlap(dataclean.yVB, "Left shoulder FOU", 0, 10))
# print(hma_overlap(dataclean.yB, "Interior FOU", 0, 10))
# print(hma_overlap(dataclean.ySB, "Interior FOU", 0, 10))
# print(hma_overlap(dataclean.yF, "Interior FOU", 0, 10))
# print(hma_overlap(dataclean.ySG, "Interior FOU", 0, 10))
# print(hma_overlap(dataclean.yG, "Interior FOU", 0, 10))
# print(hma_overlap(dataclean.yVG, "Right shoulder FOU", 0, 10))

def hma_olap_remove(x, c):
    # Given interval array x and FOU class, remove overlap from original intervals
    # For Left or Right shoulder FOUs, this leaves a single array of smaller intervals
    # For Interior FOUs, this leaves a nested 2-vector of arrays of smaller intervals for the left/right sections of the FOU, respectively
    out = []
    if c == "Left shoulder FOU":
        bmin = min(x[:, 1])
        for i in range(len(x)):
            out.append([bmin, x[i, 1]])
    elif c == "Right shoulder FOU":
        amax = max(x[:, 0])
        for i in range(len(x)):
            out.append([x[i, 0], amax])
    elif c == "Interior FOU":
        out0 = []
        out1 = []
        amax = max(x[:, 0])
        bmin = min(x[:, 1])
        for i in range(len(x)):
            out0.append([x[i, 0], amax])
            out1.append([bmin, x[i, 1]])
        out = [out0, out1]
    return out

# Debugging tests for hma_olap_remove
# VBr = hma_olap_remove(dataclean.yVB, "Left shoulder FOU")
# print(VBr)

# Br = hma_olap_remove(dataclean.yB, "Interior FOU")
# print(Br)

# SGr = hma_olap_remove(dataclean.ySG, "Interior FOU")
# print(SGr[0])
# print(SGr[1])

# Gr = hma_olap_remove(dataclean.yG, "Interior FOU")
# print(Gr)

# VGr = hma_olap_remove(dataclean.yVG, "Right shoulder FOU")
# print(VGr)

def aleft(xr, x0):
    # Calculate a parameters for left-hand side of Interior or Right shoulder FOU
    # See eq. (5) in HMA paper
    # xr is the set of reduced intervals for the left-hand side from the hmaolapremove function
    # x0 is the left bound
    # oleft is the left bound of the overlap interval of the original interval set
    # Thus, oleft will be the max right bound of the reduced intervals xr
    xr = np.array(xr)
    oleft = max(xr[:, 0])
    intlengths = np.abs(xr[:, 0] - oleft)
    # mLH is the mean of interval lengths wrt the left bound of the overlap interval (oleft)
    mLH = oleft - np.mean(intlengths)
    sLH = statistics.stdev(intlengths)
    a_left = max(x0, oleft - 3 * np.sqrt(2) * sLH)
    a_right = min(oleft, 6 * mLH + 3 * np.sqrt(2) * sLH - 5 * oleft)
    # Now test to if the order is sensible, and if not, reverse them
    if a_left <= a_right:
        out = [a_left, a_right]
    else:
        out = [max(x0, a_right), min(oleft, a_left)]
    # out_0/out_1 are the UMF/LMF intersections with the x-axis, respectively
    return out

def aright(xr, x1):
    # Calculate a parameters for right-hand side of Interior or Left shoulder FOU
    # See eq. (6) in HMA paper
    # xr is the set of reduced intervals for the right-hand side from the hmaolapremove function
    # x1 is the right bound
    # oright is the right bound of the overlap interval of the original interval set
    # Thus, oright will be the min left bound of the reduced intervals xr
    xr = np.array(xr)
    oright = min(xr[:, 1])
    intlengths = np.abs(xr[:, 1] - oright)
    #mRH is the mean of interval length wrt the right bound of the overlap interval (oright)
    mRH = oright + np.mean(intlengths)
    sRH = statistics.stdev(intlengths)
    b_right = min(x1, oright + 3 * np.sqrt(2) * sRH)
    b_left = max(oright, 6 * mRH - 3 * np.sqrt(2) * sRH - 5 * oright)
    # Now test to if the order is sensible, and if not, reverse them
    if b_left <= b_right:
        out = [b_left, b_right]
    else:
        out = [max(oright, b_right), min(x1, b_left)]
    # out_0/out_1 are the UMF/LMF intersections with the x-axis, respectively
    return out

# Debugging tests for aleft and aright
# print(aleft(VGr, 0))
# print(aleft(SGr[0], 0))
# print(aright(VBr, 10))
# print(aright(SGr[1], 10))

def hma_map(x, c, x0, x1):
    """Implement the mapping of fuzzy part step 4
    x is the set of reduced intervals from the hmaolapremove function"""
    if c == "Interior FOU":
        out = [aleft(x[0], x0), aright(x[1], x1)]
    elif c == "Left shoulder FOU":
        out = [aright(x, x1)]
    elif c == "Right shoulder FOU":
        out = [aleft(x, x0)]
        # out will be a single 2-vector for left or right-shoulder FOU, a nested 2-vector of 2-vectors for interior FOUs
    return out

# Debugging tests for hma_map
# print(hma_map(Br, "Interior FOU", 0, 10)[1])

def hma_trap(x, x0, x1):
    #Go from an array of intervals x to the UMF/LMF trapezoidal parameters
    c = hma_fou_class0(x, x0, x1)
    # compute overlap interval
    olap_interval = hma_overlap(x, c, x0, x1)
    # compute reduced intervals with overlap removed
    x_reduced = hma_olap_remove(x, c)
    # compute the trapezoidal parameters
    tp = hma_map(x_reduced, c, x0, x1)
    # calculate trapezoid parameters for different FOUs
    if c == "Interior FOU":
        # tp is a nested 2-vector of 2-vectors for left/right UMF/LMF x-axis intercepts
        out_UMF = [tp[0][0], olap_interval[0], olap_interval[1], tp[1][1]]
        out_LMF = [tp[0][1], olap_interval[0], olap_interval[1], tp[1][0]] 
    elif c == "Left shoulder FOU":
        # tp is a 2-vector of UMF/LMF x-axis intercepts
        out_UMF = [x0, x0, olap_interval[1], tp[0][1]]
        out_LMF = [x0, x0, olap_interval[1], tp[0][0]]
    elif c == "Right shoulder FOU":
        # tp is a 2-vector of UMF/LMF x-axis intercepts
        out_UMF = [tp[0][0], olap_interval[0], x1, x1]
        out_LMF = [tp[0][1], olap_interval[0], x1, x1]
    return [out_UMF, out_LMF]

G = hma_trap(dataclean.yVG, 0, 10)
print(G[0])
print(G[1])

def trap_z(x, h):
    """Trapzoidal function with h parameters"""
    return [noncenteral_tstatic_cdf.trapz(x, 1, h[0][0], h[1][0], h[2][0], h[3][0]), noncenteral_tstatic_cdf.trapz(x, 1, h[0][1], h[1][1], h[2][1], h[3][1])]