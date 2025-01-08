import numpy as np


def trap(x, ht, lb, lt, rt, rb):
    # Check if x is outside the bounds of the trapezoid (either to the left of lb or to the right of rb)
    if x < lb or x > rb:
        return 0

    # Check if x is between the left top and right top corners (flat top part of the trapezoid)
    if lt <= x <= rt:
        return ht

    # Linear interpolation on the left slope of the trapezoid
    if lb <= x < lt:
        return ht * (x - lb) / (lt - lb)

    # Linear interpolation on the right slope of the trapezoid
    if rt < x <= rb:
        return ht * (rb - x) / (rb - rt)

    # If x is exactly on the edge, handle potentially as a flat top (this line might be redundant depending on real edge behavior)
    return 0
