import math

import numpy as np
import statistics
import openpyxl
from scipy.integrate import quad
from scipy.optimize import root
from scipy.stats import chi2, norm


def round_above_threshold(n):
    if n - int(n) >= 0.98: 
        return round(n)
    else: 
        return n

def trap(x, ht, lb, lt, rt, rb):
    # trapezoid of height ht, left base lb, left top lt, right top rt, right base rb
    global tr
    if x < lb:
        return 0
    if x > rb:
        return 0
    if lt <= x <= rt:
        return ht
    if lb <= x < lt:
        tr = ht * (x - lb) / (lt - lb)
    if rt < x <= rb:
        tr = ht * (rb - x) / (rb - rt)
    return tr


def bad_data(x, x0, x1):
    # eliminates invalid intervals
    y = None
    x = np.array(x)
    a = x[:, 0]
    b = x[:, 1]
    for i in range(0, len(x)):
        # remove rows with infeasible endpoints
        if ((x0 <= a[i] < b[i] <= x1) and (b[i] - a[i] < x1 - x0)):
            if y is None:
                y = [[a[i], b[i]]]
            else:
                y.append([a[i], b[i]])
    return y


# data = [[0, 10], [-1, 10], [0, 11], [5, 4], [1, 9], [0, 8], [3, 3], [50, 10]]

# print(bad_data(data, 0, 10))

# web_data_copy = openpyxl.load_workbook('excel/WEBdatacopy.xlsx')

# wdc_sheet = web_data_copy['data']

# range_values = wdc_sheet['AI2:AJ175']


# for row in range_values:
#    for cell in row:
#        print(cell.value, end=',')
#    print()

def box_and_whisker(x):
    # calculate quartiles and inter-quartile ranges on right or left interval endpoints
    # x is an n-vector of right or left interval endpoints
    # calculate quartile index
    global q75, q25
    nq = np.floor(len(x) / 4)

    # skip this test if nq = 0
    if (nq == 0):
        return x

    # find first and third quartile values
    remainder = np.mod(len(x), 4)
    y = np.sort(x)
    if (remainder == 0):
        q25 = (y[int(nq)] + y[int(nq) - 1]) / 2
        q75 = (y[int(3 * nq)] + y[int(3 * nq) - 1]) / 2
    elif (remainder == 1):
        q25 = (y[int(nq)] + y[int(nq) - 1]) / 2
        q75 = (y[int(3 * nq)] + y[int(3 * nq + 1)]) / 2
    elif (remainder == 2):
        q25 = y[int(nq)]
        q75 = y[int(3 * nq + 1)]
    elif (remainder == 3):
        q25 = y[int(nq)]
        q75 = y[int(3 * nq + 2)]

    # find inner quartile range and bounds of valid endpoints
    iqr = q75 - q25
    xmin = q25 - 1.5 * iqr
    xmax = q75 + 1.5 * iqr
    return [xmin, xmax]


def outlier_test(x):
    # flags outlier endpoints
    # x is an n x 2 vector of right and left interval endpoints
    # calculate box and whisker test on x[0] and x[1]
    x = np.array(x)
    x0 = box_and_whisker(x[:, 0])
    x1 = box_and_whisker(x[:, 1])
    y = []

    # eliminate outlier endpoints in x using box and whisker test
    for i in range(0, len(x)):
        if (x0[0] <= x[i, 0] <= x0[1] and x1[0] <= x[i, 1] <= x1[1]):
            y.append(x[i])
    if len(y) == 0:
        return None

    y = np.array(y)
    # y has first pass outliers eliminated
    # repeat box and whisker test on L=y[1]-y[0]
    L = y[:, 1] - y[:, 0]
    Lbw = box_and_whisker(L)
    z = []
    for i in range(0, len(y)):
        if (Lbw[0] <= L[i] <= Lbw[1]):
            z.append(y[i])
    if len(z) == 0:
        return None
    return np.array(z)


outlietest = [[0, 6],
              [2, 4],
              [3, 4],
              [5, 6],
              [4, 10],
              [5, 5.5],
              [5.2, 5.3]]


# print(outlier_test(outlietest))


def fr(y, a):
    r = y
    func = lambda r: norm.cdf(y + r) - norm.cdf(y - r) - (1 - a)
    return root(func, r).x[0]


def fk(xk, a, n):
    def integrand(y):
        term1 = ((n - 1) * fr(y, a) ** 2) / (xk ** 2)
        term2 = n - 1
        pchisq = 1 - chi2.cdf(term1, term2)
        term3 = (-1 / 2) * n * y ** 2
        return ((2 * np.sqrt(n)) / (np.sqrt(2 * np.pi))) * pchisq * np.exp(term3)

    result, error = quad(integrand, 0, np.inf)
    return result


def kk(a, y, n):
    xk = 2
    func = lambda xk: fk(xk, a, n) - (1 - y)
    return root(func, xk).x[0]


# print(kk(0.05,0.05,30))

def tolerance(x, m, sigma, k):
    # flags valuews of x where either endpoint is outside the tolerance limits
    upperlim = m + k * sigma
    lowerlim = m - k * sigma
    for i in range(len(x)):
        if lowerlim <= x[i] <= upperlim:
            continue
        else:
            x[i] = -1000
    return x


def reasonable(x, ml, mr, sigma_l, sigma_r):
    # eliminate unreasonable intervals that do not overlap all other intervals or have too little overlap
    # x is an n x 2 vector of intervals
    # ml, mr, sigma_l, sigma_r are means and standard deviations
    # If sigma_l or sigma_r is zero, all intervals overlap by definition
    if sigma_l == 0 or sigma_r == 0:
        return x
    # If sigma)l == sigma_r, the solution to Eq. (A-6) in Ref. [1] is the mean of the means
    if sigma_l == sigma_r:
        zeta = (ml + mr) / 2
    else:
        zeta_1 = ((mr * sigma_l ** 2 - ml * sigma_r ** 2) - sigma_l * sigma_r * np.sqrt(
            (ml - mr) ** 2 + 2 * (sigma_l ** 2 - sigma_r ** 2) * math.log(sigma_l / sigma_r))) / (
                             sigma_l ** 2 - sigma_r ** 2)
        zeta_2 = ((mr * sigma_l ** 2 - ml * sigma_r ** 2) + sigma_l * sigma_r * np.sqrt(
            (ml - mr) ** 2 + 2 * (sigma_l ** 2 - sigma_r ** 2) * math.log(sigma_l / sigma_r))) / (
                             sigma_l ** 2 - sigma_r ** 2)

        if ml <= zeta_1 <= mr:
            zeta = zeta_1
        else:
            zeta = zeta_2

    y = []

    for i in range(len(x)):

        # maybe the one line I am not proud of, because of how python handles floating point numbers, it is necessary to round the final value
        if (2 * ml - zeta <= x[i][0] < zeta < x[i][1] <= round_above_threshold(2 * mr - zeta)):
            y.append(x[i])
    return y

# test = [[5, 6], [3, 4], [4, 7], [4, 6], [4, 7], [4, 5], [4.2, 5.3]]

# ml = statistics.mean([item[0] for item in test])
# mr = statistics.mean([item[1] for item in test])

# sigma_l = np.std([item[0] for item in test])
# sigma_r = np.std([item[1] for item in test])

# print(reasonable(test, ml, mr, sigma_l, sigma_r))


#test = [[-1000.0, 2.5], [3.0, 3.5], [0.8, 1.5], [1.0, 3.5], [4.0, 8.0], [0.0, 3.0], [0.5, 1.5], [0.0, 2.5], [0.0, 1.0], [1.0, 2.1], [1.0, 2.5], [0.0, 3.0], [0.0, 1.0], [1.0, 3.0], [0.0, 2.0], [0.3, 2.0], [1.0, 3.0], [0.0, 1.0], [3.0, 7.0], [0.0, 2.0], [0.0, 1.0], [0.0, 2.0], [0.0, 2.0], [0.0, 1.5], [0.0, 3.0], [1.0, 3.0], [0.0, 0.5], [0.0, 2.0], [0.0, 2.0], [0.0, 3.0], [0.0, 2.5], [0.25, 2.0], [8.0, 9.6], [0.0, 1.0], [1.5, 3.0], [0.0, 2.0], [0.0, 2.0], [0.2, 2.0], [0.0, 1.0], [0.0, 1.0]]

#ml = 0.3208333333333333
#mr = 2.0444444444444443

#sigma_l = 0.5171200967323204
#sigma_r = 0.7671898375979573

#print(reasonable(test, ml, mr, sigma_l, sigma_r))

