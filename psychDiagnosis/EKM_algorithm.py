import numpy as np

def wa(x, w):
    """ Compute the weighted average of x with weights w. """
    num = np.dot(w.T, x)  # Dot product of weights and values
    den = np.sum(w)       # Sum of weights
    return num / den


def selectf(z, x, i):
    """ Select the ith component of the vector function z(x). """
    return z(x)[i]


def lIWA(x, c, d, k):
    """ Compute left interval bound. """
    num1 = np.sum(x[:k] * d[:k])
    num2 = np.sum(x[k:] * c[k:])
    den1 = np.sum(d[:k])
    den2 = np.sum(c[k:])
    return (num1 + num2) / (den1 + den2)

def rIWA(x, c, d, k):
    """ Compute right interval bound. """
    num1 = np.sum(x[:k] * c[:k])
    num2 = np.sum(x[k:] * d[k:])
    den1 = np.sum(c[:k])
    den2 = np.sum(d[k:])
    return (num1 + num2) / (den1 + den2)

def augment(arr1, arr2, arr3):
    """
    Augment three arrays into a single array with three columns.
    """
    return np.column_stack((arr1, arr2, arr3))

def csort(arr, col):
    """
    Custom sort function to sort a 2D array based on a specific column.
    """
    return arr[arr[:, col].argsort()]

def kmalg(x, w):
    """
    Enhanced Karnik-Mendel algorithm for interval weighted average.
    Parameters:
    x : array : array of interval endpoints for x(i): [xmin(i), xmax(i)]
    w : array : array of interval endpoints for w(i): [wmin(i), wmax(i)]
    Returns:
    out : array : [yL, yR, kL, kR]
    """
    N = x.shape[0]

    # Enhanced KM algorithm for yL
    aug = augment(x[:, 0], w[:, 0], w[:, 1])
    aug = csort(aug, 0)
    xminsort = aug[:, 0]
    wminsort = aug[:, 1]
    wmaxsort = aug[:, 2]

    k = int(round(N / 2.4))
    if k < 1:
        k = 1

    a = np.sum(xminsort[:k] * wmaxsort[:k]) + np.sum(xminsort[k:] * wminsort[k:])
    b = np.sum(wmaxsort[:k]) + np.sum(wminsort[k:])

    y = a / b if b != 0 else 0

    for kprime in range(1, N):
        if xminsort[kprime - 1] <= y <= xminsort[kprime]:
            break

    while kprime != k:
        s = np.sign(kprime - k)
        aprime = a + s * np.sum(xminsort[min(k, kprime):max(k, kprime)] * (wmaxsort[min(k, kprime):max(k, kprime)] - wminsort[min(k, kprime):max(k, kprime)]))
        bprime = b + s * np.sum(wmaxsort[min(k, kprime):max(k, kprime)] - wminsort[min(k, kprime):max(k, kprime)])

        yprime = aprime / bprime if bprime != 0 else 0

        y = yprime
        a = aprime
        b = bprime
        k = kprime

        for kprime in range(1, N):
            if xminsort[kprime - 1] <= y <= xminsort[kprime]:
                break

    yL = y
    kL = k - 1

    # KM algorithm for yR
    aug = augment(x[:, 1], w[:, 0], w[:, 1])
    aug = csort(aug, 0)
    xmaxsort = aug[:, 0]
    wminsort = aug[:, 1]
    wmaxsort = aug[:, 2]

    k = int(round(N / 1.7))
    if k < 1:
        k = 1

    a = np.sum(xmaxsort[:k] * wminsort[:k]) + np.sum(xmaxsort[k:] * wmaxsort[k:])
    b = np.sum(wminsort[:k]) + np.sum(wmaxsort[k:])

    y = a / b if b != 0 else 0

    for kprime in range(1, N):
        if xmaxsort[kprime - 1] <= y <= xmaxsort[kprime]:
            break

    while kprime != k:
        s = np.sign(kprime - k)
        aprime = a - s * np.sum(xmaxsort[min(k, kprime):max(k, kprime)] * (wmaxsort[min(k, kprime):max(k, kprime)] - wminsort[min(k, kprime):max(k, kprime)]))
        bprime = b - s * np.sum(wmaxsort[min(k, kprime):max(k, kprime)] - wminsort[min(k, kprime):max(k, kprime)])

        yprime = aprime / bprime if bprime != 0 else 0

        y = yprime
        a = aprime
        b = bprime
        k = kprime

        for kprime in range(1, N):
            if xmaxsort[kprime - 1] <= y <= xmaxsort[kprime]:
                break

    yR = y
    kR = k - 1

    return np.array([yL, yR, kL, kR])

# # Example data input
# x = np.array([[0.5, 1.5], [1.8, 4.2], [3.7, 6.3], [5.5, 8.5], [8.2, 9.8]])
# w = np.array([[1, 3], [0.6, 1.4], [5.1, 6.9], [2.4, 5.6], [7, 9]])

# # Calculate the Karnik-Mendel algorithm result
# numberresult = kmalg(x, w)
# print("Result of kmalg:", numberresult)

# print((numberresult[0] + numberresult[1])/2)