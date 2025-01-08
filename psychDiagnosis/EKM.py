import numpy as np


def h(z, r):
    if r != 0:
        return z ** (1 / r)
    else:
        return np.exp(z)


def hinv(z, r):
    if r != 0:
        if r < 0 and z == 0:
            return float('inf')  # Infinity
        else:
            return z ** r
    else:
        if z != 0:
            return np.log(z)
        else:
            return float('-inf')  # Negative infinity


def wpmekml(x, w, r):
    """
    EKM algorithm for weighted power mean with r finite; left endpoint
    x is an array of interval endpoints for x(i):  [xmin(i),xmax(i)]
    w is an array of interval endpoints for w(i):  [wmin(i),wmax(i)]
    """
    x = np.array(x)
    w = np.array(w)

    N = x.shape[0]

    # If r is negative, check for zero elements in x
    if r < 0:
        for i in range(x.shape[0]):
            if x[i, 0] == 0:
                return 0

    # Augment and sort
    aug = np.c_[x[:, 0], w]
    aug = aug[aug[:, 0].argsort()]

    xminsort = aug[:, 0]
    u = aug[:, 1]
    v = aug[:, 2]

    # Initialize k (the switching point)
    k = round(N / 2.4) - 1

    a = np.sum([hinv(xminsort[i], r) * v[i] for i in range(k + 1)]) + np.sum(
        [hinv(xminsort[i], r) * u[i] for i in range(k + 1, N)])
    b = np.sum(v[:k + 1]) + np.sum(u[k + 1:])
    y = a / b
    yL = h(y, r)
    kL = k

    for kprime in range(N - 1):
        if xminsort[kprime] <= h(y, r) <= xminsort[kprime + 1]:
            break

    if xminsort[0] == xminsort[N - 1]:
        return xminsort[0]

    while kprime != k:
        s = np.sign(kprime - k)

        aprime = a + s * np.sum(
            [hinv(xminsort[i], r) * (v[i] - u[i]) for i in range(min(k, kprime) + 1, max(k, kprime) + 1)])
        bprime = b + s * np.sum([v[i] - u[i] for i in range(min(k, kprime) + 1, max(k, kprime) + 1)])

        yprime = aprime / bprime
        y = yprime
        a = aprime
        b = bprime
        k = kprime
        yL = h(y, r)
        kL = k

        for kprime in range(N - 1):
            if xminsort[kprime] <= h(y, r) <= xminsort[kprime + 1]:
                break

    return yL


def wpmekmr(x, w, r):
    """
    EKM algorithm for weighted power mean with r finite; right endpoint
    x is an array of interval endpoints for x(i):  [xmin(i),xmax(i)]
    w is an array of interval endpoints for w(i):  [wmin(i),wmax(i)]
    """
    x = np.array(x)
    w = np.array(w)

    N = x.shape[0]

    # Augment and sort
    aug = np.c_[x[:, 1], w]
    aug = aug[aug[:, 0].argsort()]

    xmaxsort = aug[:, 0]
    v = aug[:, 1]
    u = aug[:, 2]

    # Initialize k (the switching point)
    k = round(N / 1.7) - 1

    a = np.sum([hinv(xmaxsort[i], r) * v[i] for i in range(k + 1)]) + np.sum(
        [hinv(xmaxsort[i], r) * u[i] for i in range(k + 1, N)])
    b = np.sum(v[:k + 1]) + np.sum(u[k + 1:])
    y = a / b
    yR = h(y, r)
    kR = k

    for kprime in range(N - 1):
        if xmaxsort[kprime] <= h(y, r) <= xmaxsort[kprime + 1]:
            break

    if xmaxsort[0] == xmaxsort[N - 1]:
        return xmaxsort[0]

    while kprime != k:
        s = np.sign(kprime - k)

        aprime = a + s * np.sum(
            [hinv(xmaxsort[i], r) * (v[i] - u[i]) for i in range(min(k, kprime) + 1, max(k, kprime) + 1)])
        bprime = b + s * np.sum([v[i] - u[i] for i in range(min(k, kprime) + 1, max(k, kprime) + 1)])

        yprime = aprime / bprime
        y = yprime
        a = aprime
        b = bprime
        k = kprime
        yR = h(y, r)
        kR = k

        for kprime in range(N - 1):
            if xmaxsort[kprime] <= h(y, r) <= xmaxsort[kprime + 1]:
                break

    return yR


def wpminfendpts(x, r):
    if r == float('-inf'):
        return np.min(x[:, 0]), np.min(x[:, 1])
    elif r == float('inf'):
        return np.max(x[:, 0]), np.max(x[:, 1])


def t2wpm(x, w, r):
    if r == float('inf') or r == float('-inf'):
        return wpminfendpts(x, r)
    else:
        return wpmekml(x, w, r), wpmekmr(x, w, r)


def wpm(x, w, r):
    if 0 < abs(r) < float('inf'):
        xx = x ** r
        return (w * xx).sum() / w.sum() ** (1 / r)
    elif r == 0:
        return np.product(x ** (w / w.sum()))
    elif r == float('-inf'):
        return np.min(x)
    elif r == float('inf'):
        return np.max(x)
