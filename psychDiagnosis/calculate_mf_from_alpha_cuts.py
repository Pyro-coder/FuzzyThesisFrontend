def mu_sf(x, a):
    """
    Type-1 MF corresponding to an (m+1)x3 array of α-cuts A

    Args:
    x : float
    A : numpy array of shape (m+1, 3)

    Returns:
    mu : float
    """
    mu = 0
    m = a.shape[0]

    # Set mu = largest alpha for x lying within the corresponding alpha cut interval of S
    for j in range(m):
        if a[j, 0] <= x <= a[j, 1]:
            mu = a[j, 2]

    # Set mu = its maximum at the top alpha cut if its interval is zero length (e.g., for Gaussian MFs)
    if a[m - 1, 0] == a[m - 1, 1] and abs(x - a[m - 1, 0]) < 0.001:
        mu = a[m - 1, 2]

    return mu
