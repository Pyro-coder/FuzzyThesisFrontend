import numpy as np


def umf(x, ml, mr, sigma, ks):
    if ml - ks * sigma <= x <= ml:
        return np.exp(-0.5 * ((x - ml) / sigma) ** 2)
    elif ml <= x <= mr:
        return 1
    elif mr < x <= mr + ks * sigma:
        return np.exp(-0.5 * ((x - mr) / sigma) ** 2)
    else:
        return 0


def lmf(x, ml, mr, sigma, ks):
    if mr - ks * sigma <= x <= (ml + mr) / 2:
        return np.exp(-0.5 * ((x - mr) / sigma) ** 2)
    elif (ml + mr) / 2 < x <= ml + ks * sigma:
        return np.exp(-0.5 * ((x - ml) / sigma) ** 2)
    else:
        return 0


def fouset(umf, lmf, xmin, xmax, xinc, yinc):
    """
    Create a set S of points (x, f(x)) belonging to the FOU bounded by lmf(x) and umf(x)

    Args:
    umf : function
        Upper membership function
    lmf : function
        Lower membership function
    xmin : float
        Minimum x value
    xmax : float
        Maximum x value
    xinc : float
        Increment for x
    yinc : float
        Increment for y

    Returns:
    S : numpy array
        Set of points (x, y) belonging to the FOU
    """
    S = []
    i = 1

    x_values = np.arange(xmin, xmax + xinc, xinc)
    y_values = np.arange(yinc, 1 + 2 * yinc, yinc)

    for x in x_values:
        upper = umf(x)
        lower = lmf(x)

        for y in y_values:
            if lower < y < upper:
                S.append([x, y])
                i += 1

    return np.array(S)


def alpha_fouset(au, al, xinc, yinc):
    S = []
    i = 1
    j = 0
    while j < min(len(al), len(au)):
        x = au[j][0] + xinc
        while x <= al[j][0] - xinc:
            S.append((x, au[j][2]))
            x += xinc
        x = al[j][1] + xinc
        while x <= au[j][1] - xinc:
            S.append((x, au[j][2]))
            x += xinc
        j += 1
    return S

# Parameters
# ml, mr, sigma, ks = -2, 2, 2, 3
# xmin, xmax, xinc, yinc = -10, 10, 0.01, 0.01

# # Generate FOU set
# x_coords, y_coords = fouset(partial(umf, ml=ml, mr=mr, sigma=sigma, ks=ks), 
#                            partial(lmf, ml=ml, mr=mr, sigma=sigma, ks=ks), 
#                            xmin, xmax, xinc, yinc)

# Create the plot
# plt.figure(figsize=(10, 6))
# plt.scatter(x_coords, y_coords, c='blue', s=1, label='FOU Points')  # s=1 for smaller point size
# plt.title('Fuzzy Output (FOU) Set of Points')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend()
# plt.grid(True)
# plt.show()
