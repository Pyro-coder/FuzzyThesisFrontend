import numpy as np
import matplotlib.pyplot as plt

from .max_alpha_for_lmf import alpha_max_lmf
from .EKM import t2wpm
from .trapezoid import trap
from .calculate_mf_from_alpha_cuts import mu_sf


def alpha_to_alpha_t2wpm(aux, alx, auw, alw, r):
    """
    Calculate α-cuts of Type-2 fuzzy weighted average given UMF/LMF α-cuts of inputs.

    Parameters:
        aux (list of list of np.array): Nested n-vectors, each element containing an array of UMF α-cuts.
        alx (list of list of np.array): Nested n-vectors, each element containing an array of LMF α-cuts.
        auw (list of list of np.array): Nested n-vectors, each element containing an array of UMF weights.
        alw (list of list of np.array): Nested n-vectors, each element containing an array of LMF weights.
        r (float): Parameter for the weighted power mean.

    Returns:
        list: A nested 2-vector, each element containing an array of corresponding α-cut intervals of the WPM UMF/LMF, respectively.
    """

    n = len(aux)
    m = len(aux[0])

    xuab = []
    wucd = []

    for j in range(m):
        xuab_j = []
        wucd_j = []
        for i in range(n):
            if i == 0:
                xuab_j = aux[i][j]
                wucd_j = auw[i][j]
            else:
                xuab_j = np.vstack((xuab_j, aux[i][j]))
                wucd_j = np.vstack((wucd_j, auw[i][j]))
        xuab.append(xuab_j)
        wucd.append(wucd_j)

    hmin = alpha_max_lmf(alx, alw)

    xlab = []
    wlcd = []

    if r != -np.inf and r != np.inf:
        for j in range(hmin[0] + 1):
            xlab_j = []
            wlcd_j = []
            for i in range(n):
                if i == 0:
                    xlab_j = alx[i][j]
                    wlcd_j = alw[i][j]
                else:
                    xlab_j = np.vstack((xlab_j, alx[i][j]))
                    wlcd_j = np.vstack((wlcd_j, alw[i][j]))
            xlab.append(xlab_j)
            wlcd.append(wlcd_j)

    elif r == -np.inf:
        xminleft = [np.inf] * (hmin[0] + 1)
        xminright = [np.inf] * (hmin[0] + 1)
        for j in range(hmin[0] + 1):
            for i in range(n):
                if len(alx[i]) - 1 >= j:
                    xminleft[j] = min(xminleft[j], alx[i][j][0])
                    xminright[j] = min(xminright[j], alx[i][j][1])
            xlab.append(np.array([[xminleft[j], xminright[j], j / (m - 1)]]))

    elif r == np.inf:
        xmaxleft = [-np.inf] * (hmin[0] + 1)
        xmaxright = [-np.inf] * (hmin[0] + 1)
        for j in range(hmin[0] + 1):
            for i in range(n):
                if len(alx[i]) - 1 >= j:
                    xmaxleft[j] = max(xmaxleft[j], alx[i][j][0])
                    xmaxright[j] = max(xmaxright[j], alx[i][j][1])
            xlab.append(np.array([[xmaxleft[j], xmaxright[j], j / (m - 1)]]))

    out = [[], []]

    for j in range(m):
        zu = t2wpm(xuab[j], wucd[j], r)
        if j == 0:
            out[0] = np.array([[zu[0], zu[1], xuab[j][0][2]]])
        else:
            out[0] = np.vstack((out[0], [zu[0], zu[1], xuab[j][0][2]]))

    for j in range(hmin[0] + 1):
        if r != -np.inf and r != np.inf:
            zl = t2wpm(xlab[j], wlcd[j], r)
        else:
            zl = xlab[j]
        if j == 0:
            out[1] = np.array([[zl[0], zl[1], j / (m - 1)]])
        else:
            out[1] = np.vstack((out[1], [zl[0], zl[1], j / (m - 1)]))

    return out


def apply_trapezoidal_mfs(x, arrays):
    """Apply trapezoidal membership functions to x for a given list of parameter arrays."""
    results = []
    for array in arrays:
        # Calculate membership value for each array and add to results
        results.append(trap(x, *array))
    return np.array(results)


# # Membership function parameters
# zu0 = np.array([1, 0.1, 0.8, 1.2, 2])
# zl0 = np.array([0.8, 0.5, 0.8, 1.2, 1.5])
# wu0 = np.array([1, 0, 0.7, 1.3, 2])
# wl0 = np.array([0.8, 0, 1, 1, 2])
# zu1 = np.array([1, 0.1, 1, 2, 3])
# zl1 = np.array([0.7, 0.5, 1.5, 1.8, 2.5])
# wu1 = np.array([1, 0, 1.8, 2.2, 3])
# wl1 = np.array([0.6, 1, 1.5, 2, 2.5])
#
#
# def zu(x):
#     """Compute trapezoidal MFs for zu parameters."""
#     params = [zu0, zu1]  # zu1
#     results = [trap(x, *param) for param in params]
#     return np.array(results)
#
#
# def zl(x):
#     """Compute trapezoidal MFs for zl parameters."""
#     params = [zl0, zl1]  # zl1
#     results = [trap(x, *param) for param in params]
#     return np.array(results)
#
#
# def wu(x):
#     """Compute trapezoidal MFs for wu parameters."""
#     params = [wu0, wu1]  # wu1
#     results = [trap(x, *param) for param in params]
#     return np.array(results)
#
#
# def wl(x):
#     """Compute trapezoidal MFs for wl parameters."""
#     params = [wl0, wl1]  # wl1
#     results = [trap(x, *param) for param in params]
#     return np.array(results)
#
#
# def z0u(x):
#     """Function to extract the first element from the vector returned by zu(x)."""
#     return select_f(zu, x, 0)  # 0 indicates the first element
#
#
# # Function to extract the second element from the vector returned by zu(x)
# def z1u(x):
#     return select_f(zu, x, 1)
#
#
# # Function to extract the first element from the vector returned by zl(x)
# def z0l(x):
#     return select_f(zl, x, 0)
#
#
# # Function to extract the second element from the vector returned by zl(x)
# def z1l(x):
#     return select_f(zl, x, 1)
#
#
# # Function to extract the first element from the vector returned by wu(x)
# def w0u(x):
#     return select_f(wu, x, 0)
#
#
# # Function to extract the second element from the vector returned by wu(x)
# def w1u(x):
#     return select_f(wu, x, 1)
#
#
# # Function to extract the first element from the vector returned by wl(x)
# def w0l(x):
#     return select_f(wl, x, 0)
#
#
# # Function to extract the second element from the vector returned by wl(x)
# def w1l(x):
#     return select_f(wl, x, 1)
#
#
# z0fou = fouset(z0u, z0l, 0, 10, .01, 0.012)
# z1fou = fouset(z1u, z1l, 0, 10, .01, 0.013)
# w0fou = fouset(w0u, w0l, 0, 10, .01, 0.012)
# w1fou = fouset(w1u, w1l, 0, 10, .01, 0.013)

# u = np.arange(0, 3.01, 0.01)

# jz0 = range(len(z0fou))
# jz1 = range(len(z1fou))


# x_coords_z0 = [coord[0] for coord in z0fou]
# y_coords_z0 = [coord[1] for coord in z0fou]

# x_coords_z1 = [coord[0] for coord in z1fou]
# y_coords_z1 = [coord[1] for coord in z1fou]

# plt.scatter(x_coords_z0, y_coords_z0, label='z0fou')
# plt.scatter(x_coords_z1, y_coords_z1, label='z1fou')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Scatter Plot of z0fou and z1fou')
# plt.legend()
# plt.show()


# # Construct the support intervals matrix for zu and wu
# xsup = np.array([
#     [zu0[1], zu0[4]],  # Extracting the second and fifth elements of the first trapezoidal function
#     [zu1[1], zu1[4]]  # Same extraction for the second function
# ])
#
# wsup = np.array([
#     [wu0[1], wu0[4]],  # Similarly for wu
#     [wu1[1], wu1[4]]
# ])
#
# # Call the αfwa function with the specified parameters
# Au = alpha_fwa(zu, wu, xsup, wsup, 100, 300)
#
# # Construct the support intervals matrix for zl and wl
# xsup = np.array([
#     [zl0[1], zl0[4]],  # Extracting specific elements for zl
#     [zl1[1], zl1[4]]
# ])
#
# wsup = np.array([
#     [wl0[1], wl0[4]],  # Similarly for wl
#     [wl1[1], wl1[4]]
# ])
#
# # Call the αfwa function to compute the alpha cuts for the LMF
# Al = alpha_fwa(zl, wl, xsup, wsup, 100, 300)


# def lwa_umf(x):
#     return mu_sf(x, Au)
#
#
# def lwa_lmf(x):
#     return mu_sf(x, Al)
#
#
# c = t2_centroid(Au, Al, 300)
# # print(c)
#
# lwa_fou = fouset(lwa_umf, lwa_lmf, 0, 10, 0.05, 0.012)
#
# cl = c[0]
# cr = c[1]
#
# m = defuzz(c)


# print(m)


def t2wpmtroid(Au, Al, N, r):
    """
    WPM centroid interval of Type-2 MF computed using WPM algorithm.
    Au and Al are arrays of α-cuts of upper and lower MFs; N is # of x slices.
    """
    xinc = (Au[0, 1] - Au[0, 0]) / N
    x = [Au[0, 0] + i * xinc for i in range(N + 1)]
    xx = np.column_stack((x, x))  # xx is an array of intervals of zero width

    # Generate lower and upper MF values for each x
    w = np.zeros((N + 1, 2))
    for i in range(N + 1):
        w[i, 0] = mu_sf(x[i], Al)
        w[i, 1] = mu_sf(x[i], Au)

    # Ensure w intervals are not zero width
    w[w[:, 1] - w[:, 0] == 0, 1] += 0.001

    # Compute centroid interval using WPMEKM
    centr = t2wpm(xx, w, r)
    out = np.array([centr[0], centr[1]])

    return out


# # 1. Create a range of x values
# x_values = np.arange(0, 10, 0.01)
#
# # 2. Calculate the corresponding y values for lwa_umf and lwa_lmf
# y_values_umf = [lwa_umf(x) for x in x_values]
# y_values_lmf = [lwa_lmf(x) for x in x_values]
#
# # 3. Plot lwa_umf and lwa_lmf
# plt.plot(x_values, y_values_umf, label='lwa_umf')
# plt.plot(x_values, y_values_lmf, label='lwa_lmf')
#
# # 4. Plot lwa_FOU
# x_coords_fou = [coord[0] for coord in lwa_FOU]
# y_coords_fou = [coord[1] for coord in lwa_FOU]
# plt.scatter(x_coords_fou, y_coords_fou, c='blue', s=1, label='lwa_FOU')
#
# # 5. Plot the centroid
# plt.scatter(c[0], 1, c='red', label='Centroid Left')
# plt.scatter(c[1], 1, c='red', label='Centroid Right')
#
# # 6. Plot the defuzzified centroid
# plt.scatter(m, 1, c='purple', label='Defuzzified Centroid')
#
# # 7. Add labels, a legend, and a title
# plt.xlabel('x')
# plt.ylabel('Membership Degree')
# plt.title('Plot of lwa_umf, lwa_lmf, lwa_FOU, Centroid, and Defuzzified Centroid')
# plt.legend()
#
# # 8. Display the plot
# plt.show()

# print(t2wpmtroid(Au, Al, 1000, 0))

# def fz0(x):
#     return np.array([[z0u(x)], [z0l(x)]])
#
#
# def fw0(x):
#     return np.array([[w0u(x)], [w0l(x)]])
#
#
# def fz1(x):
#     return np.array([[z1u(x)], [z1l(x)]])
#
#
# def fw1(x):
#     return np.array([[w1u(x)], [w1l(x)]])
#
#
# Az0 = alpha_t2(fz0, 300, 0, 4, 100, 300)
# Aw0 = alpha_t2(fw0, 300, 0, 4, 100, 300)
# Az1 = alpha_t2(fz1, 300, 0, 4, 100, 300)
# Aw1 = alpha_t2(fw1, 300, 0, 4, 100, 300)
#
#
# def lwa_u(x):
#     return mu_sf(x, Au)
#
#
# def lwa_l(x):
#     return mu_sf(x, Al)
#
#
# lwa_fou = fouset(lwa_u, lwa_l, 0, 10, 0.04, 0.012)
#
# c = t2_centroid(Au, Al, 300)
# cl = c[0]
# cr = c[1]
# m = defuzz(c)
#
# auz = [Az0[0], Az1[0]]
# alz = [Az0[1], Az1[1]]
#
# auw = [Aw0[0], Aw1[0]]
# alw = [Aw0[1], Aw1[1]]
#
# r = -10
#
# alpha_wpm = alpha_to_alpha_t2wpm(auz, alz, auw, alw, r)
#
#
# def wpm_u(x):
#     return mu_sf(x, alpha_wpm[0])
#
#
# def wpm_l(x):
#     return mu_sf(x, alpha_wpm[1])
#
#
# wpm_fou = fouset(wpm_u, wpm_l, 0, 10, 0.04, 0.013)

# # PLOTTING CODE HERE
# x_values = np.arange(0, 10.01, 0.01)
#
# # Calculate the corresponding y values for lwa_umf and lwa_lmf
# y_values_umf = [lwa_umf(x) for x in x_values]
# y_values_lmf = [lwa_lmf(x) for x in x_values]
#
# # Create a high-quality plot
# plt.figure(figsize=(12, 8), dpi=300)
#
# # Plot lwa_umf and lwa_lmf
# plt.plot(x_values, y_values_umf, label='lwa_umf', color='red')
# plt.plot(x_values, y_values_lmf, label='lwa_lmf', color='red')
#
# # Plot lwa_FOU
# x_coords_fou_lwa = [coord[0] for coord in lwa_fou]
# y_coords_fou_lwa = [coord[1] for coord in lwa_fou]
# plt.scatter(x_coords_fou_lwa, y_coords_fou_lwa, c='red', s=1, label='lwa_fou')
#
# # Plot wpmU and wpmL
# y_values_wpmU = [wpm_u(x) for x in x_values]
# y_values_wpmL = [wpm_l(x) for x in x_values]
# plt.plot(x_values, y_values_wpmU, label='wpm_u', color='blue')
# plt.plot(x_values, y_values_wpmL, label='wpm_l', color='blue')
#
# # Plot wpm_FOU
# x_coords_fou_wpm = [coord[0] for coord in wpm_fou]
# y_coords_fou_wpm = [coord[1] for coord in wpm_fou]
# plt.scatter(x_coords_fou_wpm, y_coords_fou_wpm, c='blue', s=1, label='wpm_fou')
#
# # Plot the centroids
# plt.scatter(cl, 1, c='green', label='Centroid Left')
# plt.scatter(cr, 1, c='green', label='Centroid Right')
#
# # Plot the defuzzified centroid
# plt.scatter(m, 1, c='blue', label='Defuzzified Centroid')
#
# # Add labels, a legend, and a title
# plt.xlabel('x')
# plt.ylabel('Membership Degree')
# plt.title('LWA and WPM Type-2 Fuzzy Sets')
# plt.legend()
#
# # Display the plot
# plt.show()

# print("-inf: ", t2wpmtroid(alpha_wpm[0], alpha_wpm[1], 1000, float('-inf')))
# print("-10: ", t2wpmtroid(alpha_wpm[0], alpha_wpm[1], 1000, -10))
# print("-1: ", t2wpmtroid(alpha_wpm[0], alpha_wpm[1], 1000, -1))
# print("0: ", t2wpmtroid(alpha_wpm[0], alpha_wpm[1], 1000, 0))
# print("1: ", t2wpmtroid(alpha_wpm[0], alpha_wpm[1], 1000, 1))
# print("0.5: ", t2wpmtroid(alpha_wpm[0], alpha_wpm[1], 1000, 0.5))
# print("2: ", t2wpmtroid(alpha_wpm[0], alpha_wpm[1], 1000, 2))
# print("10: ", t2wpmtroid(alpha_wpm[0], alpha_wpm[1], 1000, 10))
# print("inf: ", t2wpmtroid(alpha_wpm[0], alpha_wpm[1], 1000, float('inf')))
