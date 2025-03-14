import os

from . import fou_points
from . import trapezoid
from . import trapezoid as tpz
from . import conj_disj as cd
from . import alpha_cut_power_mean as acpm
from . import alpha_cuts_UMF_LMF as acul
import matplotlib.pyplot as plt

from .calculate_mf_from_alpha_cuts import mu_sf

import numpy as np
import pandas as pd

from .fou_points import fouset
from .t2_centroid import t2_centroid, defuzz


def very_bad_lower(x):
    return tpz.trap(x, 1, 0, 0, 1.5, 3.707)


def very_bad_upper(x):
    return tpz.trap(x, 1, 0, 0, 1.5, 3.997)


def bad_lower(x):
    return tpz.trap(x, 1, 1.255, 2, 2.5, 4.625)


def bad_upper(x):
    return tpz.trap(x, 1, 0, 2, 2.5, 4.944)


def somewhat_bad_lower(x):
    return tpz.trap(x, 1, 1.5, 3, 3.5, 4.99)


def somewhat_bad_upper(x):
    return tpz.trap(x, 1, 0.957, 3, 3.5, 6.11)


def fair_lower(x):
    return tpz.trap(x, 1, 2.907, 5, 5.5, 7.717)


def fair_upper(x):
    return tpz.trap(x, 1, 2.893, 5, 5.5, 7.756)


def somewhat_good_lower(x):
    return tpz.trap(x, 1, 4.835, 6.5, 6.75, 8.62)


def somewhat_good_upper(x):
    return tpz.trap(x, 1, 4.595, 6.5, 6.75, 10)


def good_lower(x):
    return tpz.trap(x, 1, 5.409, 7, 7.5, 8.577)


def good_upper(x):
    return tpz.trap(x, 1, 4.648, 7, 7.5, 9.616)


def very_good_lower(x):
    return tpz.trap(x, 1, 7.944, 8, 10, 10)


def very_good_upper(x):
    return tpz.trap(x, 1, 6.172, 8, 10, 10)


def antonym(f, x):
    return f(10 - x)


def complement(f, x):
    return 1 - f(x)


def z7_lower(x):
    """
    Array of LMFs for 7-word vocabulary.
    Args:
        x: The input to the functions.
    Returns:
        A NumPy array containing the results of the functions applied to x.
    """
    return np.array([
        very_bad_lower(x),
        bad_lower(x),
        somewhat_bad_lower(x),
        fair_lower(x),
        somewhat_good_lower(x),
        good_lower(x),
        very_good_lower(x)
    ])


def z7_upper(x):
    """
    Array of UMFs for 7-word vocabulary.
    Args:
        x: The input to the functions.
    Returns:
        A NumPy array containing the results of the functions applied to x.
    """
    return np.array([
        very_bad_upper(x),
        bad_upper(x),
        somewhat_bad_upper(x),
        fair_upper(x),
        somewhat_good_upper(x),
        good_upper(x),
        very_good_upper(x)
    ])


# next, compute the alpha cuts for each of the vocabulary words

def f_very_bad(x):
    return np.array([z7_upper(x)[0], z7_lower(x)[0]])


def f_bad(x):
    return np.array([z7_upper(x)[1], z7_lower(x)[1]])


def f_somewhat_bad(x):
    return np.array([z7_upper(x)[2], z7_lower(x)[2]])


def f_fair(x):
    return np.array([z7_upper(x)[3], z7_lower(x)[3]])


def f_somewhat_good(x):
    return np.array([z7_upper(x)[4], z7_lower(x)[4]])


def f_good(x):
    return np.array([z7_upper(x)[5], z7_lower(x)[5]])


def f_very_good(x):
    return np.array([z7_upper(x)[6], z7_lower(x)[6]])


alpha_very_bad = acul.alpha_t2(f_very_bad, 1000, 0, 10, 100, 300)
alpha_bad = acul.alpha_t2(f_bad, 1000, 0, 10, 100, 300)
alpha_somewhat_bad = acul.alpha_t2(f_somewhat_bad, 1000, 0, 10, 100, 300)
alpha_fair = acul.alpha_t2(f_fair, 1000, 0, 10, 100, 300)
alpha_somewhat_good = acul.alpha_t2(f_somewhat_good, 1000, 0, 10, 100, 300)
alpha_good = acul.alpha_t2(f_good, 1000, 0, 10, 100, 300)
alpha_very_good = acul.alpha_t2(f_very_good, 1000, 0, 10, 100, 300)


def alpha_antonym(a):
    """
    Perform transformations on the input array `a` according to the specified logic.

    Args:
        a: A NumPy array with at least 3 columns.

    Returns:
        out: A NumPy array with the transformed values.
    """
    rows, cols = a.shape
    out = np.copy(a)

    for i in range(rows):
        out[i, 1] = 10 - a[i, 0]
        out[i, 0] = 10 - a[i, 1]
        out[i, 2] = a[i, 2]

    return out


alpha_7 = np.array([alpha_very_bad, alpha_bad, alpha_somewhat_bad, alpha_fair, alpha_somewhat_good, alpha_good,
                    alpha_very_good])

file_path = os.path.join(os.path.dirname(__file__), "excel/PCLRWords.xlsx")
words = pd.read_excel(file_path, sheet_name="Words", usecols="A:H", skiprows=2, nrows=20, header=None)

weight_words = pd.read_excel(file_path, sheet_name="Words", usecols="B:F", skiprows=26, nrows=1, header=None)

score_sheet = pd.read_excel(file_path, sheet_name="Scores", usecols="A:C", nrows=21)

scores = score_sheet['Scoring']
trait_weights = score_sheet['Weights']

vocab = score_sheet['Factors']
vocab = vocab.to_frame()
vocab['Vocab'] = 7


def create_scoredata():
    out = pd.DataFrame(index=score_sheet['Factors'], columns=['Alpha Cuts'])
    for i in range(20):
        for j in range(vocab.iloc[i, 1]):
            if scores[i] == words.at[i, j + 1]:
                out.at[score_sheet['Factors'][i], 'Alpha Cuts'] = alpha_7[j]
    return out


score_data = create_scoredata()

# Normalize factor names by stripping whitespace and converting to lowercase
score_data.index = score_data.index.str.strip().str.lower()

# Normalize your factor lists in the same way
interpersonal_factors = [
    "glib", "grandiosity", "conning", "pathological lying"
]

affective_factors = [
    "lack of remorse", "callousness", "shallow affect", "acceptance of responsibilities of actions"
]

lifestyle_factors = [
    "need for stimulation", "realistic long term goals", "impulsivity", "irresponsibility", "parasitic lifestyle",
    "number of short term marital relationships", "sexual promiscuity"
]

antisocial_factors = [
    "behavioral control", "early behavioral problems", "juvenile delinquency", "revocation of conditional release",
    "criminal versatility"
]

# Filter the score_data DataFrame for each category:
interpersonal_data = score_data.loc[interpersonal_factors]
affective_data = score_data.loc[affective_factors]
lifestyle_data = score_data.loc[lifestyle_factors]
antisocial_data = score_data.loc[antisocial_factors]

# # Print or further process the filtered data
# print("\nInterpersonal Data:")
# print(interpersonal_data)
#
# print("\nAffective Data:")
# print(affective_data)
#
# print("\nLifestyle Data:")
# print(lifestyle_data)


#
# print("\nAntisocial Data:")
# print(antisocial_data)


# Define 5 word vocabulary

def Ul(x):
    return trapezoid.trap(x, 1, 0, 0, 1, 3)


def Uu(x):
    return trapezoid.trap(x, 1, 0, 0, 1, 5.96)


def MLUl(x):
    return trapezoid.trap(x, 1, 1.98, 3, 3.5, 4.6)


def MLUu(x):
    return trapezoid.trap(x, 1, 0, 3, 3.5, 6.32)


def MLIl(x):
    return trapezoid.trap(x, 1, 4.62, 6.1, 6.5, 7.69)


def MLIu(x):
    return trapezoid.trap(x, 1, 3.98, 6.1, 6.5, 9.04)


def VIl(x):
    return trapezoid.trap(x, 1, 6.77, 8, 10, 10)


def VIu(x):
    return trapezoid.trap(x, 1, 4.52, 8, 10, 10)

def Ml(x):
    return trapezoid.trap(x, 1, 10, 10, 10, 10)

def Mu(x):
    return trapezoid.trap(x, 1, 10, 10, 10, 10)


Ufou = fou_points.fouset(Uu, Ul, 0, 10, 0.1, 0.035)
MLUfou = fou_points.fouset(MLUu, MLUl, 0, 10, 0.09, 0.025)
MLIfou = fou_points.fouset(MLIu, MLIl, 0, 10, 0.011, 0.0355)
VIfou = fou_points.fouset(VIu, VIl, 0, 10, 0.095, 0.0255)
# add fou for Maximum if needed


def z5_lower(x):
    return np.array([Ul(x), MLUl(x), MLIl(x), VIl(x), Ml(x)])


def z5_upper(x):
    return np.array([Uu(x), MLUu(x), MLIu(x), VIu(x), Mu(x)])


def f_U(x):
    return np.array([z5_upper(x)[0], z5_lower(x)[0]])


def f_MLU(x):
    return np.array([z5_upper(x)[1], z5_lower(x)[1]])


def f_MLI(x):
    return np.array([z5_upper(x)[2], z5_lower(x)[2]])


def f_VI(x):
    return np.array([z5_upper(x)[3], z5_lower(x)[3]])

def f_M(x):
    return np.array([z5_upper(x)[4], z5_lower(x)[4]])


alpha_U = acul.alpha_t2(f_U, 1000, 0, 10, 100, 300)
alpha_MLU = acul.alpha_t2(f_MLU, 1000, 0, 10, 100, 300)
alpha_MLI = acul.alpha_t2(f_MLI, 1000, 0, 10, 100, 300)
alpha_VI = acul.alpha_t2(f_VI, 1000, 0, 10, 100, 300)
alpha_M = acul.alpha_t2(f_M, 1000, 0, 10, 100, 300)

alpha_5 = np.array([alpha_U, alpha_MLU, alpha_MLI, alpha_VI, alpha_M])

vocab_5 = score_sheet['Weights']
vocab_5 = vocab_5.to_frame()
vocab_5['Vocab'] = 5


def create_weight_data():
    out = pd.DataFrame(index=score_sheet['Factors'], columns=['Alpha Cuts'])
    for i in range(20):
        for j in range(vocab_5.iloc[i, 1]):
            if trait_weights[i] == weight_words.at[0, j + 1]:
                # Convert the array to a list before assigning it to the DataFrame
                out.at[score_sheet['Factors'][i], 'Alpha Cuts'] = alpha_5[j]
                break
    return out


weight_data = create_weight_data()

# # Print the actual factor names in weight_data for debugging
# print("Actual factor names in weight_data:")
# print(weight_data.index.tolist())

# Normalize factor names by stripping whitespace and converting to lowercase in weight_data
weight_data.index = weight_data.index.str.strip().str.lower()

# Filter the weight_data DataFrame for each category:
interpersonal_weight_data = weight_data.loc[interpersonal_factors]
affective_weight_data = weight_data.loc[affective_factors]
lifestyle_weight_data = weight_data.loc[lifestyle_factors]
antisocial_weight_data = weight_data.loc[antisocial_factors]


# # Print or further process the filtered data
# print("\nInterpersonal Weight Data:")
# print(interpersonal_weight_data)
#
# print("\nAffective Weight Data:")
# print(affective_weight_data)
#
# print("\nLifestyle Weight Data:")
# print(lifestyle_weight_data)
#
# print("\nAntisocial Weight Data:")
# print(antisocial_weight_data)


def one_minus(x):
    """
    Compute (1 - x) α-cuts of an array of α-cuts x.

    Parameters:
    x (numpy.ndarray): A 2D array where each row represents an α-cut with three elements.

    Returns:
    numpy.ndarray: A 2D array with the same shape as x, containing the transformed α-cuts.
    """
    # Initialize the output array with the same shape as x
    out = np.zeros_like(x)

    # Iterate over each row in x
    for i in range(x.shape[0]):
        # Perform the transformations as specified in the Mathcad program
        out[i, 0] = 1 - x[i, 1]
        out[i, 1] = 1 - x[i, 0]
        out[i, 2] = x[i, 2]

    return out


def alpha_w(w, n):
    """
    Compute (n + 1) α-cuts array and its complement for a constant w.

    Parameters:
    w (float or numpy.ndarray): A constant value or an array.
    n (int): The number of α-cuts.

    Returns:
    tuple: A tuple containing two numpy.ndarrays. The first is the α-cuts array
           (augmented), and the second is its complement (1 - α).
    """
    # If w is a scalar, convert it to a numpy array
    w = np.asarray(w)

    # Initialize the output arrays
    wout = np.zeros((n + 1,) + np.shape(w))
    alpha = np.zeros(n + 1)

    # Compute the α-cuts array and α values
    for i in range(n + 1):
        wout[i] = w  # This works because w is now ensured to be an array
        alpha[i] = i / n

    # Augment wout with alpha
    out0 = np.column_stack((wout, wout, alpha))

    # Compute the complement using the oneminus function
    out1 = one_minus(out0)

    return [out0, out1]


def compute_alpha(x, y, r1, r2, P, R, w_func, delta_plus_func, delta_minus_func):
    """
    Generalized function to compute α-cuts of variant partial absorption operators.

    Parameters:
    x, y (list): Nested 2-vectors of upper (0) and lower (1) α-cuts of IT2 MFs.
    r1, r2 (float): Exponents for conjunction and disjunction.
    P, R (float): Penalty and reward parameters.
    w_func (function): Function to compute weights.
    delta_plus_func (function): Function to compute delta plus.
    delta_minus_func (function): Function to compute delta minus.

    Returns:
    list: A nested 2-vector containing arrays of corresponding α-cut intervals.
    """
    # Compute weights corresponding to P and R
    w = w_func(r1, r2, P, R)

    # Calculate delta_plus and delta_minus to verify correct weights
    delta_plus = delta_plus_func(w[0], w[1], r1, r2)
    delta_minus = delta_minus_func(w[0], w[1], r1, r2)

    # Compute the α-cuts of the conjunction/disjunction of x and y
    conj_disj = acpm.alpha_to_alpha_t2wpm(
        aux=[x[0], y[0]],
        alx=[x[1], y[1]],
        auw=alpha_w(w[0], 100),  # UMF α-cuts for weights
        alw=alpha_w(w[0], 100),  # LMF α-cuts for weights
        r=r1
    )

    # Compute the α-cuts of the disjunction/conjunction of x and conj_disj
    out = acpm.alpha_to_alpha_t2wpm(
        aux=[x[0], conj_disj[0]],
        alx=[x[1], conj_disj[1]],
        auw=alpha_w(w[1], 100),  # UMF α-cuts for weights
        alw=alpha_w(w[1], 100),  # LMF α-cuts for weights
        r=r2
    )

    return out


def alpha_cpa(x, y, r1=1.449, r2=-10, P=-25, R=15):
    return compute_alpha(x, y, r1, r2, P, R, cd.wdc, cd.dc_delta_plus, cd.dc_delta_minus)


def alpha_dpa(x, y, r1=-10, r2=1.449, P=-25, R=15):
    return compute_alpha(x, y, r1, r2, P, R, cd.wcd, cd.cd_delta_plus, cd.cd_delta_minus)


def alpha_calc(A, W, r):
    """
    Calculate alpha-cut arrays of Weighted Product Model (WPM) for attribute A data array
    with weights Wts array.
    r is the WPM exponent.

    Parameters:
        A (np.ndarray): Attribute data array with lower and upper bounds.
        W (np.ndarray): Weights array with lower and upper bounds.
        r (float): WPM exponent.

    Returns:
        np.ndarray: The resulting alpha-cut array.
    """
    # Check if A has more than one row
    if A.shape[0] > 1:
        # Extract xU and xL arrays
        xU = np.zeros(A.shape[0])
        xL = np.zeros(A.shape[0])
        for i in range(A.shape[0]):
            xU[i] = A[i, 0]
            xL[i] = A[i, 1]

        # Extract wU and wL arrays
        wU = np.zeros(W.shape[0])
        wL = np.zeros(W.shape[0])
        for i in range(W.shape[0]):
            wU[i] = W[i, 0]
            wL[i] = W[i, 1]

    # Calculate alpha_out
    alpha_out = A[0, 1]  # Default value if no calculations needed

    if A.shape[0] > 1:
        alpha_out = acpm.alpha_to_alpha_t2wpm(xU, xL, wU, wL, r)

    return alpha_out



def interpersonal_desirable(exponent):
    xDU = [ interpersonal_data.values[0][0][0], interpersonal_data.values[1][0][0] ]
    xDL = [ interpersonal_data.values[0][0][1], interpersonal_data.values[1][0][1] ]

    wDU = [ interpersonal_weight_data.values[0][0][0], interpersonal_weight_data.values[1][0][0] ]
    wDL = [ interpersonal_weight_data.values[0][0][1], interpersonal_weight_data.values[1][0][1] ]

    iD = acpm.alpha_to_alpha_t2wpm(xDU, xDL, wDU, wDL, exponent)

    return alpha_dpa(interpersonal_data.values[2][0].tolist(), iD)

def interpersonal_calc():
    desirable = interpersonal_desirable(2)
    return alpha_cpa(interpersonal_data.values[3][0].tolist(), desirable)


def affective_calc(mandatory_exponent, desired_exponent):
    xMU = [ affective_data.values[0][0][0], affective_data.values[1][0][0] ]
    xML = [ affective_data.values[0][0][1], affective_data.values[1][0][1] ]

    wMU = [ affective_weight_data.values[0][0][0], affective_weight_data.values[1][0][0] ]
    wML = [ affective_weight_data.values[0][0][1], affective_weight_data.values[1][0][1] ]

    alpha_mandatory = acpm.alpha_to_alpha_t2wpm(xMU, xML, wMU, wML, mandatory_exponent)

    xDU = [ affective_data.values[2][0][0], affective_data.values[3][0][0] ]
    xDL = [ affective_data.values[2][0][1], affective_data.values[3][0][1] ]

    wDU = [ affective_weight_data.values[2][0][0], affective_weight_data.values[3][0][0] ]
    wDL = [ affective_weight_data.values[2][0][1], affective_weight_data.values[3][0][1] ]

    alpha_desired = acpm.alpha_to_alpha_t2wpm(xDU, xDL, wDU, wDL, desired_exponent)

    return alpha_cpa(alpha_mandatory, alpha_desired)


def lifestyle_calc(exponent):
    xDU = [ lifestyle_data.values[0][0][0], lifestyle_data.values[1][0][0], lifestyle_data.values[5][0][0] ]
    xDL = [ lifestyle_data.values[0][0][1], lifestyle_data.values[1][0][1], lifestyle_data.values[5][0][1] ]

    wDU = [ lifestyle_weight_data.values[0][0][0], lifestyle_weight_data.values[1][0][0], lifestyle_weight_data.values[5][0][0] ]
    wDL = [ lifestyle_weight_data.values[0][0][1], lifestyle_weight_data.values[1][0][1], lifestyle_weight_data.values[5][0][1] ]

    lifestyle_desirable = acpm.alpha_to_alpha_t2wpm(xDU, xDL, wDU, wDL, exponent)

    xSU = [ lifestyle_data.values[2][0][0], lifestyle_data.values[3][0][0], lifestyle_data.values[4][0][0],
            lifestyle_data.values[6][0][0] ]
    xSL = [lifestyle_data.values[2][0][1], lifestyle_data.values[3][0][1], lifestyle_data.values[4][0][1],
           lifestyle_data.values[6][0][1]]

    wSU = [lifestyle_weight_data.values[2][0][0], lifestyle_weight_data.values[3][0][0], lifestyle_weight_data.values[4][0][0],
           lifestyle_weight_data.values[6][0][0] ]
    wSL = [lifestyle_weight_data.values[2][0][1], lifestyle_weight_data.values[3][0][1], lifestyle_weight_data.values[4][0][1],
           lifestyle_weight_data.values[6][0][1]]

    lifestyle_sufficient = acpm.alpha_to_alpha_t2wpm(xSU, xSL, wSU, wSL, exponent)

    return alpha_dpa(lifestyle_sufficient, lifestyle_desirable)

def antisocial_calc(exponent):
    xDU = [ antisocial_data.values[0][0][0], antisocial_data.values[3][0][0] ]
    xDL = [ antisocial_data.values[0][0][1], antisocial_data.values[3][0][1] ]

    wDU = [ antisocial_weight_data.values[0][0][0], antisocial_weight_data.values[3][0][0] ]
    wDL = [ antisocial_weight_data.values[0][0][1], antisocial_weight_data.values[3][0][1] ]

    antisocial_desirable = acpm.alpha_to_alpha_t2wpm(xDU, xDL, wDU, wDL, exponent)

    xSU = [ antisocial_data.values[1][0][0], antisocial_data.values[2][0][0], antisocial_data.values[4][0][0] ]
    xSL = [ antisocial_data.values[1][0][1], antisocial_data.values[2][0][1], antisocial_data.values[4][0][1] ]

    wSU = [ antisocial_weight_data.values[1][0][0], antisocial_weight_data.values[2][0][0], antisocial_weight_data.values[4][0][0] ]
    wSL = [ antisocial_weight_data.values[1][0][1], antisocial_weight_data.values[3][0][1], antisocial_weight_data.values[4][0][1] ]

    antisocial_sufficient = acpm.alpha_to_alpha_t2wpm(xSU, xSL, wSU, wSL, exponent)

    return alpha_dpa(antisocial_sufficient, antisocial_desirable)


def criminal_psychopathy(exponent):
    interpersonal = interpersonal_calc()
    affective = affective_calc(mandatory_exponent=-10, desired_exponent=1)
    lifestyle = lifestyle_calc(2)
    antisocial = antisocial_calc(2)

    interpersonal_weight = alpha_5[4]
    affective_weight = alpha_5[4]
    lifestyle_weight = alpha_5[3]
    antisocial_weight = alpha_5[4]

    xDU = [interpersonal[0], affective[0], lifestyle[0], antisocial[0]]
    xDL = [interpersonal[1], affective[1], lifestyle[1], antisocial[1]]

    wDU = [interpersonal_weight[0], affective_weight[0], lifestyle_weight[0], antisocial_weight[0]]
    wDL = [interpersonal_weight[1], affective_weight[1], lifestyle_weight[1], antisocial_weight[1]]

    return acpm.alpha_to_alpha_t2wpm(xDU, xDL, wDU, wDL, exponent)




def pb_lifestyle(exponent):
    xDU = [ lifestyle_data.values[2][0][0], lifestyle_data.values[3][0][0], lifestyle_data.values[4][0][0],
            lifestyle_data.values[5][0][0], lifestyle_data.values[6][0][0] ]
    xDL = [ lifestyle_data.values[2][0][1], lifestyle_data.values[3][0][1], lifestyle_data.values[4][0][1],
            lifestyle_data.values[5][0][1], lifestyle_data.values[6][0][1] ]

    wDU = [ alpha_5[2][0], alpha_5[1][0], alpha_5[3][0], alpha_5[0][0], alpha_5[1][0] ]
    wDL = [ alpha_5[2][1], alpha_5[1][1], alpha_5[3][1], alpha_5[0][1], alpha_5[1][1] ]

    lifestyle_desirable = acpm.alpha_to_alpha_t2wpm(xDU, xDL, wDU, wDL, exponent)

    xSU = [ lifestyle_data.values[0][0][0], lifestyle_data.values[1][0][0] ]
    xSL = [ lifestyle_data.values[0][0][1], lifestyle_data.values[1][0][1] ]

    wSU = [alpha_5[3][0], alpha_5[1][0] ]
    wSL = [alpha_5[3][1], alpha_5[1][1] ]

    lifestyle_sufficient = acpm.alpha_to_alpha_t2wpm(xSU, xSL, wSU, wSL, exponent)

    return alpha_dpa(lifestyle_sufficient, lifestyle_desirable)


def pb_interpersonal(exponent):
    interpersonal_desired = interpersonal_data.values[3][0]
    xSU = [ interpersonal_data.values[0][0][0], interpersonal_data.values[1][0][0] ]
    xSL = [ interpersonal_data.values[0][0][1], interpersonal_data.values[1][0][1] ]

    wSU = [ alpha_5[4][0], alpha_5[4][0] ]
    wSL = [ alpha_5[4][1], alpha_5[4][1] ]

    interpersonal_sufficient = acpm.alpha_to_alpha_t2wpm(xSU, xSL, wSU, wSL, exponent)

    interpersonal_d = alpha_dpa(interpersonal_sufficient, interpersonal_desired)

    return alpha_cpa(interpersonal_data.values[2][0], interpersonal_d)


def pb_affective(exponent):
    xMU = [ affective_data.values[0][0][0], affective_data.values[1][0][0] ]
    xML = [ affective_data.values[0][0][1], affective_data.values[1][0][1] ]

    wMU = [ alpha_5[4][0], alpha_5[4][0] ]
    wML = [ alpha_5[4][1], alpha_5[4][1] ]

    affective_mandatory = acpm.alpha_to_alpha_t2wpm(xMU, xML, wMU, wML, -10)

    xDU = [affective_data.values[2][0][0], affective_data.values[3][0][0]]
    xDL = [affective_data.values[2][0][1], affective_data.values[3][0][1]]

    wDU = [alpha_5[2][0], alpha_5[3][0]]
    wDL = [alpha_5[2][1], alpha_5[3][1]]

    affective_desired = acpm.alpha_to_alpha_t2wpm(xDU, xDL, wDU, wDL, exponent)

    return alpha_cpa(affective_mandatory, affective_desired)

def professionally_beneficial_psychopathy(exponent):
    lifestyle = pb_lifestyle(2)
    interpersonal = pb_interpersonal(1)
    affective = pb_affective(2)

    lifestyle_weight = alpha_5[2]
    interpersonal_weight = alpha_5[4]
    affective_weight = alpha_5[4]

    xDU = [ lifestyle[0], interpersonal[0], affective[0] ]
    xDL = [ lifestyle[1], interpersonal[1], affective[1] ]

    wDU = [ lifestyle_weight[0], interpersonal_weight[0], affective_weight[0] ]
    wDL = [ lifestyle_weight[1], interpersonal_weight[1], affective_weight[1] ]

    return acpm.alpha_to_alpha_t2wpm(xDU, xDL, wDU, wDL, exponent)


def nc_interpersonal(exponent):
    xDU = [ interpersonal_data.values[3][0][0], interpersonal_data.values[2][0][0] ]
    xDL = [ interpersonal_data.values[3][0][1], interpersonal_data.values[2][0][1] ]

    wDU = [ alpha_5[2][0], alpha_5[3][0] ]
    wDL = [ alpha_5[2][1], alpha_5[3][1] ]

    interpersonal_d = acpm.alpha_to_alpha_t2wpm(xDU, xDL, wDU, wDL, exponent)

    xSU = [interpersonal_data.values[0][0][0], interpersonal_data.values[1][0][0]]
    xSL = [interpersonal_data.values[0][0][1], interpersonal_data.values[1][0][1]]

    wSU = [alpha_5[4][0], alpha_5[4][0]]
    wSL = [alpha_5[4][1], alpha_5[4][1]]

    interpersonal_sufficient = acpm.alpha_to_alpha_t2wpm(xSU, xSL, wSU, wSL, exponent)

    return alpha_dpa(interpersonal_sufficient, interpersonal_d)

def nc_affective():
    xDU = [ affective_data.values[0][0][0], affective_data.values[1][0][0] ]
    xDL = [ affective_data.values[0][0][1], affective_data.values[1][0][1] ]

    wDU = [ alpha_5[3][0], alpha_5[4][0] ]
    wDL = [ alpha_5[3][1], alpha_5[4][1] ]

    affective_d = acpm.alpha_to_alpha_t2wpm(xDU, xDL, wDU, wDL, 2)

    affective_desirable = alpha_dpa(affective_data.values[2][0], affective_d)

    return alpha_cpa(affective_data.values[3][0], affective_desirable)

def nc_lifestyle():
    xDU = [ lifestyle_data.values[2][0][0], lifestyle_data.values[3][0][0], lifestyle_data.values[4][0][0] ]
    xDL = [ lifestyle_data.values[2][0][1], lifestyle_data.values[3][0][1], lifestyle_data.values[4][0][1] ]

    wDU = [ alpha_5[3][0], alpha_5[4][0], alpha_5[3][0] ]
    wDL = [ alpha_5[3][1], alpha_5[4][1], alpha_5[3][1] ]

    lifestyle_desirable = acpm.alpha_to_alpha_t2wpm(xDU, xDL, wDU, wDL, 1)

    xSU = [lifestyle_data.values[0][0][0], lifestyle_data.values[1][0][0], lifestyle_data.values[5][0][0], lifestyle_data.values[6][0][0] ]
    xSL = [lifestyle_data.values[0][0][1], lifestyle_data.values[1][0][1], lifestyle_data.values[5][0][1], lifestyle_data.values[6][0][1] ]

    wSU = [alpha_5[4][0], alpha_5[4][0], alpha_5[2][0], alpha_5[4][0] ]
    wSL = [alpha_5[4][1], alpha_5[4][1], alpha_5[2][1], alpha_5[4][1] ]

    lifestyle_sufficient = acpm.alpha_to_alpha_t2wpm(xSU, xSL, wSU, wSL, 1)

    return alpha_dpa(lifestyle_sufficient, lifestyle_desirable)

def noncriminal_psychopathy(exponent):
    interpersonal = nc_interpersonal(1)
    affective = nc_affective()
    lifestyle = nc_lifestyle()

    interpersonal_weight = alpha_5[3]
    affective_weight = alpha_5[2]
    lifestyle_weight = alpha_5[4]

    xDU = [ interpersonal[0], affective[0], lifestyle[0] ]
    xDL = [ interpersonal[1], affective[1], lifestyle[1] ]

    wDU = [ interpersonal_weight[0], affective_weight[0], lifestyle_weight[0] ]
    wDL = [ interpersonal_weight[1], affective_weight[1], lifestyle_weight[1] ]

    return acpm.alpha_to_alpha_t2wpm(xDU, xDL, wDU, wDL, exponent)


def load_score_data(file_path):
    score_sheet = pd.read_excel(file_path, sheet_name="Scores", usecols="A:C", nrows=21)
    scores = score_sheet['Scoring']
    vocab = score_sheet['Factors'].str.strip().str.lower()

    # Create DataFrame for Alpha Cuts
    out = pd.DataFrame(index=vocab, columns=['Alpha Cuts'])
    for i in range(20):
        for j in range(7):  # Assuming 7 vocabulary terms
            if scores[i] == words.at[i, j + 1]:
                out.at[vocab[i], 'Alpha Cuts'] = alpha_7[j]
    return out



def load_weight_data(file_path):
    score_sheet = pd.read_excel(file_path, sheet_name="Scores", usecols="A:C", nrows=21)
    vocab = score_sheet['Factors'].str.strip().str.lower()
    trait_weights = score_sheet['Weights']

    # Create DataFrame for Alpha Cuts
    out = pd.DataFrame(index=vocab, columns=['Alpha Cuts'])
    for i in range(20):
        for j in range(5):  # Assuming 5 weight vocabulary terms
            if trait_weights[i] == weight_words.at[0, j + 1]:
                out.at[vocab[i], 'Alpha Cuts'] = alpha_5[j]
                break
    return out


def get_filtered_data(file_path):
    score_data = load_score_data(file_path)
    weight_data = load_weight_data(file_path)

    # Normalize factor names for filtering
    score_data.index = score_data.index.str.strip().str.lower()
    weight_data.index = weight_data.index.str.strip().str.lower()

    # Define factor categories
    interpersonal_factors = [
        "glib", "grandiosity", "conning", "pathological lying"
    ]
    affective_factors = [
        "lack of remorse", "callousness", "shallow affect", "acceptance of responsibilities of actions"
    ]
    lifestyle_factors = [
        "need for stimulation", "realistic long term goals", "impulsivity", "irresponsibility",
        "parasitic lifestyle", "number of short term marital relationships", "sexual promiscuity"
    ]
    antisocial_factors = [
        "behavioral control", "early behavioral problems", "juvenile delinquency", "revocation of conditional release",
        "criminal versatility"
    ]

    # Filter data for each category
    data = {
        "interpersonal_data": score_data.loc[interpersonal_factors],
        "affective_data": score_data.loc[affective_factors],
        "lifestyle_data": score_data.loc[lifestyle_factors],
        "antisocial_data": score_data.loc[antisocial_factors],
        "interpersonal_weight_data": weight_data.loc[interpersonal_factors],
        "affective_weight_data": weight_data.loc[affective_factors],
        "lifestyle_weight_data": weight_data.loc[lifestyle_factors],
        "antisocial_weight_data": weight_data.loc[antisocial_factors]
    }
    return data



# Generate and Save Psychopathy Plots
def calculate_and_save_psychopathy(psych_score, title, output_dir):
    def upper_MF(x):
        return mu_sf(x, psych_score[0])

    def lower_MF(x):
        return mu_sf(x, psych_score[1])

    psych_centroid = t2_centroid(psych_score[0], psych_score[1], 300)
    fou_psych = fouset(upper_MF, lower_MF, 0, 10, 0.2, 0.012)
    dcs = defuzz(psych_centroid)

    centroid_left = psych_centroid[0]
    centroid_right = psych_centroid[1]

    x = np.arange(0, 10.01, 0.01)
    j_range = np.arange(0, fou_psych.shape[0])

    X1 = x
    X2 = x
    X3 = fou_psych[j_range, 0]

    Y1 = np.array([upper_MF(xi) for xi in x])
    Y2 = np.array([lower_MF(xi) for xi in x])
    Y3 = fou_psych[j_range, 1]

    scatter_x = [centroid_left, centroid_right, dcs]
    scatter_y = [1, 1, 1]

    plt.figure(figsize=(10, 6))
    plt.plot(X1, Y1, label="csU(x)", lw=3, zorder=2)
    plt.plot(X2, Y2, label="csL(x)", lw=3, zorder=1)
    plt.scatter(X3, Y3, label="fou", color='red', marker='+', zorder=2)
    plt.scatter(scatter_x, scatter_y, label="Centroids", color='blue', marker='D', s=40, zorder=3)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save the plot with higher resolution
    output_path = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(output_path, dpi=300)  # Set dpi to 300 for high resolution
    plt.close()

    return output_path



def generate_plots(output_dir, file_path):
    # Reload filtered data with the file path
    data = get_filtered_data(file_path)

    # Set global variables for data
    global interpersonal_data, affective_data, lifestyle_data, antisocial_data
    global interpersonal_weight_data, affective_weight_data, lifestyle_weight_data, antisocial_weight_data

    interpersonal_data = data["interpersonal_data"]
    affective_data = data["affective_data"]
    lifestyle_data = data["lifestyle_data"]
    antisocial_data = data["antisocial_data"]

    interpersonal_weight_data = data["interpersonal_weight_data"]
    affective_weight_data = data["affective_weight_data"]
    lifestyle_weight_data = data["lifestyle_weight_data"]
    antisocial_weight_data = data["antisocial_weight_data"]

    # Generate plots
    psych_score_criminal = criminal_psychopathy(1)
    plot1 = calculate_and_save_psychopathy(psych_score_criminal, "Criminal Psychopathy", output_dir)

    psych_score_professional = professionally_beneficial_psychopathy(1)
    plot2 = calculate_and_save_psychopathy(psych_score_professional, "Professionally Beneficial Psychopathy", output_dir)

    psych_score_noncriminal = noncriminal_psychopathy(2)
    plot3 = calculate_and_save_psychopathy(psych_score_noncriminal, "Non-Criminal Psychopathy", output_dir)

    return [plot1, plot2, plot3]


def calculate_psychopathy_centroids(excel_file_path):
    """
    Load the Excel file (using the same format as in the current code) and calculate
    the defuzzified centroids for Criminal Psychopathy, Professionally Beneficial Psychopathy,
    and Non-Criminal Psychopathy.

    Parameters:
        excel_file_path (str): Path to the Excel file (e.g., "excel/PCLRWords.xlsx")

    Returns:
        dict: A dictionary with keys "Criminal Psychopathy", "Professionally Beneficial Psychopathy",
              and "Non-Criminal Psychopathy" mapping to their corresponding defuzzified centroids.
    """
    # Load the filtered data and set the global variables needed by the psychopathy functions
    data = get_filtered_data(excel_file_path)
    global interpersonal_data, affective_data, lifestyle_data, antisocial_data
    global interpersonal_weight_data, affective_weight_data, lifestyle_weight_data, antisocial_weight_data

    interpersonal_data = data["interpersonal_data"]
    affective_data = data["affective_data"]
    lifestyle_data = data["lifestyle_data"]
    antisocial_data = data["antisocial_data"]

    interpersonal_weight_data = data["interpersonal_weight_data"]
    affective_weight_data = data["affective_weight_data"]
    lifestyle_weight_data = data["lifestyle_weight_data"]
    antisocial_weight_data = data["antisocial_weight_data"]

    # Compute the psychopathy scores for each type.
    # (Using exponent values consistent with the existing implementation.)
    psych_criminal = criminal_psychopathy(1)
    psych_prof = professionally_beneficial_psychopathy(1)
    psych_noncriminal = noncriminal_psychopathy(2)

    # For each psychopathy score, calculate the type-2 centroid and then defuzzify.
    centroid_criminal = t2_centroid(psych_criminal[0], psych_criminal[1], 300)
    dcs_criminal = defuzz(centroid_criminal)

    centroid_prof = t2_centroid(psych_prof[0], psych_prof[1], 300)
    dcs_prof = defuzz(centroid_prof)

    centroid_noncriminal = t2_centroid(psych_noncriminal[0], psych_noncriminal[1], 300)
    dcs_noncriminal = defuzz(centroid_noncriminal)

    # Return the defuzzified centroids in a dictionary.
    return {
        "Criminal Psychopathy": dcs_criminal,
        "Professionally Beneficial Psychopathy": dcs_prof,
        "Non-Criminal Psychopathy": dcs_noncriminal
    }
