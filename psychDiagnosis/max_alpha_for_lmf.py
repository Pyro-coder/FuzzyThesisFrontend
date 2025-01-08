import numpy as np

def alpha_max_lmf(ax, aw):
    """
    Determines the minimum number of alpha-cuts across nested vectors for x and w,
    and returns the highest alpha value from the smallest set.

    Parameters:
    - ax: Nested list or array containing alpha-cuts arrays for x.
    - aw: Nested list or array containing alpha-cuts arrays for w.

    Returns:
    - Tuple containing the index of the minimum number of rows and the highest alpha value.
    """
    # Initialize with infinity to find the minimum
    imax = float('inf')
    amax = 1  # Assuming alpha values are normalized between 0 and 1

    # Check each array in ax
    for i in range(len(ax)):
        current_length = len(ax[i])
        if current_length < imax:
            imax = current_length
            amax = ax[i][-1][2]  # Assuming the last element in each alpha-cut array is the alpha value

    # Check each array in aw
    for i in range(len(aw)):
        current_length = len(aw[i])
        if current_length < imax:
            imax = current_length
            amax = aw[i][-1][2]  # Assuming the last element in each alpha-cut array is the alpha value

    return (imax - 1, amax)  # Returns 0-based index and the highest alpha value

# Example usage
ax_example = [np.array([[0, 1, 0.1], [1, 2, 0.2], [2, 3, 0.3]]),
              np.array([[0, 1, 0.05], [1, 2, 0.15]])]  # Smaller set
aw_example = [np.array([[0, 1, 0.1], [1, 2, 0.2]])]

# Call the function
# imax, amax = alpha_max_lmf(ax_example, aw_example)
# print("Index of the smallest number of alpha-cuts:", imax)
# print("Highest alpha value from the smallest set:", amax)
