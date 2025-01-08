def select_f(z, x, i):
    # Evaluate the vector function z at x
    vector = z(x)
    # Return the ith component of the vector
    # Note: Python uses 0-based indexing, so you might need to adjust i if coming from 1-based systems
    return vector[i]
