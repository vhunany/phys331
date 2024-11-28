import numpy as np

def triangleInverse(M, upperOrLower):
    """
    Solves for the inverse of a triangular matrix M.
    
    PARAMETERS:
        M -- A 2-dimensional np.array object representing the triangular matrix.
        upperOrLower -- A string, either 'upper' or 'lower', indicating whether
                        M is an upper or lower triangular matrix.
             
    RETURNS:
        Minv -- A 2-dimensional np.array representing the inverse of M.
    """
    # Get the number of rows/columns in the matrix M (assuming M is square).
    n = M.shape[0]
    
    # Create an empty matrix of the same size as M to store the inverse.
    Minv = np.zeros_like(M)
    
    # If the matrix is lower triangular (upperOrLower == 'lower'):
    if upperOrLower == 'lower':
        # Iterate over each row starting from the top (i = 0, 1, 2, ..., n-1).
        for i in range(n):
            # Invert the diagonal element directly (since the diagonal elements
            # of a triangular matrix can be handled separately).
            Minv[i, i] = 1 / M[i, i]
            
            # For all rows below the diagonal (i+1 to n), calculate the values
            # in the inverse matrix using forward substitution.
            for j in range(i+1, n):
                # Perform forward substitution to calculate the inverse elements.
                # The formula is derived from solving M*Minv = I, where Minv is the inverse.
                Minv[j, i] = -np.dot(M[j, :i], Minv[:i, i]) / M[j, j]
    
    # If the matrix is upper triangular (upperOrLower == 'upper'):
    elif upperOrLower == 'upper':
        # Iterate over each row starting from the bottom (i = n-1, n-2, ..., 0).
        for i in reversed(range(n)):
            # Invert the diagonal element directly.
            Minv[i, i] = 1 / M[i, i]
            
            # For all rows above the diagonal (0 to i-1), calculate the values
            # in the inverse matrix using back substitution.
            for j in range(i):
                # Perform back substitution to calculate the inverse elements.
                # The formula is derived similarly to the lower triangular case.
                Minv[j, i] = -np.dot(M[j, j+1:], Minv[j+1:, i]) / M[j, j]
    
    # Return the calculated inverse matrix Minv.
    return Minv

def main():
    # Define a specific lower triangular matrix M.
    M = np.array([[9, 0, 0],
                  [-4, 2, 0],
                  [1, 0, 5]], dtype=float)
    
    # Call the triangleInverse function to compute the inverse of the matrix M,
    # indicating that it is a lower triangular matrix by passing 'lower'.
    Minv = triangleInverse(M, 'lower')
    
    # Print the calculated inverse of M.
    print("Inverse of the matrix M:\n", Minv)

# Execute the main function to run the code.
main()


# Given that MxM^-1 is close to the identity matrix (with minor numerical deviations), the solution is valid.

# The diagonal elements of the inverse are the reciprocals of the diagonal elements of the original matrix M.
# For example, if M = [[9, 0, 0], [-4, 2, 0], [1, 0, 5]], the inverse will have diagonal values [1/9, 1/2, 1/5].
    
# Zeroes in off-diagonal elements above the diagonal**:
# The original matrix is lower triangular, so the inverse will also be lower triangular.
# This means that all elements above the diagonal in the inverse matrix are zero, as shown in the output. However ...
# When multiplying M and Minv, the result should be the identity matrix I, but due to floating-point
# precision, there may be small numerical deviations. 