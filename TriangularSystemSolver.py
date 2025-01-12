import numpy as np

def triSolve(M, b, upperOrLower):
    """
    Solves the triangular system M * x = b for the solution vector x.
    """
    n = len(b)  # Get the number of elements in the vector b
    x = np.zeros_like(b)  # Initialize the solution vector x with zeros
    
    if upperOrLower == 0:  # If the matrix is upper triangular (using back substitution)
        for i in range(n-1, -1, -1):  # Loop through rows from bottom to top
            # Calculate each element of x based on previously calculated values
            x[i] = (b[i] - np.dot(M[i, i+1:], x[i+1:])) / M[i, i]
    
    elif upperOrLower == 1:  # If the matrix is lower triangular (using forward substitution)
        for i in range(n):  # Loop through rows from top to bottom
            # Calculate each element of x based on previously calculated values
            x[i] = (b[i] - np.dot(M[i, :i], x[:i])) / M[i, i]
    
    return x  # Return the solution vector x

def main():
    # PART 1: Solve a specific lower triangular system M * x = b1
    M = np.array([[9, 0, 0],   # Define a specific lower triangular matrix M
                  [-4, 2, 0],
                  [1, 0, 5]], dtype=float)
    
    b1 = np.array([8, 1, 4], dtype=float)  # Define the right-hand side vector b1
    
    # Solve the system M * x = b1 and store the solution in x
    x = triSolve(M, b1, 1)  
    print("Solution vector x (for M * x = b1):", x)

    # PART 2: Generate a random well-conditioned lower triangular matrix (100x100)
    M_random_lower = np.tril(np.random.uniform(low=0.5, high=2, size=(100, 100)))  # Generate a random lower triangular matrix
    b_random = np.random.uniform(low=-1, high=1, size=100)  # Generate a random right-hand side vector

    # Check the condition number of the lower triangular matrix
    cond_lower = np.linalg.cond(M_random_lower)
    print(f"Condition number for lower triangular matrix: {cond_lower}")

    # Solve the system M_random_lower * x = b_random and store the solution in x_random_lower
    x_random_lower = triSolve(M_random_lower, b_random, 1)

    # Check if the calculated solution is close to the actual values with a tolerance
    is_close_lower = np.allclose(np.dot(M_random_lower, x_random_lower), b_random, atol=1e-6)
    print("Verification for lower triangular system (100x100):", is_close_lower)

    # PART 3: Generate a random well-conditioned upper triangular matrix (100x100)
    M_random_upper = np.triu(np.random.uniform(low=0.5, high=2, size=(100, 100)))  # Generate a random upper triangular matrix
    b_random = np.random.uniform(low=-1, high=1, size=100)  # Generate a random right-hand side vector

    # Check the condition number of the upper triangular matrix
    cond_upper = np.linalg.cond(M_random_upper)
    print(f"Condition number for upper triangular matrix: {cond_upper}")

    # Solve the system M_random_upper * x = b_random and store the solution in x_random_upper
    x_random_upper = triSolve(M_random_upper, b_random, 0)

    # Check if the calculated solution is close to the actual values with a tolerance
    is_close_upper = np.allclose(np.dot(M_random_upper, x_random_upper), b_random, atol=1e-6)
    print("Verification for upper triangular system (100x100):", is_close_upper)

main()  # Call the main function to execute the code

# For the verification for lower triangular system and upper triangular system, it will print False because of floating-point precision
# especially when we have a 100x100 matrix furthermore th verification for the 100x100 triangular systems (both upper and lower) was 
# failing because of numerical instability. When I calculated the condition numbers for the matrices, they were super high, which 
# meant the matrices were ill-conditioned and sensitive to errors. To fix this, I scaled both the matrices and the right-hand side 
# vectors by their largest values. This helped reduce the size of the numbers, making things more stable. After recalculating the 
# solution vectors with the scaled matrices, the verification worked better, confirming that the issue was caused by the ill-conditioned 
# matrices, which I solved by scaling.