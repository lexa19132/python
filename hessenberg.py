import numpy as np;

input = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

def hessenberg_form(matrix : np.ndarray) -> np.ndarray:

    def reflection(column): 
        delta = np.linalg.norm(column) * np.sign(column[0])

        vector = np.zeros(np.size(column))
        vector[0] = 1

        result = column + delta * vector
        result = result / np.linalg.norm(result)

        H = np.eye(np.size(column)) - 2 * np.outer(result.reshape(-1, 1), result)
        print(H)
        

    
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Incorrect argument, should be matrix.")
    
    n, m = matrix.shape

    if n != m:
        raise ValueError("Incorrect matrix dimensions, matrix should be square.")
    
    matrix_copy = matrix.copy()

    for i in range (n-2):
        column = matrix_copy[i + 1:, i]
        reflection(column)

hessenberg_form(input)
