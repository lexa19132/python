import numpy as np;
from typing import Callable;

def max_eigenvalue_power_method(input : np.ndarray, vector : np.ndarray, iterations : int) -> float:
    if (vector.size != input.shape[0]):
        raise ValueError("Vector length must match matrix dimensions")
    
    matrix : np.ndarray = input.copy()
    q : np.ndarray = vector.copy().reshape(-1, 1) / np.linalg.norm(vector)

    for i in range(0, iterations):
        z = matrix @ q
        q = (z / np.linalg.norm(z)).reshape(-1, 1)
        my_lambda = (matrix @ q).reshape(1, -1) @ q
    return my_lambda

def QR_using_orthogonalization (input : np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    matrix : np.ndarray = input.copy()

    iterations : int = matrix.shape[1]
    
    Q : np.ndarray = np.zeros(matrix.shape)
    R : np.ndarray = np.zeros((matrix.shape[1], matrix.shape[1]))

    for i in range(0, iterations):
        vector = matrix[:, i]

        sum = np.zeros(vector.size)
        for j in range(0, i):
            orthogonal_vector = Q[:, j]
            R[j, i] = orthogonal_vector @ vector.reshape(-1 , 1)
            projection = orthogonal_vector * R[j, i]
            sum += projection
            
        result_vector = vector - sum
        R[i, i] = np.linalg.norm(result_vector)
        result_vector = result_vector / R[i, i]
        Q[:, i] = result_vector.flatten()

    return (Q, R)

def QR_eigenvalues(input : np.ndarray, algorithm : Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], repeats : int) -> tuple[float]:
    A : np.ndarray = input.copy()
    for i in range(0, repeats):
        QR = algorithm(A)
        Q : np.ndarray = QR[0]
        R : np.ndarray = QR[1]
        A = R @ Q 
    return np.diagonal(A)