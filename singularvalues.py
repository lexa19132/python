import numpy as np;
import eigenvalues as eig;

def singular_values(input : np.ndarray, repeats: int = 100) -> list[float]:

    matrix: np.ndarray = input.copy()

    return np.sqrt(eig.QR_eigenvalues(matrix @ matrix.T, eig.QR_using_orthogonalization, repeats))
