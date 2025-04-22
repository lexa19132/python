import numpy as np;
import eigenvalues as eig;
from typing import Callable;

def singular_values(input : np.ndarray, repeats: int = 100, algorithm : Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] = eig.QR_using_orthogonalization) -> tuple[float]:

    matrix: np.ndarray = input.copy()

    return np.sqrt(eig.QR_eigenvalues(matrix @ matrix.T, algorithm, repeats))

def SVD_decompostion(input : np.ndarray, repeats : int = 100,  algorithm : Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]  = eig.QR_using_orthogonalization) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    #U, S, V
    
    return 0