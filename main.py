import numpy as np;
import eigenvalues as eigv;
import singularvalues as singv

input_matrix = np.array([[1, 1/2, 1/3, 1/4, 1/5],
                        [1/2, 1/3, 1/4, 1/5, 1/6],
                        [1/3, 1/4, 1/5, 1/6, 1/7,],
                        [1/4, 1/5, 1/6, 1/7, 1/8],
                        [1/5, 1/6, 1/7, 1/8, 1/9]])

input_vector = np.array([1, 1 ,1, 1, 1])
input_iterations = 100

input_test_matrix = np.array([[4, 1],
                              [1, 3]])

# print(eig.max_eigenvalue_power_method(input_matrix, input_vector, input_iterations))
print(np.linalg.eigvals(input_matrix))

# print(eig.QR_using_orthogonalization(input_test_matrix))
print(eigv.QR_eigenvalues(input_matrix, eigv.QR_using_orthogonalization, 50))

print(singv.singular_values(input_matrix))

U, S, Vt = np.linalg.svd(input_matrix, full_matrices=False)
print(S)

