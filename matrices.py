import numpy as np
import math
from collections import Counter
from scipy.stats import unitary_group
from itertools import combinations_with_replacement
from multiprocessing import Pool, cpu_count
from utility import calculate_factor, generate_column_combinations, squared_permanent 

class SubmatrixGenerator:
    """
    A class to generate submatrices from a unitary matrix of the form U(1,2,3,...,n|w1,w2,w3,...,wn),
    where the w indices are all possible combinations of column indices in increasing order.
    """

    def __init__(self, parallel=False):
        self.parallel = parallel
        
    @staticmethod
    def create_submatrix(matrix, combination):
        """
        Create a submatrix based on a column combination.
        :param matrix: The original matrix.
        :param combination: A tuple representing the column indices (1-based).
        :return: The generated submatrix.
        """
        return matrix[:, [col - 1 for col in combination]]  # Convert 1-based to 0-based indexing

    def generate_submatrices(self, matrix, column_combinations):
        """
        Generate submatrices based on the provided column combinations using optional parallel processing.
        :param matrix: The original unitary matrix.
        :param column_combinations: List of column index combinations.
        :return: List of submatrices corresponding to each column combination.
        """
        if self.parallel:
            with Pool(cpu_count()) as pool:
                submatrices = pool.starmap(self.create_submatrix, [(matrix, comb) for comb in column_combinations])
        else:
            submatrices = [self.create_submatrix(matrix, comb) for comb in column_combinations]

        return submatrices
        
    def calculate_normalization_coefficients(self, column_combinations):
        """
        Calculate the normalization coefficients based on the multiplicity of indices in each combination
        using optional parallel processing.
        :param column_combinations: List of column index combinations.
        :return: List of normalization coefficients for each combination.
        """

        # def calculate_factor(combination):
        #     multiplicity = Counter(combination)
        #     return np.prod([math.factorial(count) ** 2 for count in multiplicity.values()])
        
        if self.parallel:
            with Pool(cpu_count()) as pool:
                normalization_factors = pool.map(calculate_factor, column_combinations)
        else:
            normalization_factors = [calculate_factor(comb) for comb in column_combinations]

        return normalization_factors


class UnitaryMatrix:
    """
    A class for generating random unitary matrices and testing their properties.
    """

    @staticmethod
    def generate_random_unitary(size):
        """
        Generate a random unitary matrix using the Haar measure.
        :param size: Size of the unitary matrix (n x n).
        :return: A random unitary matrix of the specified size.
        """
        z = np.random.randn(size, size) + 1j * np.random.randn(size, size)
        q, r = np.linalg.qr(z)
        d = np.diagonal(r)
        ph = d / np.abs(d)
        return q * ph

    @staticmethod
    def generate_parametrized_unitary(theta, phi, lam, size=2):
        """
        Generate a unitary matrix based on the general form parameterized by theta, phi, and lambda.
        :param theta: Angle theta for the unitary matrix.
        :param phi: Angle phi for the unitary matrix.
        :param lam: Angle lambda for the unitary matrix.
        :param size: Size of the unitary matrix. Default is 2 (1-qubit).
        :return: Parametrized unitary matrix.
        """
        if size == 2:
            return np.array([
                [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
                [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)],
            ])
        else:
            return unitary_group.rvs(size)

    @staticmethod
    def dagger(matrix):
        """
        Compute the Hermitian conjugate (dagger) of a matrix.
        :param matrix: The input matrix.
        :return: The Hermitian conjugate of the matrix.
        """
        return np.conj(matrix.T)

    @staticmethod
    def is_unitary(matrix, tol=1e-10):
        """
        Tests if a matrix is unitary by checking U * U^dagger = I.
        :param matrix: The matrix to test.
        :param tol: Tolerance for numerical checks.
        :return: True if the matrix is unitary, False otherwise.
        """
        identity = np.eye(matrix.shape[0], dtype=matrix.dtype)
        return np.allclose(identity, matrix @ np.conj(matrix.T), atol=tol)

class UnitaryMultiplier:
    """
    A class to generate two unitaries based on theta, phi parameters, and calculate their matrix multiplication.
    Includes an option for parallel computation of the matrix multiplication.
    """

    def __init__(self, parallel=False):
        self.parallel = parallel

    def compute_multiplication(self, u1, u2):
        """
        Compute the matrix multiplication of two unitaries.
        :param u1: The first unitary matrix.
        :param u2: The second unitary matrix.
        :return: The resulting matrix multiplication of the unitary and the new unitary's dagger.
        """
        u2_dagger = UnitaryMatrix.dagger(u2)

        if self.parallel:
            return self._parallel_matrix_multiplication(u1, u2_dagger)
        else:
            return np.dot(u1, u2_dagger)

    @staticmethod
    def _parallel_matrix_multiplication(u1, u2):
        """
        Perform parallel matrix multiplication for matrices.
        :param u1: First matrix.
        :param u2: Second matrix.
        :return: The resulting matrix.
        """
        def compute_element(args):
            i, j = args
            return np.sum(u1[i, :] * u2[:, j])

        # Create tasks for each element of the resulting matrix
        n = u1.shape[0]
        tasks = [(i, j) for i in range(n) for j in range(n)]

        with Pool(cpu_count()) as pool:
            results = pool.map(compute_element, tasks)

        # Assemble the resulting matrix
        result = np.array(results).reshape(n, n)
        return result
    
class OpticalNetwork:
    """
    A class to construct an optical network of beam splitters and compute the final unitary matrix.
    """

    def __init__(self, connections, parameters):
        """
        Initialize the optical network with beam splitter connections and parameters.
        
        :param connections: List of beam splitter connections (e.g., [[4,5],[3,4], ...])
        :param parameters: List of corresponding parameters [[alpha1, phi1], [alpha2, phi2], ...].
                           If phi is not provided, it defaults to 0.
        """
        if len(connections) != len(parameters):
            raise ValueError("Connections and parameters lists must have the same length.")

        # Normalize connections to ensure [a, b] == [b, a]
        self.connections = connections #[sorted(pair) for pair in connections]

        # Ensure phi defaults to 0 if not provided
        self.parameters = [(param[0], param[1] if len(param) > 1 else 0) for param in parameters]

        # Determine number of modes (network size)
        self.n = max(max(pair) for pair in self.connections)

    def list_to_matrix(self, indices, alpha, phi=0):
        """
        Generates an n x n beam splitter matrix based on indices and parameters.
        
        :param indices: List [a, b] (1-based indexing).
        :param alpha: Beam splitter angle (radians).
        :param phi: Phase shift (radians), defaults to 0.
        :return: The corresponding n x n unitary matrix.
        """
        a, b = sorted(indices)[0] - 1, sorted(indices)[1] - 1  # Convert to 0-based indexing
        matrix = np.eye(self.n, dtype=complex)  # Identity matrix
        
        
        # Beam splitter elements
        matrix[a, a] = np.exp(1j * phi) * np.cos(alpha)
        matrix[a, b] = np.exp(1j * phi) * np.sin(alpha)
        matrix[b, a] = -np.sin(alpha)
        matrix[b, b] = np.cos(alpha)
        
        if indices[0]< indices[1]:
            matrix = matrix.conj().T
        return matrix

    def multiply_sparse_beamsplitter(self, D, indices, unitary):
        """
        Efficiently multiplies the given matrix D with a beam splitter transformation.

        :param D: Regular n x n matrix.
        :param indices: List [a, b] (1-based indexing).
        :param unitary: Unitary matrix from which beam splitter values are extracted.
        :return: The transformed matrix after applying the beam splitter.
        """
        a, b = indices[0] - 1, indices[1] - 1  # Convert to 0-based indexing
        M = np.copy(D) 

        U_aa = unitary[a, a]  
        U_ab = unitary[a, b] 
        U_ba = unitary[b, a]  
        U_bb = unitary[b, b] 

        for j in range(D.shape[1]):
            M[a, j] = U_aa * D[a, j] + U_ab * D[b, j]
            M[b, j] = U_ba * D[a, j] + U_bb * D[b, j]
        return M

    def compute_final_unitary(self):
        """
        Computes the final unitary transformation matrix of the optical network.
        
        :return: The final n x n unitary matrix.
        """
        # Start with the identity matrix
        final_matrix = np.eye(self.n, dtype=complex)

        # Reverse the connections list for right-to-left multiplication
        reversed_connections = self.connections[::-1]
        reversed_parameters = self.parameters[::-1]

        # Perform matrix multiplications
        for (indices, params) in zip(reversed_connections, reversed_parameters):
            alpha, phi = params  # Extract alpha and phi
            # print([alpha, phi])
            unitary = self.list_to_matrix(indices, alpha, phi)
            # print(unitary)
            final_matrix = self.multiply_sparse_beamsplitter(final_matrix, indices, unitary)

        return final_matrix

def frobenius_norm(A,B):
    return np.linalg.norm(A-B, 'fro')

def mse(A,B):
    return np.mean((A-B)**2)