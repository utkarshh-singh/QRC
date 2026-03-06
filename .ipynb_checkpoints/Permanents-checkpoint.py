import numpy as np
from itertools import combinations
from multiprocessing import Pool, cpu_count
from utility import calculate_factor, generate_column_combinations, squared_permanent 
from matrices import SubmatrixGenerator
from tqdm import tqdm

class RyserPermanent:
    """
    A class to compute the permanent of a square matrix using Ryser's formula.
    Includes serial, parallel, and batched parallel implementations.
    """

    def __init__(self, parallel=False, batch_size=1000):
        self.parallel = parallel
        self.batch_size = batch_size

    @staticmethod
    def _ryser_subset_worker(args):
        """
        Worker function to compute the contribution of a subset to the permanent.
        :param args: Tuple containing the matrix, subset, and sign.
        :return: Contribution of the subset.
        """
        matrix, subset, sign = args
        row_sums = np.sum(matrix[:, subset], axis=1)
        product_of_sums = np.prod(row_sums)
        return sign * product_of_sums

    @staticmethod
    def _compute_batch(args):
        """
        Static method to compute the contribution of a batch of subsets to the permanent.
        :param args: Tuple containing the matrix and the batch (list of (subset, sign) tuples).
        :return: The contribution of the batch to the permanent.
        """
        matrix, batch = args
        result = 0
        for subset, sign in batch:
            row_sums = np.sum(matrix[:, subset], axis=1)
            result += sign * np.prod(row_sums)
        return result

    @staticmethod
    def _batch_generator(iterable, batch_size):
        """
        Yield successive batches from an iterable.
        :param iterable: The input iterable.
        :param batch_size: The size of each batch.
        """
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _compute_batched_parallel(self, matrix):
        """
        Parallelized calculation of the permanent using batched Ryser's formula.
        :param matrix: A square matrix (unitary or otherwise).
        :return: The permanent of the matrix.
        """
        n = matrix.shape[0]
        tasks = []
        for subset_size in range(1, n + 1):
            subsets = list(combinations(range(n), subset_size))
            sign = (-1) ** (n - subset_size)
            for batch in self._batch_generator([(s, sign) for s in subsets], self.batch_size):
                tasks.append((matrix, batch))

        with Pool(cpu_count()) as pool:
            results = pool.map(self._compute_batch, tasks)

        return sum(results)

    def compute(self, matrix):
        """
        Calculate the permanent of a square matrix using Ryser's formula.
        :param matrix: A square matrix (unitary or otherwise).
        :return: The permanent of the matrix.
        """
        n = matrix.shape[0]
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("The matrix must be square to calculate the permanent.")

        if self.parallel and n > 24:
            return self._compute_batched_parallel(matrix)

        if self.parallel:
            # Prepare arguments for parallel processing
            tasks = []
            for subset_size in range(1, n + 1):
                subsets = combinations(range(n), subset_size)
                sign = (-1) ** (n - subset_size)
                for subset in subsets:
                    tasks.append((matrix, subset, sign))

            with Pool(cpu_count()) as pool:
                results = pool.map(self._ryser_subset_worker, tasks)
            return sum(results)

        else:
            # Serial computation of the permanent
            permanent = 0
            for subset_size in range(1, n + 1):
                subsets = combinations(range(n), subset_size)
                sign = (-1) ** (n - subset_size)
                for subset in subsets:
                    row_sums = np.sum(matrix[:, subset], axis=1)
                    permanent += sign * np.prod(row_sums)
            return permanent

class ClassicalCoincidence:
    """
    A class to calculate the classical correlation function by summing up normalized permanents.
    """

    def __init__(self, unitary_matrix, parallel=False, info = False, perm_list=True):
        self.unitary_matrix = unitary_matrix
        self.parallel = parallel
        self.info = info
        self.perm_list = perm_list
        
    @staticmethod
    def compute_permanent(submatrix):
        """
        Static method to compute the permanent of a submatrix.
        :param submatrix: A submatrix of the unitary matrix.
        :return: The permanent of the submatrix.
        """
        return RyserPermanent(parallel=False).compute(submatrix)

    def calculate(self):
        """
        Calculate the classical correlation function.
        :return: The classical correlation value.
        """
        n = self.unitary_matrix.shape[0]

        # Generate column combinations
        generator = SubmatrixGenerator(parallel=self.parallel)
        column_combinations = generate_column_combinations(n)

        # Info printing
        if self.info:
            print(f"Matrix size: {n}x{n}")
            print(f"Number of submatrices to compute permanents for: {len(column_combinations)}")

        # Calculate normalization coefficients
        normalization_factors = generator.calculate_normalization_coefficients(column_combinations)

        # Generate submatrices
        submatrices = generator.generate_submatrices(self.unitary_matrix, column_combinations)

        # Calculate permanents, mod-square, normalize, and sum
        total_correlation = 0

        if self.parallel:
            with Pool(cpu_count()) as pool:
                permanents = list(tqdm(pool.imap(self.compute_permanent, submatrices), total=len(submatrices), desc="Calculating permanents"))
        else:
            permanents = [
                self.compute_permanent(submatrix)
                for submatrix in submatrices]# tqdm(submatrices, desc="Calculating permanents", total=len(submatrices))]
        
        permanent_values = []  

        for permanent, norm_factor in zip(permanents, normalization_factors):
            modulus_square = np.abs(permanent) / norm_factor
            modulus_square = modulus_square ** 2
            permanent_values.append(modulus_square) 
            total_correlation += modulus_square 
            
        permanent_values = np.array(permanent_values)
        
        if self.perm_list:
            return permanent_values 
        else:
            return total_correlation