import numpy as np
import scipy as sp
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace

class QuantumCircuitInfo:
    """
    A class to obtain various metrics and information about a quantum circuit.
    """
    def __init__(self, circuit):
        """
        Initialize with a QuantumCircuit object.

        Parameters:
        circuit (QuantumCircuit): The quantum circuit to analyze.
        """
        self.circuit = circuit.decompose().decompose()

    def get_depth(self):
        """Returns the depth of the circuit."""
        return self.circuit.depth()

    def get_num_qubits(self):
        """Returns the number of qubits in the circuit."""
        return self.circuit.num_qubits

    def get_num_parameters(self):
        """Returns the number of parameters in the circuit."""
        return self.circuit.num_parameters

    def get_parameters(self):
        """Returns the parameters of the circuit."""
        params = self.circuit.parameters
        return [params[i].name for i in range(len(params))]

    def get_gates(self):
        """Returns the number of each type of gate in the circuit."""
        return self.circuit.count_ops()

    def get_num_cnots(self):
        """Returns the number of CNOT gates in the circuit."""
        return self.circuit.count_ops().get('cx', 0)

    def get_size(self):
        """Returns the size of the circuit."""
        return self.circuit.size()

    def meyer_wallach(self, sample=1024):
        """
        Returns the Meyer-Wallach entanglement measure for the circuit.

        Parameters:
        sample (int): Number of samples to use for the calculation.
        """
        size = self.circuit.num_parameters
        res = np.zeros(sample, dtype=complex)
        N = self.circuit.num_qubits

        for i in range(sample):
            params = np.random.uniform(-np.pi, np.pi, size)
            ansatz = self.circuit.assign_parameters(params)
            ansatz.remove_final_measurements()
            U = Statevector(ansatz)
            entropy = 0
            qb = list(range(N))

            for j in range(N):
                dens = partial_trace(U, qb[:j] + qb[j + 1:]).data
                trace = np.trace(dens ** 2)
                entropy += trace

            entropy /= N
            res[i] = 1 - entropy
        return 2 * np.sum(res).real / sample

    def pqc_integral(self, samples=2048):
        """
        Returns the expressibility measure for the circuit.

        Parameters:
        samples (int): Number of samples to use for the calculation.
        """
        N = self.circuit.num_qubits
        size = self.circuit.num_parameters
        randunit_density = DensityMatrix(np.zeros((2**N, 2**N), dtype=complex))

        for _ in tqdm(range(samples), desc="Calculating PQC Integral"):
            params = np.random.uniform(-np.pi, np.pi, size)
            ansatz = self.circuit.assign_parameters(params)
            ansatz.remove_final_measurements()
            U = Statevector(ansatz)
            randunit_density += DensityMatrix(U, dims=None)
        return np.array(randunit_density / samples)

    def random_unitary(self, N):
        """
        Return a Haar distributed random unitary from U(N).

        Parameters:
        N (int): The dimension of the unitary matrix.
        """
        Z = np.random.randn(N, N) + 1.0j * np.random.randn(N, N)
        Q, R = sp.linalg.qr(Z)
        D = np.diag(np.diagonal(R) / np.abs(np.diagonal(R)))
        return np.dot(Q, D)

    def haar_integral(self, num_qubits, samples):
        """
        Return calculation of Haar Integral for a specified number of samples.

        Parameters:
        num_qubits (int): Number of qubits in the circuit.
        samples (int): Number of samples to use for the calculation.
        """
        N = 2**num_qubits
        randunit_density = np.zeros((N, N), dtype=complex)

        zero_state = np.zeros(N, dtype=complex)
        zero_state[0] = 1

        for _ in tqdm(range(samples), desc="Calculating Haar Integral"):
            A = np.matmul(zero_state, self.random_unitary(N)).reshape(-1, 1)
            randunit_density += np.kron(A, A.conj().T)

        randunit_density /= samples

        return randunit_density

    def expressibility(self, samples=2048):
        """
        Returns the entangling capacity of the circuit.

        Parameters:
        samples (int): Number of samples to use for the calculation.
        """
        num_qubits = self.circuit.num_qubits
        haar_result = self.haar_integral(num_qubits, samples)
        pqc_result = self.pqc_integral(samples)
        return np.linalg.norm(haar_result - pqc_result)

    def get_all_info(self, EXP = False):
        """Returns all information about the quantum circuit."""
        num_qubits = self.circuit.num_qubits
        if num_qubits <= 5 or EXP:
            expressibility = self.expressibility()
        else:
            expressibility = "NA"
            print("Calculating Expressibility will take time for this circuit. If you still want to calculate it, please set EXP = True in the get_all_info() call.")
        return {
            'Circuit Depth': self.get_depth(),
            'Number of qubits': self.get_num_qubits(),
            'Number of parameters': self.get_num_parameters(),
            'Parameters in the circuit': self.get_parameters(),
            'Gates in the circuit': dict(self.get_gates()),
            'Number of CNOTs in the circuit': self.get_num_cnots(),
            'Circuit size': self.get_size(),
            'Entangling_capacity (meyer_wallach)': self.meyer_wallach(),
#             'expressibility': self.expressibility(),
            'Expressibility': expressibility
        }
