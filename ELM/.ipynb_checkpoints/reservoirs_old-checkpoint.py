from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import qiskit_ibm_runtime as qir
from qiskit_ibm_runtime import SamplerV2 as Sampler2, QiskitRuntimeService
from qiskit.primitives import StatevectorSampler as Sampler
# from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator
import qiskit_aer.noise as noise
import numpy as np
from tqdm import tqdm
# import strawberryfields as sf
# from strawberryfields.ops import *
from concurrent.futures import ProcessPoolExecutor
from circuits import CPCircuit
from matrices import OpticalNetwork
from Permanents import RyserPermanent, ClassicalCoincidence
from utility import MetaFibonacci, mapping, CPaction
from itertools import product


def filter_top_states(prob_counts, k):
    sorted_states = sorted(prob_counts.items(), key=lambda x: x[1], reverse=True)
    filtered_counts = {state: prob for state, prob in sorted_states[:k]}
    return filtered_counts

def initialize_full_key_dict(size):
    """
    Creates a dictionary with all possible binary strings of a given size as keys.
    Initializes all values to 0.
    """
    all_keys = [''.join(p) for p in product('01', repeat=size)]
    return {key: 0 for key in all_keys}

def refined_counts(result_counts, size):
    """
    Updates the full dictionary with actual result counts.
    If a key exists in result_counts, update it. Otherwise, keep it as 0.
    """
    full_dict = initialize_full_key_dict(size)
    for key, value in result_counts.items():
        if key in full_dict:
            full_dict[key] = value
    return full_dict

def process_counts(counts, n_qubits, key_type="hexa"):
    binary_counts = {}
    for k, v in counts.items():
        if key_type == "hexa":
            binary_key = format(int(k, 16), f'0{n_qubits}b')
        elif key_type == "decimal":
            binary_key = format(int(k), f'0{n_qubits}b')
        elif key_type == "binary":
            binary_key = format(int(k, 2), f'0{n_qubits}b')
        else:
            raise ValueError("key_type must be one of: 'hexa', 'decimal', 'binary'")
        binary_counts[binary_key] = v

    all_bitstrings = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
    full_counts = {b: binary_counts.get(b, 0) for b in all_bitstrings}
    return full_counts

class CPRC:
    def __init__(self, dim, circuit=None, reps=1, execution_mode='simulation', K_params= None, CP_params = None, backend=None, shots=1024, optimization_level=3, kernel=False, noise_level =None, meas_limit=None, ETE= False):
        """
        Initialize the QuantumKernel with dimension, repetitions, and execution parameters.

        :param dim: Dimension of the feature map.
        :param reps: Number of repetitions for the feature map.
        :param execution_mode: Mode of execution ('simulation', 'real_device', 'fake_simulation').
        :param backend: Backend for execution (required if execution_mode is 'real_device' or 'fake_simulation').
        :param shots: Number of shots for the quantum execution.
        :param optimization_level: Optimization level for the transpiler.
        """
        self.dim = dim
        self.circuit = circuit
        self.reps = reps
        self.execution_mode = execution_mode
        self.backend = backend
        self.shots = shots
        self.optimization_level = optimization_level
#         self.shot_percentage = shot_percentage
        self.kernel = kernel
        self.noise_level = noise_level
        self.K_params= K_params if K_params is not None else np.random.uniform(0.0, 1.0, dim)
        self.CP_params = CP_params
        self.meas_limit = meas_limit
        self.ETE = ETE

    def service_(self):
        service = self.backend.service()
        return service, self.backend
        
    def CPMap(self):
        """
        Generate the Feature Map using the given dimensions and repetitions.

        :return: QuantumCircuit object representing the feature Map.
        """
        if self.meas_limit is None:
            qc = CPCircuit(num_features=self.dim, reps=self.reps, CP_params=self.CP_params, ETE = self.ETE).CPMap()
        else:
            n_qbits = CPCircuit(num_features=self.dim, reps=self.reps)._num_qubits()
            meas_q = int(n_qbits * self.meas_limit)
            qc= QuantumCircuit(n_qbits, meas_q)
            cpcircuit= CPCircuit(num_features=self.dim, reps=self.reps, CP_params=self.CP_params, ETE = self.ETE).CPMap()
            qc.append(cpcircuit, range(meas_q))
        return qc

    # def _simulate(self, qc):
    #     """
    #     Run the simulation on the quantum circuit.

    #     :param qc: QuantumCircuit to be simulated.
    #     :return: Fidelity value from the simulation result.
    #     """
    #     num_qubits = qc.num_qubits
    #     sampler = Sampler()
    #     res = sampler.run([qc]).result()
    #     counts_ = res[0].data.meas.get_counts()
    #     counts = refined_counts(counts_, num_qubits) #np.array(list(counts.values()))/self.shots
    #     fid_ = counts.values()
    #     fid = np.array(list(fid_))/self.shots
    #     return fid #fid #.get(0, 0.0)

    def _simulate(self, qc):
        """
        Run the simulation on the quantum circuit.

        :param qc: QuantumCircuit to be simulated.
        :return: Fidelity value from the simulation result.
        """
        # print(qc)
        sampler = Sampler()
        res = sampler.run(qc).result()
        # counts = res.quasi_dists[0]
        # fid = counts.values()
#         if self.shot_percentage is None:
#             limit = 2**(qc.num_qubits)
#         else:
#             limit = self.shot_percentage
#         fid = filter_top_states(counts, limit).values()
        fid = res.quasi_dists[0].values()
        # print(fid)
        return np.array(list(fid))#.get(0, 0.0)

    def _simulateDM(self, qc):
        from qiskit.quantum_info import DensityMatrix
        res = DensityMatrix(qc)
        return res
    
    def _simulateSTT(self, qc):
        from qiskit.quantum_info import Statevector
        res = Statevector(qc)
        return res

    def _simulate_with_noise(self, qc):
        num_qubits = qc.num_qubits
        backend = AerSimulator()
        if self.noise_level is None:
            print("Noise level is not set. Returning the exact simulation")
            results = self._simulate(qc)
        else:
            alpha, beta = self.noise_level, self.noise_level * 10
            noise_model = self.get_depolarizing_noise_model(alpha, beta)
            qc_transpiled = transpile(qc, backend)
            job = backend.run(qc_transpiled, shots=self.shots, noise_model=noise_model)
            result = job.result()
            counts_ = result.get_counts()
            counts = refined_counts(counts_, num_qubits)
            fidelity = counts.values() #get('0' * num_qubits, 0) 
            results = np.array(list(fidelity))/self.shots
        return results

    def _run_on_real_device(self, qc, fake_simulation):
        """
        Run the quantum circuit on a real quantum device or simulated real backend.

        :param qc: QuantumCircuit to be run.
        :param fake_simulation: Whether to run on a simulated real backend.
        :return: Fidelity value from the execution result.
        """
        num_qubits_ = qc.num_qubits
        if fake_simulation:
            # service = QiskitRuntimeService()
#             real_backend = service.backend(self.backend)

            aer_simulator = AerSimulator.from_backend(self.backend)
            pass_manager = generate_preset_pass_manager(backend=aer_simulator, optimization_level=self.optimization_level)
            qc_ = pass_manager.run(qc)
            job = aer_simulator.run([qc_]).result()
            res = job.results[0]
            counts = res.data.counts
            fid = process_counts(counts, n_qubits = num_qubits_, key_type="hexa").values()
            return np.array(list(fid)) / self.shots
        else:
            pass_manager = generate_preset_pass_manager(backend=self.backend, optimization_level=self.optimization_level)
            qc = pass_manager.run(qc)
            backend=self.backend
            sampler = Sampler2(mode = backend)
            sampler.options.default_shots = self.shots
            job = sampler.run([qc])
            print(f"Job ID is {job.job_id()}")
            result = job.result()
            counts = result[0].data.c.get_counts() 
#             print(counts)
            fid = process_counts(counts, n_qubits = num_qubits_, key_type="binary").values()
#             print(fid)
            return np.array(list(fid))/self.shots, job

    def retrieve_job_result(self, job_id):
        job = self.backend.service.job(job_id)
        result = job.result()
        counts = result[0].data.c.get_counts()
        fid = process_counts(counts, n_qubits=self.dim, key_type="binary").values()
        return np.array(list(fid)) / self.shots

    
    def qc_func(self, x):
        if self.circuit is None:
            feature_map = self.CPMap()
        else:
            feature_map = self.circuit 
            
        if self.kernel:
            # p_length = len(x)//2
            val1 = x #[:p_length]
            val2 = self.K_params
#             print(len(val1), len(val2))
            circ1 = feature_map.assign_parameters(val1)
            circ2 = feature_map.assign_parameters(val2).inverse()
            qc = circ1.compose(circ2)
            # qc = qc_.decompose().decompose().decompose().decompose().decompose().decompose()
        else:
            qc = feature_map.assign_parameters(x)
        
        if self.meas_limit is None:
            qc.measure_all()
            # print(qc.num_qubits, qc.cregs)
        else:
            meas_q = int(qc.num_qubits * self.meas_limit)
            qc.measure(range(meas_q), range(meas_q))

        if self.execution_mode == 'simulation':
            return self._simulate(qc)
        elif self.execution_mode == 'DM':
            qc.remove_final_measurements()
            return self._simulateDM(qc)
        elif self.execution_mode == 'STT':
            qc.remove_final_measurements()
            return self._simulateSTT(qc)
        elif self.execution_mode == 'noise':
            return self._simulate_with_noise(qc)
        elif self.execution_mode == 'real_device':
            return self._run_on_real_device(qc, fake_simulation=False)
    
        elif self.execution_mode == 'fake_simulation':
            return self._run_on_real_device(qc, fake_simulation=True)
        else:
            raise ValueError("Invalid execution mode. Choose from 'simulation', 'real_device', or 'fake_simulation'.")
            
    @staticmethod
    def get_depolarizing_noise_model(prob_1, prob_2):
        noise_model = noise.NoiseModel()
        error_1 = noise.depolarizing_error(prob_1, 1)
        error_2 = noise.depolarizing_error(prob_2, 2)
        noise_model.add_all_qubit_quantum_error(error_1, ['rz', 'sx', 'x'])
        noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
        return noise_model
    

class GBPermanents:
    def __init__(self, network_connections, classical=False, kernel= False, use_angle=True, n_jobs=-1):
        self.network_connections = network_connections
        self.m = len(self.network_connections)
        self.half_m = self.m // 2  # Half the number of beam splitters
        self.classical = classical
        self.use_angle = use_angle
        self.n_jobs = n_jobs
        self.kernel = kernel

    def encode_input(self, X, epsilon):
        if self.kernel:
            input_length=self.half_m
        else:
            input_length = self.m
            
        X = np.array(X).flatten()
        if X.shape[0] < input_length:
            padding = np.zeros(input_length - X.shape[0])  
            encoded_values = np.concatenate((X, padding)) + epsilon
        else:
            encoded_values = X[:input_length] + epsilon
        angles = encoded_values if self.use_angle else np.zeros_like(encoded_values)
        phases = np.zeros_like(encoded_values) if self.use_angle else encoded_values
        return angles, phases

    def compute(self, X, eps=0.001):
        if self.kernel:
            angles_Xi, phi_Xi = self.encode_input(X, epsilon=0)
            angles_Xj, phi_Xj = self.encode_input(X, epsilon=eps)
            new_angles_Xj, new_phi_Xj = np.flip(angles_Xj), np.flip(phi_Xj)
            angles_param = np.concatenate((angles_Xi, new_angles_Xj))
            phases_param = np.concatenate((phi_Xi, new_phi_Xj))
            parameters = list(zip(angles_param, phases_param))
        else:
            angles_X, phi_X = self.encode_input(X, epsilon=0)
            parameters = list(zip(angles_X, phi_X))
        
        optical_network = OpticalNetwork(self.network_connections, parameters)
        unitary_matrix = optical_network.compute_final_unitary()

        if self.classical:
            net_value = ClassicalCoincidence(unitary_matrix, parallel=False, info = False, perm_list=True).calculate()
        else:
            net_value = RyserPermanent(parallel=False).compute(unitary_matrix)
        
        return abs(net_value)
    
    
class ClassicalReservoir:
    def __init__(self, input_dim, reservoir_size=100, spectral_radius=0.9, sparsity=0.1):
        """
        Implements a classical Echo State Network (ESN) reservoir.
        """
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.W_in = np.random.uniform(-1, 1, (reservoir_size, input_dim))
        self.W_res = self._initialize_reservoir_weights()
        self.state = np.zeros(reservoir_size)
    
    def _initialize_reservoir_weights(self):
        """Initializes a sparse random reservoir matrix with spectral scaling."""
        W = np.random.randn(self.reservoir_size, self.reservoir_size) * (np.random.rand(self.reservoir_size, self.reservoir_size) < self.sparsity)
        eigenvalues = np.linalg.eigvals(W)
        W /= np.max(np.abs(eigenvalues)) / self.spectral_radius  # Normalize spectral radius
        return W
    
    def compute(self, x):
        """Updates and returns the reservoir state."""
        self.state = np.tanh(np.dot(self.W_in, x) + np.dot(self.W_res, self.state))
        return self.state