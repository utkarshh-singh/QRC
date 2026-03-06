from qiskit import QuantumCircuit, QuantumRegister 
from qiskit.circuit import Parameter, ParameterVector
from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import qiskit_ibm_runtime as qir
from qiskit_ibm_runtime import SamplerV2 as Sampler2, QiskitRuntimeService
from qiskit.primitives import StatevectorSampler
from qiskit_aer import AerSimulator
from utility import MetaFibonacci, mapping, CPaction
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


class CPCircuit:
    def __init__(self, num_features, reps=1, insert_barriers = False, CP_params= None, CP_last_layer = False, ETE = False): # alpha = -np.pi/3, beta = np.pi/6, gamma = -np.pi/9, param1 = np.pi/7, param2=np.pi/9, param3=-np.pi/7):
        """
        
        Args:
        num_features (int): A parameter representing number of features in the dataset.
        reps (int): The number of repeated circuits, has a min. value of 1.
        insert_barriers (Bool): If True, barriers are inserted after each mapping layers.
        CP_last_layer (Bool): If true, CP layers will be applied on the last layer
        CP_params (list, optional): A list of parameters for CMap and PMap circuit, or the Hyperparameters of the CPKernel.
                                 Defaults to a specific list of values if not provided.
                                 [alpha, beta, gamma, param1, param2, param3]
                                 
        """
        
        self.num_features = num_features
        self.reps = reps
        self.insert_barriers = insert_barriers
        self.CP_last_layer = CP_last_layer 
        self.ETE = ETE
        
        # default parameters if none are provided
        if CP_params is None:
            CP_params = [
                -np.pi/3, 
                np.pi/6, 
                -np.pi/9, 
                np.pi/7, 
                np.pi/9, 
                -np.pi/7
            ]
            
        self.alpha = CP_params[0]
        self.beta = CP_params[1]
        self.gamma = CP_params[2]
        self.param1 = CP_params[3]
        self.param2 = CP_params[4]
        self.param3 = CP_params[5]
#         self.alpha = Parameter('alpha')
#         self.beta = Parameter('beta')
#         self.gamma = Parameter('gamma')
#         self.param1 = Parameter('param1')
#         self.param2 = Parameter('param2')
#         self.param3 = Parameter('param3')

        """
        mapping_list (list): A list to provide number of features in each mapping layer
        qubits (int): Number of qubits in the circuit
        params (Parameter vector): List of parameters to go in the feature map circuit
        
        """

        self.mapping_list = mapping(self.num_features).mapping_list()
        self.qubits = self.mapping_list[0]
        self.params = ParameterVector('X', self.num_features)

        
    def cmap(self):
        """
        A 2-qubit quantum circuit inspired from the convolutional layer of QCNN
        """
        q1=QuantumRegister(1, 'q1')
        q2=QuantumRegister(1, 'q2')
        target = QuantumCircuit(q1,q2, name = 'C-Map')
        target.rz(-np.pi / 2, q2)
        target.cx(q2, q1)
        target.rz(self.alpha, q1)
        target.ry(self.beta, q2)
        target.cx(q1, q2)
        target.ry(self.gamma, q2)
        target.cx(q2, q1)
        target.rz(np.pi / 2, q1)
        return target

    def pmap(self):
        """
        A 2-qubit quantum circuit inspired from the pooling layer of QCNN
        """
        q1 = QuantumRegister(1, 'q1')
        q2 = QuantumRegister(1, 'q2')
        target = QuantumCircuit(q1, q2, name = 'P-Map')
        target.rz(-np.pi / 2, q2)
        target.cx(q2, q1)
        target.rz(self.param1, q1)
        target.ry(self.param2, q2)
        target.cx(q1, q2)
        target.ry(self.param3, q2)
        return target
    
    def _num_qubits(self):
        return self.qubits

    def CPMap(self):
        if self.num_features < 2:
            
            """
            ValueError: If the feature dimension is smaller than 2.
            """
            raise ValueError(
                "The CPMap contains 2-local interactions and cannot be "
                f"defined for less than 2 qubits. You provided {self.num_features}."
            )
        if self.num_features == 2:
            print('For 2-dimensional data, CPKernel gets converted to ZFeatureMap')
            pass
        mapping_list_ = self.mapping_list
        last_rep = mapping_list_[-1]
        qc = QuantumCircuit(self.qubits, name = 'CPKernel')
        
        for rep in range(self.reps):
            shift = 0
            for k in range(len(mapping_list_)):
                for i in range(mapping_list_[k]):
                    qc.h(i)
                    qc.p(self.params[i+shift], i)
                if mapping_list_[k] > last_rep:   
                    c_list = CPaction(mapping_list_[k], ETE = self.ETE).cmap_list()
                    for j in range(len(c_list)):
                        qc.append(self.cmap(), c_list[j])
#                         if self.barrier is True:
#                             qc.barrier()
                    p_list = CPaction(mapping_list_[k]).pmap_list()
                    for l in range(len(p_list)):
                        qc.append(self.pmap(), p_list[l])   
#                         if self.barrier is True:
#                             qc.barrier()
                elif mapping_list_[k] == last_rep and last_rep>1 and self.CP_last_layer is True:  
                    c_list = CPaction(mapping_list_[k], ETE = self.ETE).cmap_list()
                    for j in range(len(c_list)):
                        qc.append(self.cmap(), c_list[j])
#                         if self.barrier is True:
#                             qc.barrier()
                    p_list = CPaction(mapping_list_[k]).pmap_list()
                    for l in range(len(p_list)):
                        qc.append(self.pmap(), p_list[l])   
#                         if self.barrier is True:
#                             qc.barrier()
                else:
                    break

                shift = shift + self.mapping_list[k]
                if self.insert_barriers is True:
                    qc.barrier()
                else:
                    pass
        return qc