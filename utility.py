import numpy as np
import math
from collections import Counter
from scipy.stats import unitary_group
from itertools import combinations_with_replacement
from multiprocessing import Pool, cpu_count
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from sklearn.metrics import matthews_corrcoef, roc_auc_score, precision_recall_curve, auc

class MetaFibonacci:
    def __init__(self, n):
        self.n = n + 1
        self.sequence = [1, 1]
        self._generate_sequence()
        
    def _generate_sequence(self):
        """
        Generate the meta-Fibonacci sequence up to the nth term.
        """
        for i in range(3, self.n + 1):
            next_term = self.sequence[i - self.sequence[i - 3] - 2] + self.sequence[i - self.sequence[i - 2] - 1]
            self.sequence.append(next_term)
    
    def get_sequence(self):
        """
        Returns the meta-Fibonacci sequence up to the nth term.
        
        Returns:
        list: The meta-Fibonacci sequence up to the nth term.
        """
        return self.sequence[1:]
    
    def num_qubits(self):
        """
        Convert a feature to a number of qubits based on the meta-Fibonacci sequence.
        
        Returns:
        int: The number of qubits corresponding to the feature.
        """
        number_of_qubits = self.sequence[1:]
        return number_of_qubits[self.n - 2]

class mapping:
    def __init__(self, num_features):
        self.num_features = num_features
        self.qubits = MetaFibonacci(self.num_features).num_qubits()
        
    def mapping_list(self):
        """
        Halve the value of m (number of qubits) until the sum of the list equals number of features.

        Parameters:
        m (int): The initial value to be halved.

        Returns:
        list: A list of halved values.
        """
        m = self.qubits
        encoding_list = [m]
        while sum(encoding_list) < self.num_features:
            diff = self.num_features - sum(encoding_list)
            m //= 2  # Integer division to halve the value of m
            encoding_list.append(min(m, diff))
        return encoding_list

class CPaction:
    def __init__(self, qubits, ETE= True):
        self.qubits = qubits
        self.ETE = ETE
    def cmap_list(self):
        q = self.qubits
        input_list_=[]
        if q >2:
            for i in range(q):
                lst = [i,i+1]
                input_list_.append(lst)
        else:
            input_list_.append([0,1])
        input_list = input_list_[:-1]    
        even_list = [input_list[i] for i in range(len(input_list))  if i % 2 == 0]
        odd_list = [input_list[i] for i in range(len(input_list)) if i % 2 != 0]
        extra_ = [[q-1,0]]
        c_list = [*even_list, *odd_list, *extra_]
        if self.ETE:
            return c_list
        else:
            return c_list[:-1]
    
    def pmap_list(self):
        q = self.qubits
        p_list=[]
        for i in range(q//2):
            lst = [i,i+q-q//2]
            p_list.append(lst)
        return p_list 


class Evaluator:
    def __init__(self, y_true, y_pred, y_pred_prob=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_prob = y_pred_prob
        self.TN, self.FP, self.FN, self.TP = confusion_matrix(y_true, y_pred).ravel()
    
    def accuracy(self):
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
    
    def recall(self):
        return self.TP / (self.TP + self.FN)
    
    def precision(self):
        return self.TP / (self.FP + self.TP)
    
    def f1_score(self):
        precision = self.precision()
        recall = self.recall()
        return 2 * (precision * recall) / (precision + recall)
    
    def mcc(self):
        return matthews_corrcoef(self.y_true, self.y_pred)
    
    def confusion_matrix(self):
        return confusion_matrix(self.y_true, self.y_pred)
    
    def roc_auc(self):
        if self.y_pred_prob is not None:
            return roc_auc_score(self.y_true, self.y_pred_prob)
        else:
            raise ValueError("Probability predictions are required for ROC AUC.")
    
    def pr_auc(self):
        if self.y_pred_prob is not None:
            precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred_prob)
            return auc(recall, precision)
        else:
            raise ValueError("Probability predictions are required for PR AUC.")
    
    def evaluate(self):
        print('Evaluation Metrics:')
        print('Accuracy: ', self.accuracy())
        print('Recall: ', self.recall())
        print('F1 Score: ', self.f1_score())
        print('Precision: ', self.precision())
        print('Matthews Correlation Coefficient: ', self.mcc())
        
        if self.y_pred_prob is not None:
            print('ROC AUC: ', self.roc_auc())
            print('PR AUC: ', self.pr_auc())
        
        print('\nConfusion Matrix:')
        print('TN, FP, FN, TP')
        print(self.confusion_matrix().ravel())
        
    @staticmethod
    def print_results(results):
        print('Best Parameters: {}\n'.format(results.best_params_))

# # Example usage
# y_true = [0, 1, 1, 0, 1]
# y_pred = [0, 1, 0, 0, 1]
# y_pred_prob = [0.1, 0.9, 0.4, 0.2, 0.8]  # example probabilities

# evaluator = Evaluator(y_true, y_pred, y_pred_prob)
# evaluator.evaluate()

def calculate_factor(combination):
    multiplicity = Counter(combination)
    return np.prod([math.factorial(count) for count in multiplicity.values()])

def generate_column_combinations(n):
    """
    Generate all possible column index combinations in increasing order for an n x n matrix.
    :param n: Size of the matrix (n x n).
    :return: Generator yielding tuples representing the column index combinations.
    """
    indices = range(1, n + 1)  # Column indices start from 1 to n
    combs = [i for i in combinations_with_replacement(indices, n)]
    return combs

def generate_parameters(thetas, network_connections, phases, chi, epsilon):
    """
    Generates a list of parameters for an optical network based on given thetas, phases, and chi.
   
    :param thetas: Array of theta values.
    :param network_connections: List of beam splitter connections (defines order).
    :param phases: Array of phase values.
    :param chi: Array representing perturbations.
    :param epsilon: Small perturbation factor.
    :return: List of parameters [[theta, phase], ...] following the network order.
    """
    required_length = len(network_connections)//2 
    if len(thetas) != required_length or len(chi) != required_length: 
        raise ValueError(f"Length of thetas and chi must be {required_length}, but got " f"{len(thetas)} (thetas) and {len(chi)} (chi).") 
    thetas_perturbed = thetas + epsilon * chi
    forward_parameters = [[thetas[i], phases[i]] for i in range(len(thetas))]
    reverse_parameters = [[thetas_perturbed[i], phases[i]] for i in reversed(range(len(thetas)))]
    parameters = forward_parameters + reverse_parameters
    return parameters

class BSNetwork:
    """
    Class to generate beam splitter networks for optical quantum circuits.
    """

    def __init__(self, q_modes, network_type="Z", reps=1, Kernel=True):
        """
        Initializes the beam splitter network.

        :param q_modes: Number of quantum modes.
        :param network_type: "Z" for Zeilinger-type, "S" for basic, "Custom" for mode-based pairing.
        :param reps: Number of times to repeat the pattern (for "Custom" type).
        """
        self.q_modes = q_modes
        self.network_type = network_type
        self.reps = reps
        self.Kernel = Kernel
        self.network_connections = self._generate_network()

    def _generate_network(self):
        """
        Generates the beam splitter connections based on the selected network type.

        :return: List of beam splitter connections.
        """
        network_connections = []

        if self.network_type == "S":
            # Forward (descending) connections
            for i in range(self.q_modes, 1, -1):
                network_connections.append([i, i - 1])
            if self.Kernel: # Reverse (ascending) connections
                for i in range(1, self.q_modes):
                    network_connections.append([i, i + 1])

        elif self.network_type == "Z":  # Zeilinger-type network
            nc = []
            for j in range(1, self.q_modes):
                for i in range(self.q_modes, j, -1):
                    network_connections.append([i, i - 1])
                    nc.append([i, i - 1])
            if self.Kernel:
                reversed_list = [[k[1], k[0]] for k in nc[::-1]]
                network_connections.extend(reversed_list)

        elif self.network_type == "Custom":  # Custom mode-based pairing
            base_pattern = []
            # The base pattern (without first-last connection)
            for i in range(self.q_modes, 1, -2):  
                base_pattern.append([i, i - 1])

            for i in range(len(base_pattern) - 1):
                base_pattern.append([base_pattern[i][1], base_pattern[i + 1][0]])

            if self.q_modes % 2 != 0:
                base_pattern.append([2, 1])

            full_pattern = base_pattern * self.reps  # Repeat the pattern

            if self.Kernel:
                # Reverse the full pattern to create the cancellation sequence
                reversed_list = [[k[1], k[0]] for k in full_pattern[::-1]]
                network_connections = full_pattern + reversed_list
            
        else:
            raise ValueError("Invalid network type. Choose 'Z', 'Simple', or 'Custom'.")

        return network_connections

    def get_network(self):
        """
        Returns the generated beam splitter network.

        :return: List of beam splitter connections.
        """
        return self.network_connections

    def get_info(self):
        """
        Returns the network information.

        :return: Dictionary containing network type, total beam splitters, and number of features encoded.
        """
        total_beam_splitters = len(self.network_connections)
        num_features = total_beam_splitters // 2  # Half of the total list

        return {
            "Network Type": self.network_type,
            "Total Beam Splitters": total_beam_splitters,
            "Max Feature Capacity": num_features
        }
    
def squared_permanent(z):
    perm_sqrd = z.real*z.real + z.imag*z.imag
    return perm_sqrd