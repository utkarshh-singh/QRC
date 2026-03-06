from qiskit import QuantumCircuit
from qiskit.primitives import BackendEstimator, Estimator, BackendSampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from botocore.exceptions import NoCredentialsError
from qiskit_braket_provider import AWSBraketProvider
import boto3
from typing import Optional

class QuantumBackendSelector:
    """
    This class selects and initializes the quantum computing backend based on user preferences.

    Attributes:
        use_simulation (bool): If True, uses a basic simulator. If False, use real hardware or fake backend.
        provider (str): 'IBMQ' or 'AWS' to specify the provider.
        ibmq_token (Optional[str]): IBMQ access token.
        ibmq_instance (Optional[str]): IBMQ instance name.
        aws_device_name (Optional[str]): AWS device name.
    """
    def __init__(self, use_simulation=True, provider='IBMQ', ibmq_token=None, ibmq_instance=None, aws_device_name=None, backend_name=None):
        self.use_simulation = use_simulation
        self.provider = provider
        self.ibmq_token = ibmq_token
        self.ibmq_instance = ibmq_instance
        self.aws_device_name = aws_device_name
        self.backend_name = backend_name
        self.backend = None
        self.estimator_ = None
        self.sampler_ = None
        self.initialize_backend()

    def initialize_backend(self):
        """Initializes the quantum computing backend based on the user's choice."""
        if self.use_simulation:
            self.initialize_simulation_backend()
        else:
            if self.provider == 'IBMQ':
                self.initialize_ibmq_backend()
            elif self.provider == 'AWS':
                self.initialize_aws_backend()
            else:
                print("Invalid provider. Using Qiskit primitive Estimator (simulation).")
                self.initialize_simulation_backend()

    def initialize_simulation_backend(self):
        """Initializes a basic simulator backend."""
        print("Using Qiskit primitive Estimator (simulation).")
        self.backend = None  # No specific backend for basic simulation
        self.estimator_ = Estimator()

    def initialize_ibmq_backend(self):
        """Initializes the IBMQ backend, handling real or fake backend based on user choice."""
        # Prompt for IBMQ token if not provided
        if not self.ibmq_token:
            print("IBMQ token not provided. Please enter your IBMQ token.")
            self.ibmq_token = input("Enter IBMQ token: ")

        try:
            service = QiskitRuntimeService(channel="ibm_quantum", token=self.ibmq_token)
            
            # Prompt for IBMQ instance if not provided
            if not self.ibmq_instance:
                instances = service.instances()
                print("Available instances:", instances)
                self.ibmq_instance = input("Select an IBMQ instance: ")
            
            service = QiskitRuntimeService(channel="ibm_quantum", token=self.ibmq_token, instance=self.ibmq_instance)
            
            # Prompt for IBMQ backend if not provided
            if not self.backend_name:
                backends = service.backends()
                print("Available IBMQ backends:")
                for backend in backends:
                    print(f"- {backend}")
                self.backend_name = input("Select an IBMQ backend: ")

            self.backend = service.backend(self.backend_name)

            # Prompt the user for real hardware or fake backend simulation
            use_fake_backend = input("Do you want to use a fake backend simulation? (yes/no): ").strip().lower()
            if use_fake_backend == "yes":
                self.backend = AerSimulator.from_backend(self.backend)
                print(f"Using fake backend simulation based on {self.backend_name}.")
            else:
                print(f"Using real IBMQ backend {self.backend_name}.")
            
            self.estimator_ = BackendEstimator(backend=self.backend)
            self.sampler_ = BackendSampler(backend=self.backend)
        
        except Exception as e:
            print(f"Failed to initialize IBMQ backend: {e}")
            self.initialize_simulation_backend()

    def initialize_aws_backend(self):
        """Initializes the AWS quantum computing backend, handling real or fake backend based on user choice."""
        try:
            boto3.client('sts').get_caller_identity()  # Verify AWS credentials
            
            # Prompt for AWS device if not provided
            if not self.aws_device_name:
                provider = AWSBraketProvider()
                devices = provider.backends()
                print("Available AWS devices:")
                for device in devices:
                    print(f"- {device}")
                self.aws_device_name = input("Select an AWS device: ")

            self.backend = AWSBraketProvider().get_backend(self.aws_device_name)

            # Prompt the user for real hardware or fake backend simulation
            use_fake_backend = input("Do you want to use a fake backend simulation? (yes/no): ").strip().lower()
            if use_fake_backend == "yes":
                self.backend = AerSimulator()  # Create a basic simulator since AWS does not have a direct fake backend simulation
                print("Using a basic simulator as fake backend.")
            else:
                print(f"Using real AWS backend {self.aws_device_name}.")
            
            self.estimator_ = BackendEstimator(backend=self.backend)
            self.sampler_ = BackendSampler(backend=self.backend)
        
        except NoCredentialsError:
            print("AWS credentials not found. Please configure your AWS environment.")
            self.initialize_simulation_backend()
        except Exception as e:
            print(f"Failed to initialize AWS backend: {e}")
            self.initialize_simulation_backend()

    def get_backend_info(self):
        """Returns information about the selected backend."""
        return self.backend, self.estimator_, self.sampler_, self.provider


# Example usage
# selector = QuantumBackendSelector()#use_simulation=False, provider="IBMQ", ibmq_token="b1a2d1fc2db82ef2c15a39696ab1f988f2d78e99fe2347f139713edd78cb423b988068abd1bbbe775af15d8eb299135c995bd3dcb027937d565df4fe6cd9ef54")
# backend_info = selector.get_backend_info()
# print(backend_info)