import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.svm import SVR, SVC
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.linalg import eigvals
from properties import res_task, memory_capacity, verify_ESP, lyapunov_exponent, separation_property, plot_ESP_evolution
from utils import ReservoirWrapper

class ClassicalRC:
    """
    A class to perform classical Reservoir Computing (Echo State Networks) with multiple regression options.
    The class can also run models without a reservoir if needed.
    """

    def __init__(self, input_size, reservoir_size=100, spectral_radius=0.9, sparsity=0.1, noise=0.01, use_reservoir=True, seed=42):
        """
        Initializes the classical reservoir computing model.

        :param input_size: Number of input features (past time steps).
        :param reservoir_size: Number of reservoir neurons.
        :param spectral_radius: Largest eigenvalue of the reservoir weight matrix.
        :param sparsity: Fraction of reservoir connections set to zero.
        :param noise: Noise added to reservoir activations.
        :param use_reservoir: If False, the model runs without a reservoir (direct regression).
        :param seed: Random seed for reproducibility.
        """
        np.random.seed(seed)
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.use_reservoir = use_reservoir

        if use_reservoir:
            self._initialize_reservoir()

    def _initialize_reservoir(self):
        """
        Initializes the reservoir weight matrix with the desired spectral radius and sparsity.
        """
        W = np.random.rand(self.reservoir_size, self.reservoir_size) - 0.5  # Random values in range [-0.5, 0.5]
        W[np.random.rand(*W.shape) > self.sparsity] = 0  # Introduce sparsity

        # Normalize spectral radius
        eigenvalues = np.abs(eigvals(W))
        W *= self.spectral_radius / (max(eigenvalues) + 1e-10)

        self.W_reservoir = W
        self.W_input = np.random.rand(self.reservoir_size, self.input_size) - 0.5

    def transform(self, X):
        """
        Transforms input data through the reservoir if enabled, otherwise returns X directly.

        :param X: Input data of shape (samples, input_size)
        :return: Reservoir state activations (or raw input if reservoir is disabled)
        """
        if not self.use_reservoir:
            return X  # Skip reservoir

        num_samples = X.shape[0]
        reservoir_states = np.zeros((num_samples, self.reservoir_size))

        for i in range(num_samples):
            input_projection = np.dot(self.W_input, X[i])
            if i == 0:
                reservoir_states[i] = np.tanh(input_projection)
            else:
                reservoir_states[i] = np.tanh(np.dot(self.W_reservoir, reservoir_states[i - 1]) + input_projection)

            reservoir_states[i] += self.noise * np.random.randn(self.reservoir_size)  # Add noise

        return reservoir_states

    def train(self, X_train, y_train, model_type="ridge", alpha=1.0):
        """
        Trains a regression model on the transformed (or raw) data.

        :param X_train: Input training data.
        :param y_train: Training targets.
        :param model_type: Regression model type ("linear", "ridge", "mlp", "svr").
        :param alpha: Regularization strength for Ridge regression.
        """
        X_train_transformed = self.transform(X_train)

        if model_type == "linear":
            self.model = LinearRegression()
        elif model_type == "ridge":
            self.model = Ridge(alpha=alpha)
        elif model_type == "mlp":
            self.model = MLPRegressor(hidden_layer_sizes=(50, 50), activation="relu", max_iter=1000, random_state=42)
        elif model_type == "svr":
            self.model = make_pipeline(StandardScaler(), SVR(kernel="rbf"))
        else:
            raise ValueError("Invalid model type. Choose from ['linear', 'ridge', 'mlp', 'svr'].")

        self.model.fit(X_train_transformed, y_train)

    def predict(self, X_test):
        """
        Predicts output using the trained regression model.

        :param X_test: Input test data.
        :return: Predictions.
        """
        X_test_transformed = self.transform(X_test)
        return self.model.predict(X_test_transformed)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model's performance using Mean Squared Error (MSE).

        :param X_test: Test input data.
        :param y_test: Test targets.
        :return: Mean Squared Error (MSE).
        """
        y_pred = self.predict(X_test)
        return mean_squared_error(y_test, y_pred)

    def plot_predictions(self, y_test, y_pred, title="Reservoir Computing Prediction"):
        """
        Plots true vs. predicted values.

        :param y_test: Ground truth values.
        :param y_pred: Predicted values.
        :param title: Plot title.
        """
        plt.figure(figsize=(12, 5))
        plt.plot(y_test[:200], label="True", linestyle="dashed", color="black")
        plt.plot(y_pred[:200], label="Predicted", color="blue", alpha=0.7)
        plt.legend()
        plt.title(title)
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.grid(True)
        plt.show()

    def compute_reservoir_properties(self):
        """
        Computes and prints reservoir properties: spectral radius and sparsity.
        """
        spectral_radius_actual = max(np.abs(eigvals(self.W_reservoir)))
        sparsity_actual = np.mean(self.W_reservoir == 0)

        print(f"Reservoir Properties:")
        print(f"- Spectral Radius (Desired): {self.spectral_radius}")
        print(f"- Spectral Radius (Actual): {spectral_radius_actual:.4f}")
        print(f"- Sparsity (Desired): {self.sparsity}")
        print(f"- Sparsity (Actual): {sparsity_actual:.4f}")
        
    def check_properties(self, X, y, X_train, y_train, X_test, y_test, threshold=1e-2, plot=False):
        """
        Computes and prints various reservoir properties.

        :param X: Input data for computing separation property.
        :param y: Target output for memory capacity computation.
        :param X_train: Training inputs for NARMA10 task.
        :param y_train: Training targets for NARMA10 task.
        :param X_test: Test inputs for NARMA10 task.
        :param y_test: Test targets for NARMA10 task.
        """
        X_reservoir = self.transform(X)  # Transform input through reservoir
        X_train_reservoir = self.transform(X_train)
        X_test_reservoir = self.transform(X_test)

        # Ensure the two sequences have the same length for ESP verification
        min_length = min(X_train_reservoir.shape[0], X_test_reservoir.shape[0])
        X_train_reservoir = X_train_reservoir[:min_length]
        X_test_reservoir = X_test_reservoir[:min_length]


        print("Reservoir Property Evaluations:")
        print(f"- Reservoir Task RMSE: {res_task(X_reservoir, y):.4f}")
        print(f"- Memory Capacity: {memory_capacity(X_reservoir, y):.4f}")
        print(f"- Verify Echo State Property (ESP): {verify_ESP(X_train_reservoir, X_test_reservoir, threshold=threshold)}")
#         print(f"- NARMA10 Task RMSE: {narma10_task(X_train_reservoir, y_train):.4f}")
        print(f"- Largest Lyapunov Exponent: {lyapunov_exponent(X_reservoir):.4f}")
        print(f"- Separation Property: {separation_property(X_reservoir, X):.4f}")
        if plot:
            plot_ESP_evolution(X_train_reservoir, X_test_reservoir)


# Example Usage
# reservoir_model = ClassicalReservoirComputing(input_size=10, reservoir_size=100, use_reservoir=True)
# reservoir_model.train(X_train, y_train, model_type="ridge")
# y_pred = reservoir_model.predict(X_test)
# mse = reservoir_model.evaluate(X_test, y_test)
# reservoir_model.plot_predictions(y_test, y_pred, title="Reservoir Computing Prediction")
# reservoir_model.compute_reservoir_properties()


def extract_expectation_values(prob_list):
    n_qubits = int(np.log2(len(prob_list)))
    z_expectations = np.zeros(n_qubits)
    num_states = 2 ** n_qubits 
    for state_index in range(num_states):
        bitstring = format(state_index, f"0{n_qubits}b")  # Convert index to binary bitstring
        bit_values = np.array([1 if bit == '0' else -1 for bit in bitstring])  # Convert to ±1
        z_expectations += prob_list[state_index] * bit_values  # Weighted sum
    return z_expectations

class ESNetwork:
    """
    Universal Echo State Network (ESN) supporting different reservoir functions and regression models.
    """

    def __init__(self, reservoir, dim=4, regularization=1e-6, alpha=0.8, show_progress=True, 
                 approach='feedback', model_type='ridge', limit=None, cpk=False, save_states=True):
        """
        Initializes the ESN with the specified parameters.
        """
        self.reservoir = ReservoirWrapper(reservoir)
        self.dim = dim
        self.alpha = alpha
        self.show_progress = show_progress
        self.approach = approach
        self.regularization = regularization
        self.model_type = model_type.lower()
        self.model = self._initialize_model()
        self.limit = limit
        self.cpk = cpk
        self.prev_output = np.zeros(dim)  # Initialize feedback memory
        self.save_states = save_states  # If True, saves quantum states
        self.saved_quantum_states = None  # Placeholder for saved quantum states

        # Ensure CP Map uses kernel=True if cpk is enabled
        if self.cpk:
            self.reservoir.kernel = True

        # Warning if using feedback without CP Feature Map
        if self.approach == 'feedback' and not self.cpk:
            print("[WARNING] Using feedback without CP Feature Map (cpk=False). "
                  "This may reduce non-linearity and memory effects.")

    def _initialize_model(self):
        """Initializes the chosen regression model."""
        models = {
            'ridge': Ridge(alpha=self.regularization),
            'lasso': Lasso(alpha=self.regularization),
            'linear': LinearRegression(),
            'svr': SVR(),
            'svc': SVC(),
        }
        return models.get(self.model_type, None) or ValueError("Invalid model_type. Choose from 'ridge', 'lasso', 'linear', 'svr'.")

    def _apply_feedback(self, x):
        """Handles feedback mechanism based on the selected approach."""
        if self.approach == 'feedback':
            if self.cpk:
                prev_out = extract_expectation_values(self.prev_output)
                prev_output_k = np.zeros(self.dim)
                prev_output_k[:len(prev_out)] = prev_out*self.alpha
                return np.concatenate((x, prev_output_k))
            else:
                out_vals = self.prev_output#[0:self.dim]
                return self.alpha * x + (1 - self.alpha) * out_vals #
        else:
            return x  # No modification for time-multiplexing

    def _process_quantum_state(self, quantum_state):
        """Processes quantum state based on the 'limit' parameter."""
        out_length = int(self.limit * len(quantum_state)) if self.limit else len(quantum_state)
        return quantum_state[:out_length].flatten()

    def fit(self, X, Y, washout=200, load_saved=False):
        if load_saved and self.saved_quantum_states is not None:
            print("Using previously saved quantum states.")
            quantum_states = self.saved_quantum_states
        else:
            quantum_states = []
            iterator = tqdm(X, desc="Training Progress", unit=" sample", disable=not self.show_progress)

            for idx, x in enumerate(iterator):
                x = np.array(x, dtype=np.float32)
                modified_input = self._apply_feedback(x)

                quantum_state = self.reservoir.compute(modified_input)
                
                if idx >= washout:
                    processed_state = self._process_quantum_state(quantum_state)
                    quantum_states.append(processed_state)
                # quantum_states.append(self._process_quantum_state(quantum_state))

                if self.approach == 'feedback':
                    self.prev_output = quantum_state  # Update feedback memory

            if self.save_states:
                self.saved_quantum_states = np.array(quantum_states)  # Save states for later use

        Y = Y[washout:]
        self.model.fit(np.array(quantum_states), Y)

    def get_saved_states(self):
        """Returns saved quantum states."""
        if self.saved_quantum_states is None:
            raise ValueError("No quantum states have been saved yet. Run `fit()` first.")
        return self.saved_quantum_states

    def save_states_to_file(self, filename="quantum_states.pkl"):
        """Saves quantum states to a file."""
        if self.saved_quantum_states is None:
            raise ValueError("No quantum states available to save.")
        with open(filename, "wb") as f:
            pickle.dump(self.saved_quantum_states, f)
        print(f"Quantum states saved to {filename}")

    def load_states_from_file(self, filename="quantum_states.pkl"):
        """Loads quantum states from a file."""
        with open(filename, "rb") as f:
            self.saved_quantum_states = pickle.load(f)
        print(f"Quantum states loaded from {filename}")


    def predict(self, X, n=None, m=None, initial_input=None, X_test=None):
        """
        Unified prediction function:
            - If `n` is None, performs standard one-step predictions.
            - If `n` is provided, iteratively predicts `n` future values.
            - If `m` and `X_test` are provided, resets to ground truth after every `m` predictions.

        Parameters:
            X (array-like): Input time series data.
            n (int, optional): Number of future values to predict.
            m (int, optional): Reset interval for X_test.
            initial_input (array-like, optional): Starting input for multi-step predictions.
            X_test (array-like, optional): Ground truth data for periodic resets.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call `fit` first.")

        quantum_states = []
        predictions = []

        # Case 1: Standard Prediction
        if n is None:
            iterator = tqdm(X, desc="Prediction Progress", unit=" sample", disable=not self.show_progress)

            for x in iterator:
                x = np.array(x, dtype=np.float32)
                modified_input = self._apply_feedback(x)

                quantum_state = self.reservoir.compute(modified_input)
                quantum_states.append(self._process_quantum_state(quantum_state))

                if self.approach == 'feedback':
                    self.prev_output = quantum_state  # Update feedback memory

            return self.model.predict(np.array(quantum_states))

        # Case 2: Multi-Step Future Prediction
        current_input = np.array(initial_input).reshape(1, -1)
        test_index = 0  # Track X_test reset index

        iterator = tqdm(range(n), desc="Sequential Prediction Progress", unit=" step", disable=not self.show_progress)

        for i in iterator:
            modified_input = self._apply_feedback(current_input[0])

            quantum_state = self.reservoir.compute(modified_input)
            processed_q_state = self._process_quantum_state(quantum_state)
            print("Q_state:", processed_q_state)
            next_pred = self.model.predict(processed_q_state.reshape(1, -1))
            predictions.append(next_pred.flatten()[0])

            self.prev_output = quantum_state.flatten()  # Update feedback

            if m and (i + 1) % m == 0 and X_test is not None and test_index < len(X_test):
                current_input = np.array(X_test[test_index]).reshape(1, -1)
                test_index += 1
            else:
                current_input = np.roll(current_input, -1)
                current_input[0, -1] = next_pred.flatten()[0]

        return np.array(predictions)

    # def predict(self, X):
    #     if self.model is None:
    #         raise ValueError("Model has not been trained yet. Call `fit` first.")

    #     quantum_states = []
    #     iterator = tqdm(X, desc="Prediction Progress", unit=" sample", disable=not self.show_progress)

    #     for x in iterator:
    #         x = np.array(x, dtype=np.float32)
    #         modified_input = self._apply_feedback(x)

    #         quantum_state = self.reservoir.compute(modified_input)
    #         quantum_states.append(self._process_quantum_state(quantum_state))

    #         if self.approach == 'feedback':
    #             self.prev_output = quantum_state  # Update feedback memory

    #     return self.model.predict(np.array(quantum_states))
    