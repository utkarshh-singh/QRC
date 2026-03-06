import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import scipy.linalg
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def res_task(reservoir_states, targets):
    model = Ridge(alpha=1.0).fit(reservoir_states, targets)
    predictions = model.predict(reservoir_states)
    return np.sqrt(mean_squared_error(targets, predictions))

def memory_capacity(reservoir_states, inputs, max_delay=10):
    """
    Computes the memory capacity of the reservoir.

    Parameters:
        reservoir_states (np.array): Saved quantum reservoir states.
        inputs (np.array): Original input data.
        max_delay (int): Maximum delay to test for memory retention.

    Returns:
        float: Total memory capacity (sum of individual memory scores).
    """
    scores = []
    for delay in range(1, max_delay + 1):
        X_train = reservoir_states[:-delay]  # Remove last few samples for training
        y_train = inputs[delay:]  # Predict past values

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        
        score = model.score(X_train, y_train)  # R² Score
        scores.append(score)

    return np.sum(scores)


def verify_ESP(reservoir_states_1, reservoir_states_2, threshold=1e-2):
    """
    Verifies the Echo State Property (ESP) by checking if two reservoirs with different initial states 
    converge to the same state over time.

    Parameters:
        reservoir_states_1 (np.array): Reservoir states from normal input.
        reservoir_states_2 (np.array): Reservoir states from perturbed input.
        threshold (float): The difference threshold to confirm ESP satisfaction.

    Returns:
        bool: True if ESP is satisfied, False otherwise.
    """
    # Compute normalized differences to avoid scaling issues
    differences = np.linalg.norm(reservoir_states_1 - reservoir_states_2, axis=1)
    mean_difference = np.mean(differences[-10:])  # Mean of last 10 differences

    return mean_difference < threshold


def narma10_task(reservoir_states, targets):
    """
    Trains a Ridge Regression model on NARMA-10 and computes prediction error.

    Parameters:
        reservoir_states (np.array): Saved quantum reservoir states.
        targets (np.array): NARMA-10 target outputs.

    Returns:
        float: RMSE score.
    """
    model = Ridge(alpha=1.0)
    model.fit(reservoir_states, targets)
    
    predictions = model.predict(reservoir_states)
    return np.sqrt(mean_squared_error(targets, predictions))

def separation_property(reservoir, X1, X2):
    """
    Computes the separation property by measuring the difference in reservoir states for similar inputs.
    """
    X1 = np.array(X1, dtype=np.float32)
    X2 = np.array(X2, dtype=np.float32)
    
    # Compute reservoir states
    state1 = reservoir.apply_reservoir(X1)
    state2 = reservoir.apply_reservoir(X2)
    
    # Compute Euclidean distance between final states
    distance = np.linalg.norm(state1 - state2)
    return distance


def plot_ESP_evolution(reservoir_states_1, reservoir_states_2):
    """
    Plots the difference between two reservoir state trajectories over time.
    """
    differences = np.linalg.norm(reservoir_states_1 - reservoir_states_2, axis=1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(differences, label="State Differences", color="red")
    plt.axhline(y=0.1, linestyle="dashed", color="black", label="Threshold")
    plt.xlabel("Time Steps")
    plt.ylabel("State Difference Norm")
    plt.title("ESP Convergence Behavior")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    


# 1. Largest Lyapunov Exponent (LLE)
def lyapunov_exponent(reservoir, X, delta=1e-5, steps=100):
    """
    Computes the Lyapunov exponent of the reservoir by measuring divergence of nearby trajectories.
    """
    X = np.array(X, dtype=np.float32)
    
    # Apply small perturbation to the initial input
    X_perturbed = X.copy()
    X_perturbed[0] += delta  # Small perturbation to first input
    
    # Run both sequences through the reservoir
    original_states = reservoir.apply_reservoir(X)
    perturbed_states = reservoir.apply_reservoir(X_perturbed)
    
    # Compute the divergence over time
    divergences = np.linalg.norm(perturbed_states - original_states, axis=1)
    lyapunov_exp = np.mean(np.log(divergences + 1e-8))  # Avoid log(0)
    
    return lyapunov_exp


# 2. Washout Test (State Convergence)
def washout_test(reservoir, X, washout_steps=50):
    """
    Tests the washout property by analyzing how long the initial condition affects the reservoir.
    """
    X = np.array(X, dtype=np.float32)
    states = reservoir.apply_reservoir(X)
    
    # Compute state differences before and after washout_steps
    initial_state = states[0]
    final_state = states[washout_steps]
    
    influence = np.linalg.norm(final_state - initial_state) / np.linalg.norm(initial_state + 1e-8)
    return influence


# 3. Spectral Radius Approximation (for Linear Reservoirs)
def spectral_radius(W):
    """
    Computes the spectral radius (largest eigenvalue magnitude) of a reservoir weight matrix.
    :param W: Reservoir weight matrix
    """
    eigenvalues = np.linalg.eigvals(W)
    return max(abs(eigenvalues))

# 4. Information Processing Capacity (Memory and Nonlinearity)
def memory_capacity(reservoir, X, max_lag=10):
    """
    Computes memory capacity by checking how well the reservoir remembers past inputs.
    """
    X = np.array(X, dtype=np.float32)
    states = reservoir.apply_reservoir(X)
    
    # Determine the valid range to avoid mismatches
    min_length = len(X) - max_lag

    # Ensure targets are shaped correctly
    targets = np.array([X[max_lag:min_length, i] for i in range(X.shape[1]) for lag in range(1, max_lag + 1)]).T

    # Train a simple linear model to predict past inputs
    model = LinearRegression()
    model.fit(states[max_lag:min_length], targets)

    # Ensure predictions match target size
    predictions = model.predict(states[max_lag:min_length])
    min_target_length = min(predictions.shape[0], targets.shape[0])
    
    # Compute correlation only for valid indices
    correlation = np.mean([
        np.corrcoef(targets[:min_target_length, i], predictions[:min_target_length, i])[0, 1]
        for i in range(X.shape[1])
    ])
    
    return correlation