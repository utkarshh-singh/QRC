import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import square

def mixed_waves(n_samples, n, window_size=4, prediction_horizon=20, plot=False, seed=None, return_dataframe=False, min_plot_points=100):
    """
    Generates a dataset with randomly mixed sine and square waves, where each wave segment has n points.
    
    Parameters:
        total_points (int): Total points in the dataset.
        n (int): Number of points per wave segment.
        plot (bool): If True, plots the first 100 points of the dataset.
        seed (int, optional): Random seed for reproducibility. If None, randomness is not controlled.
        return_dataframe (bool): If True, returns a DataFrame. If False, returns X (signal values), y (labels as 0,1), and y_ (next time-step values).

    Returns:
        pd.DataFrame or (X, y, y_): A DataFrame containing time values, signal values, and wave types if return_dataframe=True.
                                     Otherwise, returns X (signal values), y (binary labels where sine=0, square=1), and y_ (next time-step values).
    """
    if seed is not None:
        np.random.seed(seed)  # Set random seed for reproducibility

    def generate_wave(wave_type, x):
        """Generates y values for either a sine or square wave based on input x positions."""
        if wave_type == "sine":
            return 0.5 * np.sin(2 * np.pi * x)
        elif wave_type == "square":
            return 0.5 * square(2 * np.pi * x)

    num_segments = n_samples // n
    adjusted_total_points = num_segments * n  # Ensure multiple of n

    x_positions = np.linspace(0, 1, n, endpoint=False)
    time_values = np.linspace(0, 10, adjusted_total_points, endpoint=False)
    signal_values = np.zeros(adjusted_total_points)
    wave_types = []

    for i in range(num_segments):
        wave_choice = np.random.choice(["sine", "square"])
        start_idx = i * n
        end_idx = start_idx + n
        signal_values[start_idx:end_idx] = generate_wave(wave_choice, x_positions)
        wave_types.extend([wave_choice] * n)

    wave_types = np.array(wave_types)

    # Convert wave types to binary labels (sine=0, square=1)
    labels = np.array([0 if wt == "sine" else 1 for wt in wave_types])
    
    # Apply sliding window
    X, Y, Y_ = [], [], []
    for i in range(len(signal_values) - window_size-prediction_horizon):
        X.append(signal_values[i:i + window_size])
        Y.append(labels[i + window_size+prediction_horizon])
        Y_.append(signal_values[i + window_size+prediction_horizon])  # Next value in time-series

    X, Y, Y_ = np.array(X), np.array(Y), np.array(Y_)

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f"Lag {i+1}" for i in range(window_size)])
    df["Target"] = Y
    df["Next_Value"] = Y_

    if plot:
        plot_points = min(min_plot_points, adjusted_total_points)
        plt.figure(figsize=(8, 3))
        plt.plot(time_values[:plot_points], signal_values[:plot_points], color='orange', alpha=0.7, label='Wave Signal')
        plt.scatter(time_values[:plot_points], signal_values[:plot_points], color='blue', s=10, label='Sample Points')
        plt.xlabel("Time (µs)")
        plt.ylabel("Amplitude")
        plt.xlim(time_values[0], time_values[plot_points - 1])
        plt.ylim(-0.55, 0.55)
        plt.title("Randomly Mixed Sine and Square Waves (First 100 Points)")
        plt.grid(True)
        plt.show()

    if return_dataframe:
        return df
    else:
        return X, Y, Y_
    

def narma(n_samples=1000, order=10, window_size=4, prediction_horizon=20, plot=False, seed=None, return_dataframe=False, min_plot_points=100):
    """
    Generates a NARMA (Nonlinear AutoRegressive Moving Average) dataset.

    Parameters:
        n_samples (int): Number of samples to generate.
        order (int): Order of the NARMA system.
        plot (bool): If True, plots the first 500 points of the dataset.
        seed (int, optional): Random seed for reproducibility. If None, randomness is not controlled.
        return_dataframe (bool): If True, returns a DataFrame. If False, returns X (input signal u) and y (output signal y).

    Returns:
        pd.DataFrame or (X, y): A DataFrame containing input (u) and output (y) time series if return_dataframe=True.
                                Otherwise, returns X (input signal u) and y (output signal y).
    """
    
    if seed is not None:
        np.random.seed(seed) 

    u = np.random.uniform(0, 0.5, n_samples)  # Random input signal
    y = np.zeros(n_samples)  # Output signal initialized

    # Generate NARMA sequence
    for t in range(order, n_samples - 1):
        y[t + 1] = (0.3 * y[t] + 0.05 * y[t] * np.sum(y[t - order:t]) 
                     + 1.5 * u[t - order] * u[t] + 0.1)

    # Apply sliding window
    X, Y = [], []
    for i in range(len(y) - window_size-prediction_horizon):
        X.append(u[i:i + window_size])
        Y.append(y[i + window_size+prediction_horizon])

    X, Y = np.array(X), np.array(Y)

    df = pd.DataFrame(X, columns=[f"Lag {i+1}" for i in range(window_size)])
    df["Target"] = Y
    
    if plot:
        plot_points = min(min_plot_points, n_samples)
        plt.figure(figsize=(8, 3))
        plt.plot(range(plot_points), u[:plot_points], color='blue', alpha=0.7, label='Input (u)')
        plt.plot(range(plot_points), y[:plot_points], color='orange', alpha=0.7, label='Output (y)')
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title(f"NARMA Time Series (First {plot_points} Points)")
        plt.legend()
        plt.grid(True)
        plt.show()

    if return_dataframe:
        return df
    else:
        return X, Y

# df_narma = generate_narma(n_samples=1000, order=10, plot=True, seed=42, return_dataframe=True)
# X_narma, y_narma = generate_narma(n_samples=1000, order=10, plot=False, seed=42, return_dataframe=False)


def MG_series_old(n_samples=20000, b=0.1, c=0.3, tau=18, window_size=4, prediction_horizon=20, time_steps=4, initial_conditions=None, plot=False, return_dataframe=False, min_plot_points=100):
    """
    Generates a Mackey-Glass time series dataset and applies a sliding window for inputs and targets.

    Parameters:
        N (int): Number of data points to generate.
        b (float): Decay rate parameter.
        c (float): Production rate parameter.
        tau (int): Time delay in the system.
        window_size (int): Number of past values to use as input for each prediction.
        initial_conditions (list, optional): Initial values to start the series.
        plot (bool): If True, plots the first 500 points of the dataset.
        return_dataframe (bool): If True, returns a DataFrame. If False, returns X (time series values) and y (shifted targets).

    Returns:
        pd.DataFrame or (X, y): A DataFrame containing input sequences and targets if return_dataframe=True.
                                Otherwise, returns X (input sequences) and y (targets).
    """
    if initial_conditions is None:
        initial_conditions = [
            0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076, 1.1352,
            1.1485, 1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756,
            1.0739, 1.0759
        ]
    
    if len(initial_conditions) <= tau:
        padding = [initial_conditions[-1]] * (tau + 1 - len(initial_conditions))
        initial_conditions += padding
    # Initialize the time series with given initial conditions
    y = initial_conditions.copy()
    N = n_samples
    start_index = max(len(initial_conditions) - 1, tau)
    # Compute the Mackey-Glass series
    for n in range(start_index, N + 99):
        y.append(y[n] - b * y[n] + c * y[n - tau] / (1 + y[n - tau] ** 10))

    # Trim the initial transient phase (first 100 data points)
    y = np.array(y[100:])

    # Create input sequences and target values using a sliding window
    X, Y = [], []
    for i in range(len(y) - window_size-prediction_horizon):
        X.append(y[i : i + window_size])  # Last `window_size` values as input
        Y.append(y[i + window_size+prediction_horizon])  # Next value as target

    X, Y = np.array(X), np.array(Y) 

    # Convert to DataFrame
    df = pd.DataFrame(X, columns=[f"Lag {i+1}" for i in range(window_size)])
    df["Target"] = Y

    if plot:
        plot_points = min(min_plot_points, len(Y))
        plt.figure(figsize=(8, 3))
        plt.plot(range(plot_points), Y[:plot_points], color='blue', alpha=0.7, label='Mackey-Glass Targets')
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title(f"Mackey-Glass Time Series Targets (First {plot_points} Points)")
        plt.legend()
        plt.grid(True)
        plt.show()

    if return_dataframe:
        return df
    else:
        return X, Y
    
    
def MG_series(n_samples=20000, b=0.1, c=0.2, tau=17, window_size=4, prediction_horizon=20, time_step=1,
              initial_conditions=None, plot=False, return_dataframe=False, min_plot_points=100):
    """
    Generates a Mackey-Glass time series dataset and applies a time-stepped sliding window for inputs and targets.

    Parameters:
        n_samples (int): Number of data points to generate after removing transients.
        b (float): Decay rate parameter.
        c (float): Production rate parameter.
        tau (int): Time delay in the system.
        window_size (int): Number of past values to use as input for each prediction.
        prediction_horizon (int): Steps ahead to predict the target value.
        time_step (int): Interval between time steps in input sequences.
        initial_conditions (list, optional): Initial values to start the series.
        plot (bool): If True, plots the target sequence.
        return_dataframe (bool): If True, returns a DataFrame.
        min_plot_points (int): Minimum number of points to plot.

    Returns:
        pd.DataFrame or (X, y): Feature matrix and target values, or a DataFrame if specified.
    """

    if initial_conditions is None:
        initial_conditions = [
            0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076, 1.1352,
            1.1485, 1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756,
            1.0739, 1.0759
        ]

    if len(initial_conditions) <= tau:
        padding = [initial_conditions[-1]] * (tau + 1 - len(initial_conditions))
        initial_conditions += padding

    # Generate Mackey-Glass sequence
    y = initial_conditions.copy()
    start_index = max(len(initial_conditions) - 1, tau)
    for n in range(start_index, n_samples + 100):
        y.append(y[n] - b * y[n] + c * y[n - tau] / (1 + y[n - tau] ** 10))

    # Remove transients
    y = np.array(y[100:])  # Now len(y) = n_samples

    # Check if parameters are valid
    total_required_length = window_size * time_step + prediction_horizon
    if len(y) < total_required_length:
        raise ValueError("Not enough data points to form even one training example. "
                         f"Got {len(y)} points, but need at least {total_required_length}.")

    # Create input-output pairs
    X, Y = [], []
#     max_i = len(y) - window_size * time_step - prediction_horizon
    for i in range((n_samples//time_step)-(prediction_horizon+window_size-1)):
        X.append(y[i*time_step : i*time_step + window_size]) 
        Y.append(y[i*time_step + window_size+prediction_horizon])
#         x_seq = [y[i + j * time_step] for j in range(window_size)]
#         target = y[i + window_size * time_step + prediction_horizon]
#         X.append(x_seq)
#         Y.append(target)

    X, Y = np.array(X), np.array(Y)

    # Plot if requested
    if plot:
        plot_points = min(min_plot_points, len(Y))
        plt.figure(figsize=(8, 3))
        plt.plot(range(plot_points), Y[:plot_points], color='blue', alpha=0.7, label='Mackey-Glass Targets')
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title(f"Mackey-Glass Time Series Targets (First {plot_points} Points)")
        plt.legend()
        plt.grid(True)
        plt.show()

    if return_dataframe:
        df = pd.DataFrame(X, columns=[f"Lag {i+1}" for i in range(window_size)])
        df["Target"] = Y
        return df

    return X, Y
