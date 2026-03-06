import os
import json
import pickle
import numpy as np
import glob
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression
from datasets import MG_series, mixed_waves, narma
from reservoirs import CPRC, GBPermanents, ClassicalReservoir #, GBSampling 
from tqdm import tqdm

# Base directory for storing results
BASE_DIR = "QRC/results"

# ===============================
# Data Handler Class
# ===============================

class DataHandler:
    """Handles dataset loading, preprocessing, and transformation."""
    @staticmethod
    def load_dataset(dataset_name, **kwargs):
        """Load dataset and prepare it using a sliding window."""
        if dataset_name == "narma":
            return narma(**kwargs)
        elif dataset_name == "mackey_glass":
            return MG_series(**kwargs)
        elif dataset_name == "mixed_waves":
            return mixed_waves(**kwargs)
        else:
            raise ValueError(f"Dataset {dataset_name} not recognized.") 
            
# ===============================
# Experiment Manager Class
# ===============================

class ExperimentManager:
    """Manages model training, saving, loading, and results storage."""
    def __init__(self, dataset_name, reservoir_name, run_id):
        self.dataset_name = dataset_name
        self.reservoir_name = reservoir_name
        self.run_id = run_id
        self.run_dir = os.path.join(BASE_DIR, dataset_name, reservoir_name, f"Run_{run_id}")
        os.makedirs(self.run_dir, exist_ok=True)

    def save_model(self, model, model_id):
        model_path = os.path.join(self.run_dir, f"GBSR_model_{model_id}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Model saved at: {model_path}")

    def save_results(self, results):
        results_path = os.path.join(self.run_dir, f"results_{self.run_id}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"✅ Results saved at: {results_path}")

    def load_results(self):
        results_path = os.path.join(self.run_dir, f"results_{self.run_id}.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                return json.load(f)
        else:
            print(f"⚠️ Results file not found: {results_path}")
            return None

# ===============================
# Prediction Handler Class
# ===============================

class Predictor:
    """Handles prediction using saved models with optional resetting every `m` steps."""
    @staticmethod
    def load_model(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def predict(X_test, dataset_name, reservoir_name, n_predictions=50, reset_m=None):
        """Predict next values using trained models, with optional reset every `m` steps."""
        model_files = glob.glob(f"{BASE_DIR}/{dataset_name}/{reservoir_name}/Run_*/GBSR_model_*.pkl")
        predictions_dict = {}
        for model_file in model_files:
            model = Predictor.load_model(model_file)
            future_predictions = list(X_test[-1])
            for i in range(n_predictions):
                next_input = np.array(future_predictions[-4:]).reshape(1, -1)
                next_pred = model.predict(next_input)[0]
                future_predictions.append(next_pred)
                if reset_m and (i + 1) % reset_m == 0:
                    future_predictions[-1] = X_test[i, -1]
            predictions_dict[model_file] = future_predictions[-n_predictions:]
        return predictions_dict

# ===============================
# Visualization Class
# ===============================

class Visualizer:
    """Handles plotting of predictions vs. true values."""
    @staticmethod
    def plot_predictions(predictions_dict, X_test, y_test, n_predictions=50):
        plt.figure(figsize=(10, 5))
        last_10_x = list(X_test[-1])
        for model_name, predictions in predictions_dict.items():
            full_x = last_10_x + predictions
            plt.plot(range(len(last_10_x)), last_10_x, label="Previous X", linestyle='dotted', color='black')
            plt.plot(range(len(last_10_x), len(full_x)), y_test[:n_predictions], label="True X", linestyle='solid', color='blue')
            plt.plot(range(len(last_10_x), len(full_x)), predictions, label=f"{model_name} Pred", linestyle='dashed')
        plt.axvline(len(last_10_x), color='r', linestyle='--', label="Future Predictions Start")
        plt.title("Predictions of Saved Models")
        plt.legend()
        plt.show()

# ===============================
# Wrapper Class for different reservoirs
# ===============================
class ReservoirWrapper:
    def __init__(self, function, **kwargs):
        """
        Wrapper to standardize different reservoir computing functions.
        :param function: The function to be used as a reservoir.
        :param kwargs: Additional parameters for the function.
        """
        self.function = function
        self.kwargs = kwargs

    def retrieve_job_result(self, job_id):
        return self.reservoir.retrieve_job_result(job_id)

    def compute(self, x, job_id=None):
        """
        Standardized method to compute the function.
        :param x: Input data.
        :return: Processed data.
        """
        if isinstance(self.function, GBPermanents):
            return self.function.compute(x, **self.kwargs)
        # elif isinstance(self.function, GBSampling):
        #     return self.function.Probs(x, **self.kwargs)
        elif isinstance(self.function, CPRC):
            if job_id is not None:
                return self.function.retrieve_job_result(job_id)
            return self.function.qc_func(x)
        # elif isinstance(self.function, CPRC):
        #     return self.function.qc_func(x)
        elif isinstance(self.function, ClassicalReservoir):
            return self.function.compute(x)
        else:
            raise ValueError("Unsupported function type.")


def closed_loop_predict_from_X_test(model, X_test, m=None, steps=None):
    """
    Closed-loop prediction using standard predict at each step.
    - Feeds X_test[i] to the model.
    - Uses prediction to overwrite the last value of X_test[i+1],
      unless a reset is due, based on interval `m`.

    Parameters:
        model: Trained model with a .predict() method.
        X_test: 2D numpy array of shape (n_samples, window_size).
        m: Reset interval (int). After every m steps, do not modify X_test[i+1].
           Set m=1 to fully follow ground truth (teacher forcing).
           Set m=None to disable resets (pure closed-loop).
        steps: Number of prediction steps (default: len(X_test) - 1).

    Returns:
        np.ndarray: Array of predicted values.
    """
    model.show_progress=False
    X_test = np.array(X_test).copy()  # Don't touch original
    window_size = X_test.shape[1]

    if steps is None:
        steps = len(X_test) - 1

    predictions = []

    for i in tqdm(range(steps), desc="Closed-loop Prediction"):
        current_input = X_test[i].reshape(1, -1)
        pred = model.predict(current_input)[0]
        predictions.append(pred)

        if i + 1 < len(X_test):
            if m is None or (m > 0 and (i + 1) % m != 0):
                # Modify the next window's last value
                X_test[i + 1, -1] = pred
            else:
                # Reset: do nothing, use the original X_test[i+1]
                pass

    return np.array(predictions)



def plot_closed_loop_forecast(predictions, ground_truth, title="Closed-Loop Forecast", 
                               start_idx=0, horizon=None, save_path=None, show=True):
    """
    Plots closed-loop predictions vs. ground truth.

    Parameters:
        predictions (array-like): Forecasted values.
        ground_truth (array-like): True values to compare against.
        start_idx (int): Starting index in ground_truth where prediction begins.
        horizon (int): Number of steps to plot. If None, plot full prediction.
        save_path (str): If provided, saves the figure to this path.
        show (bool): If True, displays the plot.
    """
    predictions = np.array(predictions).flatten()
    ground_truth = np.array(ground_truth).flatten()

    if horizon is None:
        horizon = len(predictions)

    end_idx = start_idx + horizon
    plt.figure(figsize=(14, 5))
    plt.plot(range(start_idx, end_idx), ground_truth[start_idx:end_idx], label='Ground Truth', color='black', linewidth=2)
    plt.plot(range(start_idx, end_idx), predictions[:horizon], label='Prediction', linestyle='--', color='red')
    
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
