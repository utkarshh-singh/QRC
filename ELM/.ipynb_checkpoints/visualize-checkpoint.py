import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)

def elm_metrics_dict(y_test, predictions, scores=None):
    """
    Returns metrics in the SAME structure as run_classical_benchmarks(...),
    so you can drop ELM into the same plotting functions.

    scores: optional continuous scores for ROC-AUC
      - binary: shape (n_samples,)
      - multiclass: shape (n_samples, n_classes)
    """
    y_test = np.asarray(y_test)
    predictions = np.asarray(predictions)
    n_classes = int(len(np.unique(y_test)))
    avg = "binary" if n_classes == 2 else "macro"

    m = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, average=avg, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, average=avg, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, average=avg, zero_division=0)),
    }

    if scores is not None:
        try:
            if n_classes == 2:
                m["roc_auc"] = float(roc_auc_score(y_test, scores))
            else:
                m["roc_auc_ovr_macro"] = float(roc_auc_score(y_test, scores, multi_class="ovr", average="macro"))
        except Exception:
            pass

    return m


def add_elm_to_results(results, y_test, predictions, scores=None, name="quantum_elm"):
    """
    Adds ELM into the results dict (same schema as classical baselines):
      results[name] = {"best_params": ..., "cv_best_balanced_acc": ..., "test_metrics": {...}}
    """
    results = dict(results)  # copy
    results[name] = {
        "best_params": {"note": "QuantumELM (fixed reservoir)"},
        "cv_best_balanced_acc": None,
        "test_metrics": elm_metrics_dict(y_test, predictions, scores=scores),
    }
    return results


def print_key_scores(y_test, predictions):
    """
    Your exact style (Macro metrics) but also prints balanced accuracy.
    """
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(y_test, predictions),
        "Precision (Macro)": precision_score(y_test, predictions, average="macro", zero_division=0),
        "Recall (Macro)": recall_score(y_test, predictions, average="macro", zero_division=0),
        "F1 (Macro)": f1_score(y_test, predictions, average="macro", zero_division=0),
    }

    print("\n--- Key Scores ---")
    for name, score in metrics.items():
        print(f"{name}: {score:.4f}")

    return metrics

def results_to_dataframe(results, metrics_order=None):
    """
    Convert results dict (from run_classical_benchmarks + quantum_elm entry)
    into a DataFrame with rows=models and columns=metrics.

    results[model]["test_metrics"] is expected.
    """
    rows = {}
    for model_name, entry in results.items():
        tm = entry.get("test_metrics", {})
        rows[model_name] = tm

    df = pd.DataFrame.from_dict(rows, orient="index")

    # Default metric ordering (keeps only those that exist)
    if metrics_order is None:
        metrics_order = [
            "balanced_accuracy", "accuracy", "f1", "precision", "recall",
            "roc_auc", "roc_auc_ovr_macro"
        ]
    cols = [c for c in metrics_order if c in df.columns] + [c for c in df.columns if c not in metrics_order]
    df = df[cols]

    return df


def plot_all_scores_per_model(results, title="All metrics per model", sort_by="balanced_accuracy"):
    """
    Grouped bar chart:
      - x-axis: models
      - grouped bars: metrics

    This is what you asked for: compare all scores for every model (not one metric at a time).
    """
    df = results_to_dataframe(results)

    # Optional sorting
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    if df.shape[1] == 0:
        raise ValueError("No numeric metrics found to plot.")

    models = df.index.tolist()
    metrics = df.columns.tolist()

    x = np.arange(len(models))
    width = 0.85 / len(metrics)

    plt.figure(figsize=(max(12, len(models) * 0.8), 5))
    for j, met in enumerate(metrics):
        plt.bar(x + (j - (len(metrics) - 1) / 2) * width, df[met].values, width, label=met)

    plt.xticks(x, models, rotation=35, ha="right")
    plt.ylim(0, 1.0)  # classification metrics mostly in [0,1]; if you include others remove this
    plt.ylabel("score")
    plt.title(title)
    plt.legend(ncols=min(4, len(metrics)))
    plt.tight_layout()
    plt.show()

    return df


def plot_scores_heatmap(results, title="Metric heatmap", sort_by="balanced_accuracy", cmap="viridis"):
    """
    Heatmap view is often clearer when you have many models/metrics.
    """
    df = results_to_dataframe(results)

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    df = df.select_dtypes(include=[np.number])

    plt.figure(figsize=(max(10, df.shape[1] * 1.2), max(4, df.shape[0] * 0.6)))
    plt.imshow(df.values, aspect="auto", interpolation="nearest", cmap=cmap)
    plt.colorbar(label="score")
    plt.xticks(np.arange(df.shape[1]), df.columns, rotation=35, ha="right")
    plt.yticks(np.arange(df.shape[0]), df.index)
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return df


# ---- Example usage ----
# results = run_classical_benchmarks(...)
# results_all = add_elm_to_results(results, y_test, pred_elm, scores=scores_elm, name="quantum_elm")
#
# df = plot_all_scores_per_model(results_all, title="All scores per model", sort_by="balanced_accuracy")
# print(df.round(4))
#
# df2 = plot_scores_heatmap(results_all, title="Heatmap: all scores", sort_by="balanced_accuracy")