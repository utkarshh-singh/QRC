import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_entropy_kd(entropies, params, qubit_index, mean_kd, std_kd, kd_error_bars=True, savefile=False):
    if isinstance(entropies[0], dict):
        entropy_values = [e[qubit_index] for e in entropies]
        avg_entropy_values = [np.mean(list(e.values())) for e in entropies]
    else:
        entropy_values = entropies
        avg_entropy_values = entropy_values

    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()
    ax2.spines.right.set_position(("axes", 1.1))

    # Plot entropy and average entropy
    l1, = ax1.plot(params, entropy_values, 'o-', color='cyan', label=f'Entropy (Qubit {qubit_index})')
    l2, = ax1.plot(params, avg_entropy_values, 'x--', color='red', label='Avg Entropy (All Qubits)')

    # Plot KD with error bars
    if kd_error_bars:
        l3 = ax2.errorbar(params, mean_kd, yerr=std_kd, fmt='o-', color='purple', ecolor='gray', capsize=3, label='Mean KD ± Std')
    else:
        l3, = ax2.plot(params, mean_kd, '^-', color='purple', label='Mean KD')

    # Axis labeling
    ax1.set_xlabel("Parameter Index")
    ax1.set_ylabel('Entropy & Avg Entropy', color='blue')
    ax2.set_ylabel("Mean KD", color='purple')

    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='purple')

    # Legend and title
    lines = [l1, l2, l3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11)

    plt.title(f'Entropy, Average Entropy, and Mean KD (Qubit {qubit_index})')
    plt.tight_layout()
    if savefile:
        plt.savefig(f"results/Entropy_Avg_KD_Qubit_{qubit_index}.pdf")
    plt.show()



def plot_MC_vs_tau(MC, params, qubit_index, additional_metric=None, metric_label='RMSE'):
    tau_values = params #list(range(1, len(entropies) + 1))
    entropy_values = MC #[entry[qubit_index] for entry in entropies]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot entropy on primary axis
    ax1.plot(tau_values, entropy_values, marker='o', color='blue', label=f'Entropy Qubit {qubit_index}')
    ax1.set_xlabel('Value of First Parameter')
    ax1.set_ylabel('Entropy', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    # Optional secondary axis
    if additional_metric is not None:
        ax2 = ax1.twinx()
        ax2.plot(tau_values, additional_metric, marker='x', color='red', label=metric_label)
        ax2.set_ylabel(metric_label, color='red')
        ax2.tick_params(axis='y', labelcolor='red')

    plt.title(f'Qubit {qubit_index} Entropy vs Value of First Parameter' + (f' + {metric_label}' if additional_metric is not None else ''))
    fig.tight_layout()
    plt.show()


def plot_entropy_vs_data(entropies, params, qubit_index, additional_metric=None, metric_label='RMSE'):
    tau_values = params #list(range(1, len(entropies) + 1))
    if type(entropies[0]) is dict:
        entropy_values = [entry[qubit_index] for entry in entropies]
    else:
        entropy_values =  entropies
    fig, ax1 = plt.subplots(figsize=(16, 10))

    # Plot entropy
    ax1.plot(tau_values, entropy_values, marker='o', color='blue', label=f'Entropy Qubit {qubit_index}')
    ax1.set_xlabel('Value of First Parameter')
    if type(entropies[0]) is dict:
        ax1.set_ylabel('Entropy', color='blue')
    else:
        ax1.set_ylabel('Memory Capacity', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    # Entropy min/max
    min_e, max_e = min(entropy_values), max(entropy_values)
    min_idx_e, max_idx_e = entropy_values.index(min_e), entropy_values.index(max_e)
    min_tau_e, max_tau_e = tau_values[min_idx_e], tau_values[max_idx_e]

    ax1.plot(min_tau_e, min_e, 'v', color='green', markersize=10, label='Min Entropy')
    ax1.plot(max_tau_e, max_e, '^', color='orange', markersize=10, label='Max Entropy')

    ax1.annotate(f'Min: {min_e:.3f}\nTau: {min_tau_e}',
                 (min_tau_e, min_e), textcoords="offset points", xytext=(0, -30),
                 ha='center', color='green', arrowprops=dict(arrowstyle='->', color='green'))

    ax1.annotate(f'Max: {max_e:.3f}\nTau: {max_tau_e}',
                 (max_tau_e, max_e), textcoords="offset points", xytext=(0, 15),
                 ha='center', color='orange', arrowprops=dict(arrowstyle='->', color='orange'))

    # Secondary metric (e.g., RMSE)
    if additional_metric is not None:
        ax2 = ax1.twinx()
        ax2.plot(tau_values, additional_metric, marker='x', color='red', label=metric_label)
        ax2.set_ylabel(metric_label, color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # RMSE min/max
        min_m, max_m = min(additional_metric), max(additional_metric)
        min_idx_m, max_idx_m = additional_metric.index(min_m), additional_metric.index(max_m)
        min_tau_m, max_tau_m = tau_values[min_idx_m], tau_values[max_idx_m]

        ax2.plot(min_tau_m, min_m, 'v', color='purple', markersize=10, label=f'Min {metric_label}')
        ax2.plot(max_tau_m, max_m, '^', color='brown', markersize=10, label=f'Max {metric_label}')

        ax2.annotate(f'Min: {min_m:.3f}\nTau: {min_tau_m}',
                     (min_tau_m, min_m), textcoords="offset points", xytext=(0, -30),
                     ha='center', color='purple', arrowprops=dict(arrowstyle='->', color='purple'))

        ax2.annotate(f'Max: {max_m:.3f}\nTau: {max_tau_m}',
                     (max_tau_m, max_m), textcoords="offset points", xytext=(0, 15),
                     ha='center', color='brown', arrowprops=dict(arrowstyle='->', color='brown'))

        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    else:
        ax1.legend(loc='upper left')

    plt.title(f'Qubit {qubit_index} Entropy vs Value of First Parameter' + (f' + {metric_label}' if additional_metric else ''))
    fig.tight_layout()
    plt.show()


def plot_all_metrics_together(entropies, memory_capacities, rmse, params, qubit_index):
    if isinstance(entropies[0], dict):
        entropy_values = [e[qubit_index] for e in entropies]
    else:
        entropy_values = entropies

    x_vals = np.arange(len(MC))

    fig, ax1 = plt.subplots(figsize=(16, 10))

    if not (len(entropy_values) == len(memory_capacities) == len(rmse) == len(params)):
        raise ValueError("All input lists must be of the same length.")

    def find_extrema(values, name):
        min_val = min(values)
        max_val = max(values)
        min_idx = values.index(min_val)
        max_idx = values.index(max_val)
        print(f"{name} Min: ({params[min_idx]}, {min_val:.4f})")
        print(f"{name} Max: ({params[max_idx]}, {max_val:.4f})")
        return (params[min_idx], min_val), (params[max_idx], max_val)

    entropy_min, entropy_max = find_extrema(entropy_values, "Entropy")
    mc_min, mc_max = find_extrema(memory_capacities, "Memory Capacity")
    rmse_min, rmse_max = find_extrema(rmse, "RMSE")

    fig, ax1 = plt.subplots(figsize=(18, 13))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.1))

    # Plot each line
    l1, = ax1.plot(params, entropy_values, color='blue', marker='o', label=f'Entropy (Qubit {qubit_index})')
    l2, = ax2.plot(params, memory_capacities, color='green', marker='^', label='Memory Capacity')
    l3, = ax3.plot(params, rmse, color='red', marker='s', label='RMSE')

    # Set labels
    ax1.set_xlabel("Parameter Index")
    ax1.set_ylabel(f'Entropy (Qubit {qubit_index})', color='blue')
    ax2.set_ylabel("Memory Capacity", color='green')
    ax3.set_ylabel("RMSE", color='red')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='green')
    ax3.tick_params(axis='y', labelcolor='red')

    # Function to annotate extrema with vertical lines
    def annotate_extremum(ax, point, color, label):
        x, y = point
        ax.plot(x, y, 'o', color=color, markersize=8)
        ax.axvline(x=x, color=color, linestyle='--', linewidth=1.2)
        ax.annotate(f'{label}\n({x}, {y:.3f})',
                    xy=(x, y), xytext=(5, 15), textcoords='offset points',
                    ha='left', color=color, fontsize=12,
                    arrowprops=dict(arrowstyle='->', color=color))

    # Annotate all extrema
    annotate_extremum(ax1, entropy_min, 'cyan', 'Min Entropy')
    annotate_extremum(ax1, entropy_max, 'navy', 'Max Entropy')
    annotate_extremum(ax2, mc_min, 'lime', 'Min MC')
    annotate_extremum(ax2, mc_max, 'darkgreen', 'Max MC')
    annotate_extremum(ax3, rmse_min, 'pink', 'Min RMSE')
    annotate_extremum(ax3, rmse_max, 'darkred', 'Max RMSE')

    # Combine all legends
    lines = [l1, l2, l3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title(f'Entropy, Memory Capacity, and RMSE vs Parameter (Qubit {qubit_index})')
    plt.tight_layout()
    # plt.grid(True)
    plt.savefig(f"results/Qubit_{qubit_index}.pdf")
    plt.show()

# Extended plot function with extremum annotation
def plot_all_metrics_extended(entropies, memory_capacities, rmse, params, qubit_index, mean_kd, std_kd, kd_error_bars=True, savefile=False):
    if isinstance(entropies[0], dict):
        entropy_values = [e[qubit_index] for e in entropies]
        avg_entropy_values = [np.mean(list(e.values())) for e in entropies]
    else:
        entropy_values = entropies
        avg_entropy_values = entropy_values

    # Identify extrema
    def find_extrema(values):
        min_val = min(values)
        max_val = max(values)
        min_idx = values.index(min_val)
        max_idx = values.index(max_val)
        return (params[min_idx], min_val), (params[max_idx], max_val)

    extrema = {
        "Entropy": find_extrema(entropy_values),
        "AvgEntropy": find_extrema(avg_entropy_values),
        "MC": find_extrema(memory_capacities),
        "RMSE": find_extrema(rmse),
        "KD": find_extrema(list(mean_kd))
    }

    fig, ax1 = plt.subplots(figsize=(18, 13))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.1))
    ax4 = ax1.twinx()
    ax4.spines.right.set_position(("axes", 1.2))

    # Plot metrics
    l1, = ax1.plot(params, entropy_values, 'o-', color='blue', label=f'Entropy (Qubit {qubit_index})')
    l2, = ax1.plot(params, avg_entropy_values, 'x--', color='black', label='Avg Entropy (All Qubits)')
    l3, = ax2.plot(params, memory_capacities, '^-', color='green', label='Memory Capacity')
    l4, = ax3.plot(params, rmse, 's-', color='red', label='RMSE')
    if kd_error_bars:
        l5 = ax4.errorbar(params, mean_kd, yerr=std_kd, fmt='o-', color='purple', ecolor='gray', capsize=3, label='Mean KD ± Std')
    else:
        l5, = ax4.plot(params, mean_kd, '^--', color='purple', label='Mean KD')


    # Annotate extrema
    def annotate_extremum(ax, point, color, label):
        x, y = point
        ax.plot(x, y, 'o', color=color, markersize=8)
        ax.axvline(x=x, color=color, linestyle='--', linewidth=1.2)
        ax.annotate(f'{label}\n({x}, {y:.3f})',
                    xy=(x, y), xytext=(5, 15), textcoords='offset points',
                    ha='left', color=color, fontsize=11,
                    arrowprops=dict(arrowstyle='->', color=color))

    annotate_extremum(ax1, extrema["Entropy"][0], 'cyan', 'Min Entropy')
    annotate_extremum(ax1, extrema["Entropy"][1], 'navy', 'Max Entropy')
    annotate_extremum(ax1, extrema["AvgEntropy"][0], 'skyblue', 'Min Avg Entropy')
    annotate_extremum(ax1, extrema["AvgEntropy"][1], 'darkblue', 'Max Avg Entropy')
    annotate_extremum(ax2, extrema["MC"][0], 'lime', 'Min MC')
    annotate_extremum(ax2, extrema["MC"][1], 'darkgreen', 'Max MC')
    annotate_extremum(ax3, extrema["RMSE"][0], 'pink', 'Min RMSE')
    annotate_extremum(ax3, extrema["RMSE"][1], 'darkred', 'Max RMSE')
    annotate_extremum(ax4, extrema["KD"][0], 'orchid', 'Min KD')
    annotate_extremum(ax4, extrema["KD"][1], 'indigo', 'Max KD')

    # Axis labels
    ax1.set_xlabel("Parameter Index")
    ax1.set_ylabel('Entropy & Avg Entropy', color='blue')
    ax2.set_ylabel("Memory Capacity", color='green')
    ax3.set_ylabel("RMSE", color='red')
    ax4.set_ylabel("Mean KD", color='purple')

    # Axis colors
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='green')
    ax3.tick_params(axis='y', labelcolor='red')
    ax4.tick_params(axis='y', labelcolor='purple')

    # Combine legend
    lines = [l1, l2, l3, l4, l5]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11)

    plt.title(f'All Metrics with Extremum Annotations (Qubit {qubit_index})')
    plt.tight_layout()
    if savefile:
        plt.savefig(f"results/Extended_Qubit_{qubit_index}_with_extrema.pdf")
    plt.show()