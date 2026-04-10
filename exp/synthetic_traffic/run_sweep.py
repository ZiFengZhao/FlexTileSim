import os
import subprocess
import numpy as np
import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt

EXECUTABLE = "./main"
CONFIG_FILE = "cfg.txt"
RESULTS_DIR = "results"
PKT_SIZE = 20

SCENARIOS = {
    "Steady": {
        "params": {"alpha": 1.0, "beta": 1.0},
        "inj_max": 0.018
    },
    "Low Burst": {
        "params": {"alpha": 0.005, "beta": 0.01},
        "inj_max": 0.016
    },
    "Mid Burst": {
        "params": {"alpha": 0.0025, "beta": 0.01},
        "inj_max": 0.012
    },
    "High Burst": {
        "params": {"alpha": 0.0025, "beta": 0.02},
        "inj_max": 0.012
    }
}

def get_path(name, suffix="ca"):
    safe_name = name.lower().replace(" ", "_")
    return os.path.join(RESULTS_DIR, f"{safe_name}_{suffix}.dat")

def load_ca_data(path):
    if not os.path.exists(path):
        return None, None
    try:
        data = np.loadtxt(path)
        if data.ndim == 1:
            return [data[0]], [data[1]]
        return data[:, 0].tolist(), data[:, 1].tolist()
    except Exception as e:
        print(f"Warning: Failed to load {path}: {e}")
        return None, None

def save_analytical_results(path, x_list, y_list):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(path, 'w') as f:
        f.write("# Injection_Rate\tAverage_Packet_Latency\n")
        for x, y in zip(x_list, y_list):
            f.write(f"{x:.6f}\t{y:.4f}\n")

def update_config_params(name, lines, inj_rate, alpha, beta):
    new_lines = []
    inj_is_burst = 1 if "Burst" in name else 0
    for line in lines:
        sline = line.strip()
        if sline.startswith('inj_rate='):
            new_lines.append(f"inj_rate={inj_rate}\n")
        elif sline.startswith('network_dim='):
            new_lines.append(f"network_dim=8\n")
        elif sline.startswith('inj_is_burst='):
            new_lines.append(f"inj_is_burst={inj_is_burst}\n")
        elif sline.startswith('alpha='):
            new_lines.append(f"alpha={alpha}\n")
        elif sline.startswith('beta='):
            new_lines.append(f"beta={beta}\n")
        elif sline.startswith("noc_model_type="):
            new_lines.append("noc_model_type=2\n")
        elif sline.startswith("decay_factor="):
            new_lines.append("decay_factor=0.10\n")
        elif sline.startswith("sampling_period="):
            new_lines.append("sampling_period=1000\n")
        else:
            new_lines.append(line)
    return new_lines

def run_simulation():
    try:
        res = subprocess.run(EXECUTABLE, capture_output=True, text=True, check=True, timeout=60)
        return res.stdout
    except Exception as e:
        print(f"Error executing simulation: {e}")
        return None

def parse_output(output):
    if not output: return None
    match = re.search(r"Average Packet Latency.*?:\s*([0-9.]+)", output)
    return float(match.group(1)) if match else None

def run_scenario_sweep(name, config):
    print(f"\n>>> Executing Simulation Sweep: {name}")
    if not os.path.exists(EXECUTABLE):
        print(f"Error: {EXECUTABLE} not found.")
        return [], []
        
    with open(CONFIG_FILE, 'r') as f:
        orig_lines = f.readlines()
    
    inj_start, inj_step = 0.002, 0.002
    rates = np.arange(inj_start, config['inj_max'] + 0.0001, inj_step)
    results_x, results_y = [], []

    for r in rates:
        new_cfg = update_config_params(name, orig_lines, r, config['params']['alpha'], config['params']['beta'])
        with open(CONFIG_FILE, 'w') as f: f.writelines(new_cfg)
        
        out = run_simulation()
        lat = parse_output(out)
        if lat:
            results_x.append(r)
            results_y.append(lat)
            print(f"  Inj: {r:.3f} | Lat: {lat:.2f}")

    with open(CONFIG_FILE, 'w') as f: f.writelines(orig_lines)
    return results_x, results_y

def main():
    parser = argparse.ArgumentParser(description="NoC Analytical Model Real-time Sweep & Comparison")
    parser.add_argument("--sweep", action="store_true", help="Run simulation and plot results")
    args = parser.parse_args()

    if not args.sweep:
        parser.print_help()
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_an_results = {} 
    all_ca_results = {} 

    for name, cfg in SCENARIOS.items():
        ca_path = get_path(name, "ca")
        ca_x, ca_y = load_ca_data(ca_path)
        if ca_x is None:
            print(f"Notice: CA data file '{ca_path}' not found. Plotting analytical only.")
        all_ca_results[name] = (ca_x, ca_y)

        an_x, an_y = run_scenario_sweep(name, cfg)
            
        all_an_results[name] = (an_x, an_y)

    fig, axes = plt.subplots(2, 2, figsize=(4.5, 3.0))
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    axes_flat = axes.flatten()

    for i, (name, cfg) in enumerate(SCENARIOS.items()):
        an_x, an_y = all_an_results[name]
        ca_x, ca_y = all_ca_results[name]
        
        ax_plot = axes_flat[i]
        
        if an_x:
            ax_plot.plot(an_x, an_y, 'o-', label='Analytical Model', color='blue', markersize=1.5, linewidth=1.0)
        
        if ca_x:
            ax_plot.plot(ca_x, ca_y, 's--', label='Cycle-accurate Simulation', color='red', markersize=1.5, linewidth=1.0)

        if not an_x and not ca_x:
            ax_plot.text(0.5, 0.5, "No Data to Plot", ha='center')

        ax_plot.set_title(name, fontsize=6, fontweight='bold')
        ax_plot.set_xlabel("Injection Rate (Packet/cycle/node)", fontsize=6)
        if i % 2 == 0: 
            ax_plot.set_ylabel("Average Latency (cycles)", fontsize=6)
        
        ax_plot.grid(True, linestyle='--', alpha=0.5)
        ax_plot.tick_params(labelsize=5)
        ax_plot.legend(fontsize=5, loc='upper left')
        ax_plot.set_xticks(np.arange(0.002, cfg['inj_max'] + 0.001, 0.004))

    fig_name = "./results/latency_comparison"
    plt.savefig(f"{fig_name}.png", dpi=300)
    plt.savefig(f"{fig_name}.svg")
    print(f"\nSuccess: Plot saved as '{fig_name}.png' and '{fig_name}.svg'")

if __name__ == "__main__":
    main()