import os
import subprocess
import numpy as np
import argparse
import re
import matplotlib.pyplot as plt

EXECUTABLE = "./main"
CONFIG_FILE = "cfg.txt"
RESULTS_DIR = "results"

STEADY_CONFIG = {
    "params": {"alpha": 1.0, "beta": 1.0},
    "inj_start": 0.002,
    "inj_step": 0.002,
    "num_points": 6
}

CA_REFERENCE_LATENCIES = {
    0.002: 50.12, 0.004: 53.48, 0.006: 56.60,
    0.008: 62.73, 0.010: 68.78, 0.012: 81.26
}

SAMPLING_PERIODS = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
FIXED_DECAY_FACTOR = 0.08

DECAY_FACTORS = [
    0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
    0.095, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15,
    0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
]
FIXED_SAMPLING_PERIOD = 1000

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()

def read_config_template():
    with open(CONFIG_FILE, 'r') as f:
        return f.readlines()

def write_config(lines):
    with open(CONFIG_FILE, 'w') as f:
        f.writelines(lines)

def update_config(lines, updates):
    new_lines = []
    for line in lines:
        updated = False
        for key, val in updates.items():
            if line.strip().startswith(f"{key}="):
                new_lines.append(f"{key}={val}\n")
                updated = True
                break
        if not updated:
            new_lines.append(line)
    return new_lines

def run_simulation():
    result = subprocess.run(EXECUTABLE, capture_output=True, text=True, check=True, timeout=60)
    return result.stdout

def parse_output_avg_latency(output):
    for line in output.strip().split("\n"):
        if "Average Packet Latency" in line:
            match = re.search(r"([0-9.]+)", line)
            if match:
                return float(match.group(1))
    return None

def run_injection_sweep(config_lines, inj_rate):
    updates = {
        'inj_rate': inj_rate,
        'alpha': STEADY_CONFIG["params"]["alpha"],
        'beta': STEADY_CONFIG["params"]["beta"],
        'inj_is_burst': 0
    }
    new_config = update_config(config_lines, updates)
    write_config(new_config)
    output = run_simulation()
    return parse_output_avg_latency(output)

def compute_mape(predicted, reference):
    return abs(predicted - reference) / reference * 100.0

def run_sensitivity_sweep_fixed_df(sampling_period, ca_reference):
    config_lines = read_config_template()
    updates = {'sampling_period': sampling_period, 'decay_factor': FIXED_DECAY_FACTOR, 'inj_is_burst': 0}
    config_lines = update_config(config_lines, updates)
    
    mape_list = []
    for i in range(STEADY_CONFIG["num_points"]):
        inj_rate = STEADY_CONFIG["inj_start"] + i * STEADY_CONFIG["inj_step"]
        latency = run_injection_sweep(config_lines, inj_rate)
        if latency is not None:
            mape = compute_mape(latency, ca_reference[inj_rate])
            mape_list.append(mape)
    
    write_config(read_config_template())
    return np.mean(mape_list)

def run_sensitivity_sweep_fixed_sp(decay_factor, ca_reference):
    config_lines = read_config_template()
    updates = {'sampling_period': FIXED_SAMPLING_PERIOD, 'decay_factor': decay_factor, 'inj_is_burst': 0}
    config_lines = update_config(config_lines, updates)
    
    mape_list = []
    for i in range(STEADY_CONFIG["num_points"]):
        inj_rate = STEADY_CONFIG["inj_start"] + i * STEADY_CONFIG["inj_step"]
        latency = run_injection_sweep(config_lines, inj_rate)
        if latency is not None:
            mape = compute_mape(latency, ca_reference[inj_rate])
            mape_list.append(mape)
    
    write_config(read_config_template())
    return np.mean(mape_list)

def save_results_to_file(results_sp, results_df):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, "sensitivity_results.dat")
    with open(filepath, 'w') as f:
        f.write("# sampling_period\tMAPE\n")
        for sp, mape in results_sp:
            f.write(f"{sp}\t{mape:.4f}\n")
        f.write("\n# decay_factor\tMAPE\n")
        for df, mape in results_df:
            f.write(f"{df:.6f}\t{mape:.4f}\n")

def load_results_from_file():
    filepath = os.path.join(RESULTS_DIR, "sensitivity_results.dat")
    results_sp, results_df = [], []
    with open(filepath, 'r') as f:
        section = None
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                if "sampling_period" in line:
                    section = "sp"
                elif "decay_factor" in line:
                    section = "df"
                continue
            if line:
                parts = line.split()
                if section == "sp":
                    results_sp.append((int(parts[0]), float(parts[1])))
                elif section == "df":
                    results_df.append((float(parts[0]), float(parts[1])))
    return results_sp, results_df

def plot_sensitivity_curves(results_sp, results_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))
    
    sp_values = [r[0] for r in results_sp]
    mape_values_sp = [r[1] for r in results_sp]
    ax1.plot(sp_values, mape_values_sp, 'o-', color='#2E86AB', linewidth=1, markersize=2, markerfacecolor='white', markeredgewidth=1.5)
    ax1.set_xlabel('Sliding Window Length (Cycles)', fontsize=7)
    ax1.set_ylabel('MAPE (%)', fontsize=7)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    df_values = [r[0] for r in results_df]
    mape_values_df = [r[1] for r in results_df]
    ax2.semilogx(df_values, mape_values_df, 's-', color='#A23B72', linewidth=1, markersize=2, markerfacecolor='white', markeredgewidth=1.5)
    ax2.set_xlabel('Decay Factor', fontsize=7)
    ax2.set_ylabel('MAPE (%)', fontsize=7)
    ax2.grid(True, alpha=0.3, linestyle='--', which='both')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(RESULTS_DIR, 'sensitivity_analysis.svg'), bbox_inches='tight')

def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if args.sweep:
        results_sp = []
        for sp in SAMPLING_PERIODS:
            print(f"Testing sampling_period = {sp}")
            mape = run_sensitivity_sweep_fixed_df(sp, CA_REFERENCE_LATENCIES)
            results_sp.append((sp, mape))
            print(f"  MAPE: {mape:.2f}%")
        
        results_df = []
        for df in DECAY_FACTORS:
            print(f"Testing decay_factor = {df:.4f}")
            mape = run_sensitivity_sweep_fixed_sp(df, CA_REFERENCE_LATENCIES)
            results_df.append((df, mape))
            print(f"  MAPE: {mape:.2f}%")
        
        save_results_to_file(results_sp, results_df)
        
        best_sp = min(results_sp, key=lambda x: x[1])
        best_df = min(results_df, key=lambda x: x[1])
        print(f"\nBest sampling period: {best_sp[0]} cycles (MAPE={best_sp[1]:.2f}%)")
        print(f"Best decay factor: {best_df[0]:.4f} (MAPE={best_df[1]:.2f}%)")
    
    if args.plot:
        results_sp, results_df = load_results_from_file()
        plot_sensitivity_curves(results_sp, results_df)
        print(f"Plots saved to: {RESULTS_DIR}/sensitivity_analysis.png/svg")

if __name__ == "__main__":
    main()