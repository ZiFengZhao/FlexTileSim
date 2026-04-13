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
    "num_points": 7,
    "inj_is_burst": 0,
    "ca_file": "steady_ca.dat"
}

MID_BURST_CONFIG = {
    "params": {"alpha": 0.0025, "beta": 0.01},
    "inj_start": 0.002,
    "inj_step": 0.002,
    "num_points": 5,
    "inj_is_burst": 1,
    "ca_file": "mid_burst_ca.dat"
}

SAMPLING_PERIODS = [300, 400, 500, 600, 700, 800, 900, 950, 1000, 1025, 1050, 1100, 1200, 1300, 1400, 1500, 1600]
FIXED_DECAY_FACTOR = 0.1

DECAY_FACTORS = [
    0.001, 0.0015, 0.002, 0.003, 0.0045, 0.0075, 0.008, 0.009,
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 
    0.11, 0.12, 0.13, 0.14, 0.145, 0.15,
    0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0
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


def load_ca_reference(filepath):
    reference = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                reference[float(parts[0])] = float(parts[1])
    return reference


def run_injection_sweep(config_lines, inj_rate, alpha, beta, inj_is_burst):
    updates = {
        'inj_rate': f"{inj_rate:.6f}",
        'alpha': f"{alpha:.6f}",
        'beta': f"{beta:.6f}",
        'inj_is_burst': str(inj_is_burst)
    }
    new_config = update_config(config_lines, updates)
    write_config(new_config)
    output = run_simulation()
    return parse_output_avg_latency(output)


def compute_mape(predicted, reference):
    return abs(predicted - reference) / reference * 100.0


def run_sweep_sp(sampling_period, config, ca_reference):
    config_lines = read_config_template()
    updates = {
        'sampling_period': str(sampling_period), 
        'decay_factor': f"{FIXED_DECAY_FACTOR:.6f}",
        'inj_is_burst': str(config["inj_is_burst"]),
        'alpha': f"{config['params']['alpha']:.6f}",
        'beta': f"{config['params']['beta']:.6f}"
    }
    config_lines = update_config(config_lines, updates)
    
    mape_list = []
    for i in range(config["num_points"]):
        inj_rate = config["inj_start"] + i * config["inj_step"]
        latency = run_injection_sweep(config_lines, inj_rate, config["params"]["alpha"], 
                                      config["params"]["beta"], config["inj_is_burst"])
        if latency is not None and inj_rate in ca_reference:
            mape_list.append(compute_mape(latency, ca_reference[inj_rate]))
    
    write_config(read_config_template())
    return np.mean(mape_list) if mape_list else float('inf')


def run_sweep_df(decay_factor, config, ca_reference):
    config_lines = read_config_template()
    updates = {
        'sampling_period': str(FIXED_SAMPLING_PERIOD), 
        'decay_factor': f"{decay_factor:.6f}",
        'inj_is_burst': str(config["inj_is_burst"]),
        'alpha': f"{config['params']['alpha']:.6f}",
        'beta': f"{config['params']['beta']:.6f}"
    }
    config_lines = update_config(config_lines, updates)
    
    mape_list = []
    for i in range(config["num_points"]):
        inj_rate = config["inj_start"] + i * config["inj_step"]
        latency = run_injection_sweep(config_lines, inj_rate, config["params"]["alpha"],
                                      config["params"]["beta"], config["inj_is_burst"])
        if latency is not None and inj_rate in ca_reference:
            mape_list.append(compute_mape(latency, ca_reference[inj_rate]))
    
    write_config(read_config_template())
    return np.mean(mape_list) if mape_list else float('inf')

def load_results():
    filepath = os.path.join(RESULTS_DIR, "sensitivity_results.dat")
    sp_s, df_s, sp_b, df_b = [], [], [], []
    with open(filepath, 'r') as f:
        section = None
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                if "Steady - sampling_period" in line:
                    section = "sp_s"
                elif "Steady - decay_factor" in line:
                    section = "df_s"
                elif "Mid Burst - sampling_period" in line:
                    section = "sp_b"
                elif "Mid Burst - decay_factor" in line:
                    section = "df_b"
                continue
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    if section == "sp_s":
                        sp_s.append((int(parts[0]), float(parts[1])))
                    elif section == "df_s":
                        df_s.append((float(parts[0]), float(parts[1])))
                    elif section == "sp_b":
                        sp_b.append((int(parts[0]), float(parts[1])))
                    elif section == "df_b":
                        df_b.append((float(parts[0]), float(parts[1])))
    return sp_s, df_s, sp_b, df_b


def plot_curves(sp_s, df_s, sp_b, df_b):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))
    
    ax1.semilogx([r[0] for r in df_s], [r[1] for r in df_s], 's-', color='#2E86AB', 
                 linewidth=1, markersize=3, markerfacecolor='white', markeredgewidth=1.5, label='Steady')
    ax1.semilogx([r[0] for r in df_b], [r[1] for r in df_b], '^-', color='#A23B72',
                 linewidth=1, markersize=3, markerfacecolor='white', markeredgewidth=1.5, label='Mid Burst')
    ax1.set_xlabel('Decay Factor', fontsize=7)
    ax1.set_ylabel('MAPE (%)', fontsize=7)
    ax1.grid(True, alpha=0.3, linestyle='--', which='both')
    ax1.legend(fontsize=6, loc='upper right')
    
    ax2.plot([r[0] for r in sp_s], [r[1] for r in sp_s], 'o-', color='#2E86AB',
             linewidth=1, markersize=3, markerfacecolor='white', markeredgewidth=1.5, label='Steady')
    ax2.plot([r[0] for r in sp_b], [r[1] for r in sp_b], '^-', color='#A23B72',
             linewidth=1, markersize=3, markerfacecolor='white', markeredgewidth=1.5, label='Mid Burst')
    ax2.set_xlabel('Sliding Window Length (Cycles)', fontsize=7)
    ax2.set_ylabel('MAPE (%)', fontsize=7)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=6, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sensitivity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(RESULTS_DIR, 'sensitivity_analysis.svg'), bbox_inches='tight')


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if args.sweep:
        ca_steady = load_ca_reference(os.path.join(RESULTS_DIR, STEADY_CONFIG["ca_file"]))
        ca_burst = load_ca_reference(os.path.join(RESULTS_DIR, MID_BURST_CONFIG["ca_file"]))
        print(f"Loaded {len(ca_steady)} steady, {len(ca_burst)} mid-burst CA points")
        
        print("\nSteady: sampling period sweep")
        sp_s = []
        for sp in SAMPLING_PERIODS:
            mape = run_sweep_sp(sp, STEADY_CONFIG, ca_steady)
            sp_s.append((sp, mape))
            print(f"  {sp}: {mape:.2f}%")
        
        print("\nSteady: decay factor sweep")
        df_s = []
        for df in DECAY_FACTORS:
            mape = run_sweep_df(df, STEADY_CONFIG, ca_steady)
            df_s.append((df, mape))
            print(f"  {df:.4f}: {mape:.2f}%")
        
        print("\nMid Burst: sampling period sweep")
        sp_b = []
        for sp in SAMPLING_PERIODS:
            mape = run_sweep_sp(sp, MID_BURST_CONFIG, ca_burst)
            sp_b.append((sp, mape))
            print(f"  {sp}: {mape:.2f}%")
        
        print("\nMid Burst: decay factor sweep")
        df_b = []
        for df in DECAY_FACTORS:
            mape = run_sweep_df(df, MID_BURST_CONFIG, ca_burst)
            df_b.append((df, mape))
            print(f"  {df:.4f}: {mape:.2f}%")
        
        best_sp_s = min(sp_s, key=lambda x: x[1])
        best_df_s = min(df_s, key=lambda x: x[1])
        best_sp_b = min(sp_b, key=lambda x: x[1])
        best_df_b = min(df_b, key=lambda x: x[1])
        print(f"\nBest: Steady sp={best_sp_s[0]} ({best_sp_s[1]:.2f}%), df={best_df_s[0]:.4f} ({best_df_s[1]:.2f}%)")
        print(f"Best: Mid Burst sp={best_sp_b[0]} ({best_sp_b[1]:.2f}%), df={best_df_b[0]:.4f} ({best_df_b[1]:.2f}%)")
    
    if args.plot:
        sp_s, df_s, sp_b, df_b = load_results()
        plot_curves(sp_s, df_s, sp_b, df_b)
        print(f"Plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()