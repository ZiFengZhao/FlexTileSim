import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


ONNXIM_TILE_SIMTIME = {
    "alexnet": 472.43,
    "resnet18": 975.14,
    "vgg16": 7432.56,
    "bert_base": 4203.35,
    "vit_large": 11056.02
}

MODE_MAP = {
    "Flex-Full": (0,0),
    "Flex-Hybrid1": (0,1),
    "Flex-Hybrid2": (1,0),
    "Flex-Fast": (1,1)
}

BAR_COLORS = {
    "ONNXim": "#D6E3CC", #"#EAF3E2","#D0D2D4"
    "Flex-Full": "#A2D0A4", #"#B4DEB6",#A6A8AC
    "Flex-Hybrid1": "#7BC6BE", #"#7BC6BE",#6AACD6
    "Flex-Hybrid2": "#439CC4", #"#439CC4",#0868A6
    "Flex-Fast": "#085399",
}

LINE_COLOR = "#A5678E"#AD1A60"#"#2A4C71"

def parse_sweep_results(filepath):
    """
    Returns
    -------
    sim_time_dict : dict
        model -> cores -> (noc, ddr) -> simulation_time

    cycle_dict : dict
        model -> cores -> (noc, ddr) -> total_cycles
    """

    sim_time_dict = defaultdict(lambda: defaultdict(dict))
    cycle_dict = defaultdict(lambda: defaultdict(dict))

    header_pattern = re.compile(
        r'\[([a-zA-Z0-9_]+)\s+cores=(\d+)\s+noc=(\d+)\s+ddr=(\d+)\]'
    )

    sim_time_pattern = re.compile(r'Simulation time:\s*([\d\.]+)')
    cycle_pattern = re.compile(r'Total cycles:\s*(\d+)')

    current_model = None
    current_cores = None
    current_noc = None
    current_ddr = None

    with open(filepath, "r") as f:
        for line in f:

            header_match = header_pattern.search(line)
            if header_match:
                current_model = header_match.group(1)
                current_cores = int(header_match.group(2))
                current_noc = int(header_match.group(3))
                current_ddr = int(header_match.group(4))
                continue

            sim_match = sim_time_pattern.search(line)
            if sim_match and current_model is not None:
                sim_time = float(sim_match.group(1))

                sim_time_dict[current_model][current_cores][
                    (current_noc, current_ddr)
                ] = sim_time

            cycle_match = cycle_pattern.search(line)
            if cycle_match and current_model is not None:
                cycles = int(cycle_match.group(1))

                cycle_dict[current_model][current_cores][
                    (current_noc, current_ddr)
                ] = cycles
    
    return sim_time_dict, cycle_dict

def plot_simtime_speedup(sim_time_dict,
                         bar_width=0.6,
                         fig_width_per_model=4,
                         fig_height=4, fig_name="sim_time_comparison"):
    models = list(ONNXIM_TILE_SIMTIME.keys())
    num_models = len(models)

    adjusted_width_per_model = fig_width_per_model * 0.9  
    fig, axes = plt.subplots(1, num_models, figsize=(num_models * adjusted_width_per_model, fig_height))
    plt.subplots_adjust(wspace=0.3)  
    if num_models == 1:
        axes = [axes]
    
    for idx, model in enumerate(models):
        ax = axes[idx]

        tile_num = 32 # compare 32-core cfg
        model_data = sim_time_dict[model][tile_num]

        times = []
        labels = []

        onnx_time = ONNXIM_TILE_SIMTIME[model]
        times.append(onnx_time)
        labels.append("ONNXim")

        for mode, key in MODE_MAP.items():
            if key in model_data:
                times.append(model_data[key])
                labels.append(mode)

        x= np.arange(len(times))

        colors = [BAR_COLORS[l] for l in labels]

        bars = ax.bar(x, times, width=bar_width, color=colors)

        ax.set_title(model)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30)
        ax.set_ylabel("Simulation Time (s)")
        ax.set_yscale("log")

        ax2 = ax.twinx()

        speedups = [onnx_time / t for t in times]

        ax2.plot(x, speedups, marker='o', linewidth=2, color=LINE_COLOR)

        ax2.set_ylabel("Speedup vs ONNXim")

        for i,s in enumerate(speedups):
            ax2.text(i, s, f"{s:.1f}x", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{fig_name}.png", dpi=300)
    print(f"Figure saved to {fig_name}.png")
    plt.savefig(f"{fig_name}.svg", format="svg")
    print(f"Figure saved to {fig_name}.svg")
    
if __name__ == "__main__":
    sim_time_dict, _ = parse_sweep_results("../../sweep_results_20260317_1031.txt")

    plot_simtime_speedup(sim_time_dict, bar_width=0.5,
                         fig_width_per_model=4, fig_height=3.6, fig_name="speedup_comparison")