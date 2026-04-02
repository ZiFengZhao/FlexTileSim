import re
from collections import defaultdict

MODE_MAP = {
    "Flex-Full": (0,0),
    "Flex-Hybrid1": (0,1),
    "Flex-Hybrid2": (1,0),
    "Flex-Fast": (1,1),
    "FPGA": (-1,-1),
}

FPGA_RESULTS = {
    "alexnet": 6110549,
    "resnet18": 10473274,
    "vgg16": 83972155,
    "bert_base": 106012563,
    "vit_large": 151322832
}

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

def calculate_relative_errors(exec_time_dict, baseline_config=(0,0), cores=32):
 
    error_dict = defaultdict(dict)
    models = list(exec_time_dict.keys())
    
    for model in models:
        if cores in exec_time_dict[model]:
            model_data = exec_time_dict[model][cores]
            
            if baseline_config in model_data:
                baseline_time = model_data[baseline_config]
                
                for config, time in model_data.items():
                    if config != baseline_config: 
                        relative_error = ((time - baseline_time) / baseline_time) * 100
                        error_dict[model][config] = relative_error
                    else:
                        error_dict[model][config] = 0.0
    
    return error_dict

if __name__ == "__main__":
        
    _, cycle_dict = parse_sweep_results("../../sweep_results_20260321_1132.txt")
    # add fpga results
    for model in cycle_dict:
        cycle_dict[model][4][(-1, -1)] = FPGA_RESULTS[model]
    
    error_dict = calculate_relative_errors(cycle_dict, baseline_config=(-1,-1), cores=4)

    print("\n=== Detailed Relative Errors ===")
    for model in error_dict:
        print(f"\n{model}:")
        for config, error in error_dict[model].items():
            config_name = next(name for name, val in MODE_MAP.items() if val == config)
            time = cycle_dict[model][4][config]
            if config == (-1,-1):
                error = 0
            print(f"  {config_name}: {time / 1e6:.2f} M Cycles ({error:+.2f}%)")
    
    