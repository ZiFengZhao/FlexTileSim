# FlexTileSim

FlexTileSim is a performance simulation framework for fast and flexible system-level evaluation of tiled DNN accelerators (multi-core NPUs). It consists of a front-end for workload preparation and a back-end for architectural simulation.

<p align="center">
  <img src="images/flextilesim.png" width="450">
</p>

<p align="center">
  <em>FlexTileSim Framework Overview</em>
</p>

The front-end handles DNN partitioning, operator mapping, and code generation. The back-end models compute tiles interconnected via a NoC and shared off-chip memory.

FlexTileSim supports configurable architectural parameters and multiple simulation fidelity levels, enabling efficient design space exploration.

**Key capabilities:**
- Multi-fidelity system simulation  
- Online traffic-aware analytical NoC modeling  

---

## 1. Key Features

### **1.1 Multi-Fidelity Simulation**

Flexible fidelity configuration is supported for major subsystems:

**Compute Tile**
- 4-stage pipeline: Fetch, Decode, Issue, Execute  
- DMA engine  
- Compute engine (e.g., systolic array, vector unit)  
- Scoreboard for dependency tracking  

**NoC**
- Detailed mode: Booksim2 (cycle-accurate)  
- Analytical mode: traffic-aware G/G/1 model  

**DRAM**
- Detailed mode: DRAMSim3  
- Analytical mode: configurable latency/bandwidth  

---

### **1.2 Traffic-aware NoC Analytical Model**

DNN workloads exhibit dynamic traffic patterns that violate steady-state assumptions (e.g., Poisson arrival).

FlexTileSim introduces an online analytical model that:
- Continuously samples network traffic  
- Dynamically updates arrival and service parameters  

The model is integrated into an event-driven simulator, enabling closed-loop system simulation with adaptive NoC behavior.

---

## 2. Repository Structure

```text
FlexTileSim/
├── benchmarks/   # Front-end tools and workloads
├── config/       # Simulation configs
├── external/     # DRAMSim3, Booksim2
├── include/      # Headers
├── src/          # Source code
├── log/          # Execution logs
├── Env.cfg       # Environment setup
├── Makefile

## 3. Environment Setup
The back-end is developed in C++17. GCC compiler is required.
- gcc >= 9.3.1
- g++ >= 9.3.1

Python is required for front-end workload preparation.
- Python >= 3.8.5
- matplotlib >= 3.3.4
- numpy >= 1.19.5

FlexTileSim includes modified versions of ```DRAMSim3``` and ```Booksim2```
These modules will be compiled automatically during the build process.

## 4. Quick Start

FlexTileSim has been tested on ```CentOS 7```. Other Linux distributions may work but have not been fully tested.

### Step 1: Source environment

Every time a new terminal is opened, initialize the environment:

```bash
source Env.cfg
```

### Step 2: Build simulator

Compile the simulator:

```bash
make build
```

The simulator binary will be generated:
```build/npu_sim```

### Step 3: Run example simulation

Run the built-in example:

```bash
make run
```

This command executes a precompiled AlexNet Conv3 layer on a 2-tile system configuration, which serves as a quick sanity test to ensure the simulator is functioning correctly.

### Other commands

Build Booksim only:

```base
make booksim
```

Clean build files:

```bash
make clean
```

or clean all library dependencies

```bash
make distclean
```

## 5. Input and Output

### 5.1 Simulation Input

The simulator requires a configuration file describing the system architecture and simulation parameters.

Example configuration:

```text
config/alexnet_conv3_cfg.txt
```

The configuration specifies:

- Core architecture parameters

- NoC configuration

- DRAM configuration

- Simulation parameters

- Workload instruction file

### 5.2 Simulation Output

After simulation completes, the terminal prints summary statistics as follows:

```text
Simulation Statistics:

Simulation time: 3.88 s
Maximum cycle limit: 1000000
Early termination: Yes
Total cycles: 522649
Total instructions: 20882
IPS: 5388.83
```

Where IPS (Instructions Per Second) reflects the simulation speed across different modeling fidelity and DNN workloads

### Execution Trace (Optional)

Simulation traces can be enabled in the configuration file:

```text
enable_log = 1
log_dir = ./log
log_file = log.log
```

Example trace output:

```text
Initializing FlexNPUSim Simulator...

Creating Detailed Cores...
Loaded 8 instructions for core 0
Loaded 8 instructions for core 1

Creating Analytical DDR...
DDR Frequency: 3.2 GHz

Creating Detailed NoC...
Mesh Size: 2 x 2

Starting Simulation...
[Tick 0-CoreCycle 0] Core 0 Fetch ...
[Tick 100-CoreCycle 1] Core 0 Decode ...
```

These traces provide fine-grained visibility of simulator behavior, including:

- instruction execution

- NoC activity

- DRAM accesses

- cycle-level events

## 6. Simulation Configuration

FlexTileSim uses a configuration file to specify architectural parameters.
The configuration is divided into four sections.

### Compute Tile

Key parameters:

- core_num: number of compute tiles
- npu_freq: Compute tile frequency (GHz)
- systolic_array_size: dimension of the systolic array (e.g., 16x16)

### NoC

- noc_mode: model fidelity; 0 -> detailed NoC (BookSim2); 1 -> analytical model
- noc_freq: NoC frequency (GHz)

### DRAM

- ddr_mode: model fidelity; 0 -> detailed NoC (DRAMSim3); 1 -> analytical model
- ddr_bandwidth:  DRAM bandwidth (GB/s); only valid in analytical mode

### Simulation Parameters

Important options:

- max_sim_cycle: maximum allowed simulation cycles
- inst_file: workload instruction binary
- enable_log: enable detailed execution traces

## 7. Reproducing Manuscript Results

FlexTileSim provides one-click scripts to reproduce the NoC synthetic traffic experiments presented in the manuscript.

### 7.1 NoC Synthetic Traffic Experiments

These experiments evaluate the analytical NoC model under various traffic patterns and parameter configurations, generating Figure 12 and Figure 14 in the manuscript.

#### Steps to reproduce

```bash
cd exp/synthetic_traffic
./run_experiments.sh
```

### **7.2 Full-System Simulation**

This set of experiments evaluates the accuracy and efficiency of FlexTileSim under full-system DNN workloads, corresponding to Table V and Fig. 10 in the manuscript.

---

#### **Accuracy Validation (Table V)**

To reproduce the system-level accuracy validation results:

```bash
cd exp/accuracy
python3 plt_sys_accuracy.py
```

This script processes the simulation results and generates the accuracy comparison between analytical and detailed models.

#### **Large-Scale Simulation Efficiency (Fig. 10)**

To reproduce the large-scale system simulation speedup results:

```bash
cd exp/speedup
python3 plt_speedup.py
```

This script evaluates the performance improvement of different simulation fidelity modes over cycle-accurate simulation across different system scales.

Note that the above Python scripts operate on pre-generated sweep results. These results are included in the repository and can be directly used to reproduce the figures without rerunning simulations.

For users interested in regenerating the full set of experimental data, a helper script is provided at the project root:

```bash
./run_large_scale_sweep.sh
```

This script explores multiple DNN workloads and core-count configurations to produce the complete dataset used in the evaluation. Depending on the system configuration and selected modeling fidelity, execution time may vary.
