#ifndef __CONFIG_HPP__
#define __CONFIG_HPP__

#include <cstddef>
#include <string>

struct Config {
    // NPU Core Config
    int core_num = 2;  // number of NPU cores
    int systolic_array_size = 16;
    int ce_cycle_factor = 1;
    int ce_cycle_bias = 2;
    int dma_bus_width = 16;  // bytes
    int inst_buffer_depth = 4;
    int dma_queue_depth = 2;
    int ce_queue_depth = 2;
    double npu_freq = 1;  // NPU Core Running Frequency (GHz)

    // NoC Config
    int noc_mode = 1;            // 1: analytic 0: detail
    double noc_freq = 1;         // NoC Running Frequency (GHz)
    int noc_link_width = 16;     // bytes
    int noc_link_latency = 1;    // cycles
    int noc_router_latency = 2;  // cycles
    int noc_request_queue_depth = 4;
    // NoC queueing parameters
    int an_noc_method = 0;                   // 2: traffic-aware 1: queueing with fixed parameters 0: hop-based
    int noc_sampling_period = 1000;          // sampling period (cycles)
    int noc_num_saved_samples = 5;           // maximum number of samples
    int noc_traffic_weight_mode = 0;         // 0: linear 1: exponential
    double noc_traffic_weight_factor = 0.5;  // weight factor for traffic-aware queueing

    // DDR Config
    int ddr_mode = 1;  // 1: analytic 0: detail
    std::string dramsim_cfg_path = "";
    double ddr_freq = 2.13;  // GHz
    double ddr_bandwidth = 20.0;
    int tCAS = 15;   // CAS Latency (DDR cycle)
    int tBASE = 50;  // tBASE (DDR cycle)
    int tRCD = 15;   // RAS to CAS Latency (DDR cycle)
    int tRP = 15;    // Precharge Latency (DDR cycle)
    int tBURST = 4;  // Burst Latency (ns)
    int BL = 8;      // DDR Burst Length (通常为8bits)
    int ddr_bus_width = 64;
    int ddr_request_queue_depth = 8;
    int ddr_max_outstanding = 32;

    // Simulation Config
    std::string inst_file = "";
    std::string log_dir = "./log";
    std::string log_file = "example.log";
    bool enable_log = false;
    bool enable_inst_trace = true;
    int max_sim_cycle = 100000;

    static Config from_file(const std::string& path);
};

#endif  // __CONFIG_HPP__