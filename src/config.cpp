#include "config.hpp"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

static std::string trim(std::string s) {
    while (!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
    while (!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    return s;
}

Config Config::from_file(const std::string& path) {
    Config c;
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Cannot open config: " + path);

    std::string line;
    while (std::getline(ifs, line)) {
        auto pos = line.find('#');
        if (pos != std::string::npos) line = line.substr(0, pos);
        line = trim(line);
        if (line.empty()) continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = trim(line.substr(0, eq));
        std::string v = trim(line.substr(eq + 1));
        // core config
        if (k == "core_num")
            c.core_num = std::stoi(v);
        else if (k == "systolic_array_size")
            c.systolic_array_size = std::stoi(v);
        else if (k == "ce_cycle_factor")
            c.ce_cycle_factor = std::stoi(v);
        else if (k == "ce_cycle_bias")
            c.ce_cycle_bias = std::stoi(v);
        else if (k == "dma_bus_width")
            c.dma_bus_width = std::stoi(v);
        else if (k == "dma_queue_depth")
            c.dma_queue_depth = std::stoi(v);
        else if (k == "ce_queue_depth")
            c.ce_queue_depth = std::stoi(v);
        else if (k == "npu_freq")
            c.npu_freq = std::stod(v);

        // noc config
        else if (k == "noc_mode")
            c.noc_mode = std::stoi(v);
        else if (k == "noc_freq")
            c.noc_freq = std::stod(v);
        else if (k == "noc_link_width")
            c.noc_link_width = std::stoi(v);
        else if (k == "noc_link_latency")
            c.noc_link_latency = std::stoi(v);
        else if (k == "noc_router_latency")
            c.noc_router_latency = std::stoi(v);
        else if (k == "noc_request_queue_depth")
            c.noc_request_queue_depth = std::stoi(v);
        else if (k == "an_noc_method")
            c.an_noc_method = std::stoi(v);
        else if (k == "noc_sampling_period")
            c.noc_sampling_period = std::stoi(v);
        else if (k == "noc_num_saved_samples")
            c.noc_num_saved_samples = std::stoi(v);
        else if (k == "noc_traffic_weight_mode")
            c.noc_traffic_weight_mode = std::stoi(v);
        else if (k == "noc_traffic_weight_factor")
            c.noc_traffic_weight_factor = std::stod(v);

        // ddr config
        else if (k == "ddr_mode")
            c.ddr_mode = std::stoi(v);
        else if (k == "ddr_freq")
            c.ddr_freq = std::stod(v);
        else if (k == "ddr_bandwidth")
            c.ddr_bandwidth = std::stod(v);
        else if (k == "tBASE")
            c.tBASE = std::stoi(v);
        else if (k == "tCAS")
            c.tCAS = std::stoi(v);
        else if (k == "tRCD")
            c.tRCD = std::stoi(v);
        else if (k == "tRP")
            c.tRP = std::stoi(v);
        else if (k == "tBURST")
            c.tBURST = std::stoi(v);
        else if (k == "BL")
            c.BL = std::stoi(v);
        else if (k == "ddr_bus_width")
            c.ddr_bus_width = std::stoi(v);
        else if (k == "ddr_request_queue_depth")
            c.ddr_request_queue_depth = std::stoi(v);
        else if (k == "ddr_max_outstanding")
            c.ddr_max_outstanding = std::stoi(v);
        else if (k == "max_sim_cycle")
            c.max_sim_cycle = std::stoi(v);
        else if (k == "dramsim_cfg_path")
            c.dramsim_cfg_path = v;

        // sim config
        else if (k == "inst_file")
            c.inst_file = v;
        else if (k == "log_file")
            c.log_file = v;
        else if (k == "log_dir")
            c.log_dir = v;
        else if (k == "enable_log")
            c.enable_log = std::stoi(v);
        else if (k == "enable_inst_trace")
            c.enable_inst_trace = std::stoi(v);
        /*Add more configuration options here*/
    }
    return c;
}