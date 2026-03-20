#include <cassert>
#include <fstream>
#include <iostream>

#include "config.hpp"
#include "simulator.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << std::endl;
        return 1;
    }
    std::string cfg_file = argv[1];

    Config cfg = Config::from_file(cfg_file);
    int max_sim_cycle = cfg.max_sim_cycle;
    std::cout << "Starting to create simulator..." << std::endl;
    Simulator simulator(max_sim_cycle, cfg);

    // run simulation
    simulator.run();
}