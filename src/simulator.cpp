#include "simulator.hpp"

#include <map>

static BaseModule* g_noc_ptr = NULL;

int GetSimTime() {
    if (!g_noc_ptr) {
        return 0;
    }
    return g_noc_ptr->getTrafficManager()->getTime();
}

class Stats;
Stats* GetStats(const std::string& name) {
    if (!g_noc_ptr || !g_noc_ptr->getTrafficManager()) {
        return NULL;
    }
    return g_noc_ptr->getTrafficManager()->getStats(name);
}

/* printing activity factor*/
bool gPrintActivity;

int gK;  // radix
int gN;  // dimension
int gC;  // concentration

int gNodes;

// generate nocviewer trace
bool gTrace;

ostream* gWatchOut = NULL;

Simulator::Simulator(uint64_t max_sim_cycle, const Config& cfg)
    : cfg(cfg),
      logger(cfg.log_dir + "/" + cfg.log_file, cfg.enable_log),
      _max_sim_cycle(max_sim_cycle),
      cur_tick(0),
      finished_cores_cnt(0) {
    logger.log("Initializing FlexNPUSim Simulator...");
    std::map<int, std::pair<int, string>> valid_core_configs = {
        {1, {1, "./external/Booksim2/src/examples/mesh22_lat"}},
        {2, {1, "./external/Booksim2/src/examples/mesh22_lat"}},
        {4, {2, "./external/Booksim2/src/examples/mesh33_lat"}},
        {8, {2, "./external/Booksim2/src/examples/mesh55_lat"}},
        {16, {4, "./external/Booksim2/src/examples/mesh55_lat"}},
        {32, {8, "./external/Booksim2/src/examples/mesh1010_lat"}},
        {64, {16, "./external/Booksim2/src/examples/mesh1010_lat"}}};

    auto config_it = valid_core_configs.find(cfg.core_num);
    if (config_it == valid_core_configs.end()) {
        std::string error_msg = "Invalid core number: " + std::to_string(cfg.core_num) +
                                ". Supported core numbers are: 1, 2, 4, 8, 16, 32, 64";
        std::cerr << "Error: " << error_msg << std::endl;
        throw std::runtime_error(error_msg);
    }
    std::vector<double> freqs = {cfg.npu_freq, cfg.ddr_freq, cfg.noc_freq};

    compute_base_tick_and_multipliers(freqs, base_tick_ps, ticks_per_period);
    uint64_t core_ticks = ticks_per_period[0];
    uint64_t ddr_ticks = ticks_per_period[1];
    uint64_t noc_ticks = ticks_per_period[2];
    logger.log("Base tick = %llu ps; core_period = %llu ticks; ddr_period = %llu ticks, noc_period = %llu ticks",
               (unsigned long long)base_tick_ps, (unsigned long long)core_ticks, (unsigned long long)ddr_ticks,
               (unsigned long long)noc_ticks);

    // Initialize NPU Cores
    int num_cores = cfg.core_num;
    cores.reserve(num_cores);
    std::cout << "Creating " << num_cores << " hybrid cores..." << std::endl;
    for (int i = 0; i < num_cores; i++) {
        cores.emplace_back(std::make_unique<DetailNPUCore>(this, i, cfg, logger, core_ticks));
    }
    std::cout << "Hybrid Cores initialized!" << std::endl;

    // Initialize DDR, Note memory controlers (MCs) are located at the bottom side in the NOC
    // and are connected to the DDR
    int num_ddr = config_it->second.first;
    memory_controllers.reserve(num_ddr);
    if (cfg.ddr_mode == 1) {
        logger.log("Creating Analytical DDR...");
        std::cout << "Creating Analytical DDR..." << std::endl;
        for (int i = 0; i < num_ddr; i++) {
            std::cout << "Creating DDR " << i << "..." << std::endl;
            memory_controllers.emplace_back(std::make_unique<AnalyticalDDR>(this, i, cfg, logger, ddr_ticks));
            std::cout << "DDR " << i << " created" << std::endl;
        }
    } else {
        logger.log("Creating Detailed DDR...");
        std::cout << "Creating Detailed DDR..." << std::endl;
        for (int i = 0; i < num_ddr; i++) {
            memory_controllers.emplace_back(std::make_unique<DetailDDR>(this, i, cfg, logger, ddr_ticks));
        }
    }

    std::vector<BaseModule*> core_ptrs_raw;
    core_ptrs_raw.reserve(cores.size());
    for (const auto& core_ptr : cores) {
        core_ptrs_raw.push_back(core_ptr.get());
    }

    std::vector<BaseModule*> mc_ptrs_raw;
    mc_ptrs_raw.reserve(memory_controllers.size());
    for (const auto& mc_ptr : memory_controllers) {
        mc_ptrs_raw.push_back(mc_ptr.get());
    }

    // Initialize NOC
    if (cfg.noc_mode) {
        logger.log("Creating Analytical NOC...");
        std::cout << "Creating Analytical NOC..." << std::endl;
        noc = std::make_unique<AnalyticalNOC>(this, cfg, logger, noc_ticks, core_ptrs_raw, mc_ptrs_raw);
    } else {
        logger.log("Creating Detailed NOC...");
        std::cout << "Creating Detailed NOC..." << std::endl;
        BookSimConfig booksim_cfg;
        std::string booksim_cfg_path = config_it->second.second;
        char* argv[] = {
            const_cast<char*>("program"),                // argv[0]
            const_cast<char*>(booksim_cfg_path.c_str())  // argv[1]
        };
        if (!ParseArgs(&booksim_cfg, 2, argv)) {
            std::cerr << "Failed to parse BookSim config file: " << booksim_cfg_path << std::endl;
            exit(1);
        }

        InitializeRoutingMap(booksim_cfg);
        gPrintActivity = booksim_cfg.GetInt("print_activity") > 0;
        gTrace = booksim_cfg.GetInt("viewer_trace") > 0;
        noc = std::make_unique<DetailNoC>(this, booksim_cfg, cfg, logger, noc_ticks, core_ptrs_raw, mc_ptrs_raw);

        g_noc_ptr = noc.get();
    }

    logger.log("Simulator Initialized");
    std::cout << "Simulator Initialized" << std::endl;
    std::cout << "max sim cycle lits: " << cfg.max_sim_cycle << std::endl;
}

Simulator::~Simulator() {}

void Simulator::run() {
    logger.log("================================================");
    logger.log("Starting Simulation...\n");
    std::cout << "Starting Simulation..." << std::endl;
    sim_start_time = std::chrono::high_resolution_clock::now();

    bool early_termination = false;

    uint64_t max_sim_tick = _max_sim_cycle * ticks_per_period[0];

    // add initial events for all components
    for (auto& core : cores) {
        core->scheduleNextEvent();
    }
    for (auto& ddr : memory_controllers) {
        ddr->scheduleNextEvent();
    }
    noc->scheduleNextEvent();

    while (cur_tick < max_sim_tick) {
        Event next_event = event_queue.top();
        event_queue.pop();

        cur_tick = next_event.tick;

        next_event.callback();
        if (isEarlyFinished()) {
            early_termination = true;
            break;
        }
    }

    sim_end_time = std::chrono::high_resolution_clock::now();
    sim_time_us = std::chrono::duration_cast<std::chrono::microseconds>(sim_end_time - sim_start_time).count();

    if (!early_termination) {
        logger.log("Simulation reached maximum cycle limit %d", _max_sim_cycle);
    } else {
        logger.log("Simulation finished early at cycle %d", cores[0]->getCycle());
    }

    print_stats();
}

void Simulator::initSyncBarrier(int sync_id, int total_sync_num) {
    auto it = _sync_states.find(sync_id);
    if (it == _sync_states.end()) {
        _sync_states[sync_id] = SyncEntry{0, 1, total_sync_num};
    } else {
        _sync_states[sync_id].sync_cnt++;
    }
}

void Simulator::sendSyncAck(int sync_id) {
    auto it = _sync_states.find(sync_id);
    if (it != _sync_states.end()) {
        it->second.currrent_acked_cnt++;
        // check if all cores have reached the sync point
        if (it->second.currrent_acked_cnt == it->second.total_sync_num) {
            _sync_states.erase(it);  // clear sync entry
            logger.log("[Tick %d] Simulator: All cores have reached sync point (sync_id=%d), erase sync entry",
                       getTick(), sync_id);
        }
    } else {
        // do nothing if the sync point has been cleared
    }
}

int Simulator::getSyncCnt(int sync_id) {
    auto it = _sync_states.find(sync_id);
    if (it != _sync_states.end()) {
        return it->second.sync_cnt;
    }
    return 0;
}

void Simulator::schedule(const Event& event) {
    event_queue.push(event);
}

void Simulator::print_stats() {
    double sim_time_s = sim_time_us / 1e6;
    uint64_t total_inst_num = 0;
    uint64_t max_sim_cycles = 0;
    uint64_t total_active_cycles = 0;

    for (auto& core : cores) {
        uint64_t c_cycle = core->getCycle();
        if (c_cycle > max_sim_cycles) {
            max_sim_cycles = c_cycle;
        }
        total_active_cycles += c_cycle;
        if (auto* detail_core = dynamic_cast<DetailNPUCore*>(core.get())) {
            total_inst_num += detail_core->getPC();
        }
    }

    double ips = total_inst_num / (sim_time_s);

    double core_pct = profile.core_time_ns / (sim_time_us * 1000) * 100.0;
    double noc_pct = profile.noc_time_ns / (sim_time_us * 1000) * 100.0;
    double ddr_pct = profile.ddr_time_ns / (sim_time_us * 1000) * 100.0;
    double other_sim_time =
        sim_time_us - (profile.core_time_ns / 1000) - (profile.noc_time_ns / 1000) - (profile.ddr_time_ns / 1000);
    double other_pct = 100.0 - core_pct - noc_pct - ddr_pct;

    logger.log("================================================");
    logger.log("Simulation Finished!");
    logger.log("Simulation Statistics:");
    // print noc & ddr mode
    logger.log("  DDR Mode: %s", cfg.ddr_mode == 0 ? "Detailed" : "Analytical");
    logger.log("  NoC Mode: %s", cfg.noc_mode == 0 ? "Detailed" : "Analytical");
    if (cfg.noc_mode == 1) {
        if (cfg.an_noc_method == 0) {
            logger.log("  NoC Method: %s", "Only Hop-based");
        } else if (cfg.an_noc_method == 1) {
            logger.log("  NoC Method: %s", "queue-based with fixed parameters");
        } else if (cfg.an_noc_method == 2) {
            logger.log("  NoC Method: %s", "queue-based with adaptive parameters");
        }
    }

    logger.log("  Simulation time: %.6f s", sim_time_s);
    logger.log("  Maximum cycle limit: %d", _max_sim_cycle);
    logger.log("  Early termination: %s", isEarlyFinished() ? "Yes" : "No");
    logger.log("  Total cycles (Max): %lu", max_sim_cycles);
    logger.log("  Total simulated instructions: %lu", total_inst_num);
    logger.log("  Simulated IPS: %.2f", ips);

    std::cout << "================================================" << std::endl;
    std::cout << "Simulation Statistics:" << std::endl;
    std::cout << "  Early termination: " << (isEarlyFinished() ? "Yes" : "No") << std::endl;
    std::cout << "  DDR Mode: " << (cfg.ddr_mode == 0 ? "Detailed" : "Analytical") << std::endl;
    std::cout << "  NoC Mode: " << (cfg.noc_mode == 0 ? "Detailed" : "Analytical") << std::endl;
    if (cfg.noc_mode == 1) {
        if (cfg.an_noc_method == 0) {
            std::cout << "  Method: Only Hop-based" << std::endl;
        } else if (cfg.an_noc_method == 1) {
            std::cout << "  Method: queue-based + fixed parameters" << std::endl;
        } else if (cfg.an_noc_method == 2) {
            std::cout << "  Method: queue-based + adaptive parameters" << std::endl;
        }
    }

    std::cout << "  Maximum cycle limit: " << _max_sim_cycle << std::endl;

    std::cout << "  Total cycles: " << max_sim_cycles << std::endl;
    std::cout << "  Total instructions: " << total_inst_num << std::endl;
    std::cout << "  IPS: " << ips << std::endl;

    std::cout << "  Simulation time: " << sim_time_s << " s" << std::endl;
    std::cout << "  Simulation Time Breakdown:" << std::endl;
    std::cout << "  Core Time: " << profile.core_time_ns / 1e9 << " s (" << core_pct << "%)" << std::endl;
    std::cout << "  NoC Time: " << profile.noc_time_ns / 1e9 << " s (" << noc_pct << "%)" << std::endl;
    std::cout << "  DDR Time: " << profile.ddr_time_ns / 1e9 << " s (" << ddr_pct << "%)" << std::endl;
    std::cout << "  Other Time: " << other_sim_time / 1e6 << " s (" << other_pct << "%)" << std::endl;
    //}
}