#ifndef __SIMULATOR_HPP__
#define __SIMULATOR_HPP__

#include <chrono>
#include <functional>
#include <memory>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include "analytical_ddr.hpp"
#include "analytical_noc.hpp"
#include "base_module.hpp"
#include "config.hpp"
#include "detail_ddr.hpp"
#include "detail_noc.hpp"
#include "detail_npu_core.hpp"
#include "logger.hpp"
#include "util.hpp"

class Simulator {
public:
    Config cfg;
    Logger logger;
    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> event_queue;

    // components
    std::vector<std::unique_ptr<BaseModule>> cores;
    std::vector<std::unique_ptr<BaseModule>> memory_controllers;
    std::unique_ptr<BaseModule> noc;

    // global sync unit
    // statistics
    uint64_t _max_sim_cycle;
    uint64_t base_tick_ps;
    uint64_t cur_tick;
    uint64_t finished_cores_cnt;
    std::vector<uint64_t> ticks_per_period;
    std::chrono::high_resolution_clock::time_point sim_start_time, sim_end_time;
    double sim_time_us;  // duration of the simulation (us)
    SimProfile profile;

    Simulator(uint64_t max_sim_cycle, const Config& cfg);
    ~Simulator();

    void run();
    void print_stats();
    void schedule(const Event& event);  // schedule an event to the event queue
    BaseModule* getCore(size_t core_id) { return cores[core_id].get(); }
    BaseModule* getDDR(size_t ddr_id) { return memory_controllers[ddr_id].get(); }
    BaseModule* getNoC() { return noc.get(); }
    uint64_t getTick() { return cur_tick; }
    uint64_t getTicksPerCoreCycle() { return ticks_per_period[0]; }
    uint64_t getTicksPerMemoryCycle() { return ticks_per_period[1]; }
    uint64_t getTicksPerNoCCycle() { return ticks_per_period[2]; }
    void incrFinishedCoresCnt() { finished_cores_cnt++; }
    bool isEarlyFinished() const { return finished_cores_cnt == cores.size(); }

    void initSyncBarrier(int sync_id, int total_sync_num);

    int getSyncCnt(int sync_id);

    void sendSyncAck(int sync_id);

private:
    struct SyncEntry {
        int currrent_acked_cnt = 0;
        int sync_cnt = 0;
        int total_sync_num = 0;
    };
    std::unordered_map<int, SyncEntry> _sync_states;  // sync_id -> SyncStatus
};

#endif  // __SIMULATOR_HPP__