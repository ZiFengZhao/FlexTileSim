#ifndef __DETAIL_DDR_HPP__
#define __DETAIL_DDR_HPP__

#include <assert.h>

#include <queue>

#include "../external/DRAMsim3/src/dramsim3.h"
#include "base_module.hpp"
#include "config.hpp"
#include "logger.hpp"
#include "simulator.hpp"
#include "util.hpp"

class Simulator;

class DetailDDR : public BaseModule {
private:
    size_t m_request_queue_depth;
    std::queue<Message> request_queue;

    // dramsim3
    std::unique_ptr<dramsim3::MemorySystem> m_dramsim;
    int dram_unit = 64;  // dram req unit size in bytes

    // dramsim3 callback
    void readComplete(uint64_t addr);
    void writeComplete(uint64_t addr);

    struct CurrentRequest {
        bool active = false;
        Message msg;
        std::vector<bool> completed_trans;
        int total_trans = 0;
        int trans_index = 0;
        int completed_count = 0;
        uint64_t start_tick = 0;
    } current_request;

    void clearCurrentRequest();

public:
    DetailDDR(Simulator* sim, int id, const Config& cfg, Logger& logger, uint64_t ticks_per_cycle);
    ~DetailDDR() = default;

    void tick() override;
    void sendMsg(Message& msg) override;
    bool recvMsg(Message& msg) override;
    void scheduleNextEvent() override;
    TrafficManager* getTrafficManager() override;
};

#endif  // __DETAIL_DDR_HPP__