#ifndef __ANALYTICAL_DDR_HPP__
#define __ANALYTICAL_DDR_HPP__
#pragma once
#include <queue>

#include "base_module.hpp"
#include "config.hpp"
#include "logger.hpp"
#include "simulator.hpp"
#include "util.hpp"

class Simulator;  // forward declaration

class AnalyticalDDR : public BaseModule {
private:
    // DDR configuration parameters
    size_t m_request_queue_depth;

    int base_latency;
    int BL;
    int ddr_bus_width;
    int burst_size;
    int ddr_bandwidth_per_cycle;
    int max_outstanding;

    std::queue<Message> request_queue;

    struct ActiveRequest {
        Message msg;
        uint64_t finish_tick;
    };

    std::vector<ActiveRequest> inflight_requests;

    bool tick_scheduled = false;

public:
    AnalyticalDDR(Simulator* sim, int id, const Config& cfg, Logger& logger, uint64_t ticks_per_cycle);
    ~AnalyticalDDR() = default;

    void tick() override;
    void sendMsg(Message& msg) override;
    bool recvMsg(Message& msg) override;
    void scheduleNextEvent() override;
    void scheduleTick(uint64_t tick);
    TrafficManager* getTrafficManager() override;

private:
    int getEstimatedLatency(const Message& msg);
};

#endif  // __ANALYTICAL_DDR_HPP__