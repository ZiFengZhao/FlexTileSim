#ifndef __DETAIL_NOC_HPP__
#define __DETAIL_NOC_HPP__

#include <assert.h>

#include <queue>
#include <vector>

#include "../external/Booksim2/src/booksim.hpp"
#include "../external/Booksim2/src/booksim_config.hpp"
#include "../external/Booksim2/src/trafficmanager.hpp"
#include "base_module.hpp"
#include "config.hpp"
#include "logger.hpp"
#include "simulator.hpp"
#include "util.hpp"

using FlitType = Flit::FlitType;

class Simulator;

class DetailNoC : public BaseModule {
private:
    // topology parameters
    int m_noc_mesh_width;
    int m_noc_mesh_height;
    int m_num_cores;
    int m_base_ddr_id;
    int m_core_num_per_row;
    size_t link_width_bytes;  // (e.g., 64 bytes)
    // Core (Compute Tiles)
    std::vector<BaseModule*> core_ptrs;
    // Memory Controller (MC)
    std::vector<BaseModule*> mc_ptrs;
    // NI parameters
    std::vector<std::queue<Message>> ni_request_queues;
    std::queue<int> active_routers;
    std::vector<BaseModule*> tile_to_module;
    size_t m_request_queue_depth;

    // Booksim parameters
    TrafficManager* traffic_manager;
    static const int MAX_FLITS_PER_PACKET = 16;
    int cur_pkt_id;
    int max_limit_sim_cycle;
    std::queue<Flit*> _pending_packets;

public:
    DetailNoC(Simulator* sim, BookSimConfig const& booksim_cfg, const Config& cfg, Logger& logger,
              uint64_t ticks_per_cycle, const std::vector<BaseModule*>& core_list,
              const std::vector<BaseModule*>& mc_list);

    ~DetailNoC() = default;

    void tick() override;
    void sendMsg(Message& msg) override;
    bool recvMsg(Message& msg) override;
    void scheduleNextEvent() override;
    Flit* getEnjectedFlit();
    TrafficManager* getTrafficManager() override;
};

#endif