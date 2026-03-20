#ifndef __ANALYTICAL_NOC_HPP__
#define __ANALYTICAL_NOC_HPP__

#include <math.h>

#include <queue>
#include <vector>

#include "base_module.hpp"
#include "config.hpp"
#include "logger.hpp"
#include "simulator.hpp"
#include "util.hpp"

class Simulator;

class AnalyticalNOC : public BaseModule {
private:
    // topology parameters
    int m_noc_mesh_width;
    int m_noc_mesh_height;
    int m_num_routers;
    int m_num_cores;
    int m_base_ddr_id;
    int m_core_num_per_row;

    // Core (Compute Tiles)
    std::vector<BaseModule*> core_ptrs;

    // Memory Controller (MC)
    std::vector<BaseModule*> mc_ptrs;

    // NI parameters
    std::vector<std::queue<Message>> ni_request_queues;
    size_t m_request_queue_depth;

    // NoC paramters
    int m_router_pipeline_latency;  // (e.g., 2 cycles)
    int m_link_latency;             //  (e.g., 1 cycle)
    size_t m_link_width_bytes;      //  (e.g., 64 bytes)
    int m_an_noc_method;

    // --- G/G/1 model parameters ---
    // double m_cs2;
    int m_sampling_period;
    int m_num_saved_sample_points;

    struct TrafficSample {
        uint64_t start_cycle;
        uint64_t end_cycle;
        std::vector<double> injection_rates;
        std::vector<std::vector<double>> traffic_matrix;
    };

    std::deque<TrafficSample> m_traffic_samples;

    struct RouterChannel {
        double lambda;
        double mu;
        double ca2;
        double cs2;
        double waiting_time;
        double service_time;       // S_r^j
        double service_time_sq;    // E[S^2]
        double base_service_time;  // S_base
    };

    std::vector<std::vector<RouterChannel>> m_router_channels;

    std::vector<std::vector<std::vector<int>>> m_routing_paths;  // [src][dst] -> path router IDs

    int getEstimatedLatency(int src_tile_id, int dst_tile_id, int payload_size);  // 估算消息在 NOC 中的传输延迟

    double allenCunneenApproximation(double lambda, double mu, double ca2, double cs2);
    void updateTrafficSampling();
    void computeQueuingParameters();
    double computeChannelWaitingTime(int src_router, int out_port);
    double computePathLatency(const std::vector<int>& path, int num_flits);
    void updateServiceTimeRecursive();
    void precomputeRoutingPaths();
    std::vector<int> getRoutingPath(int src_router, int dst_router);
    std::pair<int, int> getRouterCoords(int router_id);
    int getRouterIdFromCoords(int x, int y);
    int getDownstreamRouter(int router_id, int out_port);

    struct PacketRecord {
        int src_node;
        int dst_node;
        int packet_size_flits;
        uint64_t injection_cycle;
    };
    std::vector<PacketRecord> m_current_sample_packets;
    uint64_t m_last_sampling_cycle;
    int traffic_weight_mode;
    double traffic_weight_factor;

public:
    AnalyticalNOC(Simulator* sim, const Config& cfg, Logger& logger, uint64_t ticks_per_cycle,
                  const std::vector<BaseModule*>& core_list, const std::vector<BaseModule*>& mc_list);
    ~AnalyticalNOC() = default;

    void tick() override;
    void sendMsg(Message& msg) override;
    bool recvMsg(Message& msg) override;
    void scheduleNextEvent() override;
    TrafficManager* getTrafficManager() override;
};

#endif  // __ANALYTICAL_NOC_HPP__