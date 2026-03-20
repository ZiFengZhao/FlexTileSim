#include "analytical_noc.hpp"

AnalyticalNOC::AnalyticalNOC(Simulator* sim, const Config& cfg, Logger& logger, uint64_t ticks_per_cycle,
                             const std::vector<BaseModule*>& core_list, const std::vector<BaseModule*>& mc_list)
    : BaseModule(sim, cfg, logger, ticks_per_cycle, -1),
      core_ptrs(core_list),
      mc_ptrs(mc_list),
      traffic_weight_mode(cfg.noc_traffic_weight_mode),
      traffic_weight_factor(cfg.noc_traffic_weight_factor) {
    switch (cfg.core_num) {
        case 1:
            m_noc_mesh_width = 2;
            m_noc_mesh_height = 2;
            m_base_ddr_id = 2;
            m_core_num_per_row = 1;
            break;
        case 2:
            m_noc_mesh_width = 2;
            m_noc_mesh_height = 2;
            m_base_ddr_id = 2;
            m_core_num_per_row = 2;
            break;
        case 4:
            m_noc_mesh_width = 3;
            m_noc_mesh_height = 3;
            m_base_ddr_id = 6;
            m_core_num_per_row = 2;
            break;
        case 8:
            m_noc_mesh_width = 5;
            m_noc_mesh_height = 5;
            m_base_ddr_id = 20;
            m_core_num_per_row = 2;
            break;
        case 16:
            m_noc_mesh_width = 5;
            m_noc_mesh_height = 5;
            m_base_ddr_id = 20;
            m_core_num_per_row = 4;
            break;
        case 32:
            m_noc_mesh_width = 10;
            m_noc_mesh_height = 10;
            m_base_ddr_id = 40;
            m_core_num_per_row = 8;
            break;
        case 64:
            m_noc_mesh_width = 10;
            m_noc_mesh_height = 10;
            m_base_ddr_id = 80;
            m_core_num_per_row = 8;
            break;
        default:
            std::cerr << "Error: Invalid core number" << std::endl;
            exit(1);
    }
    m_num_routers = m_noc_mesh_width * m_noc_mesh_height;
    m_router_pipeline_latency = cfg.noc_router_latency;
    m_link_width_bytes = cfg.noc_link_width;
    m_link_latency = cfg.noc_link_latency;
    m_an_noc_method = cfg.an_noc_method;
    m_request_queue_depth = cfg.noc_request_queue_depth;
    ni_request_queues.resize(m_num_routers);
    m_num_cores = cfg.core_num;

    m_sampling_period = cfg.noc_sampling_period;
    m_num_saved_sample_points = cfg.noc_num_saved_samples;
    m_router_channels.resize(m_num_routers);
    for (int i = 0; i < m_num_routers; i++) {
        m_router_channels[i].resize(5);
        for (int j = 0; j < 5; j++) {
            m_router_channels[i][j] = {.lambda = 0.0,
                                       .mu = 1.0 / (m_router_pipeline_latency + m_link_latency),  // 基础服务率
                                       .ca2 = 1.0,                                                // 初始假设为指数分布
                                       .cs2 = 1.0,
                                       .waiting_time = 0.0,
                                       .service_time = (double)(m_router_pipeline_latency + m_link_latency),
                                       .service_time_sq = pow((double)(m_router_pipeline_latency + m_link_latency), 2)};
        }
    }

    precomputeRoutingPaths();

    m_last_sampling_cycle = 0;

    logger.log("NoC Frequency: %f GHz", cfg.noc_freq);
    logger.log("Mesh Size: %d x %d", m_noc_mesh_width, m_noc_mesh_height);
    logger.log("Router Pipeline Latency: %d", m_router_pipeline_latency);
    logger.log("Link Width: %d bytes", m_link_width_bytes);
    logger.log("Link Latency: %d", m_link_latency);
    logger.log("Request Queue Depth: %d", m_request_queue_depth);

    if (m_an_noc_method == 1 || m_an_noc_method == 2) {
        logger.log("Using queue-based analytical NoC model");
        logger.log("Sampling Period: %d cycles", m_sampling_period);
        logger.log("Sliding Window Size: %d", m_num_saved_sample_points);
    } else {
        logger.log("Using distance-based analytical NoC model");
    }
    logger.log("Analytical NoC initialized!");
}

void AnalyticalNOC::precomputeRoutingPaths() {
    m_routing_paths.resize(m_num_routers);
    for (int src = 0; src < m_num_routers; src++) {
        m_routing_paths[src].resize(m_num_routers);

        for (int dst = 0; dst < m_num_routers; dst++) {
            if (src == dst) continue;

            auto [src_x, src_y] = getRouterCoords(src);
            auto [dst_x, dst_y] = getRouterCoords(dst);

            std::vector<int> path;
            int current_x = src_x;
            int current_y = src_y;
            int current_router = src;

            // XY routing
            while (current_x != dst_x) {
                if (current_x < dst_x) {
                    current_x++;
                } else {
                    current_x--;
                }
                current_router = getRouterIdFromCoords(current_x, current_y);
                path.push_back(current_router);
            }

            while (current_y != dst_y) {
                if (current_y < dst_y) {
                    current_y++;
                } else {
                    current_y--;
                }
                current_router = getRouterIdFromCoords(current_x, current_y);
                path.push_back(current_router);
            }

            m_routing_paths[src][dst] = path;
        }
    }
}

std::vector<int> AnalyticalNOC::getRoutingPath(int src_router, int dst_router) {
    if (src_router == dst_router) {
        return {};
    }
    return m_routing_paths[src_router][dst_router];
}

std::pair<int, int> AnalyticalNOC::getRouterCoords(int router_id) {
    int x = router_id % m_noc_mesh_width;
    int y = router_id / m_noc_mesh_width;
    return {x, y};
}

int AnalyticalNOC::getRouterIdFromCoords(int x, int y) {
    return y * m_noc_mesh_width + x;
}

double AnalyticalNOC::allenCunneenApproximation(double lambda, double mu, double ca2, double cs2) {
    if (mu <= lambda || mu <= 0) {
        return 1000.0;
    }

    double rho = lambda / mu;
    if (rho >= 0.99) {
        return 1000.0;
    }

    double Wq = (rho / (1.0 - rho)) * ((ca2 + cs2) / 2.0) * (1.0 / mu);
    return std::max(0.0, Wq);
}

void AnalyticalNOC::updateTrafficSampling() {
    uint64_t current_cycle = getCycle();

    if (current_cycle - m_last_sampling_cycle >= (uint64_t)m_sampling_period) {
        if (!m_current_sample_packets.empty()) {
            TrafficSample sample;
            sample.start_cycle = m_last_sampling_cycle;
            sample.end_cycle = current_cycle;

            sample.traffic_matrix.resize(m_num_routers);
            for (int i = 0; i < m_num_routers; ++i) {
                sample.traffic_matrix[i].resize(m_num_routers, 0);
            }

            sample.injection_rates.resize(m_num_routers, 0);

            for (const auto& packet : m_current_sample_packets) {
                int src = packet.src_node;
                int dst = packet.dst_node;
                double traffic_volume = packet.packet_size_flits;

                sample.traffic_matrix[src][dst] += traffic_volume;
                sample.injection_rates[src] += 1.0;
            }

            double window_duration = static_cast<double>(current_cycle - m_last_sampling_cycle);
            for (int i = 0; i < m_num_routers; i++) {
                sample.injection_rates[i] /= window_duration;
                for (int j = 0; j < m_num_routers; j++) {
                    sample.traffic_matrix[i][j] /= window_duration;
                }
            }

            m_traffic_samples.push_back(sample);
            if (m_traffic_samples.size() > m_num_saved_sample_points) {
                m_traffic_samples.pop_front();
            }

            m_current_sample_packets.clear();

            computeQueuingParameters();
        }
    }
}

void AnalyticalNOC::updateServiceTimeRecursive() {
    for (int r = m_num_routers - 1; r >= 0; r--) {
        for (int out_port = 0; out_port < 5; out_port++) {
            auto& ch = m_router_channels[r][out_port];

            double sum_S = 0.0;
            double sum_S2 = 0.0;

            int downstream_router = getDownstreamRouter(r, out_port);

            if (downstream_router < 0) {
                // ejection channel
                ch.service_time = ch.base_service_time;
                ch.service_time_sq = ch.base_service_time * ch.base_service_time;
            } else {
                double P = 1.0;

                auto& down_ch = m_router_channels[downstream_router][0];

                double S = ch.base_service_time + down_ch.waiting_time + down_ch.service_time;

                sum_S += P * S;
                sum_S2 += P * S * S;

                ch.service_time = sum_S;
                ch.service_time_sq = sum_S2;
            }

            ch.mu = 1.0 / ch.service_time;

            ch.cs2 = ch.service_time_sq / (ch.service_time * ch.service_time) - 1.0;

            if (ch.cs2 < 0) ch.cs2 = 0;
        }
    }
}
void AnalyticalNOC::computeQueuingParameters() {
    if (m_traffic_samples.empty()) return;

    std::vector<std::vector<double>> avg_traffic_matrix(m_num_routers, std::vector<double>(m_num_routers, 0.0));
    std::vector<double> avg_injection_rates(m_num_routers, 0.0);

    double total_weight = 0.0;

    int sample_count = m_traffic_samples.size();

    for (int i = 0; i < sample_count; i++) {
        // double weight = static_cast<double>(i + 1) / sample_count;
        double weight = 1.0;
        if (traffic_weight_mode == 0) {
            // linear weighting
            weight = (1.0 - traffic_weight_factor) + traffic_weight_factor * (i + 1);
        } else if (traffic_weight_mode == 1) {
            // exponential weighting
            weight = std::pow(traffic_weight_factor, sample_count - 1 - i);
        }

        total_weight += weight;

        const auto& sample = m_traffic_samples[i];
        for (int src = 0; src < m_num_routers; src++) {
            avg_injection_rates[src] += sample.injection_rates[src] * weight;
            for (int dst = 0; dst < m_num_routers; dst++) {
                avg_traffic_matrix[src][dst] += sample.traffic_matrix[src][dst] * weight;
            }
        }
    }

    for (int src = 0; src < m_num_routers; src++) {
        avg_injection_rates[src] /= total_weight;
        for (int dst = 0; dst < m_num_routers; dst++) {
            avg_traffic_matrix[src][dst] /= total_weight;
        }
    }

    std::vector<double> arrival_intervals;
    for (const auto& sample : m_traffic_samples) {
        for (double rate : sample.injection_rates) {
            if (rate > 0) {
                arrival_intervals.push_back(1.0 / rate);
            }
        }
    }

    double ca2 = 1.0;
    if (arrival_intervals.size() > 1) {
        double sum = 0.0;
        double sum_sq = 0.0;
        for (double interval : arrival_intervals) {
            sum += interval;
            sum_sq += interval * interval;
        }
        double mean = sum / arrival_intervals.size();
        double variance = sum_sq / arrival_intervals.size() - mean * mean;
        ca2 = variance / (mean * mean);
    }

    for (int r = 0; r < m_num_routers; r++) {
        for (int out_port = 0; out_port < 5; out_port++) {
            double lambda = 0.0;

            for (int src = 0; src < m_num_routers; src++) {
                for (int dst = 0; dst < m_num_routers; dst++) {
                    if (src == dst) continue;

                    std::vector<int> path = getRoutingPath(src, dst);
                    if (!path.empty()) {
                        lambda += avg_traffic_matrix[src][dst];
                    }
                }
            }

            m_router_channels[r][out_port].lambda = lambda;
            m_router_channels[r][out_port].ca2 = ca2;

            m_router_channels[r][out_port].waiting_time = allenCunneenApproximation(
                lambda, m_router_channels[r][out_port].mu, ca2, m_router_channels[r][out_port].cs2);
        }
    }
    updateServiceTimeRecursive();
}

int AnalyticalNOC::getDownstreamRouter(int router_id, int out_port) {
    auto [x, y] = getRouterCoords(router_id);

    switch (out_port) {
        case 0:
            y -= 1;
            break;  // North
        case 1:
            x += 1;
            break;  // East
        case 2:
            y += 1;
            break;  // South
        case 3:
            x -= 1;
            break;  // West
        default:
            return -1;  // Local
    }

    if (x < 0 || y < 0 || x >= m_noc_mesh_width || y >= m_noc_mesh_height) return -1;

    return getRouterIdFromCoords(x, y);
}

double AnalyticalNOC::computeChannelWaitingTime(int router_id, int out_port) {
    if (router_id < 0 || router_id >= m_num_routers || out_port < 0 || out_port >= 5) {
        return 0.0;
    }

    if (m_an_noc_method == 2) {
        return m_router_channels[router_id][out_port].waiting_time;
    } else {
        return m_router_channels[router_id][out_port].service_time;
    }
}

double AnalyticalNOC::computePathLatency(const std::vector<int>& path, int num_flits) {
    double total_latency = 0.0;

    for (size_t i = 0; i < path.size(); i++) {
        int router_id = path[i];

        total_latency += m_router_pipeline_latency + m_link_latency;

        if (m_an_noc_method >= 1) {
            total_latency += computeChannelWaitingTime(router_id, 0);
        }
    }

    int body_flits = num_flits - 1;
    if (body_flits > 0) {
        total_latency += body_flits;
    }

    return total_latency;
}

int AnalyticalNOC::getEstimatedLatency(int src_tile_id, int dst_tile_id, int payload_size) {
    int num_flits = static_cast<int>(std::ceil(payload_size / (double)m_link_width_bytes));
    if (m_an_noc_method == 0) {
        int src_x = src_tile_id % m_noc_mesh_width;
        int src_y = src_tile_id / m_noc_mesh_width;
        int dst_x = dst_tile_id % m_noc_mesh_width;
        int dst_y = dst_tile_id / m_noc_mesh_width;
        int dx = std::abs(dst_x - src_x);
        int dy = std::abs(dst_y - src_y);
        int hops = dx + dy;

        int link_cycles = hops * m_link_latency;
        int router_cycles = (hops + 1) * m_router_pipeline_latency;
        int serialize_cycles = num_flits;

        return link_cycles + router_cycles + serialize_cycles;
    } else {
        std::vector<int> path = getRoutingPath(src_tile_id, dst_tile_id);

        if (m_an_noc_method == 2) {
            PacketRecord record = {.src_node = src_tile_id,
                                   .dst_node = dst_tile_id,
                                   .packet_size_flits = num_flits,
                                   .injection_cycle = getCycle()};
            m_current_sample_packets.push_back(record);

            updateTrafficSampling();
        }

        double latency = computePathLatency(path, num_flits);

        return static_cast<int>(std::ceil(latency));
    }
}

void AnalyticalNOC::sendMsg(Message& msg) {
    // get src/dst tile id from msg

    int src_tile_id = parseAddr(msg.src_addr, m_num_cores);
    int dst_tile_id = parseAddr(msg.dst_addr, m_num_cores);
    logger.log("[Tick %d-NoC Cycle %d] NoC Sending message from tile %d to tile %d", sim->getTick(), getCycle(),
               src_tile_id, dst_tile_id);

    // Calculate the estimated latency
    int pkt_size_in_bytes = 0;  // bytes
    if (msg.type == MsgType::LD_REQ || msg.type == MsgType::ST_REQ) {
        pkt_size_in_bytes = msg.data_size;
    } else if (msg.type == MsgType::ST_RESP || msg.type == MsgType::LD_RESP) {
        pkt_size_in_bytes = msg.request_data_size;
    }
    int noc_transfer_cycles = getEstimatedLatency(src_tile_id, dst_tile_id, pkt_size_in_bytes);
    // Convert cycles to ticks
    uint64_t noc_transfer_ticks = static_cast<uint64_t>(std::ceil(noc_transfer_cycles * m_ticks_per_cycle));
    // Schedule the next event
    uint64_t schedule_tick = sim->getTick() + noc_transfer_ticks;

    BaseModule* dst_mod = nullptr;
    bool find_success_flg = false;
    // get dst module pointer according to dst tile id
    // find in ddr channels
    for (auto& mc : mc_ptrs) {
        if ((mc->getId() + m_base_ddr_id) == dst_tile_id) {
            dst_mod = mc;
            find_success_flg = true;
            break;
        }
    }

    // find in core tile array
    if (!find_success_flg) {
        for (auto& core : core_ptrs) {
            int virtual_core_id = core->getId();
            int physical_core_id =
                (virtual_core_id / m_core_num_per_row) * m_noc_mesh_width + (virtual_core_id % m_core_num_per_row);
            assert(physical_core_id < m_noc_mesh_width * m_noc_mesh_height);
            if (physical_core_id == dst_tile_id) {
                dst_mod = core;
                break;
            }
        }
    }

    if (dst_mod == nullptr) {
        logger.log("[Tick %d-NoC Cycle %d] NoC No module found for tile %d", sim->getTick(), getCycle(), dst_tile_id);
        throw std::runtime_error("No module found for tile " + std::to_string(dst_tile_id));
    }

    Message msg_copy = msg;

    Event send_e;
    send_e.tick = schedule_tick;

    send_e.callback = [dst_mod, msg_copy]() mutable { dst_mod->recvMsg(msg_copy); };
    sim->schedule(send_e);
    logger.log(
        "[Tick %d-NoC Cycle %d] NoC Scheduled sendMsg event [from: tile%d to: tile%d], will call recvMsg at tick "
        "%d",
        sim->getTick(), getCycle(), src_tile_id, dst_tile_id, schedule_tick);
}

bool AnalyticalNOC::recvMsg(Message& msg) {
    int inj_router_id = parseAddr(msg.src_addr, m_num_cores);
    if (ni_request_queues[inj_router_id].size() >= m_request_queue_depth) {
        logger.log("[Tick %d-NoC Cycle %d] NoC Router %d: request queue is full, backpressure asserted", sim->getTick(),
                   getCycle(), inj_router_id);
        return false;
    } else {
        ni_request_queues[inj_router_id].push(msg);
        logger.log("[Tick %d-NoC Cycle %d] NoC Router %d received msg", sim->getTick(), getCycle(), inj_router_id);
        return true;
    }
}

void AnalyticalNOC::tick() {
    ScopedTimer timer(&sim->profile.noc_time_ns);

    logger.log("[Tick %d-NoC Cycle %d] NoC is ticked", sim->getTick(), getCycle());
    for (int i = 0; i < m_noc_mesh_height * m_noc_mesh_width; i++) {
        if (!ni_request_queues[i].empty()) {
            Message msg = ni_request_queues[i].front();
            sendMsg(msg);
            ni_request_queues[i].pop();
        }
    }

    if (m_an_noc_method == 2) {
        updateTrafficSampling();
    }
}

void AnalyticalNOC::scheduleNextEvent() {}

TrafficManager* AnalyticalNOC::getTrafficManager() {
    return NULL;
}