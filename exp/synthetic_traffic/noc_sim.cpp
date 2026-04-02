#include <chrono>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <vector>

using namespace std;
const int PKT_SIZE = 20;
const int MAX_PKT_NUM_PER_NODE = 800;
const int SATU_THRESHOLD = 1000;

struct Config {
    int inj_is_burst = 0;
    double burst_prob = 0.0;
    double alpha = 0.0;
    double beta = 0.0;
    double inj_rate = 0.0;
    long long sim_cycle = 0;
    int network_dim = 0;
    int hop_delay = 0;
    int buffer_size = 0;
    int sampling_period = 0;
    int noc_model_type = 0;  // 0: Zero-load, 1:M/D/1, 2:G/G/1
    double decay_factor = 0.1;
    double linear_factor = 0.1;
};

struct Packet {
    int src_x, src_y;
    int dst_x, dst_y;
    long long creation_time;
};

struct Event {
    int src_node_id;
    long long arrival_time;
    long long latency;
    bool operator>(const Event& other) const { return arrival_time > other.arrival_time; }
};

Config parse_config(const string& filename) {
    Config cfg;
    ifstream file(filename);

    string line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t pos = line.find('=');
        if (pos != string::npos) {
            string key = line.substr(0, pos);
            string val = line.substr(pos + 1);
            if (key == "inj_is_burst")
                cfg.inj_is_burst = stoi(val);
            else if (key == "burst_prob")
                cfg.burst_prob = stod(val);
            else if (key == "alpha")
                cfg.alpha = stod(val);
            else if (key == "beta")
                cfg.beta = stod(val);
            else if (key == "inj_rate")
                cfg.inj_rate = stod(val);
            else if (key == "sim_cycle")
                cfg.sim_cycle = stoll(val);
            else if (key == "network_dim")
                cfg.network_dim = stoi(val);
            else if (key == "hop_delay")
                cfg.hop_delay = stoi(val);
            else if (key == "buffer_size")
                cfg.buffer_size = stoi(val);
            else if (key == "sampling_period")
                cfg.sampling_period = stoi(val);
            else if (key == "noc_model_type")
                cfg.noc_model_type = stoi(val);
            else if (key == "decay_factor")
                cfg.decay_factor = stod(val);
            else if (key == "linear_factor")
                cfg.linear_factor = stod(val);
        }
    }
    return cfg;
}

class Node {
public:
    int id, x, y;
    queue<Packet> ni_queue;
    int pending_credits;
    int ni_capacity;
    int tx_busy_timer;
    bool in_burst_state;
    double alpha;
    double beta;
    double r1;
    Node(int id, int dim, double inj_rate, double alpha, double beta, int ni_cap = 8)
        : id(id),
          x(id % dim),
          y(id / dim),
          tx_busy_timer(0),
          in_burst_state(false),
          alpha(alpha),
          beta(beta),
          ni_capacity(ni_cap) {
        double p_on = alpha / (alpha + beta);
        r1 = inj_rate / p_on;
        if (r1 > 1.0) {
            std::cerr << "Error: Injection rate exceeds the maximum allowed rate." << std::endl;
            exit(1);
        };
        std::uniform_real_distribution<> dis(0.0, 1.0);
        std::mt19937 gen(42 + id);
        in_burst_state = (dis(gen) < p_on);
    }

    void return_credit() { pending_credits++; }

    bool try_generate(long long current_cycle, int total_nodes, mt19937& gen) {
        if (pending_credits <= 0) {
            return false;
        }
        if (ni_queue.size() < MAX_PKT_NUM_PER_NODE) {
            Packet p;
            p.src_x = x;
            p.src_y = y;
            p.creation_time = current_cycle;
            uniform_int_distribution<> dst_dis(0, total_nodes - 1);
            int dst_id = id;
            while (dst_id == id) {
                dst_id = dst_dis(gen);
            }
            p.dst_x = dst_id % static_cast<int>(std::sqrt(total_nodes));
            p.dst_y = dst_id / static_cast<int>(std::sqrt(total_nodes));
            ni_queue.push(p);
            pending_credits--;
            return true;
        } else {
            return false;
        }
    }

    bool generate_traffic(long long current_cycle, const Config& cfg, int total_nodes, mt19937& gen,
                          uniform_real_distribution<>& prob_dis) {
        bool generated = false;
        if (cfg.inj_is_burst) {
            double p = prob_dis(gen);
            if (in_burst_state) {
                if (p < beta) {
                    in_burst_state = false;
                }
            } else {
                if (p < alpha) {
                    in_burst_state = true;
                }
            }

            if (in_burst_state && prob_dis(gen) < r1) {
                generated = try_generate(current_cycle, total_nodes, gen);
            }

        } else {
            if (prob_dis(gen) < cfg.inj_rate) {
                generated = try_generate(current_cycle, total_nodes, gen);
            }
        }
        return generated;
    }
};

class M_D_1_model {
public:
    long long eval_waiting_time(double lambda, double L, double k, int hops) {
        double rho_s = lambda * L;
        double rho_c = (lambda * k / 4.0) * L;
        long long waiting_time = 0;
        if (rho_s >= 1.0 || rho_c >= 1.0) {
            waiting_time = -1;
            return waiting_time;
        }

        double W_s = (rho_s * L) / (2.0 * (1.0 - rho_s));
        double W_c = (rho_c * L) / (2.0 * (1.0 - rho_c));

        waiting_time = static_cast<long long>(std::round(W_s + hops * W_c));
        return waiting_time;
    }
};

class G_G_1_model {
private:
    struct TrafficSample {
        long long start_cycle;
        long long end_cycle;
        vector<vector<double>> traffic_matrix;
    };

    struct RouterChannel {
        double lambda = 0;
        double mu;
        double ca2;
        double cs2;
        double waiting_time;
        double service_time;
        double service_time_sq;
        double base_service_time = 2;
        std::deque<double> lambda_history;
    };

    struct PacketRecord {
        int src_node;
        int dst_node;
        int packet_size_flits;
        long long injection_cycle;
    };

    int m_network_dim;
    int m_num_routers;
    int m_hop_delay;
    int m_buffer_size;
    int m_sampling_period;
    int m_num_saved_sample_points;
    long long m_last_sampling_cycle;
    double m_decay_factor;
    double m_linear_factor;

    deque<TrafficSample> m_traffic_samples;
    vector<PacketRecord> m_current_sample_packets;
    vector<vector<RouterChannel>> m_router_channels;
    vector<vector<vector<int>>> m_routing_paths;

public:
    long long rho_satu_counts;
    long long rho_no_satu_counts;

public:
    G_G_1_model(int dim, int hop_delay, int sampling_period, int buffer_size, double decay_factor, double linear_factor)
        : m_network_dim(dim),
          m_hop_delay(hop_delay),
          m_buffer_size(buffer_size),
          m_sampling_period(sampling_period),
          m_decay_factor(decay_factor),
          m_linear_factor(linear_factor) {
        m_num_routers = dim * dim;
        m_num_saved_sample_points = 5;
        m_last_sampling_cycle = 0;

        rho_satu_counts = 0;
        rho_no_satu_counts = 0;

        double init_service_time = 3.0;
        m_router_channels.resize(m_num_routers, vector<RouterChannel>(5));
        for (int i = 0; i < m_num_routers; i++) {
            for (int j = 0; j < 5; j++) {
                m_router_channels[i][j] = {0.0,
                                           1.0 / init_service_time,
                                           1.0,
                                           1.0,
                                           0.0,
                                           init_service_time,
                                           pow(init_service_time, 2),
                                           init_service_time};
            }
        }
        precomputeRoutingPaths();
    }

    long long eval_waiting_time(int src, int dst, int num_flits, long long current_cycle) {
        m_current_sample_packets.push_back({src, dst, num_flits, current_cycle});
        if (current_cycle - m_last_sampling_cycle >= m_sampling_period) {
            updateTrafficSampling(current_cycle);
        }
        if (m_traffic_samples.empty()) {
            return 0;
        }
        const vector<int>& path = m_routing_paths[src][dst];
        double total_wq = 0.0;

        for (size_t i = 0; i < path.size(); i++) {
            int current_router = path[i];
            int out_port = 0;

            if (i + 1 < path.size()) {
                int next_router = path[i + 1];
                out_port = getOutPort(current_router, next_router);
            } else {
                out_port = 4;
            }

            double wq = m_router_channels[current_router][out_port].waiting_time;
            if (wq >= SATU_THRESHOLD) return -1;
            total_wq += wq;
        }

        return static_cast<long long>(std::round(total_wq));
    }

private:
    void precomputeRoutingPaths() {
        m_routing_paths.assign(m_num_routers, vector<vector<int>>(m_num_routers));
        for (int src = 0; src < m_num_routers; src++) {
            for (int dst = 0; dst < m_num_routers; dst++) {
                if (src == dst) continue;
                int src_x = src % m_network_dim, src_y = src / m_network_dim;
                int dst_x = dst % m_network_dim, dst_y = dst / m_network_dim;

                int curr_x = src_x, curr_y = src_y;
                m_routing_paths[src][dst].push_back(src);
                while (curr_x != dst_x) {
                    curr_x += (curr_x < dst_x) ? 1 : -1;
                    m_routing_paths[src][dst].push_back(curr_y * m_network_dim + curr_x);
                }
                while (curr_y != dst_y) {
                    curr_y += (curr_y < dst_y) ? 1 : -1;
                    m_routing_paths[src][dst].push_back(curr_y * m_network_dim + curr_x);
                }
            }
        }
    }

    int getOutPort(int current_router, int next_router) {
        int cx = current_router % m_network_dim, cy = current_router / m_network_dim;
        int nx = next_router % m_network_dim, ny = next_router / m_network_dim;
        if (ny < cy) return 0;  // North
        if (nx > cx) return 1;  // East
        if (ny > cy) return 2;  // South
        if (nx < cx) return 3;  // West
        return 4;               // Local
    }

    int getDownstreamRouter(int router_id, int out_port) {
        int x = router_id % m_network_dim, y = router_id / m_network_dim;
        if (out_port == 0)
            y -= 1;
        else if (out_port == 1)
            x += 1;
        else if (out_port == 2)
            y += 1;
        else if (out_port == 3)
            x -= 1;
        else
            return -1;

        if (x < 0 || y < 0 || x >= m_network_dim || y >= m_network_dim) return -1;
        return y * m_network_dim + x;
    }

    void updateTrafficSampling(long long current_cycle) {
        TrafficSample sample;
        sample.start_cycle = m_last_sampling_cycle;
        sample.end_cycle = current_cycle;
        sample.traffic_matrix.assign(m_num_routers, vector<double>(m_num_routers, 0.0));

        for (const auto& pkt : m_current_sample_packets) {
            sample.traffic_matrix[pkt.src_node][pkt.dst_node] += pkt.packet_size_flits;
        }

        double window_duration = static_cast<double>(current_cycle - m_last_sampling_cycle);
        for (int i = 0; i < m_num_routers; i++) {
            for (int j = 0; j < m_num_routers; j++) {
                sample.traffic_matrix[i][j] /= window_duration;
            }
        }

        m_traffic_samples.push_back(sample);
        if (m_traffic_samples.size() > m_num_saved_sample_points) {
            m_traffic_samples.pop_front();
        }

        m_current_sample_packets.clear();
        m_last_sampling_cycle = current_cycle;

        computeQueuingParameters();
    }

    void computeArrivalSCV() {
        for (int r = 0; r < m_num_routers; r++) {
            for (int p = 0; p < 5; p++) {
                auto& ch = m_router_channels[r][p];
                if (ch.lambda_history.size() < 2) {
                    ch.ca2 = 1.0;
                    continue;
                }

                double sum_dt = 0.0;
                double sum_sq_dt = 0.0;
                int count = 0;

                for (double hist_lambda : ch.lambda_history) {
                    if (hist_lambda > 1e-9) {
                        double dt = 1.0 / hist_lambda;
                        sum_dt += dt;
                        sum_sq_dt += dt * dt;
                        count++;
                    }
                }
                if (count < 2) {
                    ch.ca2 = 1.0;
                    continue;
                }
                double mean_dt = sum_dt / count;
                double var_dt = (sum_sq_dt / count) - (mean_dt * mean_dt);
                ch.ca2 = var_dt / (mean_dt * mean_dt);
                // ch.ca2 = 1;
                if (ch.ca2 < 0.05) ch.ca2 = 0.05;
                if (ch.ca2 > 5.0) ch.ca2 = 5.0;
            }
        }
    }

    vector<vector<double>> computeWeightedTrafficMatrix() {
        vector<vector<double>> avg_traffic_matrix(m_num_routers, vector<double>(m_num_routers, 0.0));
        double total_weight = 0.0;

        for (size_t i = 0; i < m_traffic_samples.size(); i++) {
            double weight = std::pow(m_decay_factor, m_traffic_samples.size() - 1 - i);
            total_weight += weight;

            for (int src = 0; src < m_num_routers; src++) {
                for (int dst = 0; dst < m_num_routers; dst++) {
                    avg_traffic_matrix[src][dst] += m_traffic_samples[i].traffic_matrix[src][dst] * weight;
                }
            }
        }

        for (int src = 0; src < m_num_routers; src++) {
            for (int dst = 0; dst < m_num_routers; dst++) {
                avg_traffic_matrix[src][dst] /= total_weight;
            }
        }
        return avg_traffic_matrix;
    }

    void initializeChannelParameters() {
        for (int r = 0; r < m_num_routers; r++) {
            for (int out_port = 0; out_port < 5; out_port++) {
                auto& ch = m_router_channels[r][out_port];
                ch.lambda = 0.0;
                ch.waiting_time = 0.0;
                ch.service_time = ch.base_service_time;
                ch.service_time_sq = ch.base_service_time * ch.base_service_time;
                ch.mu = 1.0 / ch.base_service_time;
                ch.ca2 = 1.0;
                ch.cs2 = 1.0;
            }
        }
    }

    void computeArrivalRates(const vector<vector<double>>& traffic_matrix) {
        for (int r = 0; r < m_num_routers; r++) {
            for (int p = 0; p < 5; p++) {
                m_router_channels[r][p].lambda = 0.0;
            }
        }

        for (int src = 0; src < m_num_routers; src++) {
            for (int dst = 0; dst < m_num_routers; dst++) {
                if (src == dst || traffic_matrix[src][dst] <= 0) continue;

                double intensity = traffic_matrix[src][dst];
                const vector<int>& path = m_routing_paths[src][dst];

                for (size_t i = 0; i < path.size(); i++) {
                    int curr = path[i];
                    int out_port = (i + 1 < path.size()) ? getOutPort(curr, path[i + 1]) : 4;
                    m_router_channels[curr][out_port].lambda += intensity;
                }
            }
        }

        for (int r = 0; r < m_num_routers; r++) {
            for (int p = 0; p < 5; p++) {
                auto& ch = m_router_channels[r][p];
                ch.lambda_history.push_back(ch.lambda);
                if (ch.lambda_history.size() > m_num_saved_sample_points) {
                    ch.lambda_history.pop_front();
                }
            }
        }
    }

    void computeWaitingTimes() {
        for (int r = 0; r < m_num_routers; r++) {
            for (int out_port = 0; out_port < 5; out_port++) {
                auto& ch = m_router_channels[r][out_port];

                if (ch.lambda <= 0) {
                    ch.waiting_time = 0.0;
                    continue;
                }

                double rho = ch.lambda / ch.mu;

                if (rho >= 0.99) {
                    rho = 0.99;
                    ch.waiting_time = SATU_THRESHOLD;
                    rho_satu_counts++;
                    continue;
                }

                rho_no_satu_counts++;

                double Wq = (rho / (1.0 - rho)) * ((ch.ca2 + ch.cs2) / 2.0) * (1.0 / ch.mu);

                ch.waiting_time = max(0.0, Wq);
            }
        }
    }

    void computeServiceTimesRecursive() {
        const double IB = static_cast<double>(m_buffer_size);

        for (int iter = 0; iter < 2; iter++) {
            for (int r = 0; r < m_num_routers; r++) {
                for (int j = 0; j < 5; j++) {
                    auto& ch = m_router_channels[r][j];

                    if (ch.lambda <= 1e-9) {
                        ch.service_time = ch.base_service_time;
                        ch.service_time_sq = ch.base_service_time * ch.base_service_time;
                        ch.mu = 1.0 / ch.service_time;
                        ch.cs2 = 0;
                        continue;
                    }

                    int downstream_r = getDownstreamRouter(r, j);

                    if (downstream_r < 0) {
                        ch.service_time = ch.base_service_time;
                        ch.service_time_sq = ch.base_service_time * ch.base_service_time;
                    } else {
                        int down_in_port = getReversePort(j);
                        auto& down_ch = m_router_channels[downstream_r][down_in_port];

                        double term = ch.base_service_time + down_ch.waiting_time + down_ch.service_time - IB;

                        term = std::max(ch.base_service_time, term);

                        ch.service_time = term;
                        ch.service_time_sq = term * term;
                    }

                    ch.mu = 1.0 / ch.service_time;

                    double sq_mean = ch.service_time * ch.service_time;
                    ch.cs2 = (ch.service_time_sq / sq_mean) - 1.0;

                    if (ch.cs2 < 0) ch.cs2 = 0;
                    if (ch.cs2 > 2.0) ch.cs2 = 2.0;
                }
            }
        }
    }

    void computeQueuingParameters() {
        if (m_traffic_samples.empty()) return;
        vector<vector<double>> smoothed_matrix = computeWeightedTrafficMatrix();
        computeArrivalRates(smoothed_matrix);
        computeArrivalSCV();

        for (int iter = 0; iter < 3; iter++) {
            computeServiceTimesRecursive();
            computeWaitingTimes();
        }
    }

    int getReversePort(int out_port) {
        switch (out_port) {
            case 0:
                return 2;  // North -> South
            case 1:
                return 3;  // East -> West
            case 2:
                return 0;  // South -> North
            case 3:
                return 1;  // West -> East
            case 4:
                return 4;  // Local
            default:
                return 4;
        }
    }
};

class NoCModel {
public:
    Config cfg;
    vector<Node>& nodes;
    M_D_1_model md1_model;
    G_G_1_model gg1_model;
    int hop_delay = 1;
    long long total_latency = 0;
    long long received_packets = 0;
    long long saturated_packets = 0;
    priority_queue<Event, vector<Event>, greater<Event>> completion_queue;
    int pkt_id = 0;
    NoCModel(const Config& cfg, vector<Node>& nodes_ref)
        : cfg(cfg),
          nodes(nodes_ref),
          gg1_model(cfg.network_dim, cfg.hop_delay, cfg.sampling_period, cfg.buffer_size, cfg.decay_factor,
                    cfg.linear_factor) {
        hop_delay = cfg.hop_delay;
    }

    void process_packet(const Packet& p, long long current_cycle) {
        int hops = abs(p.src_x - p.dst_x) + abs(p.src_y - p.dst_y);
        long long waiting_time = 0;

        if (cfg.noc_model_type == 1) {  // M/D/1
            double lambda = cfg.inj_rate;
            double L = PKT_SIZE;
            double k = cfg.network_dim;
            waiting_time = md1_model.eval_waiting_time(lambda, L, k, hops);

            if (waiting_time < 0) {
                saturated_packets++;
                waiting_time = SATU_THRESHOLD;
            }
        } else if (cfg.noc_model_type == 2) {  // G/G/1
            int src_id = p.src_y * cfg.network_dim + p.src_x;
            int dst_id = p.dst_y * cfg.network_dim + p.dst_x;

            waiting_time = gg1_model.eval_waiting_time(src_id, dst_id, PKT_SIZE, current_cycle);

            if (waiting_time < 0) {
                saturated_packets++;
                waiting_time = SATU_THRESHOLD;
            }
        }
        long long inj_queueing_time = current_cycle - p.creation_time;
        long long network_delay = hop_delay * hops + PKT_SIZE - 1;

        Event e;
        e.src_node_id = p.src_y * cfg.network_dim + p.src_x;
        if (cfg.noc_model_type == 1 || cfg.noc_model_type == 2) {
            e.latency = waiting_time + network_delay;
            e.arrival_time = p.creation_time + e.latency;
        } else {
            e.arrival_time = current_cycle + network_delay;
            e.latency = inj_queueing_time + network_delay;
        }
        pkt_id++;
        completion_queue.push(e);
    }

    void tick(long long current_cycle) {
        while (!completion_queue.empty() && completion_queue.top().arrival_time <= current_cycle) {
            Event e = completion_queue.top();
            nodes[e.src_node_id].return_credit();
            total_latency += completion_queue.top().latency;
            received_packets++;
            completion_queue.pop();
        }
    }

    void drain() {
        long long remaining = completion_queue.size();
        if (remaining > 0) {
            cout << "Warning: " << remaining << " packets remain in NoC at simulation end." << endl;
        }
        while (!completion_queue.empty()) {
            total_latency += completion_queue.top().latency;
            received_packets++;
            completion_queue.pop();
        }
    }
};

class TopLevelSim {
private:
    Config cfg;
    vector<Node> nodes;
    unique_ptr<NoCModel> noc;
    mt19937 gen;
    uniform_real_distribution<> prob_dis;

public:
    TopLevelSim(const Config& config) : cfg(config), gen(42), prob_dis(0.0, 1.0) {
        int total_nodes = cfg.network_dim * cfg.network_dim;
        for (int i = 0; i < total_nodes; ++i) {
            nodes.emplace_back(i, cfg.network_dim, cfg.inj_rate, cfg.alpha, cfg.beta);
        }

        noc = make_unique<NoCModel>(cfg, nodes);

        if (cfg.inj_is_burst == 1) {
            std::cout << "bursty injection" << std::endl;
        } else {
            std::cout << "bernoulli" << std::endl;
        }
        if (cfg.noc_model_type == 0) {
            std::cout << "ideal zero-load latency" << std::endl;
        } else if (cfg.noc_model_type == 1) {
            std::cout << "M/D/1 model" << std::endl;
        } else if (cfg.noc_model_type == 2) {
            std::cout << "G/G/1 model" << std::endl;
        }
    }

    void run() {
        int total_nodes = cfg.network_dim * cfg.network_dim;
        long long total_generated = 0;
        cout << "Starting simulation..." << endl;
        for (long long cycle = 0; cycle < cfg.sim_cycle; ++cycle) {
            for (int i = 0; i < total_nodes; ++i) {
                bool generated = nodes[i].generate_traffic(cycle, cfg, total_nodes, gen, prob_dis);
                if (generated) total_generated++;
            }
            for (int i = 0; i < total_nodes; ++i) {
                if (nodes[i].tx_busy_timer > 0) {
                    nodes[i].tx_busy_timer--;
                }
                if (nodes[i].tx_busy_timer == 0 && !nodes[i].ni_queue.empty()) {
                    Packet p = nodes[i].ni_queue.front();
                    nodes[i].ni_queue.pop();
                    nodes[i].tx_busy_timer = PKT_SIZE - 1;
                    noc->process_packet(p, cycle);
                }
            }
            noc->tick(cycle);
        }

        noc->drain();
        long long cur_cycle = cfg.sim_cycle;
        for (int i = 0; i < total_nodes; i++) {
            while (!this->nodes[i].ni_queue.empty()) {
                Packet p = this->nodes[i].ni_queue.front();
                noc->total_latency += (cur_cycle - p.creation_time);
                this->nodes[i].ni_queue.pop();
            }
        }

        if (noc->received_packets > 0) {
            cout << "Total latency: " << noc->total_latency << endl;
            double avg_latency = static_cast<double>(noc->total_latency) / noc->received_packets;
            cout << "Average Packet Latency: " << fixed << setprecision(2) << avg_latency << endl;
        } else {
            cout << "Average Packet Latency: N/A" << endl;
        }
        cout << "---------------------------------" << endl;
        if (cfg.noc_model_type == 2) {
            cout << "noc.gg1_model.rho_no_satu_counts=" << noc->gg1_model.rho_no_satu_counts << endl;
            cout << "noc.gg1_model.rho_satu_counts=" << noc->gg1_model.rho_satu_counts << endl;
        }
    }
};

int main(int argc, char* argv[]) {
    Config cfg = parse_config("./cfg.txt");

    TopLevelSim sim(cfg);
    auto start_time = std::chrono::high_resolution_clock::now();
    sim.run();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Simulation Time: " << duration.count() << " ms" << std::endl;
    return 0;
}