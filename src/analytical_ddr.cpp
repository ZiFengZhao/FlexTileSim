#include "analytical_ddr.hpp"

AnalyticalDDR::AnalyticalDDR(Simulator* sim, int id, const Config& cfg, Logger& logger, uint64_t ticks_per_cycle)
    : BaseModule(sim, cfg, logger, ticks_per_cycle, id) {
    base_latency = cfg.tBASE;
    max_outstanding = cfg.ddr_max_outstanding;
    BL = cfg.BL;
    ddr_bus_width = cfg.ddr_bus_width;
    m_request_queue_depth = cfg.ddr_request_queue_depth;
    burst_size = BL * ddr_bus_width / 8;
    int chan_num = 2;  // for mobile devices
    if (cfg.core_num == 32 || cfg.core_num == 64) {
        chan_num = 8;  // for large-scale performance evaluation, 8 channels
    }

    ddr_bandwidth_per_cycle = cfg.ddr_bandwidth / chan_num / cfg.ddr_freq;
    // ddr_bandwidth_types_per_cycle = (cfg.ddr_bandwidth * 1e9) / (npu_frequency_mhz * 1e6);
    if (m_id == 0) {
        logger.log("Initializing Analytical DDR Model");
        logger.log("DDR Frequency: %f GHz", cfg.ddr_freq);
        logger.log("Base Latency: %d in ddr cycles", base_latency);
        logger.log("DDR Bandwidth: %f per cycle", ddr_bandwidth_per_cycle);
        logger.log("Burst Size: %d bytes", burst_size);
        logger.log("DDR request queue depth: %d", m_request_queue_depth);
    }
}

void AnalyticalDDR::sendMsg(Message& msg) {
    assert(msg.type == MsgType::LD_RESP || msg.type == MsgType::ST_RESP);
    auto noc = sim->getNoC();
    bool isSent = noc->recvMsg(msg);
    if (isSent) {
        Event noc_tick_e;
        noc_tick_e.callback = [noc]() { noc->tick(); };
        noc_tick_e.tick = sim->getTick() + sim->getTicksPerNoCCycle();
        sim->schedule(noc_tick_e);

        logger.log("[Tick %d-DRAMCycle %d] DRAM %d Response sent to NoC, src addr=0x%x, dst addr=0x%x", sim->getTick(),
                   getCycle(), getId(), msg.src_addr, msg.dst_addr);
    }
}

bool AnalyticalDDR::recvMsg(Message& msg) {
    if (request_queue.size() >= m_request_queue_depth) {
        logger.log("[Tick %d-DRAMCycle %d] DRAM %d request queue is full, backpressure asserted", sim->getTick(),
                   getCycle(), getId());
        return false;  // DDR不能接收新的请求
    } else {
        request_queue.push(msg);
        logger.log("[Tick %d-DRAMCycle %d] Event Callback: DRAM %d received request from noc", sim->getTick(),
                   getCycle(), getId());

        scheduleTick(sim->getTick());
        return true;
    }
}

int AnalyticalDDR::getEstimatedLatency(const Message& msg) {
    int data_size = msg.data_size;
    int dram_cycles = base_latency + (data_size + ddr_bandwidth_per_cycle - 1) / ddr_bandwidth_per_cycle;

    return dram_cycles;
}

void AnalyticalDDR::scheduleNextEvent() {
    uint64_t next_tick = sim->getTick() + m_ticks_per_cycle;
    Event next_e;
    next_e.tick = next_tick;
    next_e.callback = [this]() { tick(); };
    sim->schedule(next_e);
}

void AnalyticalDDR::scheduleTick(uint64_t tick) {
    if (!tick_scheduled) {
        tick_scheduled = true;

        Event e;
        e.tick = tick;
        e.callback = [this]() {
            tick_scheduled = false;
            this->tick();
        };

        sim->schedule(e);
    }
}

void AnalyticalDDR::tick() {
    ScopedTimer timer(&sim->profile.ddr_time_ns);

    uint64_t now = sim->getTick();
    logger.log("[Tick %d-DRAMCycle %d] DRAM %d is ticked", now, getCycle(), getId());

    for (auto it = inflight_requests.begin(); it != inflight_requests.end();) {
        if (it->finish_tick <= now) {
            Message resp_msg;
            resp_msg.src_addr = it->msg.dst_addr;
            resp_msg.dst_addr = it->msg.src_addr;

            if (it->msg.type == MsgType::LD_REQ) {
                resp_msg.type = MsgType::LD_RESP;
                resp_msg.data_size = it->msg.request_data_size;
            } else {
                resp_msg.type = MsgType::ST_RESP;
                resp_msg.data_size = 1;
            }

            sendMsg(resp_msg);

            it = inflight_requests.erase(it);

        } else {
            ++it;
        }
    }

    while (!request_queue.empty() && inflight_requests.size() < max_outstanding) {
        ActiveRequest req;

        req.msg = std::move(request_queue.front());
        request_queue.pop();

        int dram_cycles = getEstimatedLatency(req.msg);

        req.finish_tick = now + dram_cycles * sim->getTicksPerMemoryCycle();

        inflight_requests.push_back(std::move(req));

        logger.log("[Tick %d] DRAM %d issue request (0x%x->0x%x)", now, getId(), inflight_requests.back().msg.src_addr,
                   inflight_requests.back().msg.dst_addr);
    }

    uint64_t next_finish = UINT64_MAX;

    for (auto& req : inflight_requests) {
        if (req.finish_tick < next_finish) next_finish = req.finish_tick;
    }

    if (next_finish != UINT64_MAX) {
        scheduleTick(next_finish);
    }
}

TrafficManager* AnalyticalDDR::getTrafficManager() {
    return NULL;
}