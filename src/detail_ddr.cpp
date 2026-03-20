#include "detail_ddr.hpp"

#include <cassert>

DetailDDR::DetailDDR(Simulator* sim, int id, const Config& cfg, Logger& logger, uint64_t ticks_per_cycle)
    : BaseModule(sim, cfg, logger, ticks_per_cycle, id) {
    std::string config_file = cfg.dramsim_cfg_path;
    std::string output_dir = "./log/dramsim3_output";

    m_dramsim = std::make_unique<dramsim3::MemorySystem>(
        config_file, output_dir, std::bind(&DetailDDR::readComplete, this, std::placeholders::_1),
        std::bind(&DetailDDR::writeComplete, this, std::placeholders::_1));
    m_request_queue_depth = cfg.ddr_request_queue_depth;

    if (m_id == 0) {
        logger.log("DDR Frequency: %2f GHz", cfg.ddr_freq);
    }
}

void DetailDDR::scheduleNextEvent() {
    uint64_t next_tick = m_cycle * m_ticks_per_cycle;
    Event next_e;
    next_e.tick = next_tick;
    next_e.callback = [this]() { tick(); };
    sim->schedule(next_e);
}

void DetailDDR::tick() {
    ScopedTimer timer(&sim->profile.ddr_time_ns);

    logger.log("[Tick %d-DRAMCycle %d] DRAM %d is ticked", sim->getTick(), getCycle(), getId());
    m_cycle++;
    m_dramsim->ClockTick();

    const uint64_t MAX_TRANS_TICKS = 100;
    if (current_request.active) {
        if (sim->getTick() - current_request.start_tick > MAX_TRANS_TICKS) {
            for (int i = 0; i < current_request.total_trans; i++) {
                current_request.completed_trans[i] = true;
            }
            current_request.completed_count = current_request.total_trans;

            Message resp_msg = current_request.msg;
            resp_msg.dst_addr = current_request.msg.src_addr;
            resp_msg.src_addr = current_request.msg.dst_addr;
            resp_msg.type = (current_request.msg.type == MsgType::ST_REQ) ? MsgType::ST_RESP : MsgType::LD_RESP;

            sendMsg(resp_msg);
            clearCurrentRequest();
        }
    }
    if (current_request.active) {
        if (current_request.trans_index < current_request.total_trans) {
            uint64_t addr = current_request.msg.dst_addr + current_request.trans_index * dram_unit;
            bool is_write = (current_request.msg.type == MsgType::ST_REQ);

            if (m_dramsim->WillAcceptTransaction(addr, is_write)) {
                m_dramsim->AddTransaction(addr, is_write);
                current_request.trans_index++;
                logger.log("[Tick %d-DRAMCycle %d] DRAM %d Sent transaction %d/%d", sim->getTick(), getCycle(), getId(),
                           current_request.trans_index, current_request.total_trans);
            }
        }
        scheduleNextEvent();
        return;
    }

    if (!request_queue.empty()) {
        Message& msg = request_queue.front();
        logger.log("[Tick %d-DRAMCycle %d] DRAM %d Processing new message (type=%s, size=%d)", sim->getTick(),
                   getCycle(), getId(), (msg.type == MsgType::ST_REQ ? "STORE_REQ" : "LOAD_REQ"), msg.data_size);

        int access_data_size;
        if (msg.type == MsgType::ST_REQ) {
            access_data_size = msg.data_size;
        } else {
            access_data_size = msg.request_data_size;
        }
        int num_trans = (access_data_size + dram_unit - 1) / dram_unit;
        logger.log("[Tick %d-DRAMCycle %d] DRAM %d Message requires %d transactions", sim->getTick(), getCycle(),
                   getId(), num_trans);

        current_request.active = true;
        current_request.msg = msg;
        current_request.total_trans = num_trans;
        current_request.trans_index = 0;
        current_request.completed_count = 0;
        current_request.completed_trans.resize(num_trans, false);
        current_request.start_tick = sim->getTick();

        request_queue.pop();

        uint64_t addr = msg.dst_addr;
        bool is_write = (msg.type == MsgType::ST_REQ);

        logger.log("[Tick %d-DRAMCycle %d] DRAM %d Attempting to send first transaction to Dramsim3", sim->getTick(),
                   getCycle(), getId());

        if (m_dramsim->WillAcceptTransaction(addr, is_write)) {
            m_dramsim->AddTransaction(addr, is_write);
            current_request.trans_index++;
            logger.log("[Tick %d-DRAMCycle %d] DRAM %d First transaction sent successfully", sim->getTick(), getCycle(),
                       getId());
        }
    }
    scheduleNextEvent();
}

void DetailDDR::sendMsg(Message& msg) {
    assert(msg.type == MsgType::LD_RESP || msg.type == MsgType::ST_RESP);
    auto noc = sim->getNoC();
    bool isSent = noc->recvMsg(msg);
    if (isSent) {
        Event noc_tick_e;
        noc_tick_e.callback = [noc]() { noc->tick(); };
        noc_tick_e.tick = sim->getTick() + sim->getTicksPerNoCCycle();
        sim->schedule(noc_tick_e);
        logger.log("[Tick %d-DRAMCycle %d] DRAM %d Response sent to NoC", sim->getTick(), getCycle(), getId());
        // print src addr for debug
        logger.log("[Tick %d-DRAMCycle %d] DRAM %d Response sent to NoC, src addr=0x%x, dst addr=0x%x", sim->getTick(),
                   getCycle(), getId(), msg.src_addr, msg.dst_addr);
    }
}

bool DetailDDR::recvMsg(Message& msg) {
    if (request_queue.size() >= m_request_queue_depth) {
        logger.log("[Tick %d-DRAMCycle %d] DRAM %d request queue is full, backpressure asserted", sim->getTick(),
                   getCycle(), getId());
        logger.log("[DRAM DEBUG] Queue size=%d active=%d trans_index=%d/%d, completed=%d", request_queue.size(),
                   current_request.active, current_request.trans_index, current_request.total_trans,
                   current_request.completed_count);

        throw std::runtime_error("DRAM request queue is full");
        return false;
    } else {
        request_queue.push(msg);
        logger.log("[Tick %d-DRAMCycle %d] Event Callback: DRAM %d received request from noc", sim->getTick(),
                   getCycle(), getId());
        return true;
    }
}

void DetailDDR::readComplete(uint64_t addr) {
    logger.log("[Tick %d-DRAMCycle %d] DRAM %d Read request completed", sim->getTick(), getCycle(), getId());

    if (!current_request.active) {
        return;
    }

    Message& msg = current_request.msg;
    uint64_t base_addr = msg.dst_addr;
    int trans_index = (addr - base_addr) / dram_unit;

    logger.log("[Tick %d-DRAMCycle %d] DRAM %d Read completion for transaction %d/%d", sim->getTick(), getCycle(),
               getId(), trans_index, current_request.total_trans);

    logger.log("[Tick %d-DRAMCycle %d] DRAM %d Checking transaction %d ...", sim->getTick(), getCycle(), getId(),
               trans_index);

    if (trans_index >= 0 && trans_index < current_request.total_trans &&
        !current_request.completed_trans[trans_index]) {
        current_request.completed_trans[trans_index] = true;
        current_request.completed_count++;
        logger.log("[Tick %d-DRAMCycle %d] DRAM %d Transaction marked as completed (%d/%d)", sim->getTick(), getCycle(),
                   getId(), current_request.completed_count, current_request.total_trans);

        if (current_request.completed_count == current_request.total_trans &&
            current_request.trans_index == current_request.total_trans) {
            Message resp_msg = msg;
            // swap msg src/dst addr
            resp_msg.dst_addr = msg.src_addr;
            resp_msg.src_addr = msg.dst_addr;
            resp_msg.type = MsgType::LD_RESP;

            sendMsg(resp_msg);

            clearCurrentRequest();
        }
    }
}

void DetailDDR::writeComplete(uint64_t addr) {
    if (!current_request.active) {
        return;
    }

    Message& msg = current_request.msg;
    uint64_t base_addr = msg.dst_addr;
    int trans_index = (addr - base_addr) / dram_unit;

    logger.log("[Tick %d-DRAMCycle %d] DRAM %d Write completion for transaction %d/%d", sim->getTick(), getCycle(),
               getId(), trans_index, current_request.total_trans);

    if (trans_index >= 0 && trans_index < current_request.total_trans &&
        !current_request.completed_trans[trans_index]) {
        current_request.completed_trans[trans_index] = true;
        current_request.completed_count++;
        logger.log("[Tick %d-DRAMCycle %d] DRAM %d Transaction marked as completed (%d/%d)", sim->getTick(), getCycle(),
                   getId(), current_request.completed_count, current_request.total_trans);

        if (current_request.completed_count == current_request.total_trans &&
            current_request.trans_index == current_request.total_trans) {
            logger.log("[Tick %d-DRAMCycle %d] DRAM %d All transactions completed, sending response", sim->getTick(),
                       getCycle(), getId());

            Message resp_msg = msg;
            // swap msg src/dst addr
            resp_msg.dst_addr = msg.src_addr;
            resp_msg.src_addr = msg.dst_addr;
            resp_msg.type = MsgType::ST_RESP;
            sendMsg(resp_msg);

            clearCurrentRequest();
        }
    }
}

void DetailDDR::clearCurrentRequest() {
    logger.log("[Tick %d-DRAMCycle %d] DRAM %d Clearing current request state", sim->getTick(), getCycle(), getId());
    current_request.active = false;
    current_request.msg = Message();
    current_request.completed_count = 0;
    current_request.total_trans = 0;
    current_request.trans_index = 0;
    current_request.completed_trans.clear();
    current_request.start_tick = 0;
    logger.log("[Tick %d-DRAMCycle %d] DRAM %d Request state cleared", sim->getTick(), getCycle(), getId());
}

TrafficManager* DetailDDR::getTrafficManager() {
    return NULL;
}