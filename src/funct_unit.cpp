#include <cassert>

#include "func_unit.hpp"
#include "util.hpp"

FunctionalUnit::FunctionalUnit(Simulator* sim, Logger& logger, size_t queue_depth)
    : simulator(sim), logger(logger), queue_depth(queue_depth) {}

bool FunctionalUnit::isFull() const {
    return issue_queue.size() >= queue_depth;
}

bool FunctionalUnit::isEmpty() const {
    return issue_queue.empty();
}

size_t FunctionalUnit::getQueueSize() const {
    return issue_queue.size();
}

size_t FunctionalUnit::getQueueDepth() const {
    return queue_depth;
}

void FunctionalUnit::issue(const Inst& inst) {
    assert(!isFull() && "Issue queue full");
    issue_queue.push_back(inst);
}

void FunctionalUnit::completeInstruction() {
    assert(state == FUState::BUSY && !issue_queue.empty());
    state = FUState::FINISH;
}

void FunctionalUnit::execute() {
    // base class default behavior: do nothing
}

void FunctionalUnit::tick() {
    // base class default behavior: do nothing
}

GDMA::GDMA(Simulator* sim, Logger& logger, int queue_depth, int bus_width)
    : FunctionalUnit(sim, logger, queue_depth),
      dma_bus_width(bus_width),
      isRespValid(false),
      dma_timeout_counter(0),
      dma_timeout_threshold(2000) {}

void GDMA::tick() {
    if (isBusy()) {
        Inst& exec_inst = issue_queue.front();
        if (exec_inst.type == InstType::SYNC) {
            int sync_id = exec_inst.sync_id;
            int total_sync_num = exec_inst.total_cores_to_sync;
            int sync_cnt = simulator->getSyncCnt(sync_id);
            if (sync_cnt == total_sync_num) {
                completeInstruction();

                simulator->sendSyncAck(sync_id);
                logger.log("[GDMA] Global Sync (inst_id=%d, sync_id=%d) finished", exec_inst.id, sync_id);
                return;
            }
        }
        dma_timeout_counter++;
        if (isRespValid == true) {
            completeInstruction();
        } else if (dma_timeout_counter > dma_timeout_threshold) {
            logger.log("[GDMA WARNING] DMA timeout! force complete inst_id=%d", exec_inst.id);
            completeInstruction();
        }
    }
}

void GDMA::execute() {
    assert(!isBusy() && !issue_queue.empty());
    assert(state == FUState::IDLE);

    Inst& exec_inst = issue_queue.front();
    if (exec_inst.type == InstType::SYNC) {
        logger.log("[GDMA] Global Sync (inst_id=%d, sync_id=%d) started", exec_inst.id, exec_inst.sync_id);
        simulator->initSyncBarrier(exec_inst.sync_id, exec_inst.total_cores_to_sync);
        state = FUState::BUSY;
        return;
    }

    int req_data_size = exec_inst.tensor_height * exec_inst.tensor_width;
    uint32_t ddr_addr = exec_inst.addr1;
    uint32_t ocm_addr = exec_inst.addr0;
    logger.log("[GDMA] %s (inst_id=%d, addr1=%d, addr0=%d, data_size=%d) started",
               exec_inst.type == InstType::MVIN ? "MVIN" : "MVOUT", exec_inst.id, ddr_addr, ocm_addr, req_data_size);
    Message req_msg;
    if (exec_inst.type == InstType::MVIN) {
        req_msg.type = MsgType::LD_REQ;
        req_msg.data_size = 1;
        req_msg.request_data_size = req_data_size;
    } else if (exec_inst.type == InstType::MVOUT) {
        req_msg.type = MsgType::ST_REQ;
        req_msg.data_size = req_data_size;
        req_msg.request_data_size = 1;
    } else {
        assert(false && "Invalid GDMA instruction type");
    }
    req_msg.dst_addr = ddr_addr;
    req_msg.src_addr = ocm_addr;

    if (sendMessage(req_msg) == true) {
        state = FUState::BUSY;
        dma_timeout_counter = 0;
        Event noc_tick_e;
        auto noc = simulator->getNoC();
        noc_tick_e.callback = [noc]() { noc->tick(); };
        noc_tick_e.tick = simulator->getTick() + simulator->getTicksPerNoCCycle();
        simulator->schedule(noc_tick_e);
        logger.log("[GDMA] Successfully sent request to NoC, GDMA set to busy now");
    } else {
        logger.log("[GDMA] NoC Request queue is full, GDMA cannot send request now");
    }
}

bool GDMA::sendMessage(Message& msg) {
    assert(msg.type == MsgType::LD_REQ || msg.type == MsgType::ST_REQ);
    logger.log("[GDMA] Sending request to NoC");
    auto noc = simulator->getNoC();
    bool isSent = noc->recvMsg(msg);
    return isSent;
}

void GDMA::recvMessage(Message& msg) {
    assert(msg.type == MsgType::LD_RESP || msg.type == MsgType::ST_RESP);
    isRespValid = true;
    logger.log("[GDMA] Receive response from DDR");
}

void GDMA::completeInstruction() {
    FunctionalUnit::completeInstruction();
    isRespValid = false;
}

ComputeEngine::ComputeEngine(Simulator* sim, Logger& logger, int queue_depth, int array_size, int ce_cycle_factor,
                             int ce_cycle_bias)
    : FunctionalUnit(sim, logger, queue_depth),
      systolic_array_size(array_size),
      ce_cycle_factor(ce_cycle_factor),
      ce_cycle_bias(ce_cycle_bias) {}

void ComputeEngine::tick() {
    if (isBusy()) {
        remaining_cycles--;
        if (remaining_cycles == 0) {
            completeInstruction();
        }
    }
}

void ComputeEngine::execute() {
    assert(!isBusy() && !issue_queue.empty());
    assert(state == FUState::IDLE);
    state = FUState::BUSY;

    Inst& exec_inst = issue_queue.front();
    remaining_cycles = calculateLatency(exec_inst);
    logger.log("[CE] will execute compute instruction: %s, estimated latency = %d cycles", exec_inst.getType().c_str(),
               remaining_cycles);
}

int ComputeEngine::calculateLatency(const Inst& inst) {
    switch (inst.type) {
        case InstType::GEMM:
            return calculateGEMMLatency(inst);
        case InstType::RET:
            return 1;
        case InstType::ADD:
            return calculateADDLatency(inst);
        case InstType::RELU:
            return calculateRELULatency(inst);
        case InstType::XPOSE:
            return calculateXPOSELatency(inst);
        case InstType::SOFTMAX:
            return calculateSOFTMAXLatency(inst);
        case InstType::LAYERNORM:
            return calculateLAYERNORMLatency(inst);
        case InstType::GELU:
            return calculateGELULatency(inst);
        default:
            std::cerr << "Unknown instruction type!" << std::endl;
            exit(1);
    }
}

int ComputeEngine::calculateGEMMLatency(const Inst& inst) {
    int cycles = systolic_array_size * 2;
    return cycles;
}

int ComputeEngine::calculateADDLatency(const Inst& inst) {
    return ce_cycle_factor * systolic_array_size + ce_cycle_bias;
}

int ComputeEngine::calculateRELULatency(const Inst& inst) {
    return ce_cycle_factor * systolic_array_size + ce_cycle_bias;
}

int ComputeEngine::calculateXPOSELatency(const Inst& inst) {
    return 1;
}

int ComputeEngine::calculateSOFTMAXLatency(const Inst& inst) {
    return ce_cycle_factor * systolic_array_size + ce_cycle_bias;
}

int ComputeEngine::calculateLAYERNORMLatency(const Inst& inst) {
    return ce_cycle_factor * systolic_array_size + ce_cycle_bias;
}

int ComputeEngine::calculateGELULatency(const Inst& inst) {
    return ce_cycle_factor * systolic_array_size + ce_cycle_bias;
}

void ComputeEngine::completeInstruction() {
    FunctionalUnit::completeInstruction();
}