#include "detail_npu_core.hpp"

#include <iomanip>
DetailNPUCore::DetailNPUCore(Simulator* sim, int id, const Config& cfg, Logger& logger, uint64_t ticks_per_cycle)
    : BaseModule(sim, cfg, logger, ticks_per_cycle, id),
      m_state(CoreState::IDLE),
      m_pc(0),
      enable_inst_trace(cfg.enable_inst_trace),
      fetch_ready(true),
      issue_ready(true),
      decode_valid(false) {
    gdma = std::make_unique<GDMA>(sim, logger, cfg.dma_queue_depth, cfg.dma_bus_width);

    ce = std::make_unique<ComputeEngine>(sim, logger, cfg.ce_queue_depth, cfg.systolic_array_size, cfg.ce_cycle_factor,
                                         cfg.ce_cycle_bias);

    if (cfg.inst_file.find(".txt") != std::string::npos) {
        loadInstFromTxt(cfg.inst_file);
    } else if (cfg.inst_file.find(".bin") != std::string::npos) {
        loadInstFromBin(cfg.inst_file);
    } else {
        std::cerr << "Invalid instruction file format!" << std::endl;
        exit(1);
    }

    if (m_id == 0) {
        logger.log("----------------------------------------------------");
        // logger.log("Initializing NPU Core %d", m_id);
        logger.log("NPU Core Frequency: %2f GHz", cfg.npu_freq);
        logger.log("GDMA created! BUS_WIDTH(bytes): %d", cfg.dma_bus_width);
        logger.log("CE created! Systolic array size: %dx%d", cfg.systolic_array_size, cfg.systolic_array_size);
    }
    logger.log("Loaded %zu instructions for core %d", instMemory.size(), m_id);
    std::cout << "Loaded " << instMemory.size() << " instructions for core " << m_id << std::endl;
}

void DetailNPUCore::loadInstFromTxt(const std::string& inst_file) {
    std::ifstream file(inst_file);
    if (!file.is_open()) {
        logger.log("Error: Unable to open file %s\n", inst_file.c_str());
        return;
    }
    std::string target_header = "# Machine Code for Core " + std::to_string(m_id);

    std::string line;
    bool loading_instructions = false;
    static const std::string NEXT_CORE_MARKER = "# Machine Code for ";
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (line.size() >= target_header.size() && line.substr(0, target_header.size()) == target_header) {
            loading_instructions = true;
            continue;
        }
        if (loading_instructions && line.substr(0, NEXT_CORE_MARKER.length()) == NEXT_CORE_MARKER) {
            loading_instructions = false;
            break;
        }
        if (line[0] == '#') continue;
        if (loading_instructions) {
            size_t comment_start = line.find("//");
            std::string machine_code = (comment_start == std::string::npos) ? line : line.substr(0, comment_start);

            machine_code.erase(machine_code.find_last_not_of(" \t\n\r\f\v") + 1);
            if (!machine_code.empty()) {
                // convert machine code from string to uint128_t
                INST_W machine_code_hex = std::stoull(machine_code, nullptr, 16);
                instMemory.push_back(machine_code_hex);
            }
        }
    }
    file.close();
}

void DetailNPUCore::loadInstFromBin(const std::string& inst_file) {
    std::ifstream file(inst_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << inst_file << std::endl;
        return;
    }

    long index_position = m_id * sizeof(CoreIndexEntry);
    file.seekg(index_position);

    CoreIndexEntry index_entry;
    file.read(reinterpret_cast<char*>(&index_entry), sizeof(CoreIndexEntry));

    if (file.gcount() != sizeof(CoreIndexEntry) || index_entry.instruction_count == 0) {
        logger.log("No instructions found for core %d in file %s.", m_id, inst_file.c_str());
        file.close();
        return;
    }

    file.seekg(index_entry.instruction_offset);

    instMemory.resize(index_entry.instruction_count);

    file.read(reinterpret_cast<char*>(instMemory.data()), index_entry.instruction_count * sizeof(INST_W));

    if (file.gcount() != index_entry.instruction_count * sizeof(INST_W)) {
        std::cerr << "Failed to read all instructions!" << std::endl;
        instMemory.clear();
        exit(1);
    }

    file.close();
}

void DetailNPUCore::fetchStage() {
    if (m_state != CoreState::BUSY && m_state != CoreState::IDLE) return;
    if (!fetch_ready || instBuffer.size() >= static_cast<size_t>(cfg.inst_buffer_depth)) return;

    INST_W inst = instMemory[m_pc];
    instBuffer.push_back(inst);

    // record instruction trace if enabled
    if (enable_inst_trace) {
        _inst_trace.push_back(InstTrace{getCycle(), getId(), m_pc, true});
        if (_inst_trace.size() >= TRACE_BUFFER_SIZE) {
            _inst_trace.clear();
        }
    }
    uint64_t lo = static_cast<uint64_t>(inst);
    uint64_t hi = static_cast<uint64_t>(inst >> 64);
    logger.log("[Tick %d-CoreCycle %d] Core %d Fetch: PC=%d, machine_code=0x%016lx_%016lx", sim->getTick(), getCycle(),
               getId(), m_pc, hi, lo);
    m_pc++;

    constexpr uint64_t LOG_INTERVAL = 1000;
    if (m_id == 0 && m_pc % LOG_INTERVAL == 0) {
        size_t total_inst = instMemory.size();
        double progress = (static_cast<double>(m_pc) / total_inst) * 100.0;
        const int bar_width = 50;
        int pos = static_cast<int>(bar_width * progress / 100.0);

        std::cout << "\r[" << std::string(pos, '=') << std::string(bar_width - pos, ' ') << "] " << std::fixed
                  << std::setprecision(2) << progress << "%"
                  << " (PC: " << m_pc << "/" << total_inst << ")" << std::flush;

        if (m_pc >= total_inst) {
            std::cout << std::endl;
        }
    }

    if (m_state == CoreState::IDLE && m_pc > 0) {
        m_state = CoreState::BUSY;
    }
    if (m_pc >= instMemory.size()) {
        fetch_ready = false;  // 指令memory中已经没有指令，取指级不再取值
        std::cout << "[Core " << getId() << "] m_pc reaches instMemory size" << std::endl;
    }
}

void DetailNPUCore::decodeStage() {
    if (instBuffer.empty()) return;

    if (issue_ready) {
        INST_W machine_code = instBuffer.front();
        fetch_decode_reg = InstDecode::decodeMachineCode(machine_code);
        decode_valid = true;
        instBuffer.pop_front();
        logger.log("[Tick %d-CoreCycle %d] Core %d Decode: InstID=%d, InstType=%s", sim->getTick(), getCycle(), getId(),
                   fetch_decode_reg.id, fetch_decode_reg.getType().c_str());
    }
}

void DetailNPUCore::issueStage() {
    issue_ready = !(gdma->isFull() || ce->isFull());

    if (decode_valid && issue_ready) {
        if (fetch_decode_reg.target_unit == UnitType::GDMA || fetch_decode_reg.target_unit == UnitType::CU) {
            gdma->issue(fetch_decode_reg);
            logger.log("[Tick %d-CoreCycle %d] Core %d Issue: Push InstID=%d to GDMA Queue", sim->getTick(), getCycle(),
                       getId(), fetch_decode_reg.id);
        } else if (fetch_decode_reg.target_unit == UnitType::CE) {
            ce->issue(fetch_decode_reg);
            logger.log("[Tick %d-CoreCycle %d] Core %d Issue: Push InstID=%d to CE Queue", sim->getTick(), getCycle(),
                       getId(), fetch_decode_reg.id);
        }

        decode_valid = false;
    }
}

void DetailNPUCore::executeStage() {
    if (ce->isBusy()) {
        ce->tick();
        logger.log("[Tick %d-CoreCycle %d] Core %d Execute: [CE] is busy, remaining cycles: %d", sim->getTick(),
                   getCycle(), getId(), ce->getRemainingCycles());
    } else if (ce->isFinish()) {
        Inst& cur_inst = ce->peek();

        _inst_trace.push_back(InstTrace{getCycle(), getId(), cur_inst.id, false});

        logger.log("[Tick %d-CoreCycle %d] Core %d Execute: [CE] completes current InstID=%d", sim->getTick(),
                   getCycle(), getId(), cur_inst.id);
        if (cur_inst.dep == DepType::NO_DEP) {
            ce->setIdle();
            ce->pop();
            logger.log(
                "[Tick %d-CoreCycle %d] Core %d Execute: [CE] current InstID=%d has no dependency, CE jumps to IDLE",
                sim->getTick(), getCycle(), getId(), cur_inst.id);
        } else if (cur_inst.dep == DepType::PRODUCER) {
            if (gdma->getConsExec() == true) {
                ce->setIdle();
                ce->pop();
                ce->setProdWait(false);
                gdma->setConsExec(false);
                logger.log(
                    "[Tick %d-CoreCycle %d] Core %d Execute: [CE] current InstID=%d is a producer, and its consumer "
                    "starts executing, CE jumps to IDLE, and clears GDMA's cons_exec flag",
                    sim->getTick(), getCycle(), getId(), cur_inst.id);
            } else {
                logger.log(
                    "[Tick %d-CoreCycle %d] Core %d Execute: [CE] current InstID=%d is a producer, but its consumer "
                    "has not started executing, CE stays in FINISH",
                    sim->getTick(), getCycle(), getId(), cur_inst.id);
            }
        } else if (cur_inst.dep == DepType::CONSUMER) {
            ce->setIdle();
            ce->pop();
            ce->setConsExec(false);
            gdma->setProdWait(false);
            logger.log("[Tick %d-CoreCycle %d] Core %d Execute: [CE] current InstID=%d is a consumer, CE jumps to IDLE",
                       sim->getTick(), getCycle(), getId(), cur_inst.id);
        } else {
            ce->setConsExec(false);
            if (gdma->getConsExec() == true) {
                ce->setIdle();
                ce->pop();
                ce->setProdWait(false);
            }
        }
    } else if (ce->isIdle()) {
        if (!ce->isEmpty()) {
            Inst& ce_inst = ce->peek();
            bool can_exec = syncCheck(ce_inst);
            if (can_exec) {
                if (ce_inst.dep == DepType::CONSUMER) {
                    ce->setConsExec(true);
                } else if (ce_inst.dep == DepType::PRODUCER) {
                    ce->setProdWait(true);
                } else if (ce_inst.dep == DepType::BOTH) {
                    ce->setConsExec(true);
                    ce->setProdWait(true);
                }
                ce->execute();
                logger.log(
                    "[Tick %d-CoreCycle %d] Core %d Execute: [CE] starts to execute InstID=%d, remaining cycles: %d",
                    sim->getTick(), getCycle(), getId(), ce_inst.id, ce->getRemainingCycles());
            } else {
                logger.log(
                    "[Tick %d-CoreCycle %d] Core %d Execute: [CE] current Inst%d is blocked by producer's exeuction, "
                    "waiting...",
                    sim->getTick(), getCycle(), getId(), ce_inst.id);
            }
        }
    }

    if (gdma->isBusy()) {
        gdma->tick();
        logger.log("[Tick %d-CoreCycle %d] Core %d Execute: [GDMA] is busy", sim->getTick(), getCycle(), getId());
    } else if (gdma->isFinish()) {
        Inst& cur_inst = gdma->peek();

        _inst_trace.push_back(InstTrace{getCycle(), getId(), cur_inst.id, false});

        logger.log("[Tick %d-CoreCycle %d] Core %d Execute: [GDMA] completes current InstID=%d", sim->getTick(),
                   getCycle(), getId(), cur_inst.id);

        if (cur_inst.dep == DepType::NO_DEP) {
            if (gdma->getProdWait() == false) {
                gdma->setIdle();
                gdma->pop();
                logger.log(
                    "[Tick %d-CoreCycle %d] Core %d Execute: [GDMA] current InstID=%d has no dependency, and "
                    "prod_wait=0, GDMA jumps to IDLE",
                    sim->getTick(), getCycle(), getId(), cur_inst.id);
            } else {
                logger.log(
                    "[Tick %d-CoreCycle %d] Core %d Execute: [GDMA] current InstID=%d has no dependency, but "
                    "prod_wait=1, GDMA stays in FINISH",
                    sim->getTick(), getCycle(), getId(), cur_inst.id);
            }
        } else if (cur_inst.dep == DepType::PRODUCER) {
            if (ce->getConsExec() == true) {
                gdma->setIdle();
                gdma->pop();
                gdma->setProdWait(false);
                ce->setConsExec(false);
                logger.log(
                    "[Tick %d-CoreCycle %d] Core %d Execute: GDMA current InstID=%d is a producer, and its consumer "
                    "starts executing, GDMA jumps to IDLE, and clears CE's cons_exec flag",
                    sim->getTick(), getCycle(), getId(), cur_inst.id);
            } else {
                if (gdma->getQueueSize() == 1) {
                    logger.log(
                        "[Tick %d-CoreCycle %d] Core %d Execute: GDMA current InstID=%d is a producer, and its "
                        "consumer has not started executing, and there is no valid inst in GDMA issue queue, GDMA "
                        "stays in FINISH",
                        sim->getTick(), getCycle(), getId(), cur_inst.id);
                } else {
                    logger.log(
                        "[Tick %d-CoreCycle %d] Core %d Execute: GDMA current InstID=%d is a producer, but its "
                        "consumer has not started executing, check if the next GDMA instruction has dependency",
                        sim->getTick(), getCycle(), getId(), cur_inst.id);
                    Inst& nxt_inst = gdma->peek_second();
                    if (nxt_inst.dep == DepType::NO_DEP) {
                        logger.log(
                            "[Tick %d-CoreCycle %d] Core %d Execute: GDMA the next GDMA instruction has no dependency, "
                            "GDMA jumps to IDLE",
                            sim->getTick(), getCycle(), getId());
                        gdma->setIdle();
                        gdma->pop();
                    } else {
                        logger.log(
                            "[Tick %d-CoreCycle %d] Core %d Execute: GDMA the next GDMA instruction has dependency, "
                            "GDMA stays in FINISH",
                            sim->getTick(), getCycle(), getId());
                    }
                }
            }
        } else if (cur_inst.dep == DepType::CONSUMER) {
            gdma->setIdle();
            gdma->pop();
            gdma->setConsExec(false);
        } else {
            gdma->setConsExec(false);
            if (ce->getConsExec() == true) {
                gdma->setIdle();
                gdma->pop();
                gdma->setProdWait(false);
            } else {
                logger.log(
                    "[Tick %d-CoreCycle %d] Core %d Execute: GDMA current InstID=%d is both producer and consumer, but "
                    "its consumer has not started executing, check if the next GDMA instruction has dependency",
                    sim->getTick(), getCycle(), getId(), cur_inst.id);
                Inst& nxt_inst = gdma->peek_second();
                if (nxt_inst.dep == DepType::NO_DEP) {
                    logger.log(
                        "[Tick %d-CoreCycle %d] Core %d Execute: GDMA the next GDMA instruction has no dependency, "
                        "GDMA jumps to IDLE",
                        sim->getTick(), getCycle(), getId());
                    gdma->setIdle();
                } else {
                    logger.log(
                        "[Tick %d-CoreCycle %d] Core %d Execute: [GDMA] the next GDMA instruction has dependency, GDMA "
                        "stays in FINISH",
                        sim->getTick(), getCycle(), getId());
                }
            }
        }
    } else if (gdma->isIdle()) {
        if (!gdma->isEmpty()) {
            Inst& gdma_inst = gdma->peek();
            bool can_exec = syncCheck(gdma_inst);
            logger.log(
                "[Tick %d-CoreCycle %d] Core %d Execute: [GDMA] check if the GDMA instruction can execute, can_exec=%d",
                sim->getTick(), getCycle(), getId(), can_exec);
            if (can_exec) {
                if (gdma_inst.dep == DepType::CONSUMER) {
                    gdma->setConsExec(true);
                } else if (gdma_inst.dep == DepType::PRODUCER) {
                    gdma->setProdWait(true);
                } else if (gdma_inst.dep == DepType::BOTH) {
                    gdma->setConsExec(true);
                    gdma->setProdWait(true);
                }
                gdma->execute();
                logger.log("[Tick %d-CoreCycle %d] Core %d Execute: GDMA starts to execute InstID=%d", sim->getTick(),
                           getCycle(), getId(), gdma_inst.id);
            } else {
                logger.log("[Tick %d-CoreCycle %d] Core %d Execute: GDMA waits to execute InstID=%d", sim->getTick(),
                           getCycle(), getId(), gdma_inst.id);
            }
        }
    }
}

bool DetailNPUCore::syncCheck(Inst& inst) {
    UnitType target_unit = inst.target_unit;
    DepType dep = inst.dep;
    bool can_exec = false;
    if (dep == DepType::NO_DEP || dep == DepType::PRODUCER) {
        can_exec = true;
    }

    if (dep == DepType::CONSUMER || dep == DepType::BOTH) {
        if (target_unit == UnitType::GDMA) {
            DepType ce_dep = ce->peek().dep;
            if (ce->isFinish() && (ce->peek().dep == DepType::PRODUCER || ce_dep == DepType::BOTH)) {
                can_exec = true;
            }
        } else if (target_unit == UnitType::CE) {
            DepType gdma_dep = gdma->peek().dep;
            if (gdma->isFinish() && (gdma_dep == DepType::PRODUCER || gdma_dep == DepType::BOTH)) {
                can_exec = true;
            } else if (gdma->isFinish() && gdma_dep == DepType::NO_DEP) {
                can_exec = true;
            }
        }
    }
    return can_exec;
}

void DetailNPUCore::scheduleNextEvent() {
    uint64_t next_tick = m_cycle * m_ticks_per_cycle;
    Event next_e;
    next_e.tick = next_tick;
    next_e.callback = [this]() { tick(); };
    sim->schedule(next_e);
}

void DetailNPUCore::tick() {
    ScopedTimer timer(&sim->profile.core_time_ns);

    logger.log("[Tick %d-CoreCycle %d] Core %d is ticked", sim->getTick(), getCycle(), getId());

    executeStage();  // E
    issueStage();    // I
    decodeStage();   // D
    fetchStage();    // F

    checkCompletion();

    m_cycle++;
    if (!isDone()) {
        scheduleNextEvent();
    }
}

void DetailNPUCore::checkCompletion() {
    if (m_state == CoreState::BUSY && m_pc >= instMemory.size() && instBuffer.empty() && !decode_valid &&
        gdma->isEmpty() && ce->isEmpty()) {
        m_state = CoreState::DONE;
        sim->incrFinishedCoresCnt();
        logger.log("[Tick %d-CoreCycle %d] Core %d: instruction execution completed at cycle %d", sim->getTick(),
                   getCycle(), getId(), getCycle());
    } else if (m_cycle >= static_cast<uint64_t>(cfg.max_sim_cycle - 1)) {
        logger.log("[Tick %d-CoreCycle %d] Core %d: Simulation timeout at cycle %d", sim->getTick(), getCycle(),
                   getId(), getCycle());
        std::cout << "Core " << getId() << ": Simulation timeout at cycle " << getCycle() << std::endl;
    }
}

void DetailNPUCore::sendMsg(Message& msg) {
    gdma->sendMessage(msg);
}

bool DetailNPUCore::recvMsg(Message& msg) {
    gdma->recvMessage(msg);
    return true;
}

void DetailNPUCore::dumpInstTrace() {
    std::string save_dir = cfg.log_dir;
    std::ofstream inst_trace_file;
    inst_trace_file.open(save_dir + "/inst_trace_core_" + std::to_string(getId()) + ".txt");
    for (InstTrace& inst : _inst_trace) {
        inst_trace_file << "[" << inst.core_cycle << "] "
                        << "inst" << inst.inst_id << " " << (inst.is_start ? "start" : "end") << std::endl;
    }
}

TrafficManager* DetailNPUCore::getTrafficManager() {
    return NULL;
}
