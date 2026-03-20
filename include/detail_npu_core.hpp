#ifndef __DETAIL_NPU_CORE_HPP__
#define __DETAIL_NPU_CORE_HPP__

#include <deque>
#include <memory>
#include <vector>

#include "base_module.hpp"
#include "config.hpp"
#include "func_unit.hpp"
#include "inst.hpp"
#include "logger.hpp"
#include "util.hpp"

class Simulator;  // forward declaration
class GDMA;
class ComputeEngine;

class DetailNPUCore : public BaseModule {
private:
    // functional units
    std::unique_ptr<GDMA> gdma;
    std::unique_ptr<ComputeEngine> ce;

    // external inst memory and internal inst buffer
    std::vector<INST_W> instMemory;
    std::deque<INST_W> instBuffer;

    // Core status
    enum class CoreState { IDLE, BUSY, DONE } m_state;
    uint64_t m_pc;

    // statistics
    struct InstTrace {
        uint64_t core_cycle;
        int core_id;
        uint64_t inst_id;
        bool is_start;
    };
    std::vector<InstTrace> _inst_trace;
    static const size_t TRACE_BUFFER_SIZE = 10000;
    bool enable_inst_trace = false;

    bool fetch_ready = true;
    bool issue_ready = true;
    bool decode_valid = false;
    Inst fetch_decode_reg;

public:
    DetailNPUCore(Simulator* sim, int id, const Config& cfg, Logger& logger, uint64_t ticks_per_cycle);
    ~DetailNPUCore() = default;

    void tick() override;
    void sendMsg(Message& msg) override;
    bool recvMsg(Message& msg) override;
    void scheduleNextEvent() override;
    TrafficManager* getTrafficManager() override;
    void loadInstFromTxt(const std::string& inst_file);
    void loadInstFromBin(const std::string& inst_file);

    void fetchStage();
    void decodeStage();
    void issueStage();
    void executeStage();
    void checkCompletion();
    bool syncCheck(Inst& inst);

    bool isDone() const { return m_state == CoreState::DONE; }
    void dumpInstTrace();
    uint64_t getPC() const { return m_pc; };
};

#endif  // __NPU_CORE_HPP__