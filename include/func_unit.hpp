#ifndef __FUNC_UNIT_HPP__
#define __FUNC_UNIT_HPP__

#include <deque>

#include "inst.hpp"
#include "logger.hpp"
#include "simulator.hpp"
#include "util.hpp"

class Simulator;

class FunctionalUnit {
public:
    Simulator* simulator;
    Logger& logger;
    std::deque<Inst> issue_queue;
    size_t queue_depth;
    int remaining_cycles = 0;
    bool prod_wait = false;
    bool cons_exec = false;

    enum class FUState { IDLE, BUSY, FINISH } state = FUState::IDLE;

public:
    FunctionalUnit(Simulator* sim, Logger& logger, size_t queue_depth);
    virtual ~FunctionalUnit() = default;

    bool isFull() const;
    bool isEmpty() const;
    bool isIdle() const { return state == FUState::IDLE; };
    bool isBusy() const { return state == FUState::BUSY; };
    bool isFinish() const { return state == FUState::FINISH; };
    void setIdle() { state = FUState::IDLE; };
    int getRemainingCycles() const { return remaining_cycles; };
    size_t getQueueSize() const;
    size_t getQueueDepth() const;
    Inst& peek() { return issue_queue.front(); };
    Inst& peek_second() { return issue_queue.at(1); };
    void pop() { issue_queue.pop_front(); };
    void setProdWait(bool flag) { prod_wait = flag; };
    bool getProdWait() { return prod_wait; };
    void setConsExec(bool flag) { cons_exec = flag; };
    bool getConsExec() { return cons_exec; };

    virtual void issue(const Inst& inst);

    virtual void execute();

    virtual void tick();

protected:
    virtual void completeInstruction();
};

class GDMA : public FunctionalUnit {
private:
    int dma_bus_width;
    bool isRespValid = false;
    int dma_timeout_counter;
    int dma_timeout_threshold;

public:
    GDMA(Simulator* sim, Logger& logger, int queue_depth, int bus_width);
    void execute() override;
    void tick() override;
    bool sendMessage(Message& msg);
    void recvMessage(Message& msg);

protected:
    void completeInstruction() override;
};

class ComputeEngine : public FunctionalUnit {
private:
    int systolic_array_size;
    int ce_cycle_factor;
    int ce_cycle_bias;
    int calculateLatency(const Inst& inst);

public:
    ComputeEngine(Simulator* sim, Logger& logger, int queue_depth, int array_size, int ce_cycle_factor,
                  int ce_cycle_bias);
    void execute() override;
    void tick() override;

private:
    int calculateGEMMLatency(const Inst& inst);
    int calculateADDLatency(const Inst& inst);
    int calculateRELULatency(const Inst& inst);
    int calculateSOFTMAXLatency(const Inst& inst);
    int calculateLAYERNORMLatency(const Inst& inst);
    int calculateXPOSELatency(const Inst& inst);
    int calculateGELULatency(const Inst& inst);
    void completeInstruction() override;
};

#endif  // __FUNC_UNIT_HPP__