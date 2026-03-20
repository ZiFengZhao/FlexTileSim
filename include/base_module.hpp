#ifndef __BASE_MODULE_HPP__
#define __BASE_MODULE_HPP__

#include <map>

#include "../external/Booksim2/src/booksim.hpp"
#include "../external/Booksim2/src/booksim_config.hpp"
#include "../external/Booksim2/src/trafficmanager.hpp"
#include "config.hpp"
#include "logger.hpp"
#include "util.hpp"
class Simulator;

class BaseModule {
protected:
    Simulator* sim;
    Config cfg;
    Logger& logger;

    int m_id;
    uint64_t m_cycle;
    uint64_t m_ticks_per_cycle = 0;

public:
    BaseModule(Simulator* sim, const Config& cfg, Logger& logger, uint64_t ticks_per_cycle, int id)
        : sim(sim), cfg(cfg), logger(logger), m_id(id), m_cycle(0), m_ticks_per_cycle(ticks_per_cycle) {};
    virtual ~BaseModule() = default;

    virtual void tick() = 0;
    virtual void scheduleNextEvent() = 0;
    virtual void sendMsg(Message& msg) = 0;
    virtual bool recvMsg(Message& msg) = 0;

    uint64_t getCycle() const { return m_cycle; }

    int getId() const { return m_id; }
    virtual TrafficManager* getTrafficManager() = 0;
};

#endif  // __BASE_MODULE_HPP__