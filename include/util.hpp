#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

class BaseModule;

// Event
struct Event {
    uint64_t tick;
    std::function<void()> callback;
    bool operator>(const Event& other) const { return tick > other.tick; }
};

enum class MsgType { LD_REQ, LD_RESP, ST_REQ, ST_RESP };

struct Message {
    MsgType type;
    int data_size = 1;
    int request_data_size = 1;  // bytes，used for load request
    uint32_t src_addr = 0;
    uint32_t dst_addr = 0;
};

inline int parseAddr(uint32_t addr, int num_cores) {
    const uint32_t DRAM_BASE = 0x80000000;
    const uint32_t DRAM_PER_CHAN = 0x08000000;  // 128MB per channel, total 128*16=2GB
    const uint32_t SRAM_PER_CORE = 0x02000000;  // 32MB

    bool is_ddr_req = addr & DRAM_BASE;

    int dst_tile_id = -1;

    if (num_cores == 1) {  // 2x2 mesh
        if (is_ddr_req) {
            dst_tile_id = 2;  // tile 2 is the ddr tile
        } else {
            // target core
            dst_tile_id = 0;  // only one core
        }
    } else if (num_cores == 2) {  // 2x2 mesh
        if (is_ddr_req) {
            dst_tile_id = 2;  // tile 2 is the ddr tile
        } else {
            // target core
            dst_tile_id = addr / SRAM_PER_CORE;  // mapping to tile0 or tile1
            // assert(dst_tile_id < 2);
        }
    } else if (num_cores == 4) {  // 3x3 mesh
        if (is_ddr_req) {
            // ddr is mapped to tile5 or tile6
            dst_tile_id = 6 + ((addr - DRAM_BASE) / DRAM_PER_CHAN);
            // assert(dst_tile_id == 6 || dst_tile_id == 7);
        } else {
            // core is mapped to tile0, tile1, tile3, tile4
            int virtual_tile_id = addr / SRAM_PER_CORE;
            // assert(virtual_tile_id < 4);
            dst_tile_id = virtual_tile_id / 2 * 3 + virtual_tile_id % 2;
        }
    } else if (num_cores == 8) {  // 5x5 mesh
        if (is_ddr_req) {
            // ddr is mapped to tile20, tile21
            dst_tile_id = 20 + ((addr - DRAM_BASE) / DRAM_PER_CHAN);
        } else {
            // core is mapped to tile0, tile1, tile5, tile6, tile10, tile11, tile15, tile16
            int virtual_tile_id = addr / SRAM_PER_CORE;
            // assert(virtual_tile_id < 8);
            dst_tile_id = virtual_tile_id / 2 * 5 + virtual_tile_id % 2;
        }
    } else if (num_cores == 16) {  // 5x5 mesh
        if (is_ddr_req) {
            // ddr is mapped to tile20, tile21, tile22, tile23
            dst_tile_id = 20 + ((addr - DRAM_BASE) / DRAM_PER_CHAN);
        } else {
            int virtual_tile_id = addr / SRAM_PER_CORE;
            // assert(virtual_tile_id < 16);
            dst_tile_id = virtual_tile_id / 4 * 5 + virtual_tile_id % 4;
        }
    } else if (num_cores == 32) {  // 10x10 mesh
        if (is_ddr_req) {
            // ddr is mapped to tile0, tile1, tile2, tile3, and tile90, tile91, tile92, tile93
            dst_tile_id = 40 + ((addr - DRAM_BASE) / DRAM_PER_CHAN);
        } else {
            int virtual_tile_id = addr / SRAM_PER_CORE;
            // assert(virtual_tile_id < 32);
            dst_tile_id = virtual_tile_id / 8 * 10 + virtual_tile_id % 8;
        }
    } else if (num_cores == 64) {  // 10x10 mesh
        if (is_ddr_req) {
            // ddr is mapped to tile80, tile81, tile82, tile83, tile84, tile85, tile86, tile87
            dst_tile_id = 80 + ((addr - DRAM_BASE) / DRAM_PER_CHAN);
        } else {
            int virtual_tile_id = addr / SRAM_PER_CORE;
            // assert(virtual_tile_id < 64);
            dst_tile_id = virtual_tile_id / 8 * 10 + virtual_tile_id % 8;
        }
    }

    if (dst_tile_id == -1) {
        // std::cerr << "Error: invalid address 0x" << std::hex << addr << std::dec << std::endl;
        exit(1);
    }
    return dst_tile_id;
}

struct SimProfile {
    uint64_t core_time_ns = 0;
    uint64_t noc_time_ns = 0;
    uint64_t ddr_time_ns = 0;
};

class ScopedTimer {
public:
    ScopedTimer(uint64_t* target_ns) : target(target_ns), start(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        *target += ns;
    }

private:
    uint64_t* target;
    std::chrono::high_resolution_clock::time_point start;
};

/*
 Utilities for precise clock ratio calculations.
 Approach:
 - Represent periods in integer picoseconds (ps) to avoid floating error.
 - period_ps = round(1000.0 / freq_ghz)  (since period_ps = 1e12 / (freq_ghz * 1e9) = 1000 / freq_ghz)
 - base_tick_ps = gcd(period_ps_i over all i)
 - ticks_per_base[i] = period_ps_i / base_tick_ps
*/
// Convert frequency in GHz to period in picoseconds (integer).
// Throws std::invalid_argument if freq_ghz <= 0.
inline std::uint64_t period_ps_from_ghz(double freq_ghz) {
    if (!(freq_ghz > 0.0)) throw std::invalid_argument("frequency must be > 0");
    // period_ps = 1000 / freq_ghz. Use rounding to nearest integer.
    double period_ps_d = 1000.0 / freq_ghz;
    // Guard: if extremely small or large, ensure within uint64_t
    if (period_ps_d < 1.0) period_ps_d = 1.0;
    if (period_ps_d > static_cast<double>(std::numeric_limits<std::uint64_t>::max()))
        throw std::overflow_error("period too large");
    return static_cast<std::uint64_t>(period_ps_d + 0.5);
}

// Compute gcd for uint64_t (std::gcd works with unsigned long long via <numeric> in C++17)
inline std::uint64_t gcd_u64(std::uint64_t a, std::uint64_t b) {
    while (b != 0) {
        std::uint64_t temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Compute gcd of a vector
inline std::uint64_t gcd_vector(const std::vector<std::uint64_t>& v) {
    if (v.empty()) throw std::invalid_argument("empty vector");
    std::uint64_t g = v[0];
    for (size_t i = 1; i < v.size(); ++i) {
        g = gcd_u64(g, v[i]);
        if (g == 1) return 1;  // early exit
    }
    return g;
}

// Given list of frequencies (in GHz), compute:
// - base_tick_ps: base tick in picoseconds (GCD of periods)
// - ticks_per_period: for each frequency, how many base ticks compose one of its periods
// Throws on invalid input.
inline void compute_base_tick_and_multipliers(const std::vector<double>& freqs_ghz, std::uint64_t& base_tick_ps_out,
                                              std::vector<std::uint64_t>& ticks_per_period_out,
                                              std::uint64_t min_base_ps = 10) {
    if (freqs_ghz.empty()) throw std::invalid_argument("freq list empty");
    std::vector<std::uint64_t> periods;
    periods.reserve(freqs_ghz.size());
    for (double f : freqs_ghz) {
        periods.push_back(period_ps_from_ghz(f));
    }

    std::uint64_t base = gcd_vector(periods);
    if (base < 1) base = 1;

    if (base < min_base_ps) {
        base = min_base_ps;
    }

    ticks_per_period_out.clear();
    ticks_per_period_out.reserve(periods.size());
    for (auto p : periods) {
        ticks_per_period_out.push_back((p + base / 2) / base);
    }
    base_tick_ps_out = base;
}

#endif  // __UTIL_HPP__