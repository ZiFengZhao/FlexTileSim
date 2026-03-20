#ifndef __INST_HPP__
#define __INST_HPP__
#include <cstdint>
#include <iostream>
#include <sstream>

using uint128_t = __uint128_t;
using INST_W = uint128_t;

enum struct InstType {
    NOP,
    MVIN,
    MVOUT,
    GEMM,
    RET,
    ADD,
    SOFTMAX,
    LAYERNORM,
    GELU,
    XPOSE,
    RELU,
    SYNC,
};

enum struct UnitType { GDMA, CE, CU };

enum struct DepType { NO_DEP, CONSUMER, PRODUCER, BOTH };

struct Inst {
    UnitType target_unit;
    InstType type = InstType::NOP;
    DepType dep = DepType::NO_DEP;
    uint64_t id = 0;
    uint32_t addr1 = 0, addr0 = 0;

    int stride = 0, tensor_height = 0, tensor_width = 0;
    int shift1 = 0;
    int shift0 = 0;
    int scale1 = 0;
    int scale0 = 0;
    int bias = 0;
    int sync_id = 0;
    int total_cores_to_sync = 0;

    std::string getType() {
        switch (type) {
            case InstType::NOP:
                return "NOP";
            case InstType::MVIN:
                return "MVIN";
            case InstType::MVOUT:
                return "MVOUT";
            case InstType::GEMM:
                return "GEMM";
            case InstType::RET:
                return "RET";
            case InstType::ADD:
                return "ADD";
            case InstType::SOFTMAX:
                return "SOFTMAX";
            case InstType::LAYERNORM:
                return "LAYERNORM";
            case InstType::GELU:
                return "GELU";
            case InstType::XPOSE:
                return "XPOSE";
            case InstType::RELU:
                return "RELU";
            case InstType::SYNC:
                return "SYNC";
            default:
                return "UNKNOWN";
        }
    }
};

#pragma pack(push, 1)
struct CoreIndexEntry {
    uint64_t instruction_offset;
    uint32_t instruction_count;
};
#pragma pack(pop)

// field extraction macros
#define OPCODE(machine_code) ((uint8_t)((machine_code) & 0xFF))
#define FUNCT0(machine_code) ((uint8_t)(((machine_code) >> 8) & 0xFF))
#define INST_ID(machine_code) ((uint8_t)((machine_code) >> 16) & 0xFF)
#define DEP(machine_code) (((machine_code) >> 24) & 0x3)
#define STRIDE(machine_code) (((machine_code) >> 26) & 0x7FFF)
#define GDMA_IMM0(machine_code) (((machine_code) >> 41) & 0x7FFF)
#define GDMA_IMM1(machine_code) (((machine_code) >> 56) & 0xFF)
#define ADDR0(machine_code) ((uint32_t)(((machine_code) >> 64) & 0xFFFFFFFF))
#define ADDR1(machine_code) ((uint32_t)(((machine_code) >> 96) & 0xFFFFFFFF))
#define CE_IMM0(machine_code) ((uint8_t)(((machine_code) >> 26) & 0xFF))
#define CE_IMM1(machine_code) ((uint8_t)(((machine_code) >> 34) & 0xFF))
#define CE_IMM2(machine_code) ((uint8_t)(((machine_code) >> 42) & 0xFF))
#define CE_IMM3(machine_code) ((uint8_t)(((machine_code) >> 50) & 0xFF))
#define CE_FUNCT1(machine_code) (((machine_code) >> 58) & 0x3F)

#define SYNC_CORE_COUNT(machine_code) ((uint32_t)(((machine_code) >> 64) & 0xFFFFFFFF))
#define SYNC_ID(machine_code) ((uint32_t)(((machine_code) >> 96) & 0xFFFFFFFF))

// opcode types
#define GDMA_T 0x01
#define CE_T 0x02
#define CU_T 0x03
// gdma instruction types
#define MVIN_T 0x00
#define MVOUT_T 0x01
// ce instruction types
#define GEMM_T 0x00
#define RET_T 0x01
#define SOFTMAX_T 0x02
#define LAYERNORM_T 0x03
#define ADD_T 0x04
#define GELU_T 0x05
#define XPOSE_T 0x06
#define RELU_T 0x07
// cu instruction types
#define SYNC_T 0x00
#define NOP_T 0x01

namespace InstDecode {
Inst decodeMachineCode(const INST_W& machine_code_str);

}  // namespace InstDecode

#endif  // __INST_HPP__