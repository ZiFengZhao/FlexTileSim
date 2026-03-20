#include "inst.hpp"

namespace InstDecode {

Inst decodeMachineCode(const INST_W& machine_code) {
    Inst inst;
    // uint128_t machine_code = hexStringToUint128(machine_code_str);
    inst.id = INST_ID(machine_code);
    inst.addr0 = ADDR0(machine_code);
    inst.addr1 = ADDR1(machine_code);

    switch (DEP(machine_code)) {
        case 0x0:
            inst.dep = DepType::NO_DEP;
            break;
        case 0x1:
            inst.dep = DepType::CONSUMER;
            break;
        case 0x2:
            inst.dep = DepType::PRODUCER;
            break;
        case 0x3:
            inst.dep = DepType::BOTH;
            break;
        default:
            break;
    }

    uint8_t funct0 = FUNCT0(machine_code);
    switch (OPCODE(machine_code)) {
        case GDMA_T:
            inst.target_unit = UnitType::GDMA;
            inst.stride = STRIDE(machine_code);
            inst.tensor_height = GDMA_IMM0(machine_code);
            inst.tensor_width = GDMA_IMM1(machine_code);
            if (funct0 == MVIN_T)
                inst.type = InstType::MVIN;
            else if (funct0 == MVOUT_T)
                inst.type = InstType::MVOUT;
            break;

        case CE_T:
            inst.target_unit = UnitType::CE;
            inst.scale0 = CE_IMM0(machine_code);
            inst.shift0 = CE_IMM1(machine_code);
            inst.scale1 = CE_IMM2(machine_code);
            inst.shift1 = CE_IMM3(machine_code);
            inst.bias = CE_FUNCT1(machine_code);
            if (funct0 == GEMM_T)
                inst.type = InstType::GEMM;
            else if (funct0 == RET_T)
                inst.type = InstType::RET;
            else if (funct0 == ADD_T)
                inst.type = InstType::ADD;
            else if (funct0 == SOFTMAX_T)
                inst.type = InstType::SOFTMAX;
            else if (funct0 == LAYERNORM_T)
                inst.type = InstType::LAYERNORM;
            else if (funct0 == GELU_T)
                inst.type = InstType::GELU;
            else if (funct0 == XPOSE_T)
                inst.type = InstType::XPOSE;
            else if (funct0 == RELU_T)
                inst.type = InstType::RELU;
            break;

        case CU_T:
            inst.target_unit = UnitType::CU;
            inst.sync_id = SYNC_ID(machine_code);
            inst.total_cores_to_sync = SYNC_CORE_COUNT(machine_code);
            if (funct0 == SYNC_T) inst.type = InstType::SYNC;
            break;

        default:
            break;
    }

    return inst;
}
}  // namespace InstDecode
