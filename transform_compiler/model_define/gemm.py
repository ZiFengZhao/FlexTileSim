import os 
import sys
import argparse
from typing import List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ops.gemm import matmul
from ops.instruction_base import Instruction

LBUF_BASE      = 0x0000_1000 
LBUF_STRIDE    = 0x0010_0000 
LBUF_PER_CORE = 0x0200_0000 
DRAM_PER_CHAN = 0x0800_0000 

DRAM_ZERO      = 0x8000_0000 
TENSOR_SIZE    = 0x0001_0000

def get_dummy_quant_params():
    return {
        "re_shift": 4, "re_scale": 4,
        "shift": 36, "scale": 2
    }

def gen_inst(M: int, N: int, K: int, sa_size: int, core_id: int, num_cores: int, num_ddr_channels: int, instr_id: int):
    instrs: List[str] = []

    LBUF_BASE_CORE = LBUF_BASE + core_id * LBUF_PER_CORE
    col = -1
    if num_cores == 1 or num_cores == 2:
        col = 1
    elif num_cores == 4 or num_cores == 8:
        col = core_id % 2
    elif num_cores == 16:
        col = core_id % 4
    elif num_cores == 32 or num_cores == 64:
        col = core_id % 8
    else:
        raise ValueError("Unsupported number of cores")
    
    DRAM_BASE_CHAN = DRAM_ZERO + col * DRAM_PER_CHAN

    DRAM_BASE_A = DRAM_BASE_CHAN + TENSOR_SIZE
    DRAM_BASE_B = DRAM_BASE_CHAN + 2 * TENSOR_SIZE
    DRAM_BASE_C = DRAM_BASE_CHAN + 3 * TENSOR_SIZE

    qp = get_dummy_quant_params()
    matmul(
        addr_a_base=DRAM_BASE_A,
        addr_b_base=DRAM_BASE_B,
        addr_c_base=DRAM_BASE_C,
        Height_A = M,
        Width_A  = K,
        Height_B = K,
        Width_B  = N,
        lbuf_base   = LBUF_BASE_CORE,
        lbuf_stride = LBUF_STRIDE,
        sa_size     = sa_size,
        dram_zero   = DRAM_ZERO,
        re_shift = qp["re_shift"],
        re_scale = qp["re_scale"],
        shift    = qp["shift"],
        scale    = qp["scale"],
        instrs = instrs,
        layer_name = f"Bench_M{M}_N{N}_K{K}",
        bias_base = None,
        has_bias  = False,
        instr_id = instr_id
    )

    return instrs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FlexNPUSim Instruction Trace for GEMM")
    
    parser.add_argument("--m", type=int, required=True, help="Matrix A Height")
    parser.add_argument("--n", type=int, required=True, help="Matrix B Width")
    parser.add_argument("--k", type=int, required=True, help="Common Dimension K")
    
    parser.add_argument("--sa_size", type=int, default=32, help="Systolic Array Size (default: 32)")
    parser.add_argument("--output", type=str, default="gemm.txt", help="Output instruction file path")
    
    args = parser.parse_args()

    gen_inst(args.m, args.n, args.k, args.sa_size, args.output)