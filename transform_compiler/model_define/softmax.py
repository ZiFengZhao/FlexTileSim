import os 
import sys
from typing import List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ops.instruction_base import Instruction

LBUF_BASE      = 0x0000_1000 
LBUF_STRIDE    = 0x0010_0000 
LBUF_PER_CORE = 0x0200_0000  
DRAM_PER_CHAN = 0x0800_0000  

DRAM_ZERO      = 0x8000_0000 
TENSOR_SIZE    = 0x0001_0000

def gen_softmax_inst(core_id: int, seq_size: int, instr_id: int):
    instrs: List[str] = []
    num = seq_size // 32
    LBUF_BASE_CORE = LBUF_BASE + core_id * LBUF_PER_CORE
    
    for i in range(num):
        instrs.append(Instruction.Softmax(
            addr_out=LBUF_BASE_CORE,
            addr_in=LBUF_BASE_CORE,
            de_scale=0,
            de_shift=0,
            shift=0,
            scale=0,
            dep = "00",
            instr_id=instr_id + i,
            comment="Softmax"
        ))
    return instrs