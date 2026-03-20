from .instruction_base import Instruction
from .relu_gemm import matmul_relu
from .gemm import matmul
from typing import List


def output_layer(addr_input, addr_output, addr_pooler_output,
                addr_pooler_weight,addr_pooler_bias,
                addr_logits_weight,addr_logits_bias,
                input_width, input_height, output_width,
                pooler_shift, pooler_scale, pooler_re_shift, pooler_re_scale,
                logits_shift, logits_scale, logits_re_shift, logits_re_scale,
                lbuf_base, lbuf_stride, sa_size, dram_zero,
                instrs, layer_name="")-> List[str]:
    
    instrs: List[str] = []

    instr_id = 0


    instr_id = matmul_relu(addr_a_base = addr_input, addr_b_base = addr_pooler_weight, addr_c_base = addr_pooler_output,
           Width_A=input_width, Height_A=input_height, Width_B=input_width, Height_B=input_width,
           lbuf_base=lbuf_base, lbuf_stride=lbuf_stride, sa_size=sa_size, dram_zero=dram_zero,
           re_shift=pooler_re_shift, re_scale=pooler_re_scale, shift=pooler_shift, scale=pooler_scale,
           instrs=instrs, layer_name=layer_name,
           bias_base=addr_pooler_bias, has_bias=True)
    instrs.append(Instruction.End())

    instr_id = matmul(addr_a_base=addr_pooler_output, addr_b_base=addr_logits_weight, addr_c_base=addr_output,
           Width_A=input_width, Height_A=input_height, Width_B=output_width, Height_B=input_width,
           lbuf_base=lbuf_base, lbuf_stride=lbuf_stride, sa_size=sa_size, dram_zero=dram_zero,
           re_shift=logits_re_shift, re_scale=logits_re_scale, shift=logits_shift, scale=logits_scale,
           instrs=instrs, layer_name=layer_name,
           bias_base=addr_logits_bias, has_bias=True)

    instrs.append(Instruction.End())

    return instrs
