from math import ceil
from . import constants as const
from .instruction_base import Instruction

def matmul(addr_a_base, addr_b_base, addr_c_base,
           Width_A, Height_A, Width_B, Height_B,
           lbuf_base, lbuf_stride, sa_size, dram_zero,
           re_shift, re_scale, shift, scale,
           instrs, layer_name="",
           bias_base=None, has_bias=False, instr_id=0):
   
    # a(i,k) × b(k,j)
    I = ceil(Height_A / sa_size)   
    J = ceil(Width_B  / sa_size)  

    for i in range(I):
        for j in range(J):
            valid_a_height = min(sa_size, Height_A - i * sa_size)  
            valid_b_width = min(sa_size, Width_B  - j * sa_size)  

            instr_id = matmul_sa_tile(
                addr_a = addr_a_base + i * sa_size * Width_A * const.ELEMENT_BYTES, 
                addr_b = addr_b_base + j * sa_size * const.ELEMENT_BYTES,        
                addr_c = addr_c_base + (i * Width_B * sa_size + j * sa_size) * const.ELEMENT_BYTES, 
                a_height = valid_a_height,
                a_width = Width_A,
                b_height = Height_B,
                b_width = valid_b_width,
                Height_A = Height_A,
                Width_B = Width_B,
                sa_size = sa_size,
                lbuf_base = lbuf_base,
                lbuf_stride = lbuf_stride,
                dram_zero = dram_zero,
                re_shift = re_shift,
                re_scale = re_scale,
                shift = shift,
                scale = scale,
                instrs = instrs,
                instr_id = instr_id,
                i = i, j = j, layer_name = layer_name,
                bias_base = bias_base, has_bias = has_bias
            )
    return instr_id

def matmul_sa_tile(addr_a, addr_b, addr_c,
                   a_height,a_width,b_height,b_width,
                   Height_A, Width_B,
                   sa_size, lbuf_base, lbuf_stride,
                   dram_zero,
                   re_shift, re_scale, shift, scale,
                   instrs, instr_id, i, j, layer_name="",
                   bias_base=None, has_bias=False):
   
    assert a_width == b_height , f"assert_error: a_width:{a_width} != b_height{b_height}"
    K_tiles = ceil(a_width / sa_size)

    c_lbuf   = lbuf_base + 4 * lbuf_stride
    bias_lbuf= lbuf_base + 5 * lbuf_stride

    for k in range(K_tiles):
        a_lbuf = lbuf_base + (k % 2) * 2 * lbuf_stride
        b_lbuf = a_lbuf + lbuf_stride

        tail = (a_width % sa_size) or sa_size
        valid_k = sa_size if (k < K_tiles - 1) else tail

        '''instrs.append(Instruction.Mvin(
            ddr_addr = addr_a + k * sa_size * const.ELEMENT_BYTES,
            ocm_addr = a_lbuf,
            Width_tensor  = valid_k,
            Height_tensor = a_height,
            stride = a_width,
            dep = "10",
            instr_id = instr_id,
            comment = f"[{layer_name}] Mvin A(i={i}, k={k})"
        ))'''
        instrs.append(Instruction.Mvin(
            ddr_addr = addr_a,
            ocm_addr = a_lbuf,
            Width_tensor  = valid_k,
            Height_tensor = a_height,
            stride = a_width,
            dep = "10",
            instr_id = instr_id,
            comment = f"[{layer_name}] Mvin A(i={i}, k={k})"
        ))
        instr_id += 1

        #Transpose A cons(01)
        instrs.append(Instruction.Transpose(
            ocm_addr_in = a_lbuf,
            ocm_addr_out = a_lbuf,
            dep = "01",
            instr_id = instr_id,
            comment = f"[{layer_name}] Transpose A"
        ))
        instr_id += 1

        # Mvin B prod(10)
        '''instrs.append(Instruction.Mvin(
            ddr_addr = addr_b + k * sa_size * Width_B * const.ELEMENT_BYTES,
            ocm_addr = b_lbuf,
            Width_tensor  = b_width,
            Height_tensor = valid_k,
            stride = Width_B,
            dep = "10",
            instr_id = instr_id,
            comment = f"[{layer_name}] Mvin B(k={k}, j={j})"
        ))'''
        instrs.append(Instruction.Mvin(
            ddr_addr = addr_b,
            ocm_addr = b_lbuf,
            Width_tensor  = b_width,
            Height_tensor = valid_k,
            stride = Width_B,
            dep = "10",
            instr_id = instr_id,
            comment = f"[{layer_name}] Mvin B(k={k}, j={j})"
        ))
        instr_id += 1

        # GEMM cons(01)
        instrs.append(Instruction.GEMM(
            ocm_addr_1 = a_lbuf,
            ocm_addr_2 = b_lbuf,
            dep = "01",
            instr_id = instr_id,
            comment = f"[{layer_name}] GEMM (i={i}, j={j}, k={k})"
        ))
        instr_id += 1
    #Mvin bias prod(10)
    if has_bias:
        instrs.append(Instruction.Mvin(
            ddr_addr = (bias_base if bias_base is not None else dram_zero),
            ocm_addr = bias_lbuf,
            Width_tensor  = sa_size,   
            Height_tensor = sa_size,
            stride = Width_B,
            dep = "10",
            instr_id = instr_id,
            comment = f"[{layer_name}] Mvin bias for C(i={i}, j={j})"
        ))
        instr_id += 1

    # RET prod cons
    instrs.append(Instruction.RET(
        ocm_addr = c_lbuf,
        bias_addr = (bias_lbuf if has_bias else 0),
        bias = (1 if has_bias else 0),
        re_shift = re_shift, re_scale = re_scale,
        shift = shift,
        scale = scale,
        dep = ("11" if has_bias else "10"),
        instr_id = instr_id,
        comment = f"[{layer_name}] RET C(i={i}, j={j})"
    ))
    instr_id += 1

    
    instrs.append(Instruction.Mvout(
        ddr_addr = addr_c,
        ocm_addr = c_lbuf,
        Width_tensor  = b_width,   
        Height_tensor = a_height,
        stride = Width_B,
        dep = "01",
        instr_id = instr_id,
        comment = f"[{layer_name}] Mvout C(i={i}, j={j})"
    ))
    instr_id += 1
 
    return instr_id
