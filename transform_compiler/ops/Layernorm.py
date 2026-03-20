from math import ceil
from .instruction_base import Instruction
from . import constants as const

def LayerNorm(input_addr, output_addr, input_height, input_width, de_shift, de_scale, shift, scale, 
            lbuf_base, instrs, instr_id, layer_name=""):

    MATRIX_WIDTH = 128
    assert input_width % MATRIX_WIDTH == 0, f"input_width must be multiple of {MATRIX_WIDTH} now is {input_width}"
    once_max_width = const.AXI_MAX_WIDTH
    addr_row_stride = input_width * const.ELEMENT_BYTES
    max_tile_height = int((const.NUMBER_BANK // 2) * const.BANK_CAPACITY// addr_row_stride)
    half_bank_stride = (const.NUMBER_BANK // 2) * const.LBUF_STRIDE

    for t in range(ceil(input_height / max_tile_height)):

        t_width = once_max_width
        t_height = min(max_tile_height, input_height - t * max_tile_height)
        repeat =  (input_width // once_max_width)
        

        instrs.append(Instruction.Mvin(
            ddr_addr = input_addr + t * input_width * max_tile_height,
            ocm_addr = lbuf_base,
            Width_tensor  = t_width,
            Height_tensor = t_height*repeat,
            stride = t_width,
            dep = "10",
            instr_id = instr_id,
            comment = f"[{layer_name}] Mvin Height={t_height} Width={t_width*repeat}"))
        instr_id += 1

        for i in range(t_height):
            if i == 0 and t_height != 1:
                instrs.append(Instruction.LayerNorm(
                    addr_out = lbuf_base + half_bank_stride + i * addr_row_stride,
                    addr_in  = lbuf_base + i * addr_row_stride,
                    de_shift = de_shift,
                    de_scale = de_scale,
                    shift    = shift,
                    scale    = scale,
                    dep      = "01",
                    instr_id = instr_id,
                    comment  = f"[{layer_name}] LayerNorm height={i}"))
            elif i == t_height-1 and t_height != 1:
                instrs.append(Instruction.LayerNorm(
                    addr_out = lbuf_base + half_bank_stride + i * addr_row_stride,
                    addr_in  = lbuf_base + i * addr_row_stride,
                    de_shift = de_shift,
                    de_scale = de_scale,
                    shift    = shift,
                    scale    = scale,
                    dep      = "10",
                    instr_id = instr_id,
                    comment  = f"[{layer_name}] LayerNorm height={i}"))
            elif t_height == 1:
                instrs.append(Instruction.LayerNorm(
                    addr_out = lbuf_base + half_bank_stride + i * addr_row_stride,
                    addr_in  = lbuf_base + i * addr_row_stride,
                    de_shift = de_shift,
                    de_scale = de_scale,
                    shift    = shift,
                    scale    = scale,
                    dep      = "11",
                    instr_id = instr_id,
                    comment  = f"[{layer_name}] LayerNorm height={i}"))
            else:
                instrs.append(Instruction.LayerNorm(
                    addr_out = lbuf_base + half_bank_stride + i * addr_row_stride,
                    addr_in  = lbuf_base + i * addr_row_stride,
                    de_shift = de_shift,
                    de_scale = de_scale,
                    shift    = shift,
                    scale    = scale,
                    dep      = "00",
                    instr_id = instr_id,
                    comment  = f"[{layer_name}] LayerNorm height={i}"))
            instr_id += 1

        instrs.append(Instruction.Mvout(
            ddr_addr = output_addr + t * input_width * max_tile_height,
            ocm_addr = lbuf_base + half_bank_stride,
            Width_tensor  = t_width,
            Height_tensor = t_height * repeat,
            stride = t_width,
            dep = "01",
            instr_id = instr_id,
            comment = f"[{layer_name}] Mvout Height={t_height} Width={t_width*repeat}"))
        instr_id += 1

    return instr_id
      